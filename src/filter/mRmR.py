from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from ..evaluation.logistic_regression import evaluate_logistic_regression_with_given_features
from ..precomputation import FeatureCorrelationMatrix

# ---------------------------------------------------------------------------
# Module-level logger (duplicate-handler guard)
# ---------------------------------------------------------------------------
logger = logging.getLogger("mRmRSelector")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# SelectionResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class SelectionResult:
    """Holds the output of a completed :py:meth:`mRmRSelector.forward_selection` run."""

    selected_features: list[str]
    performance_history: list[dict]        # one dict per step where evaluation ran
    stopping_reason: str                   # "max_features_reached" | "metric_threshold_reached" | "no_features_remaining"
    n_steps: int
    final_metrics: dict | None             # last entry of performance_history, or None
    selection_time_seconds: float          # lazy on-demand compute time only
    evaluation_time_seconds: float | None  # None if evaluation not performed

    def __str__(self) -> str:
        lines = [
            "SelectionResult:",
            f"  features selected : {self.n_steps}",
            f"  stopping reason   : {self.stopping_reason}",
            f"  selection time    : {self.selection_time_seconds:.3f}s",
        ]
        if self.evaluation_time_seconds is not None:
            lines.append(f"  evaluation time   : {self.evaluation_time_seconds:.3f}s")
        if self.final_metrics is not None:
            acc = self.final_metrics.get("accuracy", "n/a")
            f1 = (
                self.final_metrics.get("macro avg", {}).get("f1-score", "n/a")
                if isinstance(self.final_metrics.get("macro avg"), dict)
                else "n/a"
            )
            lines.append(f"  final accuracy    : {acc}")
            lines.append(f"  final macro F1    : {f1}")
        lines.append(f"  features          : {self.selected_features}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# mRmRSelector
# ---------------------------------------------------------------------------
class mRmRSelector:
    """
    Minimum Redundancy Maximum Relevance (mRmR) feature selector.

    Supports three correlation-based relevance methods (``'pearson'``,
    ``'kendall'``, ``'spearman'``).  Redundancy is the **mean** absolute
    pairwise correlation between a candidate and the already-selected set,
    computed with the **same** method used for relevance.

    The internal score matrix stores both feature-feature and
    feature-target scores under the actual column names from the loaded
    matrix (or ``y_train.name`` for the target).  No sentinel rename of
    ``__target__`` is applied — the target column is always addressed by
    ``y_train.name``.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.  NaNs are imputed with 0 internally.
    y_train : pd.Series
        Training target vector.  **Must have a ``.name`` attribute** that
        identifies the target column (e.g. ``'is_lumA'`` or ``'Lympho'``).
    relevance_method : str
        Correlation method for both relevance and redundancy.
        One of ``'pearson'`` (default), ``'kendall'``, ``'spearman'``.
    mrmr_score_method : str
        How to combine relevance and redundancy.
        ``'difference'`` (default) or ``'ratio'``.
    correlation_filepath : str | None
        Path to a pre-computed correlation matrix CSV that **includes** the
        target column under ``y_train.name``.  If the file does not exist it
        will be computed (features + target) and saved here.  ``None``
        activates lazy on-demand scoring.
    gene_expression_df, train_labels_df, val_labels_df : pd.DataFrame | None
        Domain-specific data forwarded to the LR evaluation helper.
        Evaluation is skipped if any of the three is ``None``.
    lr_C : float
        Regularisation parameter forwarded to the LR helper.
    random_seed : int | None
        Forwarded to the LR helper for reproducibility.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        relevance_method: str = "pearson",
        mrmr_score_method: str = "difference",
        correlation_filepath: str | None = None,
        # --- evaluation params (P0: domain-specific) ---
        gene_expression_df: pd.DataFrame | None = None,
        train_labels_df: pd.DataFrame | None = None,
        val_labels_df: pd.DataFrame | None = None,
        lr_C: float = np.inf,
        random_seed: int | None = None,
    ) -> None:
        _valid_methods = {"pearson", "kendall", "spearman"}
        if relevance_method not in _valid_methods:
            raise ValueError(
                f"relevance_method must be one of {_valid_methods}; got {relevance_method!r}"
            )
        if mrmr_score_method not in {"difference", "ratio"}:
            raise ValueError(
                "mrmr_score_method must be 'difference' or 'ratio'; "
                f"got {mrmr_score_method!r}"
            )
        if y_train.name is None:
            raise ValueError(
                "y_train must have a .name (e.g. pd.Series(..., name='is_lumA')). "
                "The name is used to locate the target column in the score matrix."
            )

        self.X_train = X_train
        self.y_train = y_train
        self.relevance_method = relevance_method
        self.mrmr_score_method = mrmr_score_method
        self.correlation_filepath = correlation_filepath

        self.gene_expression_df = gene_expression_df
        self.train_labels_df = train_labels_df
        self.val_labels_df = val_labels_df
        self.lr_C = lr_C
        self.random_seed = random_seed

        # NaN imputation — done once, reused everywhere
        self._X_filled: pd.DataFrame = X_train.fillna(0)

        # Target column name — always y_train.name, never a synthetic sentinel
        self._target_col: str = str(y_train.name)

        self._task_type: str = self._detect_task_type()

        # Score matrix — None until _precompute_matrix or first _get_score call
        self._score_matrix: pd.DataFrame | None = None

        logger.info(
            "mRmRSelector initialised | task_type=%s | target_col=%s | "
            "relevance_method=%s | mrmr_score_method=%s",
            self._task_type,
            self._target_col,
            self.relevance_method,
            self.mrmr_score_method,
        )

    # ------------------------------------------------------------------
    # §5.2  Task-type detection
    # ------------------------------------------------------------------
    def _detect_task_type(self) -> str:
        if self.y_train.nunique() < 0.1 * len(self.y_train):
            return "classification"
        return "regression"

    # ------------------------------------------------------------------
    # §5.3  Precomputation (Mode A — correlation_filepath provided)
    # ------------------------------------------------------------------
    def _precompute_matrix(self) -> None:
        if self.correlation_filepath is None:
            self._score_matrix = None
            logger.info("No correlation filepath — lazy on-demand scoring active")
            return

        fcm = FeatureCorrelationMatrix()

        if os.path.exists(self.correlation_filepath):
            fcm.load_correlation_matrix(self.correlation_filepath)
            self._score_matrix = fcm.correlation_matrix.abs()
            logger.info(
                "Loaded correlation matrix from %s %s",
                self.correlation_filepath,
                self._score_matrix.shape,
            )
        else:
            combined = pd.concat(
                [self._X_filled, self.y_train], axis=1
            )
            FeatureCorrelationMatrix.compute_correlation_matrix(
                combined,
                method=self.relevance_method,
                filepath=self.correlation_filepath,
            )
            fcm.load_correlation_matrix(self.correlation_filepath)
            self._score_matrix = fcm.correlation_matrix.abs()
            logger.info(
                "Computed and saved correlation matrix to %s %s",
                self.correlation_filepath,
                self._score_matrix.shape,
            )

        # Warn if the target column is missing; fill lazily
        if self._target_col not in self._score_matrix.columns:
            logger.warning(
                "Target column '%s' absent in loaded matrix — target relevance "
                "will be computed lazily",
                self._target_col,
            )
            self._score_matrix[self._target_col] = np.nan
            self._score_matrix.loc[self._target_col] = np.nan

    # ------------------------------------------------------------------
    # §5.4  Unified score access
    # ------------------------------------------------------------------
    def _lazy_init_matrix(self) -> None:
        """Initialise the score matrix to NaN on first Mode-B access."""
        cols = list(self._X_filled.columns) + [self._target_col]
        self._score_matrix = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
        np.fill_diagonal(self._score_matrix.values, 1.0)

    def _compute_raw_score(self, col_a: str, col_b: str) -> float:
        """Compute absolute correlation between two columns (or target)."""
        series_a = (
            self.y_train if col_a == self._target_col else self._X_filled[col_a]
        )
        series_b = (
            self.y_train if col_b == self._target_col else self._X_filled[col_b]
        )
        if series_a.std() == 0 or series_b.std() == 0:
            return 0.0
        r = series_a.corr(series_b, method=self.relevance_method)
        return 0.0 if pd.isna(r) else abs(r)

    def _get_score(self, col_a: str, col_b: str) -> float:
        """Return cached absolute correlation; compute and cache on miss."""
        # Lazy init (Mode B)
        if self._score_matrix is None:
            self._lazy_init_matrix()

        # Bounds-check — warn and compute live rather than KeyError
        if col_a not in self._score_matrix.index or col_b not in self._score_matrix.columns:
            logger.warning(
                "Feature not found in score matrix: ('%s', '%s') — computing live",
                col_a,
                col_b,
            )
            return self._compute_raw_score(col_a, col_b)

        val = self._score_matrix.loc[col_a, col_b]
        if not pd.isna(val):
            return float(val)

        raw = self._compute_raw_score(col_a, col_b)
        self._score_matrix.loc[col_a, col_b] = raw
        self._score_matrix.loc[col_b, col_a] = raw
        return raw

    # ------------------------------------------------------------------
    # §5.5  Relevance
    # ------------------------------------------------------------------
    def _get_relevance(self, feature: str) -> float:
        return self._get_score(feature, self._target_col)

    # ------------------------------------------------------------------
    # §5.6  Redundancy — vectorised
    # ------------------------------------------------------------------
    def _get_redundancy(self, candidate: str, selected: list[str]) -> float:
        if not selected:
            return 0.0
        # Populate any cache misses
        for s in selected:
            self._get_score(candidate, s)
        return float(self._score_matrix.loc[candidate, selected].mean())

    # ------------------------------------------------------------------
    # §5.7  mRmR score
    # ------------------------------------------------------------------
    def _compute_mrmr_score(self, relevance: float, redundancy: float) -> float:
        if self.mrmr_score_method == "difference":
            return relevance - redundancy
        # ratio
        return relevance / (redundancy + 1e-8)

    # ------------------------------------------------------------------
    # §5.8  Evaluation step
    # ------------------------------------------------------------------
    def _evaluate_step(self, selected_features: list[str]) -> dict:
        _, report = evaluate_logistic_regression_with_given_features(
            gene_expression_df=self.gene_expression_df,
            train_labels_df=self.train_labels_df,
            val_labels_df=self.val_labels_df,
            feature_list=selected_features,
            random_seed=self.random_seed,
            output_dict=True,
            lr_C=self.lr_C,
        )
        return report

    # ------------------------------------------------------------------
    # §5.9  Forward selection
    # ------------------------------------------------------------------
    def forward_selection(
        self,
        n_features_to_select: int,
        stopping_metric: str | None = None,
        stopping_threshold: float | None = None,
    ) -> SelectionResult:
        """
        Greedy forward mRmR feature selection.

        Parameters
        ----------
        n_features_to_select : int
            Maximum number of features to select.
        stopping_metric : str | None
            Metric key from the classification report to monitor for early
            stopping (e.g. ``'accuracy'`` or ``'macro avg_f1-score'``).
            Ignored when evaluation data is not provided.
        stopping_threshold : float | None
            Stop early when *stopping_metric* ≥ *stopping_threshold*.

        Returns
        -------
        SelectionResult
        """
        # ── PRE-SELECTION (not timed) ──────────────────────────────────
        self._precompute_matrix()
        logger.info(
            "Starting forward selection for up to %d features", n_features_to_select
        )

        # ── SELECTION LOOP ─────────────────────────────────────────────
        selected: list[str] = []
        unselected: list[str] = list(self.X_train.columns)
        t_select: float = 0.0
        t_eval: float = 0.0
        stopping_reason: str | None = None
        performance_history: list[dict] = []

        _eval_enabled = (
            self.gene_expression_df is not None
            and self.train_labels_df is not None
            and self.val_labels_df is not None
        )

        for step in range(n_features_to_select):
            # (a) All features already selected?
            if not unselected:
                stopping_reason = "no_features_remaining"
                logger.info("No features remaining — stopping at step %d", step)
                break

            # (b) TIMED: mRmR scoring + best-feature selection
            t0 = perf_counter()
            scores = {
                c: self._compute_mrmr_score(
                    self._get_relevance(c),
                    self._get_redundancy(c, selected),
                )
                for c in unselected
            }
            best = max(scores, key=scores.__getitem__)
            selected.append(best)
            unselected.remove(best)
            t_select += perf_counter() - t0

            logger.info(
                "Step %d: selected '%s' (%d total)", step + 1, best, len(selected)
            )

            # (c) Evaluation (optional)
            if _eval_enabled:
                t0 = perf_counter()
                metrics = self._evaluate_step(selected)
                t_eval += perf_counter() - t0
                performance_history.append(metrics)

                acc = metrics.get("accuracy", float("nan"))
                f1 = (
                    metrics.get("macro avg", {}).get("f1-score", float("nan"))
                    if isinstance(metrics.get("macro avg"), dict)
                    else float("nan")
                )
                logger.info(
                    "Step %d eval — accuracy: %.4f  macro F1: %.4f",
                    step + 1,
                    acc,
                    f1,
                )

                # Early stopping
                if stopping_metric is not None and stopping_threshold is not None:
                    metric_value = self._resolve_metric(metrics, stopping_metric)
                    if metric_value >= stopping_threshold:
                        stopping_reason = "metric_threshold_reached"
                        logger.info(
                            "Early stopping: %s=%.4f >= threshold=%.4f",
                            stopping_metric,
                            metric_value,
                            stopping_threshold,
                        )
                        break

        if stopping_reason is None:
            stopping_reason = "max_features_reached"

        # ── POST-SELECTION (not timed) ─────────────────────────────────
        logger.info(
            "Selection complete: %d features, reason=%s",
            len(selected),
            stopping_reason,
        )
        logger.info("Selection time: %.3fs", t_select)
        if _eval_enabled:
            logger.info("Evaluation time: %.3fs", t_eval)

        return SelectionResult(
            selected_features=selected,
            performance_history=performance_history,
            stopping_reason=stopping_reason,
            n_steps=len(selected),
            final_metrics=performance_history[-1] if performance_history else None,
            selection_time_seconds=t_select,
            evaluation_time_seconds=t_eval if _eval_enabled else None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_metric(report: dict, metric_key: str) -> float:
        """
        Resolve a dotted metric key from a classification_report dict.

        Supported formats
        -----------------
        * ``'accuracy'``
        * ``'macro avg_f1-score'``  →  ``report['macro avg']['f1-score']``
        * any single-level key present directly in *report*
        """
        if metric_key in report:
            v = report[metric_key]
            return float(v) if not isinstance(v, dict) else float("nan")

        # Try split on first underscore to handle 'macro avg_f1-score' style
        if "_" in metric_key:
            idx = metric_key.index("_")
            outer, inner = metric_key[:idx], metric_key[idx + 1:]
            if outer in report and isinstance(report[outer], dict):
                return float(report[outer].get(inner, float("nan")))

        logger.warning(
            "stopping_metric '%s' not found in classification report — "
            "early stopping disabled for this step",
            metric_key,
        )
        return float("-inf")
