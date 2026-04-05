from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from time import perf_counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ..evaluation.logistic_regression import evaluate_logistic_regression_with_given_features
from ..precomputation import FeatureCorrelationMatrix, FeatureRelevanceScores

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
    """Holds the output of a completed :py:meth:`mRmRSelector.forward_selection` run.

    Each entry in ``performance_history`` contains a ``'step'`` key (1-based)
    equal to the number of features selected at that evaluation point.
    """

    selected_features: list[str]
    performance_history: list[dict]        # one dict per step where evaluation ran; each has 'step' key
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

    Supports correlation-based relevance methods (``'pearson'``, ``'kendall'``,
    ``'spearman'``) as well as non-correlation methods for relevance only
    (``'mutual_information'``, ``'random_forest'``, ``'f_statistic'``).

    When ``relevance_method`` is correlation-based, **redundancy uses the same
    method** (``redundancy_method`` must match; enforced at construction per R-10).
    When ``relevance_method`` is non-correlation, ``redundancy_method`` governs
    the separate pairwise correlation matrix used for redundancy scoring.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.  NaNs are imputed with 0 internally.
    y_train : pd.Series
        Training target vector.  **Must have a ``.name`` attribute**.
    relevance_method : str
        Method for feature→target relevance.
        One of ``'pearson'`` (default), ``'kendall'``, ``'spearman'``,
        ``'mutual_information'``, ``'random_forest'``, ``'f_statistic'``.
    mrmr_score_method : str
        How to combine relevance and redundancy.
        ``'difference'`` (default) or ``'ratio'``.
    redundancy_method : str
        Correlation method for feature→feature redundancy.
        One of ``'pearson'`` (default), ``'kendall'``, ``'spearman'``.
        Must equal ``relevance_method`` when the latter is correlation-based (R-10).
    correlation_filepath : str | None
        Path to a pre-computed correlation matrix CSV.  For non-correlation
        relevance methods this matrix is used for redundancy only.
    relevance_scores_filepath : str | None
        Path to a pre-computed relevance scores CSV (non-correlation methods).
        If the file does not exist it will be computed and saved here.
        ``None`` activates lazy per-feature computation (Mode B).
    gene_expression_df, train_labels_df, val_labels_df : pd.DataFrame | None
        Domain-specific data forwarded to the LR classification evaluation helper.
        Evaluation is skipped if any of the three is ``None``.
    X_val : pd.DataFrame | None
        Generic validation feature matrix for regression evaluation.
    y_val : pd.Series | None
        Generic validation target vector for regression evaluation.
    lr_C : float
        Regularisation parameter forwarded to the LR helper.
    random_seed : int | None
        Forwarded to evaluation helpers and non-correlation relevance methods.
    """

    _CORR_METHODS = {"pearson", "kendall", "spearman"}
    _NON_CORR_METHODS = {"mutual_information", "random_forest", "f_statistic"}

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        relevance_method: str = "pearson",
        mrmr_score_method: str = "difference",
        # G4 — decoupled redundancy method
        redundancy_method: str = "pearson",
        # existing params
        correlation_filepath: str | None = None,
        # G4 — non-correlation relevance scores filepath
        relevance_scores_filepath: str | None = None,
        # --- classification evaluation params (P0: domain-specific) ---
        gene_expression_df: pd.DataFrame | None = None,
        train_labels_df: pd.DataFrame | None = None,
        val_labels_df: pd.DataFrame | None = None,
        lr_C: float = np.inf,
        random_seed: int | None = None,
        # G3 — generic regression evaluation params
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        _all_rel = self._CORR_METHODS | self._NON_CORR_METHODS
        if relevance_method not in _all_rel:
            raise ValueError(
                f"relevance_method must be one of {_all_rel}; got {relevance_method!r}"
            )
        if redundancy_method not in self._CORR_METHODS:
            raise ValueError(
                "redundancy_method must be 'pearson', 'kendall', or 'spearman'; "
                f"got {redundancy_method!r}"
            )
        # R-10: when relevance is correlation, redundancy must match
        if relevance_method in self._CORR_METHODS and redundancy_method != relevance_method:
            raise ValueError(
                f"When relevance_method is '{relevance_method}', "
                f"redundancy_method must match. Got '{redundancy_method}'."
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
        self.redundancy_method = redundancy_method
        self.correlation_filepath = correlation_filepath
        self.relevance_scores_filepath = relevance_scores_filepath

        self.gene_expression_df = gene_expression_df
        self.train_labels_df = train_labels_df
        self.val_labels_df = val_labels_df
        self.lr_C = lr_C
        self.random_seed = random_seed

        # NaN imputation — done once, reused everywhere
        self._X_filled: pd.DataFrame = X_train.fillna(0)

        # G3 — regression evaluation data
        self._X_val_filled: pd.DataFrame | None = X_val.fillna(0) if X_val is not None else None
        self._y_val: pd.Series | None = y_val

        # Target column name — always y_train.name, never a synthetic sentinel
        self._target_col: str = str(y_train.name)

        self._task_type: str = self._detect_task_type()

        # Score matrix — None until _precompute_matrix or first _get_score call
        self._score_matrix: pd.DataFrame | None = None

        # G4 — non-correlation relevance score stores
        self._relevance_scores: pd.Series | None = None   # Mode A: full precomputed Series
        self._relevance_cache: dict | None = None         # Mode B: lazy hot-cache (random_forest bulk-inits all features on first call)

        logger.info(
            "mRmRSelector initialised | task_type=%s | target_col=%s | "
            "relevance_method=%s | redundancy_method=%s | mrmr_score_method=%s",
            self._task_type,
            self._target_col,
            self.relevance_method,
            self.redundancy_method,
            self.mrmr_score_method,
        )

    # ------------------------------------------------------------------
    # Task-type detection
    # ------------------------------------------------------------------
    def _detect_task_type(self) -> str:
        if self.y_train.nunique() < 0.1 * len(self.y_train):
            return "classification"
        return "regression"

    # ------------------------------------------------------------------
    # Precomputation (Mode A)
    # ------------------------------------------------------------------
    def _precompute_matrix(self) -> None:
        """Load or compute the correlation matrix used for redundancy (and relevance when
        ``relevance_method`` is correlation-based)."""
        if self.relevance_method in self._NON_CORR_METHODS:
            # G4 — non-correlation: precompute relevance scores separately,
            # and load/compute the redundancy-only correlation matrix.
            self._precompute_relevance_scores()
            self._precompute_redundancy_matrix()
        else:
            # P0 path: combined [X ‖ y] matrix for both relevance and redundancy
            self._precompute_combined_matrix()

    def _precompute_combined_matrix(self) -> None:
        """P0 path — combined correlation matrix for correlation-based relevance."""
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
            combined = pd.concat([self._X_filled, self.y_train], axis=1)
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

    def _precompute_redundancy_matrix(self) -> None:
        """G4 — load or compute the redundancy-only pairwise correlation matrix."""
        if self.correlation_filepath is None:
            self._score_matrix = None
            logger.info(
                "No correlation filepath for redundancy — lazy on-demand scoring active"
            )
            return

        fcm = FeatureCorrelationMatrix()

        if os.path.exists(self.correlation_filepath):
            fcm.load_correlation_matrix(self.correlation_filepath)
            self._score_matrix = fcm.correlation_matrix.abs()
            logger.info(
                "Loaded redundancy matrix from %s %s",
                self.correlation_filepath,
                self._score_matrix.shape,
            )
        else:
            FeatureCorrelationMatrix.compute_correlation_matrix(
                self._X_filled,
                method=self.redundancy_method,
                filepath=self.correlation_filepath,
            )
            fcm.load_correlation_matrix(self.correlation_filepath)
            self._score_matrix = fcm.correlation_matrix.abs()
            logger.info(
                "Computed and saved redundancy matrix to %s %s",
                self.correlation_filepath,
                self._score_matrix.shape,
            )

    def _precompute_relevance_scores(self) -> None:
        """G4 Mode A — precompute/load relevance scores from file."""
        if self.relevance_scores_filepath is None:
            # Mode B — lazy hot-cache; initialise container if needed
            if self._relevance_cache is None:
                self._relevance_cache = {}
            logger.info(
                "No relevance_scores_filepath — lazy per-feature relevance computation active"
            )
            return

        frs = FeatureRelevanceScores()

        if os.path.exists(self.relevance_scores_filepath):
            frs.load_relevance_scores(self.relevance_scores_filepath)
            self._relevance_scores = frs.relevance_scores
            logger.info(
                "Loaded relevance scores from %s (%d features)",
                self.relevance_scores_filepath,
                len(self._relevance_scores),
            )
        else:
            FeatureRelevanceScores.compute_relevance_scores(
                X=self._X_filled,
                y=self.y_train,
                method=self.relevance_method,
                task_type=self._task_type,
                filepath=self.relevance_scores_filepath,
                random_seed=self.random_seed,
            )
            frs.load_relevance_scores(self.relevance_scores_filepath)
            self._relevance_scores = frs.relevance_scores
            logger.info(
                "Computed and saved relevance scores to %s",
                self.relevance_scores_filepath,
            )

    # ------------------------------------------------------------------
    # Unified score access (correlation matrix)
    # ------------------------------------------------------------------
    def _lazy_init_matrix(self) -> None:
        """Initialise the score matrix to NaN on first Mode-B access."""
        cols = list(self._X_filled.columns) + [self._target_col]
        self._score_matrix = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
        np.fill_diagonal(self._score_matrix.values, 1.0)

    def _compute_raw_score(self, col_a: str, col_b: str, method: str | None = None) -> float:
        """Compute absolute correlation between two columns (or target)."""
        _method = method if method is not None else self.relevance_method
        series_a = (
            self.y_train if col_a == self._target_col else self._X_filled[col_a]
        )
        series_b = (
            self.y_train if col_b == self._target_col else self._X_filled[col_b]
        )
        if series_a.std() == 0 or series_b.std() == 0:
            return 0.0
        r = series_a.corr(series_b, method=_method)
        return 0.0 if pd.isna(r) else abs(r)

    def _get_score(self, col_a: str, col_b: str, method: str | None = None) -> float:
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
            return self._compute_raw_score(col_a, col_b, method=method)

        val = self._score_matrix.loc[col_a, col_b]
        if not pd.isna(val):
            return float(val)

        raw = self._compute_raw_score(col_a, col_b, method=method)
        self._score_matrix.loc[col_a, col_b] = raw
        self._score_matrix.loc[col_b, col_a] = raw
        return raw

    # ------------------------------------------------------------------
    # G4 — single-feature lazy relevance computation (Mode B)
    # ------------------------------------------------------------------
    def _compute_raw_relevance(self, feature: str) -> float:
        """Compute relevance score for a single feature using the non-correlation method.

        Used only in Mode B (no ``relevance_scores_filepath``).
        """
        from sklearn.feature_selection import (
            mutual_info_classif,
            mutual_info_regression,
            f_classif,
            f_regression,
        )
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        x = self._X_filled[[feature]]  # 2-D slice required by sklearn
        y = self.y_train

        if self.relevance_method == "mutual_information":
            fn = (
                mutual_info_classif
                if self._task_type == "classification"
                else mutual_info_regression
            )
            score = fn(x, y, random_state=self.random_seed)[0]

        elif self.relevance_method == "f_statistic":
            fn = f_classif if self._task_type == "classification" else f_regression
            score = fn(x, y)[0][0]  # returns (F_scores, p_values)

        elif self.relevance_method == "random_forest":
            Cls = (
                RandomForestClassifier
                if self._task_type == "classification"
                else RandomForestRegressor
            )
            rf = Cls(n_estimators=100, random_state=self.random_seed)
            # Fit once on the full feature matrix and populate the entire cache.
            # This avoids fitting one RF per feature (O(n_features) fits) and
            # instead performs a single fit — identical to the disk-cache path.
            rf.fit(self._X_filled, y)
            importances = rf.feature_importances_
            for feat, imp in zip(self._X_filled.columns, importances):
                self._relevance_cache[feat] = (
                    0.0 if (pd.isna(imp) or np.isinf(imp)) else float(imp)
                )
            return self._relevance_cache.get(feature, 0.0)

        else:
            raise ValueError(
                f"_compute_raw_relevance called with correlation method {self.relevance_method!r}"
            )

        return 0.0 if (pd.isna(score) or np.isinf(score)) else float(score)

    # ------------------------------------------------------------------
    # G4 — relevance dispatch
    # ------------------------------------------------------------------
    def _get_relevance(self, feature: str) -> float:
        if self.relevance_method in self._CORR_METHODS:
            # P0 path: target column in combined matrix
            return self._get_score(feature, self._target_col)

        # Non-correlation path
        if self._relevance_scores is not None:
            # Mode A: fully populated Series
            return float(self._relevance_scores.get(feature, 0.0))

        # Mode B: lazy hot-cache
        if self._relevance_cache is None:
            self._relevance_cache = {}
        if feature not in self._relevance_cache:
            self._relevance_cache[feature] = self._compute_raw_relevance(feature)
        return self._relevance_cache[feature]

    # ------------------------------------------------------------------
    # G4 — redundancy with explicit method forwarding
    # ------------------------------------------------------------------
    def _get_redundancy(self, candidate: str, selected: list[str]) -> float:
        if not selected:
            return 0.0
        # Populate any cache misses using redundancy_method
        for s in selected:
            self._get_score(candidate, s, method=self.redundancy_method)
        return float(self._score_matrix.loc[candidate, selected].mean())

    # ------------------------------------------------------------------
    # mRmR score
    # ------------------------------------------------------------------
    def _compute_mrmr_score(self, relevance: float, redundancy: float) -> float:
        if self.mrmr_score_method == "difference":
            return relevance - redundancy
        # ratio
        return relevance / (redundancy + 1e-8)

    # ------------------------------------------------------------------
    # G3 — evaluation step (dispatch on task type)
    # ------------------------------------------------------------------
    def _evaluate_step(self, selected_features: list[str]) -> dict:
        if self._task_type == "classification":
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
        else:  # regression
            from ..LR_regression_baseline import train_and_evaluate_linear_regression
            _, metrics = train_and_evaluate_linear_regression(
                X_train=self._X_filled[selected_features],
                y_train=self.y_train,
                X_val=self._X_val_filled[selected_features],
                y_val=self._y_val,
                n_features=len(selected_features),
                random_seed=self.random_seed,
            )
            return metrics

    # ------------------------------------------------------------------
    # G1 — forward selection with eval_every_k + step injection
    # ------------------------------------------------------------------
    def forward_selection(
        self,
        n_features_to_select: int,
        stopping_metric: str | None = None,
        stopping_threshold: float | None = None,
        eval_every_k: int = 1,              # G1 — P1
    ) -> SelectionResult:
        """
        Greedy forward mRmR feature selection.

        Parameters
        ----------
        n_features_to_select : int
            Maximum number of features to select.
        stopping_metric : str | None
            Metric key from the evaluation report to monitor for early stopping.
            For classification: e.g. ``'accuracy'`` or ``'macro avg_f1-score'``.
            For regression: e.g. ``'r2'``.
            Ignored when evaluation data is not provided.
        stopping_threshold : float | None
            Stop early when *stopping_metric* ≥ *stopping_threshold*.
        eval_every_k : int
            Evaluate every *k*-th step (default ``1`` = every step).
            The final step is always evaluated regardless.

        Returns
        -------
        SelectionResult
        """
        if eval_every_k < 1:
            raise ValueError(f"eval_every_k must be >= 1; got {eval_every_k}")

        # ── PRE-SELECTION (not timed) ──────────────────────────────────
        self._precompute_matrix()
        logger.info(
            "Starting forward selection for up to %d features (eval_every_k=%d)",
            n_features_to_select,
            eval_every_k,
        )

        # ── SELECTION LOOP ─────────────────────────────────────────────
        selected: list[str] = []
        unselected: list[str] = list(self.X_train.columns)
        t_select: float = 0.0
        t_eval: float = 0.0
        stopping_reason: str | None = None
        performance_history: list[dict] = []

        # G3 — eval enabled for classification OR regression
        _eval_enabled = (
            self._task_type == "classification"
            and self.gene_expression_df is not None
            and self.train_labels_df is not None
            and self.val_labels_df is not None
        ) or (
            self._task_type == "regression"
            and self._X_val_filled is not None
            and self._y_val is not None
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

            # (c) G1 — evaluation predicate
            _should_eval = _eval_enabled and (
                (step + 1) % eval_every_k == 0          # every k-th step
                or (step + 1) == n_features_to_select   # always on final requested step
                or not unselected                        # always on forced-final step
            )

            if _should_eval:
                t0 = perf_counter()
                metrics = self._evaluate_step(selected)
                t_eval += perf_counter() - t0
                # G1 — inject step number (1-based)
                metrics["step"] = step + 1
                performance_history.append(metrics)

                # Log depending on task type
                if self._task_type == "classification":
                    acc = metrics.get("accuracy", float("nan"))
                    f1 = (
                        metrics.get("macro avg", {}).get("f1-score", float("nan"))
                        if isinstance(metrics.get("macro avg"), dict)
                        else float("nan")
                    )
                    logger.info(
                        "Step %d eval — accuracy: %.4f  macro F1: %.4f",
                        step + 1, acc, f1,
                    )
                else:
                    logger.info(
                        "Step %d eval — r2: %.4f  mse: %.4f  mae: %.4f",
                        step + 1,
                        metrics.get("r2", float("nan")),
                        metrics.get("mse", float("nan")),
                        metrics.get("mae", float("nan")),
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

            elif _eval_enabled and not _should_eval:
                # G1 — log skipped steps
                logger.info(
                    "Step %d: evaluation skipped (eval_every_k=%d)", step + 1, eval_every_k
                )

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
    # G3 — plot_vs_random_baseline
    # ------------------------------------------------------------------
    def plot_vs_random_baseline(
        self,
        result: SelectionResult,
        baseline_summary: pd.DataFrame,
        title_suffix: str = "",
    ) -> None:
        """
        Plot mRmR performance against a random baseline.

        Parameters
        ----------
        result : SelectionResult
            The output of :py:meth:`forward_selection`.  Each entry in
            ``result.performance_history`` must have a ``'step'`` key.
        baseline_summary : pd.DataFrame
            Wide-format summary returned by
            ``LR_random_baseline.plot_performance_with_stats(return_summary=True)``
            (classification) or
            ``LR_regression_baseline.plot_performance_with_stats(return_summary=True)``
            (regression).
        title_suffix : str
            Optional suffix appended to the figure title.
        """
        if self._task_type == "classification":
            self._plot_vs_baseline_clf(result, baseline_summary, title_suffix)
        else:
            self._plot_vs_baseline_reg(result, baseline_summary, title_suffix)

    def _plot_vs_baseline_clf(
        self,
        result: SelectionResult,
        baseline_summary: pd.DataFrame,
        title_suffix: str,
    ) -> None:
        """Classification branch of plot_vs_random_baseline."""
        mrmr_records = [
            {
                "n_features": entry["step"],
                "accuracy":   entry.get("accuracy", float("nan")),
                "macro_f1":   (
                    entry.get("macro avg", {}).get("f1-score", float("nan"))
                    if isinstance(entry.get("macro avg"), dict)
                    else float("nan")
                ),
            }
            for entry in result.performance_history
        ]
        mrmr_df = pd.DataFrame(mrmr_records)

        # Print summary table
        print("\nSummary Statistics — Random Baseline (Mean ± Std):")
        try:
            display(baseline_summary)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            print(baseline_summary.to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        panels = [
            ("accuracy", "acc_mean", "acc_std", "Accuracy"),
            ("macro_f1", "f1_mean",  "f1_std",  "Macro F1"),
        ]

        for ax, (mrmr_col, bl_mean, bl_std, title) in zip(axes, panels):
            ax.plot(
                mrmr_df["n_features"], mrmr_df[mrmr_col],
                marker="o", linewidth=2, color="steelblue", label="mRmR",
            )
            bx = baseline_summary["features_num"]
            ax.plot(
                bx, baseline_summary[bl_mean],
                marker="s", linewidth=2, color="darkorange",
                linestyle="--", label="Random baseline (mean)",
            )
            ax.fill_between(
                bx,
                baseline_summary[bl_mean] - baseline_summary[bl_std],
                baseline_summary[bl_mean] + baseline_summary[bl_std],
                alpha=0.25, color="darkorange", label="±1 std (random)",
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Number of features")
            ax.set_ylabel(title)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

        fig.suptitle(
            f"Classification: mRmR vs LR Random Baseline{title_suffix}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.show()

    def _plot_vs_baseline_reg(
        self,
        result: SelectionResult,
        baseline_summary: pd.DataFrame,
        title_suffix: str,
    ) -> None:
        """Regression branch of plot_vs_random_baseline."""
        mrmr_records = [
            {
                "n_features": entry["step"],
                "r2":  entry.get("r2",  float("nan")),
                "mse": entry.get("mse", float("nan")),
                "mae": entry.get("mae", float("nan")),
            }
            for entry in result.performance_history
        ]
        mrmr_df = pd.DataFrame(mrmr_records)

        # Print summary table
        print("\nSummary Statistics — LinReg Random Baseline (Mean ± Std):")
        try:
            display(baseline_summary)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            print(baseline_summary.to_string(index=False))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        panels = [
            ("r2",  "r2_mean",  "r2_std",  "R²"),
            ("mse", "mse_mean", "mse_std", "MSE"),
            ("mae", "mae_mean", "mae_std", "MAE"),
        ]

        for ax, (mrmr_col, bl_mean, bl_std, title) in zip(axes, panels):
            ax.plot(
                mrmr_df["n_features"], mrmr_df[mrmr_col],
                marker="o", linewidth=2, color="steelblue", label="mRmR",
            )
            bx = baseline_summary["features_num"]
            ax.plot(
                bx, baseline_summary[bl_mean],
                marker="s", linewidth=2, color="darkorange",
                linestyle="--", label="Random baseline (mean)",
            )
            ax.fill_between(
                bx,
                baseline_summary[bl_mean] - baseline_summary[bl_std],
                baseline_summary[bl_mean] + baseline_summary[bl_std],
                alpha=0.25, color="darkorange", label="±1 std (random)",
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Number of features")
            ax.set_ylabel(title)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

        fig.suptitle(
            f"Regression: mRmR vs LinReg Random Baseline{title_suffix}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_metric(report: dict, metric_key: str) -> float:
        """
        Resolve a dotted metric key from an evaluation report dict.

        Supported formats
        -----------------
        * ``'accuracy'``, ``'r2'``, ``'mse'``, ``'mae'``
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
            "stopping_metric '%s' not found in evaluation report — "
            "early stopping disabled for this step",
            metric_key,
        )
        return float("-inf")
