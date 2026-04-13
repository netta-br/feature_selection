from __future__ import annotations

import json
import logging
import os
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from ..evaluation.logistic_regression import evaluate_logistic_regression_with_given_features
from ..score_calculator import ScoreCalculator, infer_task_type
from ..score_matrix import ScoreMatrix
from ..results import SelectionResult

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
# Config loader helper
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    """Load project-wide config.json if it exists."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "config.json"
    )
    config_path = os.path.normpath(config_path)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


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
    redundancy_agg : str
        Aggregation for redundancy: ``'mean'`` or ``'max'``.
    correlation_filepath : str | None
        Path to a pre-computed correlation matrix file.  For non-correlation
        relevance methods this matrix is used for redundancy only.
    relevance_scores_filepath : str | None
        Path to a pre-computed relevance scores CSV (non-correlation methods).
        If the file does not exist it will be computed and saved here.
        ``None`` activates lazy per-feature computation (Mode B).
    X_val : pd.DataFrame | None
        Validation feature matrix.  Evaluation is skipped when ``None``.
    y_val : pd.Series | None
        Validation target vector.  Evaluation is skipped when ``None``.
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
        # redundancy aggregation
        redundancy_agg: str = "mean",
        # existing params
        correlation_filepath: str | None = None,
        # G4 — non-correlation relevance scores filepath
        relevance_scores_filepath: str | None = None,
        lr_C: float = np.inf,
        random_seed: int | None = None,
        # generic evaluation params (classification & regression)
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
        if redundancy_agg not in {"mean", "max"}:
            raise ValueError(
                f"redundancy_agg must be 'mean' or 'max'; got {redundancy_agg!r}"
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
        self.redundancy_agg = redundancy_agg
        self.correlation_filepath = correlation_filepath
        self.relevance_scores_filepath = relevance_scores_filepath

        self.lr_C = lr_C
        self.random_seed = random_seed

        # NaN imputation — done once, reused everywhere
        self._X_filled: pd.DataFrame = X_train.fillna(0)

        # evaluation data (classification & regression)
        self._X_val_filled: pd.DataFrame | None = X_val.fillna(0) if X_val is not None else None
        self._y_val: pd.Series | None = y_val

        # Target column name — always y_train.name, never a synthetic sentinel
        self._target_col: str = str(y_train.name)

        self._task_type: str = infer_task_type(y_train.values)

        # Load config for method_params
        self._cfg = _load_config().get("score_matrix", {})

        # ── New: ScoreCalculator / ScoreMatrix ─────────────────────────
        self._calc: ScoreCalculator | None = None
        self._sm: ScoreMatrix | None = None

        logger.info(
            "mRmRSelector initialised | task_type=%s | target_col=%s | "
            "relevance_method=%s | redundancy_method=%s | mrmr_score_method=%s | "
            "redundancy_agg=%s",
            self._task_type,
            self._target_col,
            self.relevance_method,
            self.redundancy_method,
            self.mrmr_score_method,
            self.redundancy_agg,
        )

    # ------------------------------------------------------------------
    # Build ScoreCalculator / ScoreMatrix (replaces _precompute_matrix)
    # ------------------------------------------------------------------
    def _build_score_calculator(self) -> None:
        """Build ScoreCalculator for Mode B, or ScoreMatrix for Mode A.

        For non-correlation relevance methods, builds two calculators:
        one for relevance (target-based) and one for redundancy (correlation-based).
        """
        # Determine the effective method for the calculator.
        # For correlation-based methods, one calculator handles both
        # relevance and redundancy.
        # For non-correlation methods, we need separate handling.
        if self.relevance_method in self._CORR_METHODS:
            effective_method = self.relevance_method
        else:
            # Non-correlation: redundancy uses redundancy_method (correlation)
            # Relevance is computed separately.
            effective_method = self.redundancy_method

        self._calc = ScoreCalculator(
            X=self._X_filled.values,
            feature_names=list(self._X_filled.columns),
            target_name=self._target_col,
            target_col=self.y_train.values,
            method=effective_method,
            redundancy_agg=self.redundancy_agg,
            method_params=self._cfg.get(effective_method, {}),
            random_seed=self.random_seed,
        )

        if self.correlation_filepath is not None:
            # Mode A — use ScoreMatrix for disk-backed precomputed matrix
            self._sm = ScoreMatrix(
                X=self._X_filled.values,
                feature_names=list(self._X_filled.columns),
                target_names=[self._target_col],
                target_data=self.y_train.values[:, None],
                method=effective_method,
                filepath=self.correlation_filepath,
                file_format=self._cfg.get("file_format", "csv"),
                method_params=self._cfg.get(effective_method, {}),
                random_seed=self.random_seed,
            )
            self._sm.precompute()

            # For Mode A, populate relevance from the precomputed matrix
            target_idx = self._sm.index[self._target_col]
            feat_names = list(self._X_filled.columns)
            feat_idxs = np.array(
                [self._sm.index[f] for f in feat_names], dtype=np.intp
            )
            relevance_from_matrix = self._sm.arr[target_idx, feat_idxs]

            # Store in calc's _relevance_vec for uniform API
            self._calc._relevance_vec = relevance_from_matrix.copy()
            self._calc._redundancy_acc = np.zeros(self._calc._N, dtype=np.float64)
            self._calc._n_selected = 0

            # For non-correlation relevance: override relevance vector
            if self.relevance_method in self._NON_CORR_METHODS:
                self._compute_noncorr_relevance()

            logger.info("Mode A: ScoreMatrix precomputed from %s", self.correlation_filepath)
        else:
            # Mode B — single-target calculator only
            self._calc.init_target()

            # For non-correlation relevance: override relevance vector
            if self.relevance_method in self._NON_CORR_METHODS:
                self._compute_noncorr_relevance()

            logger.info("Mode B: ScoreCalculator initialised (accumulator pattern)")

    # ------------------------------------------------------------------
    # Non-correlation relevance override
    # ------------------------------------------------------------------
    def _compute_noncorr_relevance(self) -> None:
        """Compute relevance using non-correlation method and override _relevance_vec."""
        if self.relevance_scores_filepath is not None and os.path.exists(
            self.relevance_scores_filepath
        ):
            # Load from file
            series = pd.read_csv(
                self.relevance_scores_filepath, index_col=0
            ).squeeze("columns")
            feat_names = list(self._X_filled.columns)
            relevance = np.array(
                [float(series.get(f, 0.0)) for f in feat_names], dtype=np.float64
            )
            self._calc._relevance_vec = relevance
            logger.info(
                "Loaded non-correlation relevance from %s",
                self.relevance_scores_filepath,
            )
            return

        # Compute using a temporary ScoreCalculator with the non-corr method
        rel_calc = ScoreCalculator(
            X=self._X_filled.values,
            feature_names=list(self._X_filled.columns),
            target_name=self._target_col,
            target_col=self.y_train.values,
            method=self.relevance_method,
            method_params=self._cfg.get(self.relevance_method, {}),
            random_seed=self.random_seed,
        )
        rel_calc.init_target()
        self._calc._relevance_vec = rel_calc._relevance_vec.copy()

        # Save if filepath provided
        if self.relevance_scores_filepath is not None:
            feat_names = list(self._X_filled.columns)
            series = pd.Series(
                self._calc._relevance_vec, index=feat_names, name="score"
            )
            series.index.name = "feature"
            dirpath = os.path.dirname(self.relevance_scores_filepath)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            series.to_csv(self.relevance_scores_filepath, header=True)
            logger.info(
                "Computed and saved non-correlation relevance to %s",
                self.relevance_scores_filepath,
            )

    # ------------------------------------------------------------------
    # mRmR score
    # ------------------------------------------------------------------
    def _compute_mrmr_score(self, relevance: float, redundancy: float) -> float:
        if self.mrmr_score_method == "difference":
            return relevance - redundancy
        # ratio
        return relevance / (redundancy + 1e-8)

    # ------------------------------------------------------------------
    # evaluation step (dispatch on task type)
    # ------------------------------------------------------------------
    def _evaluate_step(self, selected_features: list[str]) -> dict:
        if self._task_type == "classification":
            if self._X_val_filled is None or self._y_val is None:
                logger.warning(
                    "Classification evaluation skipped: X_val or y_val not provided."
                )
                return {}
            _, report = evaluate_logistic_regression_with_given_features(
                X_train=self._X_filled,
                y_train=self.y_train,
                X_val=self._X_val_filled,
                y_val=self._y_val,
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

        # ── PRE-SELECTION (timed) ──────────────────────────────────────
        t_pre = perf_counter()
        self._build_score_calculator()
        t_precompute = perf_counter() - t_pre
        logger.info(
            "Starting forward selection for up to %d features (eval_every_k=%d, precompute=%.3fs)",
            n_features_to_select,
            eval_every_k,
            t_precompute,
        )

        # ── SELECTION LOOP — accumulator pattern ───────────────────────
        N_features = self._calc._N
        feat_names = list(self._X_filled.columns)

        selected: list[str] = []
        selected_idxs: list[int] = []
        remaining_idxs = np.arange(N_features, dtype=np.intp)

        t_select: float = t_precompute
        t_eval: float = 0.0
        stopping_reason: str | None = None
        performance_history: list[dict] = []

        # eval enabled for classification OR regression
        _eval_enabled = (
            self._X_val_filled is not None
            and self._y_val is not None
        )

        for step in range(n_features_to_select):
            # (a) All features already selected?
            if len(remaining_idxs) == 0:
                stopping_reason = "no_features_remaining"
                logger.info("No features remaining — stopping at step %d", step)
                break

            # (b) TIMED: mRmR scoring + best-feature selection
            t0 = perf_counter()

            rel = self._calc.get_relevance(remaining_idxs)
            red = self._calc.get_redundancy(remaining_idxs)

            if self.mrmr_score_method == "difference":
                scores = rel - red
            else:
                scores = rel / (red + 1e-8)

            best_local = int(np.argmax(scores))
            best_idx = remaining_idxs[best_local]

            # Remove best from remaining
            remaining_idxs = np.delete(remaining_idxs, best_local)

            # Update redundancy accumulator
            self._calc.update_redundancy(best_idx, remaining_idxs)

            selected_idxs.append(int(best_idx))
            selected.append(feat_names[best_idx])
            t_select += perf_counter() - t0

            logger.info(
                "Step %d: selected '%s' (%d total)", step + 1, feat_names[best_idx], len(selected)
            )

            # (c) G1 — evaluation predicate
            _should_eval = _eval_enabled and (
                (step + 1) % eval_every_k == 0          # every k-th step
                or (step + 1) == n_features_to_select   # always on final requested step
                or len(remaining_idxs) == 0              # always on forced-final step
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

        # Build short label: method abbreviation + score method + agg (if non-default)
        _rel_abbrev = {
            "pearson": "P", "kendall": "K", "spearman": "S",
            "mutual_information": "MI", "random_forest": "RF", "f_statistic": "FS",
        }.get(self.relevance_method, self.relevance_method)
        _score_abbrev = "D" if self.mrmr_score_method == "difference" else "R"
        _agg_suffix = f"-{self.redundancy_agg}" if self.redundancy_agg != "mean" else ""
        _label = f"mRmR-{_rel_abbrev}-{_score_abbrev}-{_agg_suffix}"

        return SelectionResult(
            selected_features=selected,
            performance_history=performance_history,
            stopping_reason=stopping_reason,
            n_steps=len(selected),
            final_metrics=performance_history[-1] if performance_history else None,
            selection_time_seconds=t_select,
            evaluation_time_seconds=t_eval if _eval_enabled else None,
            task_type=self._task_type,
            label=_label,
        )

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
