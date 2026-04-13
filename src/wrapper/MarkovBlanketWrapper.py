from __future__ import annotations

import copy
import logging
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, r2_score

from ..score_calculator import ScoreCalculator, infer_task_type
from ..score_matrix import ScoreMatrix
from ..results import SelectionResult, WrapperSelectionResult

# ---------------------------------------------------------------------------
# Module-level logger (duplicate-handler guard)
# ---------------------------------------------------------------------------
logger = logging.getLogger("WrapperSelector")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# WrapperSelector
# ---------------------------------------------------------------------------
class WrapperSelector:
    """
    Unified wrapper-based feature selector supporting four algorithm variants:

    ========================  ================  =================
    ``use_su_ranking``        ``use_mb_pruning``  Algorithm
    ========================  ================  =================
    ``False``                 ``False``           SFS
    ``False``                 ``True``            SFS-MB
    ``True``                  ``False``           IWSS
    ``True``                  ``True``            **IWSS-MB** (Wang et al. 2017)
    ========================  ================  =================

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.  NaNs imputed with 0 internally.
    y_train : pd.Series
        Training target vector.  Must have a ``.name`` attribute.
    X_val : pd.DataFrame
        Validation feature matrix (used only for periodic test evaluation,
        not for selection decisions).
    y_val : pd.Series
        Validation target vector.
    use_su_ranking : bool
        If ``True`` (IWSS mode), rank candidates descending by SU(f, C) at
        init; otherwise iterate in original column order (SFS mode).
    use_mb_pruning : bool
        If ``True``, apply vectorised Approximate Markov Blanket pruning
        after each feature is accepted (Definition 5, Wang et al.).
    mb_threshold : float
        Margin added to the MB redundancy condition.  A candidate ``F_j``
        is pruned only when ``SU(F_i, F_j) - SU(F_j, C) > mb_threshold``
        (in addition to ``SU(F_i, C) >= SU(F_j, C)``).  Default ``0.0``
        reproduces the original Definition 5; positive values make pruning
        more conservative (fewer candidates pruned).
    evaluator : sklearn-compatible estimator or None
        Duck-typed ``fit`` / ``predict`` estimator.  Defaults to
        ``LogisticRegression(max_iter=1000, C=np.inf)`` for classification
        or ``Ridge(alpha=1.0)`` for regression (chosen after task-type
        detection).
    min_features : int
        Minimum set size required before a candidate can be accepted.
    cv_folds : int
        Number of folds for cross-validation inside the selection loop.
    cv_min_folds : int
        Minimum number of folds that must satisfy the improvement condition
        for a candidate to be accepted.
    patience : int or None
        Stop after this many consecutive steps with no accepted feature.
        ``None`` disables patience-based stopping.
    su_filepath : str or None
        Mode A: path to a precomputed SU matrix file.  If the file exists it
        is loaded; if it does not, the matrix is computed and saved there.
        ``None`` → Mode B (incremental lazy fill).
    random_seed : int or None
        Forwarded to evaluator and CV splitter for reproducibility.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_su_ranking: bool = True,
        use_mb_pruning: bool = True,
        mb_threshold: float = 0.0,
        evaluator=None,
        min_features: int = 1,
        cv_folds: int = 5,
        cv_min_folds: int = 3,
        patience: int | None = None,
        su_filepath: str | None = None,
        random_seed: int | None = None,
    ) -> None:
        if y_train.name is None:
            raise ValueError(
                "y_train must have a .name attribute "
                "(e.g. pd.Series(..., name='label'))."
            )

        self.use_su_ranking = use_su_ranking
        self.use_mb_pruning = use_mb_pruning
        self.mb_threshold = mb_threshold
        self.min_features = min_features
        self.cv_folds = cv_folds
        self.cv_min_folds = cv_min_folds
        self.patience = patience
        self.su_filepath = su_filepath
        self.random_seed = random_seed

        # Data
        self._X_filled: pd.DataFrame = X_train.fillna(0)
        self._X_val_filled: pd.DataFrame = X_val.fillna(0)
        self._y_train: pd.Series = y_train
        self._y_val: pd.Series = y_val
        self._target_col: str = str(y_train.name)

        # Task type detection (must precede evaluator default selection)
        self._task_type: str = infer_task_type(y_train.values)

        # Evaluator — default depends on task type
        if evaluator is None:
            if self._task_type == "classification":
                self._evaluator = LogisticRegression(
                    max_iter=1000, C=np.inf, random_state=random_seed
                )
            else:
                self._evaluator = Ridge(alpha=1.0, random_state=random_seed)
        else:
            self._evaluator = evaluator

        # ── ScoreCalculator / ScoreMatrix ──────────────────────────────
        self._calc: ScoreCalculator | None = None
        self._sm: ScoreMatrix | None = None
        self._filter_time_seconds: float = 0.0

        # Candidate list (built at init)
        self._candidates: list[str] = []

        # Build SU calculator and candidates
        self._init_candidates()

        logger.info(
            "WrapperSelector initialised | task=%s | use_su_ranking=%s | "
            "use_mb_pruning=%s | mb_threshold=%.4f | cv_folds=%d | "
            "cv_min_folds=%d | patience=%s | n_candidates=%d",
            self._task_type,
            use_su_ranking,
            use_mb_pruning,
            mb_threshold,
            cv_folds,
            cv_min_folds,
            patience,
            len(self._candidates),
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_candidates(self) -> None:
        """Build _candidates list; set up SU ScoreCalculator if needed."""
        needs_su = self.use_su_ranking or self.use_mb_pruning

        if not needs_su:
            self._candidates = list(self._X_filled.columns)
            return

        t0 = perf_counter()

        # Build ScoreCalculator with SU method
        self._calc = ScoreCalculator(
            X=self._X_filled.values,
            feature_names=list(self._X_filled.columns),
            target_name=self._target_col,
            target_col=self._y_train.values,
            method="su",
            random_seed=self.random_seed,
        )

        if self.su_filepath is not None:
            # Mode A — precomputed SU matrix
            self._sm = ScoreMatrix(
                X=self._X_filled.values,
                feature_names=list(self._X_filled.columns),
                target_names=[self._target_col],
                target_data=self._y_train.values[:, None],
                method="su",
                filepath=self.su_filepath,
                file_format="csv",  # backward compat with existing CSV files
                random_seed=self.random_seed,
            )
            self._sm.precompute()

            # Populate calc's relevance from the precomputed matrix
            target_idx = self._sm.index[self._target_col]
            feat_names = list(self._X_filled.columns)
            feat_idxs_sm = np.array(
                [self._sm.index[f] for f in feat_names], dtype=np.intp
            )
            relevance_from_matrix = self._sm.arr[target_idx, feat_idxs_sm]
            self._calc._relevance_vec = relevance_from_matrix.copy()
            self._calc._redundancy_acc = np.zeros(self._calc._N, dtype=np.float64)
            self._calc._n_selected = 0

            # Also ensure disc cache is built for compute_row calls
            self._calc._ensure_disc()
        else:
            # Mode B — single-target calculator only
            self._calc.init_target()

        self._filter_time_seconds = perf_counter() - t0

        # Build ranked candidates from relevance vector
        feat_idxs = np.arange(len(self._X_filled.columns), dtype=np.intp)
        su_vs_target = self._calc.get_relevance(feat_idxs)

        feat_cols = list(self._X_filled.columns)
        if self.use_su_ranking:
            order = np.argsort(-su_vs_target)
            self._candidates = [feat_cols[i] for i in order]
        else:
            self._candidates = feat_cols

        logger.info(
            "Filter init done in %.3fs | SU target col filled | "
            "Mode %s | top candidate: %s (SU=%.4f)",
            self._filter_time_seconds,
            "A" if self.su_filepath is not None else "B",
            self._candidates[0] if self._candidates else "—",
            float(su_vs_target[np.argsort(-su_vs_target)[0]])
            if len(feat_cols) > 0 else 0.0,
        )

    # ------------------------------------------------------------------
    # Vectorised MB pruning
    # ------------------------------------------------------------------
    def _mb_prune_vectorised(self, selected_feat: str) -> list[str]:
        """
        Vectorised Approximate Markov Blanket pruning (Definition 5).

        Uses ScoreCalculator.compute_row for the SU vector between
        the selected feature and all remaining candidates.

        Returns list of pruned feature names (removed from ``_candidates``).
        """
        if not self._candidates:
            return []
        assert self._calc is not None

        sel_idx = self._calc._index[selected_feat]
        tgt_idx = self._calc._index[self._target_col]

        cand_idxs = np.array(
            [self._calc._index[c] for c in self._candidates], dtype=np.intp
        )

        # SU(selected_feat, target) — scalar
        su_sel_C = self._calc.get_relevance(np.array([sel_idx], dtype=np.intp))[0]

        # SU(candidates, target) — vector
        su_cand_C = self._calc.get_relevance(cand_idxs)

        # SU(selected_feat, candidates) — vector via compute_row
        if self._sm is not None:
            # Mode A: read from precomputed matrix
            sm_sel_idx = self._sm.index[selected_feat]
            sm_cand_idxs = np.array(
                [self._sm.index[c] for c in self._candidates], dtype=np.intp
            )
            su_sel_cand = self._sm.arr[sm_sel_idx, sm_cand_idxs]
        else:
            # Mode B: compute on-the-fly
            su_sel_cand = self._calc.compute_row(sel_idx, cand_idxs)

        redundant_mask = (su_sel_C >= su_cand_C) & (su_sel_cand - su_cand_C > self.mb_threshold)

        pruned: list[str] = []
        surviving: list[str] = []
        for feat, flag in zip(self._candidates, redundant_mask):
            if flag:
                pruned.append(feat)
            else:
                surviving.append(feat)

        self._candidates = surviving
        return pruned

    # ------------------------------------------------------------------
    # CV evaluation — mean-only baseline
    # ------------------------------------------------------------------
    def _cv_evaluate(
        self,
        features: list[str],
        baseline_mean: float,
    ) -> tuple[float, int]:
        """
        k-fold CV on ``_X_filled[features]`` / ``y_train``.

        Acceptance criterion per fold:
          fold_score > baseline_mean

        Parameters
        ----------
        features : list[str]
            Feature set to evaluate (includes the candidate).
        baseline_mean : float
            Mean CV score of the current selected set (before adding
            candidate).  Each fold is compared against this scalar.

        Returns
        -------
        mean_score : float
        n_improved : int
            Count of folds satisfying ``fold_score > baseline_mean``.
        """
        X = self._X_filled[features].values
        y = self._y_train.values

        if self._task_type == "classification":
            splitter = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
            )
        else:
            splitter = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
            )

        fold_scores = np.empty(self.cv_folds, dtype=np.float64)
        for k, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            clf = copy.deepcopy(self._evaluator)
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[val_idx])
            if self._task_type == "classification":
                fold_scores[k] = accuracy_score(y[val_idx], y_pred)
            else:
                fold_scores[k] = r2_score(y[val_idx], y_pred)

        mean_score = float(fold_scores.mean())
        n_improved = int((fold_scores > baseline_mean).sum())
        return mean_score, n_improved

    # ------------------------------------------------------------------
    # Test/validation evaluation (outside timed loop)
    # ------------------------------------------------------------------
    def _test_evaluate(self, features: list[str]) -> dict:
        """
        Fit evaluator on full training set, score on validation set.

        Returns a metrics dict compatible with SelectionResult's
        performance_history format.
        """
        clf = copy.deepcopy(self._evaluator)
        X_tr = self._X_filled[features].values
        X_vl = self._X_val_filled[features].values
        y_tr = self._y_train.values
        y_vl = self._y_val.values

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_vl)

        if self._task_type == "classification":
            from sklearn.metrics import classification_report
            report = classification_report(y_vl, y_pred, output_dict=True)
            return report
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            return {
                "r2":  float(r2_score(y_vl, y_pred)),
                "mse": float(mean_squared_error(y_vl, y_pred)),
                "mae": float(mean_absolute_error(y_vl, y_pred)),
            }

    # ------------------------------------------------------------------
    # Metric resolution helper
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_metric(report: dict, metric_key: str) -> float:
        """Resolve a (possibly nested) metric key from an evaluation dict."""
        if metric_key in report:
            v = report[metric_key]
            return float(v) if not isinstance(v, dict) else float("nan")
        if "_" in metric_key:
            idx = metric_key.index("_")
            outer, inner = metric_key[:idx], metric_key[idx + 1:]
            if outer in report and isinstance(report[outer], dict):
                return float(report[outer].get(inner, float("nan")))
        logger.warning(
            "stopping_metric '%s' not found in report — early stopping disabled",
            metric_key,
        )
        return float("-inf")

    # ------------------------------------------------------------------
    # run() helper — IWSS: evaluate a single ranked candidate
    # ------------------------------------------------------------------
    def _run_iwss_step(
        self,
        candidate: str,
        selected: list[str],
        baseline_mean: float,
    ) -> tuple[bool, float, int, float]:
        """
        Evaluate *candidate* in IWSS mode.

        Returns
        -------
        accepted : bool
        mean_s   : float   CV mean score
        n_imp    : int     folds improved
        dt_clf   : float   wall-clock seconds for the CV call
        """
        t0 = perf_counter()
        mean_s, n_imp = self._cv_evaluate([*selected, candidate], baseline_mean)
        dt_clf = perf_counter() - t0

        accepted = (
            n_imp >= self.cv_min_folds
            and mean_s > baseline_mean
            and len(selected) + 1 >= self.min_features
        )
        return accepted, mean_s, n_imp, dt_clf

    # ------------------------------------------------------------------
    # run() helper — SFS: exhaustive search over remaining candidates
    # ------------------------------------------------------------------
    def _run_sfs_step(
        self,
        candidates_slice: list[str],
        selected: list[str],
        baseline_mean: float,
    ) -> tuple[str | None, float, float, int]:
        """
        Exhaustive best-feature search over *candidates_slice* (SFS mode).

        Evaluates every candidate and returns the one with the highest mean
        CV score that also satisfies the acceptance predicate.

        Returns
        -------
        best_feat  : str | None   best candidate, or None if none qualifies
        best_mean  : float        mean CV score of best candidate
        dt_clf     : float        total wall-clock seconds for all CV calls
        n_evals    : int          number of CV calls made
        """
        best_feat: str | None = None
        best_mean: float = baseline_mean
        best_n_imp: int = 0
        dt_clf: float = 0.0
        n_evals: int = 0

        for c in candidates_slice:
            t0 = perf_counter()
            mean_s, n_imp = self._cv_evaluate([*selected, c], baseline_mean)
            dt_clf += perf_counter() - t0
            n_evals += 1

            if (
                n_imp >= self.cv_min_folds
                and mean_s > best_mean
                and len(selected) + 1 >= self.min_features
            ):
                best_feat = c
                best_mean = mean_s
                best_n_imp = n_imp

        return best_feat, best_mean, dt_clf, n_evals

    # ------------------------------------------------------------------
    # run() helper — MB pruning (flag check lives in run())
    # ------------------------------------------------------------------
    def _prune_candidates(
        self,
        candidates: list[str],
        accepted_feat: str,
        accepted_pos: int,
    ) -> tuple[list[str], int]:
        """
        Remove ``accepted_feat`` from ``candidates`` at ``accepted_pos``,
        then apply vectorised MB pruning.

        The ``use_mb_pruning`` flag check is performed by the caller
        (``run()``); this helper assumes pruning is desired.

        Syncs ``self._candidates`` with the local list so
        ``_mb_prune_vectorised`` operates on the correct state.

        Returns
        -------
        updated_candidates : list[str]
        n_pruned           : int
        """
        # Remove the just-accepted feature from the working list
        if accepted_pos < len(candidates) and candidates[accepted_pos] == accepted_feat:
            candidates.pop(accepted_pos)

        self._candidates = candidates
        pruned = self._mb_prune_vectorised(accepted_feat)
        candidates = self._candidates
        return candidates, len(pruned)

    # ------------------------------------------------------------------
    # run() helper — test-eval predicate (pure)
    # ------------------------------------------------------------------
    def _should_test_eval(
        self,
        n_selected: int,
        last_eval_step: int,
        n_features_to_select: int,
        candidates_remaining: int,
        k: int,
    ) -> bool:
        """Return True when a test/validation evaluation should be triggered."""
        return (
            n_selected > last_eval_step
            and (
                n_selected % k == 0
                or n_selected == n_features_to_select
                or candidates_remaining == 0
            )
        )

    # ------------------------------------------------------------------
    # run() helper — timed test evaluation
    # ------------------------------------------------------------------
    def _run_test_eval(self, selected: list[str]) -> tuple[dict, float]:
        """
        Run ``_test_evaluate`` and measure wall-clock time.

        Returns
        -------
        metrics : dict
        dt      : float   seconds elapsed
        """
        t0 = perf_counter()
        metrics = self._test_evaluate(selected)
        return metrics, perf_counter() - t0

    # ------------------------------------------------------------------
    # Main selection loop
    # ------------------------------------------------------------------
    def run(
        self,
        n_features_to_select: int,
        stopping_metric: str | None = None,
        stopping_threshold: float | None = None,
        test_eval_every_k: int = 1,
    ) -> WrapperSelectionResult:
        """
        Run wrapper-based feature selection.

        Parameters
        ----------
        n_features_to_select : int
            Maximum number of features to select.
        stopping_metric : str or None
            Metric key from the test evaluation report for early stopping.
        stopping_threshold : float or None
            Stop when ``stopping_metric >= stopping_threshold``.
        test_eval_every_k : int
            Run test/validation evaluation every k selections (default 1).

        Returns
        -------
        WrapperSelectionResult
        """
        if test_eval_every_k < 1:
            raise ValueError(f"test_eval_every_k must be >= 1; got {test_eval_every_k}")

        # Local mutable copy of candidates for this run
        candidates = list(self._candidates)

        selected: list[str] = []
        cv_baseline_mean: float = 0.0

        performance_history: list[dict] = []
        stopping_reason: str | None = None

        n_selection_evals: int = 0
        t_classifier: float = 0.0
        t_test_eval: float = 0.0
        n_pruned: int = 0
        n_skipped: int = 0
        no_improvement_streak: int = 0
        last_eval_step: int = 0

        # ── START selection timer ──────────────────────────────────────
        t0_selection = perf_counter()

        i = 0
        while i < len(candidates) and len(selected) < n_features_to_select:

            if self.use_su_ranking:
                # ── IWSS mode ─────────────────────────────────────────
                accepted, mean_s, n_imp, dt = self._run_iwss_step(
                    candidates[i], selected, cv_baseline_mean
                )
                t_classifier += dt
                n_selection_evals += 1

                if accepted:
                    feat = candidates[i]
                    selected.append(feat)
                    cv_baseline_mean = mean_s
                    no_improvement_streak = 0

                    logger.info(
                        "Step %d (IWSS): selected '%s' mean_cv=%.4f n_improved=%d",
                        len(selected), feat, mean_s, n_imp,
                    )

                    # ── Test/validation evaluation ─────────────────────────
                    _n_selected = len(selected)
                    if self._should_test_eval(
                        _n_selected, last_eval_step, n_features_to_select,
                        len(candidates) - i, test_eval_every_k,
                    ):
                        metrics, dt = self._run_test_eval(selected)
                        t_test_eval += dt
                        metrics["step"] = _n_selected
                        last_eval_step = _n_selected
                        performance_history.append(metrics)

                        if self._task_type == "classification":
                            logger.info(
                                "  Test eval step %d — accuracy: %.4f",
                                _n_selected, metrics.get("accuracy", float("nan")),
                            )
                        else:
                            logger.info(
                                "  Test eval step %d — r2: %.4f",
                                _n_selected, metrics.get("r2", float("nan")),
                            )

                        if stopping_metric is not None and stopping_threshold is not None:
                            if (
                                self._resolve_metric(metrics, stopping_metric)
                                >= stopping_threshold
                            ):
                                stopping_reason = "metric_threshold_reached"
                                break

                    if self.use_mb_pruning:
                        candidates, np_ = self._prune_candidates(candidates, feat, i)
                        n_pruned += np_
                        n_skipped += np_
                        if np_:
                            logger.info("  MB pruned %d candidates", np_)
                        # pop(i) already removed the accepted feature and advanced
                        # the list — do NOT increment i here; the next unprocessed
                        # candidate is now at position i.
                        continue
                    # Without MB pruning: fall through to i += 1
                else:
                    no_improvement_streak += 1
                    logger.debug(
                        "  Candidate '%s' rejected (mean=%.4f n_imp=%d streak=%d)",
                        candidates[i], mean_s, n_imp, no_improvement_streak,
                    )
                    if (
                        self.patience is not None
                        and no_improvement_streak >= self.patience
                    ):
                        stopping_reason = "patience_exceeded"
                        break

                i += 1

            else:
                # ── SFS mode ──────────────────────────────────────────
                best, best_mean, dt, n_evals = self._run_sfs_step(
                    candidates[i:], selected, cv_baseline_mean
                )
                t_classifier += dt
                n_selection_evals += n_evals

                if best is None:
                    no_improvement_streak += 1
                    if (
                        self.patience is not None
                        and no_improvement_streak >= self.patience
                    ):
                        stopping_reason = "patience_exceeded"
                        break
                    stopping_reason = "no_improvement"
                    break

                # Move best to position i (stable ordering)
                candidates.remove(best)
                candidates.insert(i, best)
                selected.append(best)
                cv_baseline_mean = best_mean
                no_improvement_streak = 0

                logger.info(
                    "Step %d (SFS): selected '%s' mean_cv=%.4f",
                    len(selected), best, best_mean,
                )

                # ── Test/validation evaluation ─────────────────────────
                _n_selected = len(selected)
                if self._should_test_eval(
                    _n_selected, last_eval_step, n_features_to_select,
                    len(candidates) - i, test_eval_every_k,
                ):
                    metrics, dt = self._run_test_eval(selected)
                    t_test_eval += dt
                    metrics["step"] = _n_selected
                    last_eval_step = _n_selected
                    performance_history.append(metrics)

                    if self._task_type == "classification":
                        logger.info(
                            "  Test eval step %d — accuracy: %.4f",
                            _n_selected, metrics.get("accuracy", float("nan")),
                        )
                    else:
                        logger.info(
                            "  Test eval step %d — r2: %.4f",
                            _n_selected, metrics.get("r2", float("nan")),
                        )

                    if stopping_metric is not None and stopping_threshold is not None:
                        if (
                            self._resolve_metric(metrics, stopping_metric)
                            >= stopping_threshold
                        ):
                            stopping_reason = "metric_threshold_reached"
                            break

                if self.use_mb_pruning:
                    candidates, np_ = self._prune_candidates(candidates, best, i)
                    n_pruned += np_
                    n_skipped += np_
                    if np_:
                        logger.info("  MB pruned %d candidates", np_)

        # ── STOP selection timer ───────────────────────────────────────
        t_selection_loop = perf_counter() - t0_selection

        if stopping_reason is None:
            stopping_reason = (
                "max_features_reached"
                if len(selected) == n_features_to_select
                else "no_features_remaining"
            )

        logger.info(
            "Selection complete: %d features | reason=%s | "
            "total_time=%.3fs (filter=%.3fs clf=%.3fs test=%.3fs) | "
            "cv_evals=%d pruned=%d skipped=%d",
            len(selected),
            stopping_reason,
            self._filter_time_seconds + t_selection_loop,
            self._filter_time_seconds,
            t_classifier,
            t_test_eval,
            n_selection_evals,
            n_pruned,
            n_skipped,
        )

        # Build short result label from flags
        if self.use_su_ranking and self.use_mb_pruning:
            _label = "IWSS-MB"
        elif self.use_su_ranking:
            _label = "IWSS"
        elif self.use_mb_pruning:
            _label = "SFS-MB"
        else:
            _label = "SFS"

        return WrapperSelectionResult(
            selected_features=selected,
            performance_history=performance_history,
            stopping_reason=stopping_reason,
            n_steps=len(selected),
            final_metrics=performance_history[-1] if performance_history else None,
            selection_time_seconds=self._filter_time_seconds + t_selection_loop,
            evaluation_time_seconds=t_test_eval if performance_history else None,
            task_type=self._task_type,
            label=_label,
            filter_time_seconds=self._filter_time_seconds,
            classifier_time_seconds=t_classifier,
            test_evaluation_time_seconds=t_test_eval,
            n_wrapper_evaluations=n_selection_evals,
            n_candidates_pruned=n_pruned,
            n_evaluations_skipped=n_skipped,
        )
