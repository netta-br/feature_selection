from __future__ import annotations

import copy
import logging
import os
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, r2_score

from ..filter.SymmetricUncertainty import SymmetricUncertainty
from ..precomputation import SymmetricUncertaintyMatrix
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
    classifier : sklearn-compatible estimator or None
        Duck-typed ``fit`` / ``predict`` estimator.  Defaults to
        ``LogisticRegression(max_iter=1000, C=np.inf)``.
    min_features : int
        Minimum set size required before a candidate can be accepted.
    cv_folds : int
        Number of folds for cross-validation inside the selection loop.
    cv_min_folds : int
        Minimum number of folds that must satisfy the dual improvement
        condition for a candidate to be accepted.
    patience : int or None
        Stop after this many consecutive steps with no accepted feature.
        ``None`` disables patience-based stopping.
    su_filepath : str or None
        Mode A: path to a precomputed SU matrix CSV.  If the file exists it
        is loaded; if it does not, the matrix is computed and saved there.
        ``None`` → Mode B (incremental lazy fill).
    random_seed : int or None
        Forwarded to classifier and CV splitter for reproducibility.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_su_ranking: bool = True,
        use_mb_pruning: bool = True,
        classifier=None,
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

        # Classifier
        if classifier is None:
            self._classifier = LogisticRegression(
                max_iter=1000, C=np.inf, random_state=random_seed
            )
        else:
            self._classifier = classifier

        # Task type detection
        self._task_type: str = self._detect_task_type()

        # SU infrastructure (populated only when needed)
        self._X_disc: np.ndarray | None = None
        self._col_index: dict[str, int] | None = None
        self._su_arr: np.ndarray | None = None
        self._col_filled: set[int] = set()
        self._su_bulk_done: bool = False
        self._filter_time_seconds: float = 0.0

        # Candidate list (built at init)
        self._candidates: list[str] = []

        # Build SU and candidates
        self._init_candidates()

        logger.info(
            "WrapperSelector initialised | task=%s | use_su_ranking=%s | "
            "use_mb_pruning=%s | cv_folds=%d | cv_min_folds=%d | "
            "patience=%s | n_candidates=%d",
            self._task_type,
            use_su_ranking,
            use_mb_pruning,
            cv_folds,
            cv_min_folds,
            patience,
            len(self._candidates),
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _detect_task_type(self) -> str:
        if self._y_train.nunique() < 0.1 * len(self._y_train):
            return "classification"
        return "regression"

    def _init_candidates(self) -> None:
        """Build _candidates list; set up SU infrastructure if needed."""
        needs_su = self.use_su_ranking or self.use_mb_pruning

        if not needs_su:
            self._candidates = list(self._X_filled.columns)
            return

        t0 = perf_counter()

        # ── Discretise all columns once ────────────────────────────────
        all_cols = list(self._X_filled.columns) + [self._target_col]
        combined = pd.concat(
            [self._X_filled, self._y_train.rename(self._target_col)], axis=1
        )
        self._X_disc = np.stack(
            [SymmetricUncertainty.discretise(combined[c].values) for c in all_cols],
            axis=1,
        ).astype(np.int8)

        n = len(all_cols)
        self._col_index = {name: i for i, name in enumerate(all_cols)}

        # Allocate SU array
        self._su_arr = np.full((n, n), np.nan, dtype=np.float64)
        np.fill_diagonal(self._su_arr, 1.0)

        # ── Mode A: load or compute full matrix ────────────────────────
        if self.su_filepath is not None:
            self._load_or_compute_su_matrix()
        else:
            # Mode B: fill target column only at init
            self._fill_column(self._target_col)

        self._filter_time_seconds = perf_counter() - t0

        # Build ranked candidates
        tgt_idx = self._col_index[self._target_col]
        feat_cols = list(self._X_filled.columns)
        feat_idxs = np.array([self._col_index[f] for f in feat_cols], dtype=np.intp)
        su_vs_target = self._su_arr[feat_idxs, tgt_idx]

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

    def _load_or_compute_su_matrix(self) -> None:
        """
        Mode A: load SU matrix from CSV, or compute+save if missing.

        If the file already exists but does not yet contain the current target
        column (e.g. a shared SU file was built for a different task), the
        target column is appended by calling ``compute_su_matrix`` again with
        the updated target list derived from the already-loaded matrix.
        """
        assert self.su_filepath is not None
        um = SymmetricUncertaintyMatrix()
        if os.path.exists(self.su_filepath):
            um.load_su_matrix(self.su_filepath)
            # If the current target is absent from the cached matrix, extend it.
            if (
                um.su_matrix is not None
                and self._target_col not in um.su_matrix.columns
            ):
                logger.info(
                    "Target '%s' not found in existing SU matrix at %s — "
                    "extending matrix with new target column.",
                    self._target_col,
                    self.su_filepath,
                )
                SymmetricUncertaintyMatrix.compute_su_matrix(
                    X=self._X_filled,
                    y=self._y_train,
                    filepath=self.su_filepath,
                )
                um.load_su_matrix(self.su_filepath)
        else:
            SymmetricUncertaintyMatrix.compute_su_matrix(
                X=self._X_filled,
                y=self._y_train,
                filepath=self.su_filepath,
            )
            um.load_su_matrix(self.su_filepath)

        if um.su_matrix is None:
            logger.warning(
                "SU matrix failed to load from %s — falling back to Mode B",
                self.su_filepath,
            )
            self._fill_column(self._target_col)
            return

        # Overwrite _su_arr from loaded DataFrame
        df = um.su_matrix
        for col_name, col_idx in self._col_index.items():
            if col_name in df.index:
                for row_name, row_idx in self._col_index.items():
                    if row_name in df.columns:
                        val = df.loc[row_name, col_name]
                        if not np.isnan(val):
                            self._su_arr[row_idx, col_idx] = float(val)

        self._su_bulk_done = True
        self._col_filled = set(self._col_index.values())

    # ------------------------------------------------------------------
    # Incremental column-fill (Mode B)
    # ------------------------------------------------------------------
    def _fill_column(self, col_name: str) -> None:
        """
        Fill the row/column for *col_name* in ``_su_arr`` in one vectorised
        call.  No-op if already filled or ``_su_bulk_done``.
        """
        if self._su_arr is None or self._col_index is None or self._X_disc is None:
            return
        if self._su_bulk_done:
            return
        idx = self._col_index.get(col_name)
        if idx is None or idx in self._col_filled:
            return

        anchor = self._X_disc[:, idx]
        su_row = SymmetricUncertainty.compute_su_column(anchor, self._X_disc)

        self._su_arr[idx, :] = su_row
        self._su_arr[:, idx] = su_row
        self._col_filled.add(idx)

    def _get_su(self, col_a: str, col_b: str) -> float:
        """Integer-index SU access with lazy fill on miss."""
        if self._su_arr is None or self._col_index is None:
            return 0.0
        ia = self._col_index.get(col_a)
        ib = self._col_index.get(col_b)
        if ia is None or ib is None:
            return 0.0
        if ia not in self._col_filled:
            self._fill_column(col_a)
        if ib not in self._col_filled:
            self._fill_column(col_b)
        val = self._su_arr[ia, ib]
        return float(val) if not np.isnan(val) else 0.0

    # ------------------------------------------------------------------
    # Vectorised MB pruning
    # ------------------------------------------------------------------
    def _mb_prune_vectorised(self, selected_feat: str) -> list[str]:
        """
        Vectorised Approximate Markov Blanket pruning (Definition 5).

        ``_fill_column(selected_feat)`` must have been called before this.

        Returns list of pruned feature names (removed from ``_candidates``).
        """
        if not self._candidates:
            return []
        assert self._su_arr is not None
        assert self._col_index is not None

        sel_idx = self._col_index[selected_feat]
        tgt_idx = self._col_index[self._target_col]
        cand_idxs = np.array(
            [self._col_index[c] for c in self._candidates], dtype=np.intp
        )

        su_sel_C    = self._su_arr[sel_idx, tgt_idx]           # scalar
        su_cand_C   = self._su_arr[cand_idxs, tgt_idx]         # vector
        su_sel_cand = self._su_arr[sel_idx, cand_idxs]         # vector

        redundant_mask = (su_sel_C >= su_cand_C) & (su_sel_cand > su_cand_C)

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
    # CV evaluation (selection-driving, on X_train only)
    # ------------------------------------------------------------------
    def _cv_evaluate(
        self,
        features: list[str],
        baseline_scores: np.ndarray,
    ) -> tuple[float, int, np.ndarray]:
        """
        k-fold CV on ``_X_filled[features]`` / ``y_train``.

        Acceptance criterion per fold (dual condition):
          (a) fold_score_new >= baseline_mean  (beats global average baseline)
          (b) fold_score_new >  baseline_scores[k]  (beats this fold's baseline)

        Parameters
        ----------
        features : list[str]
            Feature set to evaluate (includes the candidate).
        baseline_scores : np.ndarray
            Per-fold scores of the current selected set (before adding
            candidate).  Shape ``(cv_folds,)``.

        Returns
        -------
        mean_score : float
        n_improved : int
            Count of folds satisfying the dual condition.
        fold_scores : np.ndarray
            Per-fold scores for this call (shape ``(cv_folds,)``).
        """
        X = self._X_filled[features].values
        y = self._y_train.values
        baseline_mean = float(baseline_scores.mean())

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
            clf = copy.deepcopy(self._classifier)
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[val_idx])
            if self._task_type == "classification":
                fold_scores[k] = accuracy_score(y[val_idx], y_pred)
            else:
                fold_scores[k] = r2_score(y[val_idx], y_pred)

        mean_score = float(fold_scores.mean())

        # Dual condition: (a) >= baseline_mean  AND  (b) > per-fold baseline
        improved = (fold_scores >= baseline_mean) & (fold_scores > baseline_scores)
        n_improved = int(improved.sum())

        return mean_score, n_improved, fold_scores

    # ------------------------------------------------------------------
    # Test/validation evaluation (outside timed loop)
    # ------------------------------------------------------------------
    def _test_evaluate(self, features: list[str]) -> dict:
        """
        Fit classifier on full training set, score on validation set.

        Returns a metrics dict compatible with SelectionResult's
        performance_history format.
        """
        clf = copy.deepcopy(self._classifier)
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

        # Reset mutable candidate list to a fresh copy for this run
        candidates = list(self._candidates)  # local copy; may be mutated by MB pruning

        selected: list[str] = []
        # CV baseline: per-fold scores for the *current* selected set
        cv_baseline_scores = np.zeros(self.cv_folds, dtype=np.float64)
        cv_baseline_mean: float = 0.0

        performance_history: list[dict] = []
        stopping_reason: str | None = None

        n_selection_evals: int = 0
        t_classifier: float = 0.0
        t_test_eval: float = 0.0
        n_pruned: int = 0
        n_skipped: int = 0
        no_improvement_streak: int = 0
        _last_eval_step: int = 0   # tracks step at which test eval last ran

        # ── START selection timer ──────────────────────────────────────
        t0_selection = perf_counter()

        i = 0
        while i < len(candidates) and len(selected) < n_features_to_select:

            if not self.use_su_ranking:
                # ── SFS mode: exhaustive search over remaining candidates ──
                best_feat: str | None = None
                best_mean: float = cv_baseline_mean
                best_n_improved: int = 0
                best_fold_scores: np.ndarray | None = None

                for c in candidates[i:]:
                    if self.use_mb_pruning:
                        self._fill_column(c)
                    t0_clf = perf_counter()
                    mean_s, n_imp, fold_s = self._cv_evaluate(
                        [*selected, c], cv_baseline_scores
                    )
                    t_classifier += perf_counter() - t0_clf
                    n_selection_evals += 1

                    if (
                        n_imp >= self.cv_min_folds
                        and mean_s > best_mean
                        and len(selected) + 1 >= self.min_features
                    ):
                        best_feat = c
                        best_mean = mean_s
                        best_n_improved = n_imp
                        best_fold_scores = fold_s

                if best_feat is None:
                    no_improvement_streak += 1
                    if (
                        self.patience is not None
                        and no_improvement_streak >= self.patience
                    ):
                        stopping_reason = "patience_exceeded"
                        break
                    stopping_reason = "no_improvement"
                    break

                # Move best to position i
                candidates.remove(best_feat)
                candidates.insert(i, best_feat)
                selected.append(best_feat)
                cv_baseline_scores = best_fold_scores  # type: ignore[assignment]
                cv_baseline_mean = best_mean
                no_improvement_streak = 0

                logger.info(
                    "Step %d (SFS): selected '%s' mean_cv=%.4f n_improved=%d",
                    len(selected), best_feat, best_mean, best_n_improved,
                )

                if self.use_mb_pruning:
                    # Remove the accepted feature (at position i) before pruning
                    if i < len(candidates) and candidates[i] == best_feat:
                        candidates.pop(i)
                    self._candidates = candidates  # sync for _mb_prune_vectorised
                    pruned = self._mb_prune_vectorised(best_feat)
                    candidates = self._candidates   # sync back
                    n_pruned += len(pruned)
                    n_skipped += len(pruned)
                    if pruned:
                        logger.info(
                            "  MB pruned %d candidates: %s",
                            len(pruned), pruned[:5],
                        )

            else:
                # ── IWSS mode: evaluate next ranked candidate ──────────
                candidate = candidates[i]

                if self.use_mb_pruning:
                    self._fill_column(candidate)

                t0_clf = perf_counter()
                mean_s, n_imp, fold_s = self._cv_evaluate(
                    [*selected, candidate], cv_baseline_scores
                )
                t_classifier += perf_counter() - t0_clf
                n_selection_evals += 1

                accepted = (
                    n_imp >= self.cv_min_folds
                    and mean_s > cv_baseline_mean
                    and len(selected) + 1 >= self.min_features
                )

                if accepted:
                    selected.append(candidate)
                    cv_baseline_scores = fold_s
                    cv_baseline_mean = mean_s
                    no_improvement_streak = 0

                    logger.info(
                        "Step %d (IWSS): selected '%s' mean_cv=%.4f n_improved=%d",
                        len(selected), candidate, mean_s, n_imp,
                    )

                    if self.use_mb_pruning:
                        # Remove the accepted feature at position i before pruning
                        if i < len(candidates) and candidates[i] == candidate:
                            candidates.pop(i)
                        self._candidates = candidates
                        pruned = self._mb_prune_vectorised(candidate)
                        candidates = self._candidates
                        n_pruned += len(pruned)
                        n_skipped += len(pruned)
                        if pruned:
                            logger.info(
                                "  MB pruned %d candidates: %s",
                                len(pruned), pruned[:5],
                            )
                        # i stays the same — pop(i) already advanced past the
                        # accepted feature; outer i += 1 would skip the next
                        # candidate, so we continue directly.
                        i += 1
                        continue
                    # Without MB pruning: outer i += 1 advances past accepted feature
                else:
                    no_improvement_streak += 1
                    logger.debug(
                        "  Candidate '%s' rejected (mean=%.4f n_imp=%d streak=%d)",
                        candidate, mean_s, n_imp, no_improvement_streak,
                    )
                    if (
                        self.patience is not None
                        and no_improvement_streak >= self.patience
                    ):
                        stopping_reason = "patience_exceeded"
                        break

                i += 1

            # ── Test/validation evaluation (OUTSIDE timed loop) ───────
            # Only trigger when a new feature was actually added since last eval.
            _n_selected = len(selected)
            _should_test_eval = (
                _n_selected > _last_eval_step  # a new feature was added
                and (
                    _n_selected % test_eval_every_k == 0
                    or _n_selected == n_features_to_select
                    or i >= len(candidates)
                )
            )
            if _should_test_eval:
                t0_test = perf_counter()
                metrics = self._test_evaluate(selected)
                t_test_eval += perf_counter() - t0_test
                metrics["step"] = _n_selected
                _last_eval_step = _n_selected
                performance_history.append(metrics)

                if self._task_type == "classification":
                    acc = metrics.get("accuracy", float("nan"))
                    logger.info(
                        "  Test eval step %d — accuracy: %.4f",
                        len(selected), acc,
                    )
                else:
                    logger.info(
                        "  Test eval step %d — r2: %.4f",
                        len(selected), metrics.get("r2", float("nan")),
                    )

                if stopping_metric is not None and stopping_threshold is not None:
                    if (
                        self._resolve_metric(metrics, stopping_metric)
                        >= stopping_threshold
                    ):
                        stopping_reason = "metric_threshold_reached"
                        break

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

        return WrapperSelectionResult(
            selected_features=selected,
            performance_history=performance_history,
            stopping_reason=stopping_reason,
            n_steps=len(selected),
            final_metrics=performance_history[-1] if performance_history else None,
            selection_time_seconds=self._filter_time_seconds + t_selection_loop,
            evaluation_time_seconds=t_test_eval if performance_history else None,
            task_type=self._task_type,
            filter_time_seconds=self._filter_time_seconds,
            classifier_time_seconds=t_classifier,
            test_evaluation_time_seconds=t_test_eval,
            n_wrapper_evaluations=n_selection_evals,
            n_candidates_pruned=n_pruned,
            n_evaluations_skipped=n_skipped,
        )
