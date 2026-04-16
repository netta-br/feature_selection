"""
ScoreCalculator — single-target stateful computation engine.

Operates on N features + 1 target.  Owns all score computation including the
full SU pipeline (discretisation, entropy, MI, SU), accumulator vectors,
relevance cache, and discretisation cache.

Used directly by selectors in Mode B (no file) or internally by ScoreMatrix
for per-target relevance columns.
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("ScoreCalculator")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------
def infer_task_type(y: np.ndarray, threshold: float = 0.1) -> str:
    """Infer whether the target represents classification or regression.

    Parameters
    ----------
    y : np.ndarray
        Target vector.
    threshold : float
        If ``unique(y) / len(y) < threshold`` the task is classification.

    Returns
    -------
    str
        ``'classification'`` or ``'regression'``.
    """
    return "classification" if np.unique(y).size < threshold * len(y) else "regression"


# ---------------------------------------------------------------------------
# ScoreCalculator
# ---------------------------------------------------------------------------
class ScoreCalculator:
    """Single-target stateful computation engine for feature scoring.

    Supports methods: ``'pearson'``, ``'kendall'``, ``'spearman'``, ``'su'``,
    ``'mutual_information'``, ``'random_forest'``, ``'f_statistic'``.

    Parameters
    ----------
    X : np.ndarray
        ``(n_samples, N)`` feature matrix — referenced, **not** copied.
    feature_names : list[str]
        Column names for features (length N).
    target_name : str
        Name of the single target column.
    target_col : np.ndarray
        ``(n_samples,)`` target vector — referenced, **not** copied.
    method : str
        Scoring method.
    redundancy_agg : str
        ``'mean'`` or ``'max'`` — how redundancy accumulates.
    method_params : dict | None
        Extra parameters forwarded to sklearn estimators (RF n_estimators, etc.).
    random_seed : int | None
        For reproducibility.
    """

    _CORR_METHODS = {"pearson", "kendall", "spearman"}
    _NON_CORR_METHODS = {"mutual_information", "random_forest", "f_statistic"}
    _ALL_METHODS = _CORR_METHODS | _NON_CORR_METHODS | {"su"}

    def __init__(
        self,
        X: np.ndarray,
        feature_names: list[str],
        target_name: str,
        target_col: np.ndarray,
        method: str,
        redundancy_agg: str = "mean",
        method_params: dict | None = None,
        random_seed: int | None = None,
    ) -> None:
        # ── Validation ─────────────────────────────────────────────────
        if method not in self._ALL_METHODS:
            raise ValueError(
                f"method must be one of {self._ALL_METHODS}; got {method!r}"
            )
        if target_name in feature_names:
            raise ValueError(
                f"target_name {target_name!r} must not appear in feature_names"
            )
        if redundancy_agg not in {"mean", "max"}:
            raise ValueError(
                f"redundancy_agg must be 'mean' or 'max'; got {redundancy_agg!r}"
            )

        # ── Public attributes ──────────────────────────────────────────
        self.method = method
        self.redundancy_agg = redundancy_agg
        self.method_params = method_params or {}
        self.random_seed = random_seed

        # ── Internal references (no copy) ──────────────────────────────
        self._X = X                        # (n_samples, N)
        self._target_col = target_col      # (n_samples,)
        self._target_name = target_name
        self._feature_names = list(feature_names)
        self._N = len(feature_names)       # number of features (target idx = N)

        # name → column index  (features 0…N-1, target at N)
        self._index: dict[str, int] = {
            name: i for i, name in enumerate(feature_names)
        }
        self._index[target_name] = self._N

        # ── Lazy caches ────────────────────────────────────────────────
        self._X_disc: np.ndarray | None = None       # (n_samples, N+1) int8
        self._entropies: np.ndarray | None = None    # (N+1,) float64

        # ── Accumulator state ──────────────────────────────────────────
        self._relevance_vec: np.ndarray | None = None  # (N,)
        self._redundancy_acc: np.ndarray | None = None  # (N,)
        self._n_selected: int = 0

        # ── Task type ──────────────────────────────────────────────────
        self._task_type: str = infer_task_type(target_col)

    # ------------------------------------------------------------------
    # 2c — Lazy discretisation + entropy cache
    # ------------------------------------------------------------------
    def _ensure_disc(self) -> None:
        """Discretise all N features + 1 target into 3-bin z-score scheme.

        Also computes all (N+1) Shannon entropies in a single pass.
        No-op if already cached.
        """
        if self._X_disc is not None:
            return

        # (n_samples, N+1)
        all_data = np.concatenate(
            [self._X, self._target_col[:, None]], axis=1
        )
        std = all_data.std(axis=0)
        std[std == 0] = 1.0
        z = (all_data - all_data.mean(axis=0)) / std
        self._X_disc = np.digitize(z, bins=[-0.5, 0.5]).astype(np.int8)

        # Vectorised entropy for all columns
        N_BINS = 3
        n = all_data.shape[0]
        counts = np.zeros((self._X_disc.shape[1], N_BINS), dtype=np.int64)
        for b in range(N_BINS):
            counts[:, b] = (self._X_disc == b).sum(axis=0)
        probs = counts / n
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(probs > 0, np.log(probs), 0.0)
        self._entropies = -(probs * lp).sum(axis=1)  # (N+1,)

    # ------------------------------------------------------------------
    # 2d — Fully vectorised SU row (np.bincount, no Python loop)
    # ------------------------------------------------------------------
    def _compute_su_row(
        self,
        anchor_idx: int,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        """Vectorised SU(anchor, active[k]) for each k.  No Python loop.

        Parameters
        ----------
        anchor_idx : int
            Column index into ``_X_disc`` for the anchor.
        active_indices : np.ndarray
            1-D int array of column indices for the active features.

        Returns
        -------
        np.ndarray
            ``(K,)`` float64 SU values.
        """
        self._ensure_disc()
        K = len(active_indices)
        if K == 0:
            return np.empty(0, dtype=np.float64)

        h_anchor = self._entropies[anchor_idx]              # scalar
        h_active = self._entropies[active_indices]           # (K,)

        anchor_disc = self._X_disc[:, anchor_idx]            # (n,)
        active_disc = self._X_disc[:, active_indices]        # (n, K)
        n = anchor_disc.shape[0]

        # Joint counts via np.bincount on flat index
        joint_codes = anchor_disc[:, None] * 3 + active_disc  # (n, K) values in 0..8
        offsets = np.arange(K, dtype=np.int64)[None, :] * 9   # (1, K)
        flat = (offsets + joint_codes.astype(np.int64)).ravel()
        joint_counts = np.bincount(flat, minlength=K * 9).reshape(K, 9)

        # Joint entropy  H(anchor, active[k])
        joint_probs = joint_counts / n
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(joint_probs > 0, np.log(joint_probs), 0.0)
        h_xy = -(joint_probs * lp).sum(axis=1)  # (K,)

        # MI = H(X) + H(Y) - H(X,Y)
        mi = np.maximum(h_anchor + h_active - h_xy, 0.0)

        # SU = 2 * MI / (H(X) + H(Y))
        denom = h_anchor + h_active
        with np.errstate(divide="ignore", invalid="ignore"):
            su = np.where(denom > 0, 2.0 * mi / denom, 0.0)
        return np.clip(su, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 2e — compute_row (pure computation, no side effects)
    # ------------------------------------------------------------------
    def compute_row(
        self,
        anchor: str | int,
        active_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute score vector for *anchor* vs each active feature.

        Parameters
        ----------
        anchor : str | int
            Feature name or integer index.
        active_indices : np.ndarray | None
            Integer indices into ``_index`` space.  ``None`` → all N features.

        Returns
        -------
        np.ndarray
            ``(|active|,)`` score vector.  No side effects on accumulators.
        """
        # Resolve anchor to integer index
        if isinstance(anchor, str):
            anchor_idx = self._index[anchor]
        else:
            anchor_idx = int(anchor)

        if active_indices is None:
            active_indices = np.arange(self._N, dtype=np.intp)

        # ── Dispatch on method ─────────────────────────────────────────
        if self.method == "su":
            return self._compute_su_row(anchor_idx, active_indices)

        if self.method == "pearson":
            return self._compute_pearson_row(anchor_idx, active_indices)

        if self.method in ("kendall", "spearman"):
            return self._compute_rank_corr_row(anchor_idx, active_indices)

        if self.method in self._NON_CORR_METHODS:
            # Non-correlation methods: only valid when anchor is the target
            if anchor_idx != self._N:
                raise ValueError(
                    f"Method {self.method!r} only supports target as anchor, "
                    f"got feature index {anchor_idx}"
                )
            return self._compute_sklearn_relevance(active_indices)

        raise ValueError(f"Unknown method: {self.method!r}")

    # ------------------------------------------------------------------
    # Pearson (vectorised via np.corrcoef-style manual calculation)
    # ------------------------------------------------------------------
    def _compute_pearson_row(
        self,
        anchor_idx: int,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        """Pearson |r| between anchor and each active column."""
        anchor_data = self._get_column(anchor_idx).astype(np.float64)
        active_data = self._get_columns(active_indices).astype(np.float64)

        # Centre
        anchor_centered = anchor_data - anchor_data.mean()
        active_centered = active_data - active_data.mean(axis=0)

        # Dot products
        n = len(anchor_data)
        cov = (anchor_centered[:, None] * active_centered).sum(axis=0) / n
        std_a = anchor_centered.std()
        std_active = active_centered.std(axis=0)

        denom = std_a * std_active
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(denom > 0, cov / denom, 0.0)
        return np.abs(r)

    # ------------------------------------------------------------------
    # Kendall / Spearman (via pandas for scipy backend)
    # ------------------------------------------------------------------
    def _compute_rank_corr_row(
        self,
        anchor_idx: int,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        """Rank-based |r| (kendall or spearman)."""
        anchor_series = pd.Series(self._get_column(anchor_idx))
        active_data = self._get_columns(active_indices)
        active_df = pd.DataFrame(active_data, columns=range(len(active_indices)))

        corr = active_df.corrwith(anchor_series, method=self.method)
        result = corr.values.astype(np.float64)
        result = np.nan_to_num(result, nan=0.0)
        return np.abs(result)

    # ------------------------------------------------------------------
    # sklearn-based relevance (MI, RF, F-stat)
    # ------------------------------------------------------------------
    def _compute_sklearn_relevance(
        self,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute relevance scores for active features vs target using sklearn.

        Only valid when anchor is the target column.
        """
        from sklearn.feature_selection import (
            mutual_info_classif,
            mutual_info_regression,
            f_classif,
            f_regression,
        )
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        X_active = self._X[:, active_indices]
        y = self._target_col

        if self.method == "mutual_information":
            fn = (
                mutual_info_classif
                if self._task_type == "classification"
                else mutual_info_regression
            )
            kwargs = {}
            n_neighbors = self.method_params.get("n_neighbors")
            if n_neighbors is not None:
                kwargs["n_neighbors"] = n_neighbors
            scores = fn(X_active, y, random_state=self.random_seed, **kwargs)

        elif self.method == "f_statistic":
            fn = f_classif if self._task_type == "classification" else f_regression
            scores, _ = fn(X_active, y)

        elif self.method == "random_forest":
            Cls = (
                RandomForestClassifier
                if self._task_type == "classification"
                else RandomForestRegressor
            )
            n_est = self.method_params.get("n_estimators", 100)
            max_depth = self.method_params.get("max_depth", None)
            rf = Cls(
                n_estimators=n_est,
                max_depth=max_depth,
                random_state=self.random_seed,
            )
            rf.fit(X_active, y)
            scores = rf.feature_importances_
        else:
            raise ValueError(f"Unknown sklearn method: {self.method!r}")

        scores = np.where(np.isnan(scores) | np.isinf(scores), 0.0, scores)
        return scores.astype(np.float64)

    # ------------------------------------------------------------------
    # Column access helpers
    # ------------------------------------------------------------------
    def _get_column(self, idx: int) -> np.ndarray:
        """Return 1-D column by index (feature or target)."""
        if idx == self._N:
            return self._target_col
        return self._X[:, idx]

    def _get_columns(self, idxs: np.ndarray) -> np.ndarray:
        """Return 2-D (n_samples, len(idxs)) array for given indices."""
        n = len(idxs)
        result = np.empty((self._X.shape[0], n), dtype=self._X.dtype)
        for k, idx in enumerate(idxs):
            if idx == self._N:
                result[:, k] = self._target_col
            else:
                result[:, k] = self._X[:, idx]
        return result

    # ------------------------------------------------------------------
    # 2f — init_target: compute relevance vector
    # ------------------------------------------------------------------
    def init_target(self) -> None:
        """Compute relevance for all N features vs the target.

        Stores the result in ``_relevance_vec``.
        Resets ``_redundancy_acc`` to zeros and ``_n_selected`` to 0.
        """
        target_idx = self._N
        all_feat_idxs = np.arange(self._N, dtype=np.intp)

        self._relevance_vec = self.compute_row(target_idx, all_feat_idxs)
        self._redundancy_acc = np.zeros(self._N, dtype=np.float64)
        self._n_selected = 0

        logger.info(
            "init_target complete: method=%s, N=%d features, target=%s",
            self.method, self._N, self._target_name,
        )

    # ------------------------------------------------------------------
    # 2g — update_redundancy: accumulate
    # ------------------------------------------------------------------
    def update_redundancy(
        self,
        selected_feat: str | int,
        remaining_idxs: np.ndarray,
    ) -> None:
        """Compute scores(selected_feat, remaining) and update accumulator.

        For ``mean``: ``_redundancy_acc[remaining] += new_scores; _n_selected += 1``.
        For ``max``:  ``np.maximum(_redundancy_acc[remaining], new_scores, out=...)``.
        """
        if self._redundancy_acc is None:
            raise RuntimeError("Call init_target() before update_redundancy()")

        new_scores = self.compute_row(selected_feat, remaining_idxs)

        if self.redundancy_agg == "mean":
            self._redundancy_acc[remaining_idxs] += new_scores
            self._n_selected += 1
        else:  # max
            np.maximum(
                self._redundancy_acc[remaining_idxs],
                new_scores,
                out=self._redundancy_acc[remaining_idxs],
            )
            self._n_selected += 1

    # ------------------------------------------------------------------
    # 2h — get_relevance / get_redundancy: read slices
    # ------------------------------------------------------------------
    def get_relevance(self, idxs: np.ndarray) -> np.ndarray:
        """Return ``_relevance_vec[idxs]`` — zero-copy slice.

        Parameters
        ----------
        idxs : np.ndarray
            Integer feature indices.

        Returns
        -------
        np.ndarray
            Relevance scores for the requested features.
        """
        if self._relevance_vec is None:
            raise RuntimeError("Call init_target() before get_relevance()")
        return self._relevance_vec[idxs]

    def get_redundancy(self, idxs: np.ndarray) -> np.ndarray:
        """Return normalised redundancy for the requested features.

        For ``mean``: ``_redundancy_acc[idxs] / _n_selected`` (zeros if 0).
        For ``max``:  ``_redundancy_acc[idxs]`` directly.

        Parameters
        ----------
        idxs : np.ndarray
            Integer feature indices.

        Returns
        -------
        np.ndarray
            Redundancy scores.
        """
        if self._redundancy_acc is None:
            raise RuntimeError("Call init_target() before get_redundancy()")
        if self.redundancy_agg == "mean":
            if self._n_selected == 0:
                return np.zeros(len(idxs), dtype=np.float64)
            return self._redundancy_acc[idxs] / self._n_selected
        else:  # max
            return self._redundancy_acc[idxs]

    # ------------------------------------------------------------------
    # 2i — __repr__
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ScoreCalculator(method={self.method!r}, N={self._N}, "
            f"target={self._target_name!r})"
        )
