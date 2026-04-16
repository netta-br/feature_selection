"""
ScoreMatrix — Mode A disk-backed full matrix supporting k ≥ 1 targets.

Creates per-target ``ScoreCalculator`` instances internally for target
relevance columns; feature-feature scores are computed once.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

from .score_calculator import ScoreCalculator

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("ScoreMatrix")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# ScoreMatrix
# ---------------------------------------------------------------------------
class ScoreMatrix:
    """Disk-backed full score matrix for Mode A (k ≥ 1 targets).

    Builds an ``(N+k) × (N+k)`` score array where N is the number of features
    and k the number of targets.  Feature-feature scores are computed once;
    per-target relevance rows use dedicated ``ScoreCalculator`` instances.

    Parameters
    ----------
    X : np.ndarray
        ``(n_samples, N)`` feature matrix — referenced, not copied.
    feature_names : list[str]
        Length-N list of feature column names.
    target_names : list[str]
        Length-k list of target column names.
    target_data : np.ndarray
        ``(n_samples, k)`` target data — referenced, not copied.
    method : str
        Scoring method forwarded to ``ScoreCalculator``.
    filepath : str
        Path for loading/saving the matrix (CSV or Parquet).
        The file format is **inferred from the extension** (``.csv`` →
        CSV, ``.parquet`` / ``.pq`` → Parquet).
    file_format : str
        Fallback format when the filepath extension is not recognised.
        ``'parquet'`` (default) or ``'csv'``.
    method_params : dict | None
        Extra params forwarded to ``ScoreCalculator``.
    random_seed : int | None
        For reproducibility.
    """

    # Map of recognised extensions → canonical format strings
    _EXT_TO_FORMAT: dict[str, str] = {
        ".parquet": "parquet",
        ".pq": "parquet",
        ".csv": "csv",
    }

    def __init__(
        self,
        X: np.ndarray,
        feature_names: list[str],
        target_names: list[str],
        target_data: np.ndarray,
        method: str,
        filepath: str,
        file_format: str = "parquet",
        method_params: dict | None = None,
        random_seed: int | None = None,
    ) -> None:
        self._X = X
        self._feature_names = list(feature_names)
        self._target_names = list(target_names)
        self._target_data = target_data if target_data.ndim == 2 else target_data[:, None]
        self.method = method
        self._filepath = filepath

        # Infer file format from the filepath extension; fall back to the
        # explicit *file_format* parameter only when the extension is not
        # recognised.
        ext = os.path.splitext(filepath)[1].lower()
        inferred = self._EXT_TO_FORMAT.get(ext)
        if inferred is not None and inferred != file_format:
            logger.info(
                "file_format overridden: filepath extension '%s' → '%s' "
                "(caller passed '%s')",
                ext, inferred, file_format,
            )
        self._file_format = inferred if inferred is not None else file_format
        self._method_params = method_params or {}
        self._random_seed = random_seed

        N = len(feature_names)
        k = len(target_names)
        self._N = N
        self._k = k

        # Name → index mapping  (features 0…N-1, targets N…N+k-1)
        all_names = self._feature_names + self._target_names
        self._index: dict[str, int] = {
            name: i for i, name in enumerate(all_names)
        }

        # (N+k) × (N+k) float64 row-major C-order
        self._arr = np.full((N + k, N + k), np.nan, dtype=np.float64)
        np.fill_diagonal(self._arr, 1.0)
        self._precomputed = False

    # ------------------------------------------------------------------
    # precompute
    # ------------------------------------------------------------------
    def precompute(self) -> None:
        """Load from file if it exists; otherwise compute full matrix and save.

        Feature-feature block: one ``ScoreCalculator`` (first target).
        Per-target relevance: one ``ScoreCalculator`` per target.
        """
        if self._precomputed:
            return

        if os.path.exists(self._filepath):
            self._load_from_file()
            self._precomputed = True
            return

        self._compute_full_matrix()
        self._save_to_file()
        self._precomputed = True

    # ------------------------------------------------------------------
    # _load_from_file
    # ------------------------------------------------------------------
    def _load_from_file(self) -> None:
        """Load matrix from disk, reindex to ``_index`` order, validate."""
        if self._file_format == "parquet":
            df = pd.read_parquet(self._filepath)
            # Parquet stores column names; index is first column or default
            if df.index.dtype == object:
                pass  # already has string index
            else:
                df = df.set_index(df.columns[0]) if df.columns[0] not in self._index else df
        else:
            df = pd.read_csv(self._filepath, index_col=0)

        # Reindex to our canonical order
        all_names = self._feature_names + self._target_names
        # Only use names that exist in both the file and our index
        common = [n for n in all_names if n in df.columns and n in df.index]
        if len(common) == 0:
            logger.warning(
                "Loaded matrix from %s has no overlapping columns — "
                "computing from scratch",
                self._filepath,
            )
            self._compute_full_matrix()
            self._save_to_file()
            return

        df_reindexed = df.reindex(index=all_names, columns=all_names)
        loaded = df_reindexed.values.astype(np.float64)

        # Copy non-NaN values into our array
        valid = ~np.isnan(loaded)
        self._arr[valid] = loaded[valid]

        logger.info(
            "Loaded score matrix from %s (%d×%d, %d valid entries)",
            self._filepath,
            df.shape[0],
            df.shape[1],
            int(valid.sum()),
        )

        # Check if there are still NaN entries that need computing
        still_nan = np.isnan(self._arr)
        np.fill_diagonal(still_nan, False)  # diagonal is always 1.0
        if still_nan.any():
            logger.info(
                "Matrix has %d NaN entries — filling with computation",
                int(still_nan.sum()),
            )
            self._fill_missing_entries()

    # ------------------------------------------------------------------
    # _compute_full_matrix
    # ------------------------------------------------------------------
    def _compute_full_matrix(self) -> None:
        """Compute the full (N+k)×(N+k) matrix from scratch."""
        N = self._N
        k = self._k

        # ── Feature-feature block (N×N) ────────────────────────────────
        # Use first target to create a ScoreCalculator
        first_target = self._target_data[:, 0]
        first_target_name = self._target_names[0]

        calc = ScoreCalculator(
            X=self._X,
            feature_names=self._feature_names,
            target_name=first_target_name,
            target_col=first_target,
            method=self.method,
            method_params=self._method_params,
            random_seed=self._random_seed,
        )

        if self.method == "su":
            # SU is symmetric — compute upper triangle only
            all_feat_idxs = np.arange(N, dtype=np.intp)
            for i in range(N):
                # Compute SU(i, j) for j > i
                remaining = np.arange(i + 1, N, dtype=np.intp)
                if len(remaining) == 0:
                    continue
                scores = calc.compute_row(i, remaining)
                self._arr[i, remaining] = scores
                self._arr[remaining, i] = scores  # symmetric

            logger.info("Feature-feature SU block computed (N=%d)", N)
        else:
            # Non-SU methods: compute full rows
            all_feat_idxs = np.arange(N, dtype=np.intp)
            for i in range(N):
                scores = calc.compute_row(i, all_feat_idxs)
                self._arr[i, :N] = scores

            # Apply abs for correlation methods
            if self.method in ScoreCalculator._CORR_METHODS:
                self._arr[:N, :N] = np.abs(self._arr[:N, :N])
                np.nan_to_num(self._arr[:N, :N], nan=0.0, copy=False)

            logger.info("Feature-feature block computed (N=%d, method=%s)", N, self.method)

        # ── Target relevance rows (per target) ─────────────────────────
        for t in range(k):
            target_col = self._target_data[:, t]
            target_name = self._target_names[t]
            target_idx = N + t

            calc_t = ScoreCalculator(
                X=self._X,
                feature_names=self._feature_names,
                target_name=target_name,
                target_col=target_col,
                method=self.method,
                method_params=self._method_params,
                random_seed=self._random_seed,
            )

            feat_idxs = np.arange(N, dtype=np.intp)
            # target is at index N in the ScoreCalculator's namespace
            scores = calc_t.compute_row(target_name, feat_idxs)
            self._arr[target_idx, :N] = scores
            self._arr[:N, target_idx] = scores  # symmetric fill

            logger.info(
                "Target '%s' relevance row computed (%d features)",
                target_name,
                N,
            )

        # Ensure diagonal is 1.0
        np.fill_diagonal(self._arr, 1.0)

    # ------------------------------------------------------------------
    # _fill_missing_entries
    # ------------------------------------------------------------------
    def _fill_missing_entries(self) -> None:
        """Fill any remaining NaN entries via ScoreCalculator."""
        N = self._N

        # Use first target for feature-feature block
        first_target = self._target_data[:, 0]
        first_target_name = self._target_names[0]

        calc = ScoreCalculator(
            X=self._X,
            feature_names=self._feature_names,
            target_name=first_target_name,
            target_col=first_target,
            method=self.method,
            method_params=self._method_params,
            random_seed=self._random_seed,
        )

        # Feature-feature NaNs
        for i in range(N):
            nan_mask = np.isnan(self._arr[i, :N])
            if not nan_mask.any():
                continue
            nan_idxs = np.where(nan_mask)[0].astype(np.intp)
            scores = calc.compute_row(i, nan_idxs)
            self._arr[i, nan_idxs] = scores
            self._arr[nan_idxs, i] = scores

        # Target rows
        for t in range(self._k):
            target_idx = N + t
            nan_mask = np.isnan(self._arr[target_idx, :N])
            if not nan_mask.any():
                continue

            target_col = self._target_data[:, t]
            target_name = self._target_names[t]

            calc_t = ScoreCalculator(
                X=self._X,
                feature_names=self._feature_names,
                target_name=target_name,
                target_col=target_col,
                method=self.method,
                method_params=self._method_params,
                random_seed=self._random_seed,
            )

            nan_idxs = np.where(nan_mask)[0].astype(np.intp)
            scores = calc_t.compute_row(target_name, nan_idxs)
            self._arr[target_idx, nan_idxs] = scores
            self._arr[nan_idxs, target_idx] = scores

        np.fill_diagonal(self._arr, 1.0)

    # ------------------------------------------------------------------
    # _save_to_file
    # ------------------------------------------------------------------
    def _save_to_file(self) -> None:
        """Save matrix to disk in the configured format."""
        all_names = self._feature_names + self._target_names
        df = pd.DataFrame(self._arr, index=all_names, columns=all_names)

        # Ensure directory exists
        dirpath = os.path.dirname(self._filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        if self._file_format == "parquet":
            df.to_parquet(self._filepath)
        else:
            df.to_csv(self._filepath)

        logger.info(
            "Score matrix saved to %s (%s, %d×%d)",
            self._filepath,
            self._file_format,
            *df.shape,
        )

    # ------------------------------------------------------------------
    # get — O(1) read
    # ------------------------------------------------------------------
    def get(self, row: str, col: str) -> float:
        """O(1) read from ``_arr``.

        Raises ``ValueError`` if the value is NaN.

        Parameters
        ----------
        row, col : str
            Feature or target names.

        Returns
        -------
        float
        """
        i = self._index[row]
        j = self._index[col]
        val = self._arr[i, j]
        if np.isnan(val):
            raise ValueError(
                f"Score for ({row!r}, {col!r}) is NaN — matrix not fully computed"
            )
        return float(val)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def arr(self) -> np.ndarray:
        """Direct array access for hot paths."""
        return self._arr

    @property
    def index(self) -> dict[str, int]:
        """Name → index mapping."""
        return self._index

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ScoreMatrix(method={self.method!r}, "
            f"N={self._N}, k={self._k}, "
            f"precomputed={self._precomputed})"
        )
