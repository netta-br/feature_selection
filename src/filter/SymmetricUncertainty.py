from __future__ import annotations

import numpy as np


class SymmetricUncertainty:
    """
    Static Symmetric Uncertainty (SU) computation utilities.

    No instance state — all methods are @staticmethod.
    Used by SymmetricUncertaintyMatrix (precomputation.py) and WrapperSelector
    (wrapper/MarkovBlanketWrapper.py).

    Discretisation
    --------------
    Paper-accurate 3-bin z-score scheme (Wang et al. 2017):
      Normalise to zero mean / unit std, then bin with thresholds -0.5 / +0.5.

    Mathematical definition
    -----------------------
    SU(F_i, F_j) = 2 * MI(F_i, F_j) / (H(F_i) + H(F_j))

    Values lie in [0, 1].  Returns 0.0 when either column is constant or
    entropy sum is zero.
    """

    # ------------------------------------------------------------------
    # Discretisation
    # ------------------------------------------------------------------
    @staticmethod
    def discretise(arr: np.ndarray) -> np.ndarray:
        """
        Z-score normalise *arr* then cut into 3 bins at -0.5 / +0.5:
          bin 0 : z < -0.5
          bin 1 : -0.5 <= z <= 0.5
          bin 2 : z > 0.5

        Constant columns (std == 0) are returned as all-zeros.

        Parameters
        ----------
        arr : np.ndarray
            1-D numeric array of length N_samples.

        Returns
        -------
        np.ndarray
            1-D int8 array with values in {0, 1, 2}.
        """
        std = arr.std()
        if std == 0:
            return np.zeros(len(arr), dtype=np.int8)
        z = (arr - arr.mean()) / std
        return np.digitize(z, bins=[-0.5, 0.5]).astype(np.int8)

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------
    @staticmethod
    def entropy(x: np.ndarray) -> float:
        """
        Shannon entropy H(x) in nats.

        Parameters
        ----------
        x : np.ndarray
            1-D integer (discretised) array.

        Returns
        -------
        float
            H(x) >= 0.  Returns 0.0 for constant arrays.
        """
        counts = np.bincount(x.astype(np.int64))
        probs = counts[counts > 0] / len(x)
        return float(-np.sum(probs * np.log(probs)))

    # ------------------------------------------------------------------
    # Mutual Information
    # ------------------------------------------------------------------
    @staticmethod
    def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
        """
        Mutual information MI(x, y) = H(x) + H(y) - H(x, y) in nats.

        Parameters
        ----------
        x, y : np.ndarray
            1-D integer (discretised) arrays of equal length.

        Returns
        -------
        float
            MI(x, y) >= 0.
        """
        hx = SymmetricUncertainty.entropy(x)
        hy = SymmetricUncertainty.entropy(y)

        # Joint entropy H(x, y)
        n = len(x)
        # Encode joint state as a single integer to use bincount
        n_bins_x = int(x.max()) + 1
        joint = x.astype(np.int64) * (int(y.max()) + 1) + y.astype(np.int64)
        counts = np.bincount(joint)
        probs = counts[counts > 0] / n
        hxy = float(-np.sum(probs * np.log(probs)))

        return float(max(0.0, hx + hy - hxy))

    # ------------------------------------------------------------------
    # Scalar SU
    # ------------------------------------------------------------------
    @staticmethod
    def compute_su(x: np.ndarray, y: np.ndarray) -> float:
        """
        Scalar SU(x, y) in [0, 1].

        Parameters
        ----------
        x, y : np.ndarray
            1-D integer (discretised) arrays — must already be discretised.

        Returns
        -------
        float
            SU(x, y).  Returns 0.0 if either column is constant (H = 0).
        """
        hx = SymmetricUncertainty.entropy(x)
        hy = SymmetricUncertainty.entropy(y)
        denom = hx + hy
        if denom == 0.0:
            return 0.0
        mi = SymmetricUncertainty.mutual_information(x, y)
        return float(min(1.0, 2.0 * mi / denom))

    # ------------------------------------------------------------------
    # Vectorised column SU
    # ------------------------------------------------------------------
    @staticmethod
    def compute_su_column(
        anchor: np.ndarray,
        others: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised SU(anchor, others[:, k]) for each column k.

        Parameters
        ----------
        anchor : np.ndarray
            1-D int8 array, length N_samples — the anchor feature (already
            discretised).
        others : np.ndarray
            2-D int8 array, shape (N_samples, K) — K other features (already
            discretised).

        Returns
        -------
        np.ndarray
            float64 array of length K:  SU(anchor, others[:, k]).
        """
        K = others.shape[1]
        result = np.empty(K, dtype=np.float64)
        h_anchor = SymmetricUncertainty.entropy(anchor)
        for k in range(K):
            col = others[:, k]
            h_col = SymmetricUncertainty.entropy(col)
            denom = h_anchor + h_col
            if denom == 0.0:
                result[k] = 0.0
            else:
                mi = SymmetricUncertainty.mutual_information(anchor, col)
                result[k] = min(1.0, 2.0 * mi / denom)
        return result
