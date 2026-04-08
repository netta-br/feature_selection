from __future__ import annotations

import os

import pandas as pd
import numpy as np

class FeatureCorrelationMatrix:
    """
    A class to compute, store, and retrieve feature correlation matrices.
    """
    def __init__(self):
        self.correlation_matrix = None

    @classmethod
    def compute_correlation_matrix(cls, df: pd.DataFrame, method: str = 'pearson', filepath: str = 'correlation_matrix.csv'):
        """
        Computes the pairwise correlation matrix for the given DataFrame
        and saves it to a CSV file.

        Args:
            df (pd.DataFrame): The input DataFrame containing numerical features.
            method (str): The correlation method to use. 'pearson', 'kendall', or 'spearman'.
            filepath (str): The path to save the correlation matrix CSV.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if method not in ['pearson', 'kendall', 'spearman']:
            raise ValueError("Method must be 'pearson', 'kendall', or 'spearman'.")

        print(f"Computing {method} correlation matrix...")
        # Exclude non-numeric columns from correlation calculation
        bool_df = df.select_dtypes(include=bool)
        bool_df = bool_df.astype(int)
        numeric_df = df.select_dtypes(include=np.number).merge(bool_df)
        if numeric_df.empty:
            raise ValueError("No numeric columns found in the DataFrame to compute correlation.")

        corr_matrix = numeric_df.corr(method=method)
        corr_matrix.to_csv(filepath)
        print(f"Correlation matrix saved to {filepath}")

    def load_correlation_matrix(self, filepath: str):
        """
        Loads a correlation matrix from a CSV file into the object.

        Args:
            filepath (str): The path to the correlation matrix CSV file.
        """
        if not isinstance(filepath, str):
            raise TypeError("Input 'filepath' must be a string.")

        try:
            self.correlation_matrix = pd.read_csv(filepath, index_col=0)
            print(f"Correlation matrix loaded from {filepath}")
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            self.correlation_matrix = None
        except Exception as e:
            print(f"Error loading correlation matrix: {e}")
            self.correlation_matrix = None

    def get_feature_correlation(self, feature1, feature2):
        """
        Returns the correlation between two specified features.

        Args:
            feature1 (str or int): The name or integer index of the first feature.
            feature2 (str or int): The name or integer index of the second feature.

        Returns:
            float or None: The correlation value, or None if features are not found
                           or matrix is not loaded.
        """
        if self.correlation_matrix is None:
            print("Error: Correlation matrix not loaded. Please call load_correlation_matrix first.")
            return None

        if isinstance(feature1, int):
            if not (0 <= feature1 < len(self.correlation_matrix.columns)):
                print(f"Error: Feature index {feature1} is out of bounds.")
                return None
            feature1 = self.correlation_matrix.columns[feature1]
        if isinstance(feature2, int):
            if not (0 <= feature2 < len(self.correlation_matrix.columns)):
                print(f"Error: Feature index {feature2} is out of bounds.")
                return None
            feature2 = self.correlation_matrix.columns[feature2]

        if feature1 not in self.correlation_matrix.columns or \
           feature2 not in self.correlation_matrix.columns:
            print(f"Error: One or both features ('{feature1}', '{feature2}') not found in the correlation matrix.")
            return None

        return self.correlation_matrix.loc[feature1, feature2]


class FeatureRelevanceScores:
    """
    Computes and stores feature-to-target relevance scores for non-correlation methods.

    One scalar score per feature (not pairwise).  Supported methods:
    ``'mutual_information'``, ``'random_forest'``, ``'f_statistic'``.

    The scores are saved/loaded as a two-column CSV (index = feature name,
    value column = ``'score'``).
    """

    SUPPORTED_METHODS = {"mutual_information", "random_forest", "f_statistic"}

    def __init__(self) -> None:
        self.relevance_scores: pd.Series | None = None  # index=feature name, value=score

    # ------------------------------------------------------------------
    # Classmethod: compute full-matrix scores and save to CSV
    # ------------------------------------------------------------------
    @classmethod
    def compute_relevance_scores(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        method: str,
        task_type: str,
        filepath: str,
        random_seed: int | None = None,
    ) -> None:
        """
        Compute relevance scores for all features in *X* and save to *filepath*.

        Parameters
        ----------
        X : pd.DataFrame
            Sample × feature matrix (NaNs should be imputed before calling).
        y : pd.Series
            Target vector aligned to *X*.
        method : str
            One of ``'mutual_information'``, ``'random_forest'``, ``'f_statistic'``.
        task_type : str
            ``'classification'`` or ``'regression'``.
        filepath : str
            Destination CSV path.
        random_seed : int | None
            Used by MI and RF for reproducibility.
        """
        if method not in cls.SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {cls.SUPPORTED_METHODS}; got {method!r}"
            )
        if task_type not in {"classification", "regression"}:
            raise ValueError(
                f"task_type must be 'classification' or 'regression'; got {task_type!r}"
            )

        X_arr = X.values
        y_arr = y.values

        if method == "mutual_information":
            from sklearn.feature_selection import (
                mutual_info_classif,
                mutual_info_regression,
            )
            fn = mutual_info_classif if task_type == "classification" else mutual_info_regression
            scores = fn(X_arr, y_arr, random_state=random_seed)

        elif method == "f_statistic":
            from sklearn.feature_selection import f_classif, f_regression
            fn = f_classif if task_type == "classification" else f_regression
            scores = fn(X_arr, y_arr)[0]  # (F_scores, p_values) → take F_scores

        elif method == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            Cls = (
                RandomForestClassifier if task_type == "classification"
                else RandomForestRegressor
            )
            rf = Cls(n_estimators=100, random_state=random_seed)
            rf.fit(X_arr, y_arr)
            scores = rf.feature_importances_

        # Guard NaN / inf → 0.0
        scores = np.where(np.isnan(scores) | np.isinf(scores), 0.0, scores)

        series = pd.Series(scores, index=X.columns, name="score")
        series.index.name = "feature"
        series.to_csv(filepath, header=True)
        print(f"Relevance scores ({method}, {task_type}) saved to {filepath}")

    # ------------------------------------------------------------------
    # Instance method: load from CSV
    # ------------------------------------------------------------------
    def load_relevance_scores(self, filepath: str) -> None:
        """
        Load relevance scores from *filepath* into ``self.relevance_scores``.

        Parameters
        ----------
        filepath : str
            Path to the CSV saved by :py:meth:`compute_relevance_scores`.
        """
        self.relevance_scores = pd.read_csv(filepath, index_col=0).squeeze("columns")
        print(f"Relevance scores loaded from {filepath} ({len(self.relevance_scores)} features)")


class SymmetricUncertaintyMatrix:
    """
    Compute, save, and load the full Symmetric Uncertainty matrix for Mode A
    precomputation.

    The matrix covers all feature pairs plus one or more target columns,
    so that a **single CSV** can be shared across different tasks on the same
    feature set.  Layout: square CSV with feature names (and target names) as
    both index and column headers.

    Passing multiple targets via ``y`` avoids recomputing the expensive
    feature-pair SU block when switching prediction tasks.

    The MI intermediate matrix can optionally be saved alongside the SU matrix
    via ``mi_filepath``.  If ``mi_filepath`` already exists when
    ``compute_su_matrix`` is called, the MI computation is skipped and the
    matrix is loaded from file instead (avoids recomputing expensive MI).
    """

    def __init__(self) -> None:
        self.su_matrix: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Class-method: compute full SU (and optionally MI) matrix, save CSV
    # ------------------------------------------------------------------
    @classmethod
    def compute_su_matrix(
        cls,
        X: pd.DataFrame,
        y: "pd.Series | list[pd.Series]",
        filepath: str,
        mi_filepath: str | None = None,
    ) -> None:
        """
        Compute the full SU matrix for *X* and one or more targets, then save
        to *filepath*.

        Passing multiple target Series in *y* means the expensive feature-pair
        SU block is computed **once** and all target columns are appended to
        the same matrix.  A single CSV therefore serves every task that shares
        the same feature set.

        Parameters
        ----------
        X : pd.DataFrame
            Samples × features matrix (NaNs already imputed).
        y : pd.Series or list[pd.Series]
            One or more target vectors.  Each Series must have a unique
            ``.name`` attribute that does not clash with column names in *X*.
        filepath : str
            Destination CSV path for the SU matrix.
        mi_filepath : str | None
            Optional path to save (or load) the intermediate MI matrix.
            If provided and the file already exists, MI is loaded from it
            rather than recomputed.
        """
        from .filter.SymmetricUncertainty import SymmetricUncertainty

        # Normalise y to a list of Series
        if isinstance(y, pd.Series):
            targets: list[pd.Series] = [y]
        else:
            targets = list(y)

        target_names = [str(t.name) for t in targets]
        # Validate uniqueness and no clash with feature names
        for tname in target_names:
            if tname in X.columns:
                raise ValueError(
                    f"Target name {tname!r} clashes with a feature column name."
                )
        if len(set(target_names)) != len(target_names):
            raise ValueError("Duplicate target names are not allowed.")

        feat_cols = list(X.columns)
        all_cols = feat_cols + target_names
        combined = X.copy()
        for t in targets:
            combined[str(t.name)] = t.values
        N = len(all_cols)

        # ── Discretise all columns once ────────────────────────────────
        X_disc = np.stack(
            [SymmetricUncertainty.discretise(combined[c].values) for c in all_cols],
            axis=1,
        )  # shape (n_samples, N)

        # ── MI matrix ─────────────────────────────────────────────────
        if mi_filepath is not None and os.path.exists(mi_filepath):
            mi_df = pd.read_csv(mi_filepath, index_col=0)
            # The cached MI file may have been computed with a subset of the
            # current targets; extend it if new target columns are missing.
            missing_cols = [c for c in all_cols if c not in mi_df.columns]
            if missing_cols:
                print(
                    f"MI matrix loaded from {mi_filepath} — "
                    f"extending with {len(missing_cols)} new column(s): {missing_cols}"
                )
                # Re-index to full all_cols; fill missing with NaN then compute
                mi_df = mi_df.reindex(index=all_cols, columns=all_cols)
                mi_arr = mi_df.values.astype(np.float64)
                # Fill NaN cells (new target rows/cols)
                for i, ci in enumerate(all_cols):
                    for j, cj in enumerate(all_cols):
                        if np.isnan(mi_arr[i, j]):
                            if j < i:
                                mi_arr[i, j] = mi_arr[j, i]
                            elif i == j:
                                mi_arr[i, j] = SymmetricUncertainty.entropy(X_disc[:, i])
                            else:
                                mi_arr[i, j] = SymmetricUncertainty.mutual_information(
                                    X_disc[:, i], X_disc[:, j]
                                )
                # Re-save extended MI matrix
                mi_df_new = pd.DataFrame(mi_arr, index=all_cols, columns=all_cols)
                mi_df_new.to_csv(mi_filepath)
                print(f"MI matrix (extended) saved to {mi_filepath}")
            else:
                mi_arr = mi_df.reindex(index=all_cols, columns=all_cols).values.astype(np.float64)
                print(f"MI matrix loaded from {mi_filepath}")
        else:
            mi_arr = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                for j in range(N):
                    if j < i:
                        mi_arr[i, j] = mi_arr[j, i]
                    elif j == i:
                        mi_arr[i, j] = SymmetricUncertainty.entropy(X_disc[:, i])
                    else:
                        mi_arr[i, j] = SymmetricUncertainty.mutual_information(
                            X_disc[:, i], X_disc[:, j]
                        )

            if mi_filepath is not None:
                mi_df = pd.DataFrame(mi_arr, index=all_cols, columns=all_cols)
                mi_df.to_csv(mi_filepath)
                print(f"MI matrix saved to {mi_filepath}")

        # ── Compute SU from MI and entropy ─────────────────────────────
        # entropy vector H[i] = MI[i, i] (we stored H on the diagonal above)
        H = np.diag(mi_arr).copy()          # shape (N,)
        su_arr = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                denom = H[i] + H[j]
                if denom == 0.0:
                    su_arr[i, j] = 0.0
                elif i == j:
                    su_arr[i, j] = 1.0
                else:
                    su_arr[i, j] = min(1.0, 2.0 * mi_arr[i, j] / denom)

        su_df = pd.DataFrame(su_arr, index=all_cols, columns=all_cols)
        su_df.to_csv(filepath)
        print(f"SU matrix saved to {filepath} {su_df.shape}")

    # ------------------------------------------------------------------
    # Instance method: load from CSV
    # ------------------------------------------------------------------
    def load_su_matrix(self, filepath: str) -> None:
        """
        Load a previously saved SU matrix from *filepath*.

        Parameters
        ----------
        filepath : str
            Path to the CSV saved by :py:meth:`compute_su_matrix`.
        """
        try:
            self.su_matrix = pd.read_csv(filepath, index_col=0)
            print(f"SU matrix loaded from {filepath} {self.su_matrix.shape}")
        except FileNotFoundError:
            print(f"Error: SU matrix file not found at {filepath}")
            self.su_matrix = None
        except Exception as exc:
            print(f"Error loading SU matrix: {exc}")
            self.su_matrix = None
