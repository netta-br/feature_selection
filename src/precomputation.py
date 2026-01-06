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
