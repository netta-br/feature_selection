import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import os # Import os for file path checks
from ..evaluation.logistic_regression import evaluate_logistic_regression_with_given_features

class mRmRSelector:
    """
    A class for selecting features using the Minimum Redundancy Maximum Relevance (mRmR) algorithm.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series,
                 relevance_method: str,
                 redundancy_method: str,
                 mrmr_score_method: str,
                 correlation_filepath: str = None,
                 base_gene_expression_df: pd.DataFrame = None, # Added for evaluation
                 base_train_labels_df: pd.DataFrame = None,    # Added for evaluation
                 base_val_labels_df: pd.DataFrame = None,      # Added for evaluation
                 lr_C: float = np.inf):                       # Added for evaluation

        self.X = X
        self.y = y
        self.relevance_method = relevance_method
        self.redundancy_method = redundancy_method
        self.mrmr_score_method = mrmr_score_method
        self.correlation_filepath = correlation_filepath

        self.base_gene_expression_df = base_gene_expression_df
        self.base_train_labels_df = base_train_labels_df
        self.base_val_labels_df = base_val_labels_df
        self.lr_C = lr_C

        self.selected_features = []
        self.unselected_features = list(X.columns)
        self._correlation_matrix = None
        self._relevance_scores = None
        self.performance_history = [] # To store evaluation metrics

    def _is_target_categorical(self) -> bool:
        """Determines if the target variable is categorical (for classification) or continuous (for regression)."""
        # A common heuristic: if y has fewer than 10% unique values, treat as categorical
        return self.y.nunique() < 0.1 * len(self.y)

    def _calculate_absolute_correlation(self, X_feature: pd.Series, y: pd.Series) -> float:
        """
        Calculates the absolute Pearson correlation between a single feature and the target.
        """
        # Ensure that X_feature and y are aligned by index
        common_index = X_feature.index.intersection(y.index)
        if common_index.empty:
            return 0.0 # No common samples, no correlation

        X_feature_aligned = X_feature.loc[common_index]
        y_aligned = y.loc[common_index]

        # Handle cases where correlation might be undefined (e.g., constant feature)
        if X_feature_aligned.std() == 0 or y_aligned.std() == 0:
            return 0.0

        # Pearson correlation
        correlation = X_feature_aligned.corr(y_aligned, method='pearson')
        return abs(correlation) if not pd.isna(correlation) else 0.0

    def _calculate_f_statistic(self, X_feature: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculates the F-statistic between a single feature and the target.
        Adapts for classification or regression based on the target variable.
        """
        # Ensure that X_feature and y are aligned by index
        common_index = X_feature.index.intersection(y.index)
        if common_index.empty:
            return 0.0 # No common samples, no F-statistic

        X_feature_aligned = X_feature.loc[common_index]
        y_aligned = y.loc[common_index]

        if self._is_target_categorical():
            # For classification tasks
            f_stat, _ = f_classif(X_feature_aligned, y_aligned)
        else:
            # For regression tasks
            f_stat, _ = f_regression(X_feature_aligned, y_aligned)
        return f_stat[0] # f_classif/f_regression return (array of scores, array of p-values)

    def _calculate_mutual_information(self, X_feature: pd.Series, y: pd.Series) -> float:
        """
        Calculates mutual information between a feature and the target.
        Adapts for classification or regression based on the target variable.
        """
        # Ensure that X_feature and y are aligned by index
        common_index = X_feature.index.intersection(y.index)
        if common_index.empty:
            return 0.0 # No common samples, no mutual information

        X_feature_2d = X_feature.loc[common_index].values.reshape(-1, 1)
        y_aligned = y.loc[common_index]

        if self._is_target_categorical():
            # For classification tasks
            mi_score = mutual_info_classif(X_feature_2d, y_aligned, random_state=42)[0]
        else:
            # For regression tasks
            mi_score = mutual_info_regression(X_feature_2d, y_aligned, random_state=42)[0]
        return mi_score

    def _calculate_random_forest_importance(self, X_data: pd.DataFrame, y_data: pd.Series) -> pd.Series:
        """
        Calculates feature importances using a RandomForest model.
        Adapts for classification or regression based on the target variable.
        """
        if self._is_target_categorical():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_data, y_data)
        return pd.Series(model.feature_importances_, index=X_data.columns)

    def _precompute_measures(self):
        """
        Precomputes the feature-feature correlation matrix and feature-target relevance scores.
        """
        # Ensure no NaN values in self.X before computing correlation or relevance
        X_filled = self.X.fillna(0)

        # --- Precompute Correlation Matrix ---
        if self.correlation_filepath and os.path.exists(self.correlation_filepath):
            print(f"Loading correlation matrix from {self.correlation_filepath}...")
            try:
                self._correlation_matrix = pd.read_csv(self.correlation_filepath, index_col=0)
                print("Correlation matrix loaded.")
            except Exception as e:
                print(f"Error loading correlation matrix: {e}. Recomputing.")
                self._correlation_matrix = None # Set to None to trigger recomputation

        if self._correlation_matrix is None: # If not loaded or error, compute
            print("Precomputing feature-feature correlation matrix...")
            self._correlation_matrix = X_filled.corr(method='pearson')
            if self.correlation_filepath:
                self._correlation_matrix.to_csv(self.correlation_filepath)
                print(f"Correlation matrix saved to {self.correlation_filepath}")
            else:
                print("Correlation matrix computed (not saved as no filepath provided).")

        # --- Precompute Feature-Target Relevance Scores ---
        print(f"Precomputing feature-target relevance scores using '{self.relevance_method}'...")
        if self.relevance_method == 'rf_importance':
            # RandomForest importance needs all features at once
            self._relevance_scores = self._calculate_random_forest_importance(X_filled, self.y)
        else:
            self._relevance_scores = pd.Series(dtype=float)
            for feature in X_filled.columns:
                if self.relevance_method == 'correlation':
                    relevance = self._calculate_absolute_correlation(X_filled[feature], self.y)
                elif self.relevance_method == 'f_statistic':
                    # F-stat expects 2D array, so pass a DataFrame with a single column
                    relevance = self._calculate_f_statistic(X_filled[[feature]], self.y)
                elif self.relevance_method == 'mutual_info':
                    relevance = self._calculate_mutual_information(X_filled[feature], self.y)
                else:
                    raise ValueError(f"Unknown relevance method: {self.relevance_method}")
                self._relevance_scores.loc[feature] = relevance

        print("Relevance scores precomputation complete.")
        print("Precomputation complete.")


    def _get_relevance(self, feature: str) -> float:
        """
        Retrieves the precomputed relevance score for a given feature.
        """
        if self._relevance_scores is None:
            raise ValueError("Relevance scores have not been precomputed. Call _precompute_measures first.")
        # Handle cases where feature might not be in relevance scores (e.g., if X_filled changed or specific feature dropped)
        return self._relevance_scores.get(feature, 0.0) # Return 0 if feature not found

    def _get_redundancy(self, candidate_feature: str, selected_features_list: list) -> float:
        """
        Computes the L1 norm of the redundancy between a candidate feature
        and the currently selected features.
        """
        if self.redundancy_method != 'l1_correlation':
            raise ValueError(f"Unknown redundancy method: {self.redundancy_method}. Only 'l1_correlation' is implemented.")

        if not selected_features_list:
            return 0.0

        if self._correlation_matrix is None:
            raise ValueError("Correlation matrix has not been precomputed. Call _precompute_measures first.")

        redundancy_values = []
        for s_feature in selected_features_list:
            # Check if both features exist in the correlation matrix
            if candidate_feature in self._correlation_matrix.columns and \
               s_feature in self._correlation_matrix.columns:
                correlation = self._correlation_matrix.loc[candidate_feature, s_feature]
                redundancy_values.append(abs(correlation))
            else:
                # Handle cases where a feature might be missing from the correlation matrix
                print(f"Warning: Feature '{candidate_feature}' or '{s_feature}' not found in correlation matrix. Assuming 0 redundancy.")
                redundancy_values.append(0.0) # Assume no redundancy if missing

        return sum(redundancy_values)

    def _compute_mrmr_score(self, relevance: float, redundancy: float) -> float:
        """
        Combines relevance and redundancy into an mRmR score.
        """
        if self.mrmr_score_method == 'difference':
            return relevance - redundancy
        elif self.mrmr_score_method == 'ratio':
            return relevance / (redundancy + 1e-8) # Add small epsilon to prevent division by zero
        else:
            raise ValueError("mRmR score method must be 'difference' or 'ratio'.")

    def perform_greedy_mrmr_step(self) -> str:
        """
        Identifies and selects the unselected feature with the highest mRmR score.
        Returns the selected feature name, or None if no features left to select.
        """
        if not self.unselected_features:
            return None

        best_feature = None
        max_mrmr_score = -np.inf

        for candidate_feature in self.unselected_features:
            relevance = self._get_relevance(candidate_feature)
            redundancy = self._get_redundancy(candidate_feature, self.selected_features)
            mrmr_score = self._compute_mrmr_score(relevance, redundancy)

            if mrmr_score > max_mrmr_score:
                max_mrmr_score = mrmr_score
                best_feature = candidate_feature

        if best_feature is not None:
            self.selected_features.append(best_feature)
            self.unselected_features.remove(best_feature)
            return best_feature
        else:
            return None

    def _evaluate_current_selection(self, random_seed: int = None) -> dict:
        """
        Evaluates the current set of selected features using Logistic Regression.
        Requires base_gene_expression_df, base_train_labels_df, base_val_labels_df, and lr_C
        to be set during initialization.
        """
        if not self.selected_features:
            return {"accuracy": 0.0, "macro avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}}

        if self.base_gene_expression_df is None or \
           self.base_train_labels_df is None or \
           self.base_val_labels_df is None:
            raise ValueError("Evaluation data (base_gene_expression_df, base_train_labels_df, base_val_labels_df) must be provided in __init__ for evaluation.")

        _, report = evaluate_logistic_regression_with_given_features(
            gene_expression_df=self.base_gene_expression_df,
            train_labels_df=self.base_train_labels_df,
            val_labels_df=self.base_val_labels_df,
            feature_list=self.selected_features,
            random_seed=random_seed,
            output_dict=True,
            lr_C=self.lr_C
        )
        return report

    def forward_selection(self, n_features_to_select: int,
                            stopping_metric: str = None,
                            stopping_threshold: float = None,
                            random_seed:int = None) -> tuple:
        """
        Orchestrates the greedy forward selection process to select a desired number of features.
        Includes optional early stopping based on validation performance.

        Args:
            n_features_to_select (int): The maximum number of features to select.
            stopping_metric (str, optional): The metric to monitor for early stopping (e.g., 'macro avg_f1-score', 'accuracy').
                                             If None, no early stopping is performed.
            stopping_threshold (float, optional): The threshold for the stopping_metric. If the metric
                                                  reaches or exceeds this value, selection stops.

        Returns:
            tuple: A tuple containing:
                   - list: The list of selected feature names.
                   - list: A list of dictionaries, where each dictionary contains the evaluation
                           metrics after each feature selection step.
        """
        if n_features_to_select <= 0:
            print("Number of features to select must be greater than 0.")
            return [], []

        # Ensure precomputation is done before starting selection
        if self._relevance_scores is None or self._correlation_matrix is None:
            self._precompute_measures()

        print(f"Starting mRmR forward selection for {n_features_to_select} features...")
        if stopping_metric and stopping_threshold is not None:
            print(f"Early stopping enabled: monitoring '{stopping_metric}' with threshold >= {stopping_threshold:.4f}")
        
        self.performance_history = [] # Reset performance history for this run

        for i in range(n_features_to_select):
            if not self.unselected_features:
                print(f"All available features selected. Stopping at {len(self.selected_features)} features.")
                break

            selected_feature = self.perform_greedy_mrmr_step()
            if selected_feature:
                print(f"Step {i+1}: Selected '{selected_feature}'. Total selected: {len(self.selected_features)}")

                # Evaluate current selection if evaluation data is provided
                if self.base_gene_expression_df is not None:
                    current_performance = self._evaluate_current_selection(random_seed=random_seed) 
                    self.performance_history.append(current_performance)

                    macro_avg_f1 = current_performance['macro avg']['f1-score']
                    accuracy = current_performance['accuracy']
                    print(f"  Validation Performance - Accuracy: {accuracy:.4f}, Macro Avg F1: {macro_avg_f1:.4f}")

                    # Check for early stopping
                    if stopping_metric and stopping_threshold is not None:
                        if stopping_metric == 'accuracy':
                            metric_value = accuracy
                        elif stopping_metric == 'macro avg_f1-score':
                            metric_value = macro_avg_f1
                        else:
                            # Attempt to get value from report directly
                            try:
                                # Example: 'True_f1-score' or 'False_precision'
                                parts = stopping_metric.split('_')
                                if len(parts) == 2: # e.g., 'macro avg_f1-score'
                                    metric_value = current_performance[parts[0]][parts[1]]
                                elif len(parts) == 1: # e.g., 'accuracy'
                                    metric_value = current_performance[parts[0]]
                                else:
                                    raise KeyError # force generic error
                            except KeyError:
                                print(f"Warning: Unknown stopping_metric '{stopping_metric}'. Early stopping will not be applied.")
                                metric_value = -np.inf # Effectively disable stopping

                        if metric_value >= stopping_threshold:
                            print(f"\nEarly stopping triggered at {len(self.selected_features)} features!")
                            print(f"Metric '{stopping_metric}' reached {metric_value:.4f} (threshold: {stopping_threshold:.4f}).")
                            break
            else:
                print(f"No more features to select after {len(self.selected_features)} steps.")
                break

        print("\nmRmR feature selection complete.")
        print("Final selected features count:", len(self.selected_features))
        return self.selected_features, self.performance_history