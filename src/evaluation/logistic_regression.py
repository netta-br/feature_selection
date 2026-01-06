from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def evaluate_logistic_regression_with_given_features(gene_expression_df: pd.DataFrame, 
                                                     train_labels_df: pd.DataFrame, 
                                                     val_labels_df: pd.DataFrame, 
                                                     feature_list: list[str], 
                                                     random_seed:int=None, 
                                                     output_dict:bool=False, 
                                                     lr_C:float = np.inf):
    """
    Trains and evaluates a Logistic Regression model on gene expression data using a predefined list of features.

    Args:
        gene_expression_df (pd.DataFrame): DataFrame containing gene expression data.
                                           Rows are genes, columns are samples (e.g., F1, F2, ...).
                                           Assumes the first column is 'Unnamed: 0' for gene names.
        train_labels_df (pd.DataFrame): DataFrame containing training labels.
                                        Must have 'samplename' and 'is_lumA' columns.
        val_labels_df (pd.DataFrame): DataFrame containing validation labels.
                                      Must have 'samplename' and 'is_lumA' columns.
        feature_list (list): A list of feature (gene) names to be used for training and evaluation.
        random_seed (int, optional): Seed for Logistic Regression model. Defaults to None.
        output_dict (bool, optional): Whether to return the classification report as a dictionary. Defaults to False.
        lr_C (float, optional): Regularization parameter for Logistic Regression. Defaults to np.inf.

    Returns:
        tuple: A tuple containing:
               - trained_model (LogisticRegression): The trained Logistic Regression model.
               - classification_rep (str): A string containing the classification report for the validation set.
    """
    # 1. Preprocess gene expression data: Transpose and set gene names as index
    # Make a copy to avoid modifying the original dataframe
    gene_expression_processed = gene_expression_df.copy().fillna(0)
    gene_expression_processed = gene_expression_processed.set_index('Unnamed: 0').T
    gene_expression_processed.index.name = 'samplename'

    # 2. Align dataframes
    # Ensure samples in gene_expression_processed match labels dataframes
    X_train_full = gene_expression_processed.loc[train_labels_df['samplename']]
    y_train = train_labels_df['is_lumA']

    X_val_full = gene_expression_processed.loc[val_labels_df['samplename']]
    y_val = val_labels_df['is_lumA']

    # Ensure the order of samples is consistent
    X_train_full = X_train_full.reindex(train_labels_df['samplename'])
    X_val_full = X_val_full.reindex(val_labels_df['samplename'])

    # 3. Select features using the provided feature_list
    # Check if all features in feature_list exist in the gene expression data
    missing_features = [f for f in feature_list if f not in X_train_full.columns]
    if missing_features:
        raise ValueError(f"The following features are not found in the gene expression data: {missing_features}")

    X_train = X_train_full[feature_list]
    X_val = X_val_full[feature_list]

    # 4. Train Logistic Regression model
    model = LogisticRegression(random_state=random_seed, max_iter=1000, C=lr_C)
    model.fit(X_train, y_train)

    # 5. Evaluate model
    y_pred = model.predict(X_val)
    classification_rep = classification_report(y_val, y_pred, output_dict=output_dict)

    return model, classification_rep