from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def evaluate_logistic_regression_with_given_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_list: list[str],
    random_seed: int = None,
    output_dict: bool = False,
    lr_C: float = np.inf,
) -> tuple:
    """
    Trains and evaluates a Logistic Regression model using a predefined list of features.

    The caller is responsible for providing ready-to-use DataFrames/Series
    (samples × features, index = sample IDs).

    Args:
        X_train (pd.DataFrame): Training feature matrix (samples × features).
                                Index should be sample IDs.
        y_train (pd.Series): Training binary labels. Index should be sample IDs.
        X_val (pd.DataFrame): Validation feature matrix (samples × features).
                              Index should be sample IDs.
        y_val (pd.Series): Validation binary labels. Index should be sample IDs.
        feature_list (list[str]): A list of feature names to be used for training and evaluation.
        random_seed (int, optional): Seed for Logistic Regression model. Defaults to None.
        output_dict (bool, optional): Whether to return the classification report as a dictionary. Defaults to False.
        lr_C (float, optional): Regularization parameter for Logistic Regression. Defaults to np.inf.

    Returns:
        tuple: A tuple containing:
               - trained_model (LogisticRegression): The trained Logistic Regression model.
               - classification_rep (str | dict): The classification report for the validation set.
    """
    # 1. Select features using the provided feature_list
    missing_features = [f for f in feature_list if f not in X_train.columns]
    if missing_features:
        raise ValueError(f"The following features are not found in the training data: {missing_features}")

    X_train_subset = X_train[feature_list]
    X_val_subset = X_val[feature_list]

    # 2. Train Logistic Regression model
    model = LogisticRegression(random_state=random_seed, max_iter=1000, C=lr_C)
    model.fit(X_train_subset, y_train)

    # 3. Evaluate model
    y_pred = model.predict(X_val_subset)
    classification_rep = classification_report(y_val, y_pred, output_dict=output_dict)

    return model, classification_rep
