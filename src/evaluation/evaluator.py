from __future__ import annotations

import copy

import pandas as pd


class Evaluator:
    def __init__(self, model, task_type: str, label: str = ""):
        self.model = model        # any sklearn-compatible .fit/.predict object
        self.task_type = task_type  # "classification" | "regression"
        self.label = label

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_list: list[str],
    ) -> dict:
        """Trains model on X_train[feature_list], evaluates on X_test[feature_list].
        Returns metrics dict."""
        missing = [f for f in feature_list if f not in X_train.columns]
        if missing:
            raise ValueError(f"Features missing from X_train: {missing}")

        model = copy.deepcopy(self.model)
        model.fit(X_train[feature_list], y_train)
        y_pred = model.predict(X_test[feature_list])

        if self.task_type == "classification":
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            }

            try:
                if hasattr(model, "predict_proba"):
                    classes = list(model.classes_)
                    if len(classes) == 2:
                        y_proba = model.predict_proba(X_test[feature_list])[:, 1]
                        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            except Exception:
                pass

            return metrics

        elif self.task_type == "regression":
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            mse = mean_squared_error(y_test, y_pred)
            return {
                "r2": r2_score(y_test, y_pred),
                "mse": mse,
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": mse ** 0.5,
            }

        else:
            raise ValueError(f"Unknown task_type: {self.task_type!r}")

    @staticmethod
    def calculate_performance_history(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        selected_features: list[str],
        evaluator: Evaluator,
        eval_every_k: int = 1,
    ) -> list[dict]:
        """Incrementally evaluates on selected_features[:k] for k=1..len(selected_features),
        stepping by eval_every_k. Always evaluates at the final step.
        Returns list of dicts, each with 'step' key plus metric keys."""
        n = len(selected_features)
        steps = list(range(1, n + 1, eval_every_k))
        if n not in steps:
            steps.append(n)

        history = []
        for k in steps:
            metrics = evaluator.evaluate(
                X_train, y_train, X_test, y_test, selected_features[:k]
            )
            history.append({"step": k, **metrics})

        return history


def LinearRegressionEvaluator(params: dict | None = None) -> Evaluator:
    """Returns Evaluator wrapping sklearn LinearRegression(**params)."""
    from sklearn.linear_model import LinearRegression
    p = params or {}
    return Evaluator(LinearRegression(**p), task_type="regression", label="LinearRegression")


def LogisticRegressionEvaluator(params: dict | None = None) -> Evaluator:
    """Returns Evaluator wrapping sklearn LogisticRegression(max_iter=1000, **params)."""
    from sklearn.linear_model import LogisticRegression
    p = {"max_iter": 1000, **(params or {})}
    return Evaluator(LogisticRegression(**p), task_type="classification", label="LogisticRegression")


def KNNEvaluator(params: dict | None = None) -> Evaluator:
    """Returns Evaluator wrapping sklearn KNeighborsClassifier(**params)."""
    from sklearn.neighbors import KNeighborsClassifier
    p = params or {}
    return Evaluator(KNeighborsClassifier(**p), task_type="classification", label="KNN")


def NaiveBayesEvaluator(params: dict | None = None) -> Evaluator:
    """Returns Evaluator wrapping sklearn GaussianNB(**params)."""
    from sklearn.naive_bayes import GaussianNB
    p = params or {}
    return Evaluator(GaussianNB(**p), task_type="classification", label="NaiveBayes")
