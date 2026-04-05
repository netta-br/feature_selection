from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm


def train_and_evaluate_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_features: int,
    random_seed: int | None = None,
) -> tuple[LinearRegression, dict]:
    """
    Train a Linear Regression model on randomly selected features and evaluate on validation set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : pd.Series
        Validation target vector.
    n_features : int
        Number of features to randomly select.
    random_seed : int | None
        Seed for random feature selection.

    Returns
    -------
    tuple[LinearRegression, dict]
        Trained model and metrics dict with keys 'r2', 'mse', 'mae'.
    """
    if n_features > X_train.shape[1]:
        raise ValueError(
            f"n_features ({n_features}) cannot exceed the number of available features "
            f"({X_train.shape[1]})."
        )

    np.random.seed(random_seed)
    selected = np.random.choice(X_train.columns, n_features, replace=False)

    model = LinearRegression()
    model.fit(X_train[selected], y_train)

    y_pred = model.predict(X_val[selected])

    metrics = {
        "r2":  r2_score(y_val, y_pred),
        "mse": mean_squared_error(y_val, y_pred),
        "mae": mean_absolute_error(y_val, y_pred),
    }
    return model, metrics


def validation_report_to_df(metrics_dict: dict, n_features: int) -> pd.DataFrame:
    """
    Convert a metrics dict ``{'r2': ..., 'mse': ..., 'mae': ...}`` to long format.

    Parameters
    ----------
    metrics_dict : dict
        Dict with keys 'r2', 'mse', 'mae'.
    n_features : int
        Number of features used (stored in the 'features_num' column).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns ['variable', 'value', 'features_num'].
    """
    rows = [
        {"variable": k, "value": v, "features_num": n_features}
        for k, v in metrics_dict.items()
    ]
    return pd.DataFrame(rows)


def plot_performance_with_stats(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    N_values: list[int],
    random_seed: int,
    num_runs: int,
    return_summary: bool = False,
) -> pd.DataFrame | None:
    """
    Run multiple randomised Linear Regression experiments, plot performance statistics,
    and optionally return the wide-format summary DataFrame.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : pd.Series
        Validation target vector.
    N_values : list[int]
        Feature counts to evaluate.
    random_seed : int
        Base random seed; each run uses ``random_seed + run_id``.
    num_runs : int
        Number of randomised runs per feature count.
    return_summary : bool
        If ``True``, return the wide-format summary DataFrame; otherwise return ``None``.

    Returns
    -------
    pd.DataFrame | None
        Wide-format summary with columns
        ``[features_num, r2_mean, r2_std, mse_mean, mse_std, mae_mean, mae_std]``
        when ``return_summary=True``; ``None`` otherwise.
    """
    all_rows: list[pd.DataFrame] = []

    for run_id in tqdm(range(num_runs), desc="Running LinReg experiments"):
        for n in N_values:
            _, metrics = train_and_evaluate_linear_regression(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_features=n,
                random_seed=random_seed + run_id,
            )
            row_df = validation_report_to_df(metrics, n)
            row_df["run_id"] = run_id
            all_rows.append(row_df)

    full_df = pd.concat(all_rows, ignore_index=True)

    # Aggregate: mean and std per (features_num, variable)
    agg = (
        full_df.groupby(["features_num", "variable"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Pivot to wide format
    agg_pivot = agg.pivot(index="features_num", columns="variable", values=["mean", "std"])
    agg_pivot.columns = [f"{metric}_{stat}" for stat, metric in agg_pivot.columns]
    summary_df = agg_pivot.reset_index()

    # Ensure consistent column order
    desired_cols = [
        "features_num",
        "r2_mean", "r2_std",
        "mse_mean", "mse_std",
        "mae_mean", "mae_std",
    ]
    summary_df = summary_df[[c for c in desired_cols if c in summary_df.columns]]

    print("\nSummary Statistics — Linear Regression Random Baseline (Mean ± Std):\n")
    try:
        display(summary_df)  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        print(summary_df.to_string(index=False))

    # Plot: 3 sub-panels (r2, mse, mae)
    metrics_info = [
        ("r2_mean",  "r2_std",  "R²"),
        ("mse_mean", "mse_std", "MSE"),
        ("mae_mean", "mae_std", "MAE"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = summary_df["features_num"]

    for ax, (mean_col, std_col, label) in zip(axes, metrics_info):
        if mean_col not in summary_df.columns:
            continue
        mean_vals = summary_df[mean_col]
        std_vals = summary_df[std_col].fillna(0)
        sns.lineplot(x=x, y=mean_vals, ax=ax, marker="o", label=label)
        ax.fill_between(
            x,
            mean_vals - 1.5 * std_vals,
            mean_vals + 1.5 * std_vals,
            alpha=0.2,
        )
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Number of features")
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend(title="Metric")

    fig.suptitle(
        f"Linear Regression Performance with Random Feature Selection "
        f"(Mean ± 1.5 Std over {num_runs} Runs)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    if return_summary:
        return summary_df
    return None
