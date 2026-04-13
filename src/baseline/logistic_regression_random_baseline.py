from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def train_and_evaluate_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_features: int,
    random_seed: int | None = None,
    output_dict: bool = False,
    lr_C: float = np.inf,
) -> tuple:
    """
    Trains and evaluates a Logistic Regression model using randomly selected features.

    The caller is responsible for providing ready-to-use DataFrames/Series
    (samples × features, index = sample IDs).

    Args:
        X_train (pd.DataFrame): Training feature matrix (samples × features).
        y_train (pd.Series): Training binary labels.
        X_val (pd.DataFrame): Validation feature matrix (samples × features).
        y_val (pd.Series): Validation binary labels.
        n_features (int): The number of features to randomly select.
        random_seed (int | None, optional): Seed for random feature selection. Defaults to None.
        output_dict (bool, optional): Whether to return the classification report as a dictionary. Defaults to False.
        lr_C (float, optional): Regularization parameter for Logistic Regression
                                (inverse - smaller value means more regularization).
                                Defaults to np.inf (no regularization).

    Returns:
        tuple: A tuple containing:
               - trained_model (LogisticRegression): The trained Logistic Regression model.
               - classification_rep (str | dict): The classification report for the validation set.
    """
    # 1. Validate n_features
    if n_features > X_train.shape[1]:
        raise ValueError("n_features cannot be greater than the total number of features.")

    # 2. Perform random feature selection
    np.random.seed(random_seed)
    selected_features = np.random.choice(X_train.columns, n_features, replace=False)

    X_train_subset = X_train[selected_features]
    X_val_subset = X_val[selected_features]

    # 3. Train Logistic Regression model
    model = LogisticRegression(random_state=random_seed, max_iter=1000, C=lr_C)
    model.fit(X_train_subset, y_train)

    # 4. Evaluate model
    y_pred = model.predict(X_val_subset)
    classification_rep = classification_report(y_val, y_pred, output_dict=output_dict)

    return model, classification_rep

def validation_report_to_df(validation_report, features_num):
  report = pd.DataFrame(validation_report).T.reset_index().rename(columns={'index':'class'})
  report = pd.melt(report, id_vars = ['class'], value_vars=['precision', 'recall','f1-score','support'])
  report['features_num'] = features_num
  return report.loc[report['class'] != 'accuracy']

def plot_performance_with_stats(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    N_values: list[int],
    random_seed: int,
    num_runs: int,
    lr_C: float = np.inf,
    return_summary: bool = False,
) -> pd.DataFrame | None:
    """
    Runs multiple random feature selection experiments and plots performance statistics.

    The caller is responsible for providing ready-to-use DataFrames/Series
    (samples × features, index = sample IDs).

    Args:
        X_train (pd.DataFrame): Training feature matrix (samples × features).
        y_train (pd.Series): Training binary labels.
        X_val (pd.DataFrame): Validation feature matrix (samples × features).
        y_val (pd.Series): Validation binary labels.
        N_values (list[int]): List of feature counts to evaluate.
        random_seed (int): Base random seed (varied per run).
        num_runs (int): Number of random runs per feature count.
        lr_C (float, optional): Regularization parameter. Defaults to np.inf.
        return_summary (bool, optional): Whether to return baseline summary DataFrame. Defaults to False.

    Returns:
        pd.DataFrame | None: Baseline summary if return_summary is True, else None.
    """
    all_runs_results = []
    # Also collect per-run accuracy + macro f1 for the wide-format baseline_summary
    acc_records = []

    for run_id in tqdm(range(num_runs), desc="Running experiments"):
        run_results = []
        for n in N_values:
            # Call the train_and_evaluate_logistic_regression function
            _, validation_report = train_and_evaluate_logistic_regression(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_features=n,
                random_seed=random_seed + run_id, # Vary seed for each run
                output_dict=True,
                lr_C=lr_C
            )
            # Convert validation report to DataFrame and append
            run_results.append(validation_report_to_df(validation_report, n))
            # Capture accuracy + macro f1 for baseline_summary
            acc_records.append({
                'features_num': n,
                'run_id': run_id,
                'accuracy': validation_report.get('accuracy', float('nan')),
                'macro_f1': (
                    validation_report.get('macro avg', {}).get('f1-score', float('nan'))
                    if isinstance(validation_report.get('macro avg'), dict)
                    else float('nan')
                ),
            })

        # Concatenate results for the current run and add run_id
        current_run_df = pd.concat(run_results)
        current_run_df['run_id'] = run_id
        all_runs_results.append(current_run_df)

    # Concatenate all runs' results into a single DataFrame
    full_results_df = pd.concat(all_runs_results)

    # Filter for 'macro avg' and exclude 'support'
    filtered_results = full_results_df[
        (full_results_df['class'] == 'macro avg') &
        (full_results_df['variable'] != 'support')
    ]

    # Calculate mean and standard deviation
    summary_stats = filtered_results.groupby(['features_num', 'variable'])['value'].agg(['mean', 'std']).reset_index()

    # Display the summary statistics table
    print("\nSummary Statistics (Mean +/- Std Dev):\n")
    print(summary_stats)

    # Plotting
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(data=summary_stats, x='features_num', y='mean', hue='variable', marker='o')

    # Add shaded margins for 1.5 standard deviations
    for var in summary_stats['variable'].unique():
        subset = summary_stats[summary_stats['variable'] == var]
        plt.fill_between(
            subset['features_num'],
            subset['mean'] - 1.5 * subset['std'],
            subset['mean'] + 1.5 * subset['std'],
            alpha=0.2
        )

    plt.title(f'Logistic Regression Performance with Random Feature Selection (Mean +/- 1.5 Std Dev over {num_runs} Runs)')
    plt.xlabel('Number of Features')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.legend(title='Metric')
    plt.show()

    # Build wide-format baseline_summary: [features_num, acc_mean, acc_std, f1_mean, f1_std]
    acc_df = pd.DataFrame(acc_records)
    baseline_summary = (
        acc_df.groupby('features_num')[['accuracy', 'macro_f1']]
        .agg(['mean', 'std'])
        .reset_index()
    )
    baseline_summary.columns = ['features_num', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std']

    if return_summary:
        return baseline_summary
    return None
