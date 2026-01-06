from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def train_and_evaluate_logistic_regression(gene_expression_df, train_labels_df, val_labels_df, n_features, random_seed=None, output_dict:bool=False, lr_C:float = np.inf):
    """
    Trains and evaluates a Logistic Regression model on gene expression data.

    Args:
        gene_expression_df (pd.DataFrame): DataFrame containing gene expression data.
                                           Rows are genes, columns are samples (e.g., F1, F2, ...).
                                           Assumes the first column is 'Unnamed: 0' for gene names.
        train_labels_df (pd.DataFrame): DataFrame containing training labels.
                                        Must have 'samplename' and 'is_lumA' columns.
        val_labels_df (pd.DataFrame): DataFrame containing validation labels.
                                      Must have 'samplename' and 'is_lumA' columns.
        n_features (int): The number of features (genes) to randomly select.
        random_seed (int, optional): Seed for random feature selection. Defaults to None.
        output_dict (bool, optional): Whether to return the classification report as a dictionary. Defaults to False.
        lr_C (float, optional): Regularization parameter for Logistic Regression 
                                (inverse - smaller value means more regularization). 
                                Defaults to np.inf (no regularization).

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

    # 3. Perform random feature selection
    if n_features > X_train_full.shape[1]:
        raise ValueError("n_features cannot be greater than the total number of features.")

    np.random.seed(random_seed)
    selected_features = np.random.choice(X_train_full.columns, n_features, replace=False)

    X_train = X_train_full[selected_features]
    X_val = X_val_full[selected_features]

    # 4. Train Logistic Regression model
    model = LogisticRegression(random_state=random_seed, max_iter=1000, C=lr_C)
    model.fit(X_train, y_train)

    # 5. Evaluate model
    y_pred = model.predict(X_val)
    classification_rep = classification_report(y_val, y_pred, output_dict=output_dict)

    return model, classification_rep

def validation_report_to_df(validation_report, features_num):
  report = pd.DataFrame(validation_report).T.reset_index().rename(columns={'index':'class'})
  report = pd.melt(report, id_vars = ['class'], value_vars=['precision', 'recall','f1-score','support'])
  report['features_num'] = features_num
  return report.loc[report['class'] != 'accuracy']

def plot_performance_with_stats(gene_expression_df:pd.DataFrame, 
                                train_labels_df: pd.DataFrame, 
                                val_labels_df: pd.DataFrame, 
                                N_values: list[int], 
                                random_seed: int, 
                                num_runs: int, 
                                lr_C: float = np.inf):
    all_runs_results = []

    for run_id in tqdm(range(num_runs), desc="Running experiments"):
        run_results = []
        for n in N_values:
            # Call the train_and_evaluate_logistic_regression function
            _, validation_report = train_and_evaluate_logistic_regression(
                gene_expression_df=gene_expression_df,
                train_labels_df=train_labels_df,
                val_labels_df=val_labels_df,
                n_features=n,
                random_seed=random_seed + run_id, # Vary seed for each run
                output_dict=True,
                lr_C=lr_C
            )
            # Convert validation report to DataFrame and append
            run_results.append(validation_report_to_df(validation_report, n))

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
    display(summary_stats)

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