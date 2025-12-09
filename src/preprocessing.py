import pandas as pd

def train_test_val_split(df: pd.DataFrame, random_seed:int, train_pcnt:float=0.7, val_pct:float=0.1):
    df_copy = df.copy()
    df_copy['dataset'] = None
    n_samples = len(df_copy)
    train_size = int(n_samples * train_pcnt)
    val_size = int(n_samples * val_pct)
    test_size = n_samples - train_size - val_size
    dataset_column_idx = df_copy.columns.get_loc('dataset')
    train_idx = df_copy.sample(n=train_size, random_state=random_seed).index
    df_copy.iloc[train_idx, dataset_column_idx] = 'Train'
    non_train_df_copy = df_copy.loc[df_copy.dataset != 'Train']
    validation_idx = non_train_df_copy.sample(n=val_size, random_state=random_seed).index
    df_copy.iloc[validation_idx, dataset_column_idx] = 'Validation'
    df_copy.loc[df_copy.dataset.isna(), 'dataset'] = 'Test'
    def get_dataset(ds):
        return df_copy[df_copy.dataset == ds]
    return get_dataset('Train'), get_dataset('Validation'), get_dataset('Test')


