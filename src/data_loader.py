from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .preprocessing import train_test_val_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TargetData:
    """Train/val/test splits for a single target variable."""

    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    task_type: str  # "classification" or "regression"


@dataclass
class DataBundle:
    """Complete data splits for the pipeline."""

    X_train: pd.DataFrame  # samples × features
    X_val: pd.DataFrame  # samples × features
    X_test: pd.DataFrame  # samples × features
    targets: dict[str, TargetData]  # target_name → TargetData
    feature_names: list[str]
    n_samples: int
    n_features: int


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """Loads CSV data, handles auto-transpose, merges labels, splits data."""

    def __init__(
        self,
        features_path: str,
        labels_path: str,
        sample_id_column: str = "samplename",
        auto_transpose: bool = True,
        fillna_value: float | int = 0,
    ) -> None:
        self.features_path = features_path
        self.labels_path = labels_path
        self.sample_id_column = sample_id_column
        self.auto_transpose = auto_transpose
        self.fillna_value = fillna_value

    def load_and_split(
        self,
        targets: list[dict],
        random_seed: int = 2,
        train_pct: float = 0.7,
        val_pct: float = 0.1,
    ) -> DataBundle:
        # ------------------------------------------------------------------
        # 1. Load features CSV
        # ------------------------------------------------------------------
        df = pd.read_csv(self.features_path)
        original_shape = df.shape

        # ------------------------------------------------------------------
        # 2 / 3. Auto-transpose detection or ID-column handling
        # ------------------------------------------------------------------
        if self.auto_transpose and df.shape[0] < df.shape[1]:
            # First column contains feature/gene names (row identifiers)
            df.set_index(df.columns[0], inplace=True)
            df = df.T
            df.index.name = None
            new_shape = df.shape
            logger.info(
                "Auto-transposed features: %s → %s",
                original_shape,
                new_shape,
            )
        else:
            # Check if the first column looks like an ID column
            first_col = df.columns[0]
            if (
                first_col == self.sample_id_column
                or first_col == "Unnamed: 0"
                or not pd.api.types.is_numeric_dtype(df[first_col])
            ):
                df.set_index(first_col, inplace=True)
                df.index.name = None
            logger.info(
                "Loaded features without transpose: %s", df.shape,
            )

        # ------------------------------------------------------------------
        # 4. Fill NaN values
        # ------------------------------------------------------------------
        df.fillna(self.fillna_value, inplace=True)

        # ------------------------------------------------------------------
        # 5. Load labels CSV
        # ------------------------------------------------------------------
        labels = pd.read_csv(self.labels_path)

        # ------------------------------------------------------------------
        # 6. Merge features and labels on sample IDs
        # ------------------------------------------------------------------
        feature_sample_ids = set(df.index.astype(str))
        label_sample_ids = set(labels[self.sample_id_column].astype(str))
        common_ids = feature_sample_ids & label_sample_ids

        if not common_ids:
            logger.error("No matching sample IDs found between features and labels")
            raise ValueError(
                "No matching sample IDs found between features and labels"
            )

        # Log warnings for mismatched samples
        only_in_features = feature_sample_ids - label_sample_ids
        only_in_labels = label_sample_ids - feature_sample_ids
        if only_in_features:
            logger.warning(
                "%d samples in features but not in labels: %s",
                len(only_in_features),
                sorted(only_in_features)[:5],
            )
        if only_in_labels:
            logger.warning(
                "%d samples in labels but not in features: %s",
                len(only_in_labels),
                sorted(only_in_labels)[:5],
            )

        logger.info(
            "Sample intersection: %d (features=%d, labels=%d)",
            len(common_ids),
            len(feature_sample_ids),
            len(label_sample_ids),
        )

        # Align both DataFrames to the intersection
        df.index = df.index.astype(str)
        labels[self.sample_id_column] = labels[self.sample_id_column].astype(str)
        labels = labels.set_index(self.sample_id_column)

        common_ids_sorted = sorted(common_ids)
        features_df = df.loc[common_ids_sorted]
        labels_df = labels.loc[common_ids_sorted]

        # ------------------------------------------------------------------
        # 7. Extract target columns from labels
        # ------------------------------------------------------------------
        target_series_map: dict[str, tuple[pd.Series, str]] = {}
        for target_spec in targets:
            name = target_spec["name"]
            task_type = target_spec["task_type"]
            if name not in labels_df.columns:
                logger.error(
                    "Target column '%s' not found in labels. Available: %s",
                    name,
                    list(labels_df.columns),
                )
                raise ValueError(
                    f"Target column '{name}' not found in labels. "
                    f"Available: {list(labels_df.columns)}"
                )
            target_series_map[name] = (labels_df[name], task_type)

        # ------------------------------------------------------------------
        # 8. Split into train/val/test
        # ------------------------------------------------------------------
        # train_test_val_split uses .iloc internally, so it requires a
        # default integer RangeIndex.  We save the sample-ID index,
        # reset to integers, split, then restore the sample-ID index.
        sample_ids = features_df.index.copy()
        features_int_idx = features_df.reset_index(drop=True)

        X_train, X_val, X_test = train_test_val_split(
            features_int_idx, random_seed, train_pcnt=train_pct, val_pct=val_pct
        )

        # train_test_val_split adds a 'dataset' column — drop it
        X_train = X_train.drop(columns=["dataset"])
        X_val = X_val.drop(columns=["dataset"])
        X_test = X_test.drop(columns=["dataset"])

        # Restore sample-ID index using the positional mapping
        X_train.index = sample_ids[X_train.index]
        X_val.index = sample_ids[X_val.index]
        X_test.index = sample_ids[X_test.index]

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(X_train),
            len(X_val),
            len(X_test),
        )

        # Build TargetData for each target
        targets_dict: dict[str, TargetData] = {}
        for name, (series, task_type) in target_series_map.items():
            y_train = series.loc[X_train.index]
            y_val = series.loc[X_val.index]
            y_test = series.loc[X_test.index]
            targets_dict[name] = TargetData(
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                task_type=task_type,
            )

        # ------------------------------------------------------------------
        # 9. Build and return DataBundle
        # ------------------------------------------------------------------
        return DataBundle(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            targets=targets_dict,
            feature_names=list(X_train.columns),
            n_samples=len(X_train) + len(X_val) + len(X_test),
            n_features=len(X_train.columns),
        )
