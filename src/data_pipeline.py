from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.schema import ATTACK_CLASSES, FEATURE_COLUMNS, LABEL_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "sample.parquet"
NOTEBOOK_SKEWED_FEATURES = ["flow_duration", "rate"]


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run data/fetch/fetch_sample.py or data/fetch/prepare_sample.py first."
        )

    df = pd.read_parquet(dataset_path)
    missing_columns = [col for col in FEATURE_COLUMNS + [LABEL_COLUMN] if col not in df]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}. Check src/schema.py and the prepared sample."
        )
    return df.copy()


def clean_dataset(
    df: pd.DataFrame,
    allowed_labels: list[str] | None = None,
) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=[LABEL_COLUMN])
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(0.0)

    cleaned[LABEL_COLUMN] = cleaned[LABEL_COLUMN].astype(str).str.strip()
    cleaned = cleaned[cleaned[LABEL_COLUMN] != ""].reset_index(drop=True)

    if allowed_labels is not None:
        cleaned = cleaned[cleaned[LABEL_COLUMN].isin(allowed_labels)].reset_index(drop=True)

    return cleaned


def summarize_dataset_labels(df: pd.DataFrame) -> dict[str, object]:
    labels = sorted(df[LABEL_COLUMN].astype(str).unique())
    labels_outside_schema = sorted(set(labels) - set(ATTACK_CLASSES))
    labels_missing_from_data = sorted(set(ATTACK_CLASSES) - set(labels))
    return {
        "label_count": len(labels),
        "labels": labels,
        "labels_outside_schema": labels_outside_schema,
        "labels_missing_from_data": labels_missing_from_data,
    }


def log_transform_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    transformed = df.copy()
    selected_columns = columns or NOTEBOOK_SKEWED_FEATURES
    for column in selected_columns:
        if column in transformed.columns:
            transformed[column] = transformed[column].clip(lower=0.0)
            transformed[column] = np.log1p(transformed[column])
    return transformed


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplits:
    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def oversample_training_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    train_df = X_train.copy()
    train_df[LABEL_COLUMN] = y_train.values

    class_counts = train_df[LABEL_COLUMN].value_counts()
    target_count = int(class_counts.max())

    sampled_parts: list[pd.DataFrame] = []
    for label, group in train_df.groupby(LABEL_COLUMN):
        if len(group) < target_count:
            group = group.sample(
                n=target_count,
                replace=True,
                random_state=random_state,
            )
        sampled_parts.append(group)

    oversampled = (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    X_resampled = oversampled[FEATURE_COLUMNS].copy()
    y_resampled = oversampled[LABEL_COLUMN].copy()
    return X_resampled, y_resampled


def build_training_splits(
    path: str | Path = DEFAULT_DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    allowed_labels: list[str] | None = None,
) -> DatasetSplits:
    df = load_dataset(path)
    df = clean_dataset(df, allowed_labels=allowed_labels)
    splits = split_dataset(df, test_size=test_size, random_state=random_state)
    X_train_balanced, y_train_balanced = oversample_training_set(
        splits.X_train,
        splits.y_train,
        random_state=random_state,
    )
    return DatasetSplits(
        X_train=X_train_balanced,
        X_test=splits.X_test,
        y_train=y_train_balanced,
        y_test=splits.y_test,
    )


def describe_split(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    summary = pd.concat(
        [
            y_train.value_counts().rename("train"),
            y_test.value_counts().rename("test"),
        ],
        axis=1,
    ).fillna(0)
    return summary.astype(int).sort_index()
