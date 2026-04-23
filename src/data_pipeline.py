from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    DATASET_PATH,
    IMPUTER_STRATEGY,
    PREPROCESSING_DIR,
    RANDOM_STATE,
    SKEW_MIN_NONZERO_FRACTION,
    SKEW_MIN_UNIQUE,
    SKEW_THRESHOLD,
    SPLITS_DIR,
    TEST_SIZE,
    VAL_SIZE,
)
from src.schema import ATTACK_CLASSES, FEATURE_COLUMNS, LABEL_COLUMN


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass
class PreprocessorArtifacts:
    preprocessor: Pipeline
    label_encoder: LabelEncoder
    feature_columns: list[str]
    skewed_features: list[str]
    constant_features: list[str]
    class_weights: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading & inspection
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run data/fetch/fetch_sample.py or data/fetch/prepare_sample.py first."
        )

    df = pd.read_parquet(dataset_path)
    required = FEATURE_COLUMNS + [LABEL_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            "Check src/schema.py and the prepared sample."
        )
    return df


def inspect_dataset(df: pd.DataFrame) -> dict[str, object]:
    """Summary of data-quality issues that must be resolved before modeling."""
    features = df[FEATURE_COLUMNS]
    numeric = features.apply(pd.to_numeric, errors="coerce")

    null_counts = numeric.isna().sum()
    inf_counts = np.isinf(numeric).sum()
    label_counts = df[LABEL_COLUMN].astype(str).value_counts()

    return {
        "rows": len(df),
        "feature_count": len(FEATURE_COLUMNS),
        "null_counts_per_feature": null_counts[null_counts > 0].to_dict(),
        "inf_counts_per_feature": inf_counts[inf_counts > 0].to_dict(),
        "duplicate_row_count": int(df.duplicated().sum()),
        "label_count": int(df[LABEL_COLUMN].nunique()),
        "labels_outside_schema": sorted(set(label_counts.index) - set(ATTACK_CLASSES)),
        "labels_missing_from_data": sorted(set(ATTACK_CLASSES) - set(label_counts.index)),
        "label_distribution": label_counts.to_dict(),
    }


def detect_constant_features(df: pd.DataFrame) -> list[str]:
    """Return feature columns that have a single unique value across all rows."""
    return [col for col in FEATURE_COLUMNS if df[col].nunique(dropna=False) <= 1]


def detect_skewed_features(
    X: pd.DataFrame,
    threshold: float = SKEW_THRESHOLD,
    min_unique: int = SKEW_MIN_UNIQUE,
    min_nonzero_fraction: float = SKEW_MIN_NONZERO_FRACTION,
    exclude: list[str] | None = None,
) -> list[str]:
    """
    Return features that are *meaningfully* skewed on the training split.

    A feature qualifies only if ALL three conditions hold:
      1. |skew| > threshold  (heavy tail)
      2. nunique >= min_unique  (enough distinct values to call it continuous)
      3. fraction of non-zero rows >= min_nonzero_fraction  (not a sparse indicator)

    Conditions 2 and 3 prevent Yeo-Johnson from being applied to binary /
    indicator features (protocol flags, TCP flag counts) that register huge
    mathematical skew because they are mostly zero. Power transforms cannot
    normalize those and just waste fit time.
    """
    exclude = set(exclude or [])
    candidates = [c for c in FEATURE_COLUMNS if c in X.columns and c not in exclude]

    sub = X[candidates]
    skew = sub.skew(numeric_only=True).abs()
    nunique = sub.nunique(dropna=True)
    nonzero_fraction = (sub != 0).mean()

    mask = (skew > threshold) & (nunique >= min_unique) & (nonzero_fraction >= min_nonzero_fraction)
    return skew[mask].index.tolist()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_dataset(
    df: pd.DataFrame,
    allowed_labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Minimal cleaning for the prepared parquet:

      - Drop rows with a missing or empty label.
      - Optionally filter to an allow-list of labels.
      - Drop exact duplicate rows.

    Numeric coercion and inf-replacement are **not** done here — the prepared
    parquet is schema-validated upstream (`src.data_pipeline.load_dataset` and
    `data/fetch/prepare_sample.py`). The fitted preprocessor's SimpleImputer
    remains as the defensive last line for NaN/inf that might appear at
    inference time on raw traffic.
    """
    cleaned = df.dropna(subset=[LABEL_COLUMN]).copy()
    cleaned[LABEL_COLUMN] = cleaned[LABEL_COLUMN].astype(str).str.strip()
    cleaned = cleaned[cleaned[LABEL_COLUMN] != ""]

    if allowed_labels is not None:
        cleaned = cleaned[cleaned[LABEL_COLUMN].isin(allowed_labels)]

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DatasetSplits:
    """Stratified 3-way split: train / val / test."""
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Scale val_size to be relative to the remaining (train + val) slice.
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor(skewed_features: list[str]) -> Pipeline:
    """
    Median-impute -> Yeo-Johnson power-transform on skewed columns -> standard-scale.

    Uses pandas output so downstream steps can reference features by name.
    """
    if skewed_features:
        skew_step = ColumnTransformer(
            transformers=[
                (
                    "power",
                    PowerTransformer(method="yeo-johnson", standardize=False),
                    skewed_features,
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
    else:
        skew_step = "passthrough"

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=IMPUTER_STRATEGY)),
            ("power_transform", skew_step),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def fit_preprocessor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    skew_threshold: float = SKEW_THRESHOLD,
    skew_min_unique: int = SKEW_MIN_UNIQUE,
    skew_min_nonzero_fraction: float = SKEW_MIN_NONZERO_FRACTION,
    exclude_from_skew: list[str] | None = None,
) -> PreprocessorArtifacts:
    """Detect issues on X_train, build and fit the preprocessor + label encoder."""
    constant_features = detect_constant_features(X_train)
    skewed_features = detect_skewed_features(
        X_train,
        threshold=skew_threshold,
        min_unique=skew_min_unique,
        min_nonzero_fraction=skew_min_nonzero_fraction,
        exclude=(exclude_from_skew or []) + constant_features,
    )

    preprocessor = build_preprocessor(skewed_features)
    preprocessor.fit(X_train[FEATURE_COLUMNS])

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    class_weight_array = compute_class_weight(
        class_weight="balanced",
        classes=label_encoder.classes_,
        y=y_train.values,
    )
    class_weights = {
        str(cls): float(w)
        for cls, w in zip(label_encoder.classes_, class_weight_array)
    }

    return PreprocessorArtifacts(
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        feature_columns=FEATURE_COLUMNS,
        skewed_features=skewed_features,
        constant_features=constant_features,
        class_weights=class_weights,
    )


def transform_features(
    preprocessor: Pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the fitted preprocessor and return a DataFrame with original feature order."""
    transformed = preprocessor.transform(X[FEATURE_COLUMNS])
    return transformed[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    artifacts: PreprocessorArtifacts,
    output_dir: str | Path = PREPROCESSING_DIR,
) -> Path:
    """Save preprocessor, label encoder, and a JSON manifest for inspection."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts.preprocessor, out / "preprocessor.joblib")
    joblib.dump(artifacts.label_encoder, out / "label_encoder.joblib")

    manifest = {
        "feature_columns": artifacts.feature_columns,
        "skewed_features": artifacts.skewed_features,
        "constant_features": artifacts.constant_features,
        "classes": artifacts.label_encoder.classes_.tolist(),
        "class_weights": artifacts.class_weights,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def load_artifacts(
    input_dir: str | Path = PREPROCESSING_DIR,
) -> PreprocessorArtifacts:
    src = Path(input_dir)
    preprocessor: Pipeline = joblib.load(src / "preprocessor.joblib")
    label_encoder: LabelEncoder = joblib.load(src / "label_encoder.joblib")
    manifest = json.loads((src / "manifest.json").read_text())
    return PreprocessorArtifacts(
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        feature_columns=manifest["feature_columns"],
        skewed_features=manifest["skewed_features"],
        constant_features=manifest["constant_features"],
        class_weights=manifest["class_weights"],
    )


def save_splits(
    splits: DatasetSplits,
    label_encoder: LabelEncoder,
    output_dir: str | Path = SPLITS_DIR,
) -> Path:
    """
    Save each split as a single parquet containing features + `label` (string)
    + `label_encoded` (int). One file per split keeps loading simple.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, X, y in [
        ("train", splits.X_train, splits.y_train),
        ("val", splits.X_val, splits.y_val),
        ("test", splits.X_test, splits.y_test),
    ]:
        frame = X.copy()
        frame[LABEL_COLUMN] = y.values
        frame["label_encoded"] = label_encoder.transform(y.values)
        frame.to_parquet(out / f"{name}.parquet", index=False)

    manifest = {
        "train_rows": len(splits.X_train),
        "val_rows": len(splits.X_val),
        "test_rows": len(splits.X_test),
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "encoded_label_column": "label_encoded",
        "classes": label_encoder.classes_.tolist(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def load_split(
    split: str,
    input_dir: str | Path = SPLITS_DIR,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load a previously saved split. Returns (X, y_string, y_encoded)."""
    path = Path(input_dir) / f"{split}.parquet"
    frame = pd.read_parquet(path)
    X = frame[FEATURE_COLUMNS]
    y_str = frame[LABEL_COLUMN]
    y_enc = frame["label_encoded"]
    return X, y_str, y_enc


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def describe_splits(splits: DatasetSplits) -> pd.DataFrame:
    """Per-class row counts across train/val/test. Useful for sanity checks."""
    summary = pd.concat(
        [
            splits.y_train.value_counts().rename("train"),
            splits.y_val.value_counts().rename("val"),
            splits.y_test.value_counts().rename("test"),
        ],
        axis=1,
    ).fillna(0).astype(int).sort_index()
    return summary
