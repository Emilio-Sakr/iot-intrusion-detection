"""
Central configuration for the IoT intrusion detection project.

All file paths, directory layouts, and pipeline hyperparameters live here so
the rest of the codebase has a single source of truth. Import from this module
instead of hard-coding values.

    from src.config import DATASET_PATH, RANDOM_STATE, SKEW_THRESHOLD
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = PROCESSED_DIR / "splits"

MODELS_DIR: Path = PROJECT_ROOT / "models"
PREPROCESSING_DIR: Path = MODELS_DIR / "preprocessing"
BASELINE_DIR: Path = MODELS_DIR / "baseline"

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Canonical data files
# ---------------------------------------------------------------------------

DATASET_PATH: Path = PROCESSED_DIR / "sample.parquet"
DATASET_METADATA_PATH: Path = PROCESSED_DIR / "sample_metadata.json"

SPLIT_FILES: dict[str, Path] = {
    "train": SPLITS_DIR / "train.parquet",
    "val": SPLITS_DIR / "val.parquet",
    "test": SPLITS_DIR / "test.parquet",
}
SPLITS_MANIFEST_PATH: Path = SPLITS_DIR / "manifest.json"

PREPROCESSOR_PATH: Path = PREPROCESSING_DIR / "preprocessor.joblib"
LABEL_ENCODER_PATH: Path = PREPROCESSING_DIR / "label_encoder.joblib"
PREPROCESSING_MANIFEST_PATH: Path = PREPROCESSING_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# prepare_sample.py — raw-data downsampling
# ---------------------------------------------------------------------------

# Rows read at a time from each CSV when scanning the raw dataset.
CHUNK_SIZE: int = 100_000

# Per-class downsampling cap. None -> use the median class count.
# Classes at or below the cap keep 100% of their rows (no data loss for rare
# classes). Oversampling happens nowhere; imbalance is handled via class
# weights in the loss function instead.
MAX_SAMPLES_PER_CLASS: int | None = None

# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

VAL_SIZE: float = 0.15
TEST_SIZE: float = 0.15
# train_size is 1 - VAL_SIZE - TEST_SIZE = 0.70

# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

# Yeo-Johnson power transform eligibility. A feature is transformed only if
# ALL three conditions hold on the training split:
#   1. |skew| > SKEW_THRESHOLD
#   2. nunique >= SKEW_MIN_UNIQUE
#   3. fraction of non-zero rows >= SKEW_MIN_NONZERO_FRACTION
# Conditions 2 and 3 exclude sparse indicator features (protocol flags, TCP
# flag counts) that look "skewed" mathematically but are really binary-ish —
# Yeo-Johnson cannot normalize a distribution that's almost all zeros.
SKEW_THRESHOLD: float = 1.0
SKEW_MIN_UNIQUE: int = 50
SKEW_MIN_NONZERO_FRACTION: float = 0.05

# Strategy used by SimpleImputer in the fitted pipeline. Defensive against
# NaN/inf at inference time even if the prepared parquet itself is clean.
IMPUTER_STRATEGY: str = "median"

# ---------------------------------------------------------------------------
# Environment variables (looked up at runtime by fetch_sample.py)
# ---------------------------------------------------------------------------

ENV_SAMPLE_FILE_ID: str = "SAMPLE_FILE_ID"
ENV_METADATA_FILE_ID: str = "METADATA_FILE_ID"
