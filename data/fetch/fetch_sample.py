from __future__ import annotations

import os
import sys
from pathlib import Path

import gdown
from dotenv import load_dotenv

import pandas as pd

# Project root is two levels up from data/fetch/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.config import (
    DATASET_METADATA_PATH,
    DATASET_PATH,
    ENV_METADATA_FILE_ID,
    ENV_SAMPLE_FILE_ID,
)
from src.schema import FEATURE_COLUMNS, LABEL_COLUMN

SAMPLE_FILE_ID = os.environ.get(ENV_SAMPLE_FILE_ID, "")
METADATA_FILE_ID = os.environ.get(ENV_METADATA_FILE_ID, "")

SAMPLE_OUTPUT = DATASET_PATH
METADATA_OUTPUT = DATASET_METADATA_PATH


def download_from_drive(file_id: str, output_path: Path) -> None:
    """
    Download a Google Drive file by ID using gdown.
    gdown handles the virus-scan confirmation page that Drive shows for large
    files, which causes urllib / requests to silently save an HTML page instead
    of the actual file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading -> {output_path}")
    url = f"https://drive.google.com/uc?id={file_id}"
    success = gdown.download(url, str(output_path), quiet=False)
    if not success:
        raise RuntimeError(
            f"gdown failed to download file ID {file_id!r}.\n"
            "Check that the file is shared as 'Anyone with the link can view'."
        )


def validate_sample(path: Path) -> None:
    """Verify the downloaded parquet has the columns defined in src/schema.py."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(
            f"Downloaded file is not a valid parquet — it may be an HTML error page.\n"
            f"Delete {path.name} and re-run this script.\nOriginal error: {e}"
        ) from e

    required = FEATURE_COLUMNS + [LABEL_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Downloaded sample is missing schema columns: {missing}\n"
            "The file may be outdated or from an incompatible source."
        )
    print(f"  Schema OK — {len(df):,} rows, {len(df.columns)} columns")


def main() -> None:
    if not SAMPLE_FILE_ID:
        print(
            "Error: SAMPLE_FILE_ID is not set.\n"
            "Copy .env.example to .env and fill in the Google Drive file IDs."
        )
        sys.exit(1)

    if SAMPLE_OUTPUT.exists():
        print(f"sample.parquet already exists: {SAMPLE_OUTPUT}")
    else:
        download_from_drive(SAMPLE_FILE_ID, SAMPLE_OUTPUT)
        validate_sample(SAMPLE_OUTPUT)

    if METADATA_FILE_ID:
        if METADATA_OUTPUT.exists():
            print(f"sample_metadata.json already exists: {METADATA_OUTPUT}")
        else:
            download_from_drive(METADATA_FILE_ID, METADATA_OUTPUT)

    print("\nDone.")
    print(f"  Sample:   {SAMPLE_OUTPUT}")
    if METADATA_OUTPUT.exists():
        print(f"  Metadata: {METADATA_OUTPUT}")


if __name__ == "__main__":
    main()
