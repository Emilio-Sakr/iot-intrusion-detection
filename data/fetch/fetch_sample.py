from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

# Project root is two levels up from data/fetch/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.schema import FEATURE_COLUMNS, LABEL_COLUMN

SAMPLE_FILE_ID = os.environ.get("SAMPLE_FILE_ID", "")
METADATA_FILE_ID = os.environ.get("METADATA_FILE_ID", "")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_OUTPUT = PROCESSED_DIR / "sample.parquet"
METADATA_OUTPUT = PROCESSED_DIR / "sample_metadata.json"

def build_drive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading → {output_path}")
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {output_path.name}: {e}") from e


def validate_sample(path: Path) -> None:
    """Verify the downloaded parquet has the columns defined in src/schema.py."""
    try:
        import pandas as pd
    except ImportError:
        print("  (pandas not installed — skipping schema validation)")
        return

    df = pd.read_parquet(path)
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
        download_file(build_drive_url(SAMPLE_FILE_ID), SAMPLE_OUTPUT)
        validate_sample(SAMPLE_OUTPUT)

    if METADATA_FILE_ID:
        if METADATA_OUTPUT.exists():
            print(f"sample_metadata.json already exists: {METADATA_OUTPUT}")
        else:
            download_file(build_drive_url(METADATA_FILE_ID), METADATA_OUTPUT)

    print("\nDone.")
    print(f"  Sample:   {SAMPLE_OUTPUT}")
    if METADATA_OUTPUT.exists():
        print(f"  Metadata: {METADATA_OUTPUT}")


if __name__ == "__main__":
    main()
