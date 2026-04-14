from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

def normalize_column_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

# Project root is two levels up from data/fetch/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.schema import ATTACK_CLASSES, FEATURE_COLUMNS, LABEL_COLUMN

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "sample.parquet"
METADATA_FILE = PROCESSED_DIR / "sample_metadata.json"

RANDOM_SEED = 42

# Rows read at a time from each CSV
CHUNK_SIZE = 100_000

# Downsample classes above this.  None -> median class count.
# Classes at or below this keep 100% of their rows (no oversampling here --
# that belongs in the training preprocessing step, after train/test split,
# to avoid data leakage).
MAX_SAMPLES_PER_CLASS: int | None = None

REQUIRED_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

def find_csv_files(raw_dir: Path) -> list[Path]:
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Put the raw dataset in data/raw/ first."
        )
    return csv_files


def align_chunk_to_schema(
    chunk: pd.DataFrame,
    filename: str,
    warned_missing_features: dict[str, set[str]],
) -> pd.DataFrame:
    if LABEL_COLUMN not in chunk.columns:
        raise ValueError(
            f"Missing required label column in {filename}: ['{LABEL_COLUMN}']\n"
            f"Expected columns are defined in src/schema.py."
        )

    missing_features = [c for c in FEATURE_COLUMNS if c not in chunk.columns]
    if missing_features:
        file_warned = warned_missing_features.setdefault(filename, set())
        newly_missing = [c for c in missing_features if c not in file_warned]
        if newly_missing:
            print(
                f"  Warning: {filename} is missing feature column(s) "
                f"{newly_missing}; filling with 0.0"
            )
            file_warned.update(newly_missing)
        for col in missing_features:
            chunk[col] = 0.0

    return chunk[REQUIRED_COLUMNS].copy()


def count_classes(csv_files: list[Path]) -> dict[str, int]:
    """Pass 1: count rows per class across all CSVs (label column only)."""
    totals: dict[str, int] = {}
    for csv_file in csv_files:
        for chunk in pd.read_csv(csv_file, chunksize=CHUNK_SIZE):
            chunk.columns = [normalize_column_name(c) for c in chunk.columns]
            if LABEL_COLUMN not in chunk.columns:
                continue
            for label, n in chunk[LABEL_COLUMN].value_counts().items():
                totals[str(label)] = totals.get(str(label), 0) + int(n)
    return totals


def compute_targets(
    class_totals: dict[str, int],
    cap: int,
) -> dict[str, int]:
    """Decide target sample size for each class.

    - Classes above cap -> target = cap  (downsample)
    - Everything else   -> target = actual count (keep all rows)
    """
    return {
        label: min(cap, total)
        for label, total in class_totals.items()
    }


def process_files(csv_files: list[Path]) -> tuple[pd.DataFrame, dict]:
    """
    Two-pass pipeline:
      Pass 1 -- scan CSVs to count per-class totals (fast).
      Pass 2 -- adaptive per-class sampling: rare classes keep 100% of their
                rows (no data thrown away), majority classes get downsampled
                proportionally per chunk.
      Oversampling is deliberately NOT done here -- it belongs in the training
      preprocessing step, after train/test split, to avoid data leakage.
    """

    # --- Pass 1: class counts ---
    print("Pass 1: counting class frequencies ...")
    class_totals = count_classes(csv_files)
    rows_seen = sum(class_totals.values())

    cap = MAX_SAMPLES_PER_CLASS or int(pd.Series(class_totals).median())
    targets = compute_targets(class_totals, cap)

    # Per-class fraction for Pass 2 chunked sampling.
    # Classes at or below cap keep frac=1.0 (no loss).
    sample_fracs: dict[str, float] = {}
    for label, total in class_totals.items():
        if total > cap:
            sample_fracs[label] = cap / total
        else:
            sample_fracs[label] = 1.0

    print(f"  {len(class_totals)} classes, {rows_seen:,} total rows")
    print(f"  cap={cap:,}")
    n_down = sum(1 for f in sample_fracs.values() if f < 1.0)
    n_keep = sum(1 for f in sample_fracs.values() if f == 1.0)
    print(f"  {n_down} classes will be downsampled, {n_keep} kept in full")

    # --- Pass 2: adaptive sampling ---
    print("\nPass 2: adaptive stratified sampling ...")
    sampled_parts: list[pd.DataFrame] = []
    rows_sampled = 0
    warned_missing_features: dict[str, set[str]] = {}

    for file_index, csv_file in enumerate(csv_files, start=1):
        print(f"\n[{file_index}/{len(csv_files)}] {csv_file.name}")

        for chunk_index, chunk in enumerate(
            pd.read_csv(csv_file, chunksize=CHUNK_SIZE), start=1
        ):
            chunk.columns = [normalize_column_name(c) for c in chunk.columns]
            chunk = align_chunk_to_schema(
                chunk, csv_file.name, warned_missing_features,
            )

            sampled_chunk = (
                chunk.groupby(LABEL_COLUMN, group_keys=False)
                .apply(
                    lambda x: x.sample(
                        n=max(1, int(len(x) * sample_fracs.get(x.name, 1.0))),
                        random_state=RANDOM_SEED,
                    )
                )
                .reset_index(drop=True)
            )

            rows_sampled += len(sampled_chunk)
            sampled_parts.append(sampled_chunk)

            print(
                f"  Chunk {chunk_index}: "
                f"read {len(chunk):,} -> kept {len(sampled_chunk):,}"
            )

    if not sampled_parts:
        raise ValueError("No rows were sampled from the source files.")

    df = pd.concat(sampled_parts, ignore_index=True)

    # Warn about labels outside the expected taxonomy
    unknown = set(df[LABEL_COLUMN].unique()) - set(ATTACK_CLASSES)
    if unknown:
        print(f"  Warning: unexpected label values found: {sorted(unknown)}")

    class_counts_balanced: dict[str, int] = {
        str(k): int(v) for k, v in df[LABEL_COLUMN].value_counts().items()
    }

    metadata = {
        "source_folder": str(RAW_DIR),
        "source_files_count": len(csv_files),
        "source_files": [f.name for f in csv_files],
        "rows_seen": rows_seen,
        "rows_sampled": len(df),
        "columns": list(df.columns),
        "label_column": LABEL_COLUMN,
        "max_samples_per_class": cap,
        "chunk_size": CHUNK_SIZE,
        "random_seed": RANDOM_SEED,
        "class_distribution_raw": class_totals,
        "class_targets": targets,
        "class_distribution_final": class_counts_balanced,
        "output_file": str(OUTPUT_FILE),
    }

    return df, metadata


def save_outputs(df: pd.DataFrame, metadata: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_FILE, index=False)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSaved sample:   {OUTPUT_FILE}")
    print(f"Saved metadata: {METADATA_FILE}")


def main() -> None:
    try:
        csv_files = find_csv_files(RAW_DIR)
        df, metadata = process_files(csv_files)
        save_outputs(df, metadata)
        print(f"\nDone. Final shape: {df.shape}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
