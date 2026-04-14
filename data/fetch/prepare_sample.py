from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Project root is two levels up from data/fetch/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.schema import ATTACK_CLASSES, FEATURE_COLUMNS, LABEL_COLUMN  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "sample.parquet"
METADATA_FILE = PROCESSED_DIR / "sample_metadata.json"

RANDOM_SEED = 42

# Fraction of each class sampled per chunk (first pass)
SAMPLE_FRACTION = 0.10

# Rows read at a time from each CSV
CHUNK_SIZE = 100_000

# After the first-pass stratified sampling, cap every class at this many rows
# to counteract class imbalance.
#   None  → cap at the median class count (robust default: ignores outlier-tiny
#            classes that would otherwise drag the cap down to near-zero)
#   int   → fixed cap, e.g. MAX_SAMPLES_PER_CLASS = 20_000
MAX_SAMPLES_PER_CLASS: int | None = None

# Emit a warning if any class ends up with fewer rows than this after balancing.
MIN_SAMPLES_PER_CLASS_WARN = 200

REQUIRED_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

def find_csv_files(raw_dir: Path) -> list[Path]:
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Put the raw dataset in data/raw/ first."
        )
    return csv_files


def validate_chunk_columns(chunk: pd.DataFrame, filename: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in chunk.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {filename}: {missing}\n"
            f"Expected columns are defined in src/schema.py."
        )


def process_files(csv_files: list[Path]) -> tuple[pd.DataFrame, dict]:
    """
    Two-phase pipeline:
      1. Stratified sampling — sample SAMPLE_FRACTION of each class per chunk,
         preserving class proportions while keeping memory usage bounded.
      2. Class balancing — after concatenation, cap every class at
         MAX_SAMPLES_PER_CLASS (default: minority class count) so that no
         single class dominates the final sample.
    """
    sampled_parts: list[pd.DataFrame] = []

    rows_seen = 0
    rows_sampled = 0
    class_counts_before: dict[str, int] = {}
    class_counts_after_sampling: dict[str, int] = {}

    for file_index, csv_file in enumerate(csv_files, start=1):
        print(f"\n[{file_index}/{len(csv_files)}] Processing: {csv_file.name}")

        for chunk_index, chunk in enumerate(
            pd.read_csv(csv_file, chunksize=CHUNK_SIZE), start=1
        ):
            validate_chunk_columns(chunk, csv_file.name)

            # Keep only schema-defined columns in canonical order
            chunk = chunk[REQUIRED_COLUMNS].copy()

            rows_seen += len(chunk)

            for label, count in chunk[LABEL_COLUMN].value_counts(dropna=False).items():
                class_counts_before[str(label)] = (
                    class_counts_before.get(str(label), 0) + int(count)
                )

            # --- Phase 1: stratified sampling within each chunk ---
            sampled_chunk = (
                chunk.groupby(LABEL_COLUMN, group_keys=False)
                .apply(
                    lambda x: x.sample(
                        frac=min(SAMPLE_FRACTION, 1.0),
                        random_state=RANDOM_SEED,
                    )
                )
                .reset_index(drop=True)
            )

            rows_sampled += len(sampled_chunk)

            for label, count in sampled_chunk[LABEL_COLUMN].value_counts(dropna=False).items():
                class_counts_after_sampling[str(label)] = (
                    class_counts_after_sampling.get(str(label), 0) + int(count)
                )

            sampled_parts.append(sampled_chunk)

            print(
                f"  Chunk {chunk_index}: "
                f"read {len(chunk):,} rows → kept {len(sampled_chunk):,}"
            )

    if not sampled_parts:
        raise ValueError("No rows were sampled from the source files.")

    df = pd.concat(sampled_parts, ignore_index=True)

    # --- Phase 2: class balancing ---
    counts = df[LABEL_COLUMN].value_counts()
    # Use median rather than min so a single rare class doesn't pull the cap
    # down to near-zero and discard the bulk of your training data.
    cap = MAX_SAMPLES_PER_CLASS or int(counts.median())
    print(f"\nBalancing classes — cap per class: {cap:,} rows …")
    print(f"  Class counts before balancing (min={counts.min():,}, "
          f"median={int(counts.median()):,}, max={counts.max():,})")

    df = (
        df.groupby(LABEL_COLUMN, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), cap), random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )

    # Warn about labels outside the expected taxonomy
    unknown = set(df[LABEL_COLUMN].unique()) - set(ATTACK_CLASSES)
    if unknown:
        print(f"  Warning: unexpected label values found: {sorted(unknown)}")

    # Warn about classes with very few rows
    final_counts = df[LABEL_COLUMN].value_counts()
    sparse = final_counts[final_counts < MIN_SAMPLES_PER_CLASS_WARN]
    if not sparse.empty:
        print(
            f"  Warning: {len(sparse)} class(es) have fewer than "
            f"{MIN_SAMPLES_PER_CLASS_WARN} rows after balancing — "
            "consider lowering SAMPLE_FRACTION or collecting more raw data:\n"
            + "\n".join(f"    {cls}: {n:,}" for cls, n in sparse.items())
        )

    class_counts_balanced: dict[str, int] = {
        str(k): int(v) for k, v in df[LABEL_COLUMN].value_counts().items()
    }

    metadata = {
        "source_folder": str(RAW_DIR),
        "source_files_count": len(csv_files),
        "source_files": [f.name for f in csv_files],
        "rows_seen": rows_seen,
        "rows_sampled_before_balancing": rows_sampled,
        "rows_sampled_final": len(df),
        "columns": list(df.columns),
        "label_column": LABEL_COLUMN,
        "sample_fraction": SAMPLE_FRACTION,
        "max_samples_per_class": cap,
        "chunk_size": CHUNK_SIZE,
        "random_seed": RANDOM_SEED,
        "class_distribution_raw": class_counts_before,
        "class_distribution_after_sampling": class_counts_after_sampling,
        "class_distribution_balanced": class_counts_balanced,
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
