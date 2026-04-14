# IoT Intrusion Detection

Network traffic classification for IoT intrusion detection.
The dataset covers 47 features and 10 attack/benign classes (see [src/schema.py](src/schema.py)).

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in SAMPLE_FILE_ID and METADATA_FILE_ID
```

`.env` is gitignored. Ask a data owner for the actual file IDs.

---

## Data pipeline

### Option A — Download the shared sample (teammates)

The prepared sample is hosted on Google Drive. Once `.env` is filled in:

```bash
python data/fetch/fetch_sample.py
```

Downloads to:
- `data/processed/sample.parquet`
- `data/processed/sample_metadata.json`

The script validates the downloaded file against the schema in `src/schema.py` before finishing.

---

### Option B — Build the sample from raw data (data owners)

1. Place the raw CSV files anywhere inside `data/raw/`
2. Run:

```bash
python data/fetch/prepare_sample.py
```

**What it does (two-pass pipeline):**

| Pass | Description |
|---|---|
| Pass 1: count | Scans all CSVs to count per-class totals. Computes a target for each class: classes above cap get downsampled, classes below floor get oversampled, everything else keeps 100% of its rows. |
| Pass 2: sample | Reads CSVs in chunks. Each class is sampled at its own rate -- rare classes keep every row, majority classes are cut proportionally. After concat, any class still below floor is oversampled with replacement. |
| Schema enforcement | Columns are normalized and reordered to match `FEATURE_COLUMNS + [LABEL_COLUMN]` from `src/schema.py`. Missing columns are zero-filled with a warning. |

Outputs:
- `data/processed/sample.parquet` -- balanced, schema-compliant sample
- `data/processed/sample_metadata.json` -- per-class targets, raw counts, and final distribution

**Tunable constants** (top of `data/fetch/prepare_sample.py`):

| Constant | Default | Description |
|---|---|---|
| `MAX_SAMPLES_PER_CLASS` | `None` | Downsample classes above this (`None` = median class count) |
| `MIN_SAMPLES_PER_CLASS` | `None` | Oversample classes below this with replacement (`None` = cap // 10) |
| `CHUNK_SIZE` | `100_000` | Rows read per CSV chunk |
| `RANDOM_SEED` | `42` | Reproducibility seed |

---

## Schema

Defined in [src/schema.py](src/schema.py):

- **47 features** (all `float`) — network flow statistics, TCP flag counts, protocol indicators
- **10 classes** — `BenignTraffic` + 9 attack categories
- `TrafficRecord` — Pydantic model for single-record input validation
- `PredictionResponse` — API response model (prediction, confidence, per-class probabilities)

---

## Environment variables

| Variable | Description |
|---|---|
| `SAMPLE_FILE_ID` | Google Drive file ID for `sample.parquet` |
| `METADATA_FILE_ID` | Google Drive file ID for `sample_metadata.json` |

---

## Docker

```bash
docker build -t iot-intrusion-detection .
docker run -p 8000:8000 iot-intrusion-detection
```

> `src/api.py` and `models/` are not yet implemented. The Dockerfile is a placeholder for the inference service.
