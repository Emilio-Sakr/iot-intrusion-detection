# IoT Intrusion Detection

Network traffic classification for IoT intrusion detection.
The dataset covers 47 features and 34 attack/benign classes (see [src/schema.py](src/schema.py)).

> **New to the project?** Read [TECHNICAL.md](TECHNICAL.md) for the deep dive on what the pipeline does, the math behind each step, and why each choice was made.

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
| Pass 1: count | Scans all CSVs to count per-class totals and compute a downsampling cap. |
| Pass 2: sample | Reads CSVs in chunks. Minority classes keep all rows, majority classes are reduced proportionally. |
| Schema enforcement | Columns are normalized and reordered to match `FEATURE_COLUMNS + [LABEL_COLUMN]` from `src/schema.py`. Missing columns are zero-filled with a warning. |

Outputs:
- `data/processed/sample.parquet` -- schema-compliant sample for analysis/modeling
- `data/processed/sample_metadata.json` -- per-class targets, raw counts, and final distribution

**Tunable constants** (top of `data/fetch/prepare_sample.py`):

| Constant | Default | Description |
|---|---|---|
| `MAX_SAMPLES_PER_CLASS` | `None` | Downsample classes above this (`None` = median class count) |
| `CHUNK_SIZE` | `100_000` | Rows read per CSV chunk |
| `RANDOM_SEED` | `42` | Reproducibility seed |

### Preprocessing pipeline

The full preprocessing flow lives in [notebooks/03_preprocessing_pipeline.ipynb](notebooks/03_preprocessing_pipeline.ipynb) and is driven by helpers in [src/data_pipeline.py](src/data_pipeline.py). Running the notebook top to bottom:

1. Loads `data/processed/sample.parquet` and inspects data-quality issues (NaN, inf, duplicates, label drift).
2. Cleans the dataset — drops missing labels, coerces features, replaces `±inf` with `NaN`, deduplicates.
3. **Stratified 3-way split** (70 / 15 / 15) — fitting happens on train only to avoid leakage.
4. Detects constant features (reported) and skewed features (`|skew| > 1.0`, power-transformed).
5. Fits a single `sklearn.Pipeline`: `SimpleImputer(median)` → `PowerTransformer(yeo-johnson, skewed cols)` → `StandardScaler`.
6. Fits a `LabelEncoder` and computes balanced `class_weights` (preferred over naive oversampling).
7. Persists artifacts to `models/preprocessing/` and transformed splits to `data/processed/splits/`.

### Baseline model

[notebooks/02_baseline_logistic_regression.ipynb](notebooks/02_baseline_logistic_regression.ipynb) trains a multinomial Logistic Regression on the persisted splits using the class weights from preprocessing, and saves:

- `models/baseline/logistic_regression.joblib` — the fitted model
- `models/baseline/logistic_regression_metrics.json` — accuracy, macro-F1, weighted-F1, per-class report (val + test)
- `models/baseline/logistic_regression_confusion_{val,test}.png` — row-normalized confusion matrices

Reusable evaluation helpers (used by future model notebooks too) live in [src/evaluation.py](src/evaluation.py). Re-load a saved baseline anywhere with:

```python
from src.evaluation import load_baseline
model, metrics = load_baseline("logistic_regression")
```

---

## Schema

Defined in [src/schema.py](src/schema.py):

- **47 features** (all `float`) — network flow statistics, TCP flag counts, protocol indicators
- **34 classes** — `BenignTraffic` + 33 attack categories (DDoS, DoS, Mirai, Recon, MITM/Spoofing, scanning/brute, application-layer)
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
