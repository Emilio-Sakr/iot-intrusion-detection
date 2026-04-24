# IoT Intrusion Detection

A machine-learning pipeline and serving API that classifies IoT network flows as benign or one of 33 attack types, trained on the [CIC-IoT-2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) dataset.

---

## Quick start

### Option A — pull the prebuilt image from Docker Hub

```bash
docker pull <dockerhub-user>/iot-intrusion-detection:latest
docker run --rm -p 8000:8000 <dockerhub-user>/iot-intrusion-detection:latest
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Option B — build from source

```bash
git clone https://github.com/Emilio-Sakr/iot-intrusion-detection.git
cd iot-intrusion-detection
docker build -t iot-intrusion-detection .
docker run --rm -p 8000:8000 iot-intrusion-detection
```

### Option C — run directly with Python

```bash
pip install -r requirements.txt
uvicorn src.api:app --port 8000
```

Requires Python 3.11+ and the preprocessor + trained model under `models/` (produced by the training notebooks — see below).

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check + currently loaded model name |
| `POST` | `/predict` | One `TrafficRecord` → prediction + confidence + per-class probabilities |
| `POST` | `/predict/batch` | List of `TrafficRecord` → list of predictions (micro-batched streaming) |

Schemas are defined in [src/schema.py](src/schema.py). The API applies the fitted preprocessor before inference, so clients send raw feature values.

Swap the served model via env var:

```bash
MODEL_FILENAME=random_forest.joblib uvicorn src.api:app --port 8000
```

Defaults to `three_stage_rf.joblib`.

---

## Streaming demo

Terminal-based live feed that samples records from the held-out test set and streams them to the running API with colour-coded alerts.

```bash
# terminal 1 — start the API
uvicorn src.api:app --port 8000

# terminal 2 — run the stream
python demo_stream.py
```

First run builds a small stratified fixture cached at `data/processed/demo_pool.parquet` (~15 s). Subsequent runs start instantly.

Useful flags:

```bash
python demo_stream.py --n 120 --rate 6            # longer, faster feed
python demo_stream.py --attack-ratio 0.5          # heavier alert density
```

---

## Training from scratch

The four notebooks under [notebooks/](notebooks/) form the end-to-end training pipeline. Each consumes artifacts produced by its predecessors — run them in order.

| Order | Notebook | Purpose | Wall-clock |
|---|---|---|---:|
| 1 | [01_preprocessing_pipeline.ipynb](notebooks/01_preprocessing_pipeline.ipynb) | Clean → stratified split → fit preprocessor → persist artifacts | ~5 min |
| 2 | [02_logistic_regression.ipynb](notebooks/02_logistic_regression.ipynb) | Multinomial logistic regression (linear baseline) | ~48 min |
| 3 | [03_random_forest.ipynb](notebooks/03_random_forest.ipynb) | Random Forest at a 15 % subsample | ~1 min |
| 4 | [04_three_stage_classifier.ipynb](notebooks/04_three_stage_classifier.ipynb) | Three-stage hierarchical RF (final model) | ~25 min |

Fit times measured on 8 cores / 17 GB RAM. Notebook 04 needs the full training split (~4.1 M rows); reduce `SUBSAMPLE_FRAC` if you have less RAM.

### Getting the data

The prepared sample is hosted on Google Drive. Copy `.env.example` → `.env`, obtain the file IDs from a data owner, then:

```bash
cp .env.example .env
# edit .env
python data/fetch/fetch_sample.py
```

Or rebuild from raw CSVs placed anywhere under `data/raw/`:

```bash
python data/fetch/prepare_sample.py
```

Either path produces `data/processed/sample.parquet` against which notebook 01 runs.

---

## Dataset

**CIC-IoT-2023** — Canadian Institute for Cybersecurity, University of New Brunswick.

- Dataset page: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>
- Benchmark paper: <https://www.mdpi.com/1424-8220/23/13/5941>
