"""
FastAPI inference service for the IoT intrusion detection model.

Loads the preprocessor, label encoder, and classifier once at startup, then
applies the same transform pipeline used during training to incoming raw
feature vectors. Swap the served model by setting MODEL_FILENAME to another
joblib under models/baseline/ (default: random_forest.joblib).

Run locally:
    uvicorn src.api:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import BASELINE_DIR
from src.data_pipeline import load_artifacts, transform_features
from src.schema import PredictionResponse, TrafficRecord

MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "three_stage_rf.joblib")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = BASELINE_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found: {model_path}. "
            f"Train a model or set MODEL_FILENAME to an existing file in {BASELINE_DIR}."
        )

    artifacts = load_artifacts()
    state["preprocessor"] = artifacts.preprocessor
    state["model"] = joblib.load(model_path)
    # Use the model's own class order, not the label encoder's, because
    # predict_proba(X)[i] is aligned with model.classes_[i] — not with any
    # external sort order. Mismatches here silently mis-label every row.
    state["classes"] = [str(c) for c in state["model"].classes_]
    state["model_name"] = model_path.name
    yield
    state.clear()


app = FastAPI(
    title="IoT Intrusion Detection",
    description="Classifies IoT network flow records as benign or one of several attack types.",
    version="0.1.0",
    lifespan=lifespan,
)


def _predict_frame(X: pd.DataFrame) -> list[PredictionResponse]:
    X_t = transform_features(state["preprocessor"], X)
    probs = state["model"].predict_proba(X_t)
    idx = np.argmax(probs, axis=1)
    classes = state["classes"]
    return [
        PredictionResponse(
            prediction=classes[i],
            confidence=float(p[i]),
            probabilities={c: float(v) for c, v in zip(classes, p)},
        )
        for p, i in zip(probs, idx)
    ]


@app.get("/health")
def health():
    return {
        "status": "ok" if state else "loading",
        "model": state.get("model_name"),
        "n_classes": len(state.get("classes", [])),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(record: TrafficRecord):
    X = pd.DataFrame([record.model_dump()])
    return _predict_frame(X)[0]


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(records: list[TrafficRecord]):
    if not records:
        raise HTTPException(status_code=400, detail="records must be non-empty")
    X = pd.DataFrame([r.model_dump() for r in records])
    return _predict_frame(X)
