"""
Evaluation helpers shared by every baseline / model notebook.

The intent is that any classifier (LR, RF, HGB, ...) can be evaluated and
persisted with the same one-liner pattern:

    metrics = evaluate(model, X_val, y_val, label_encoder)
    plot_confusion_matrix(model, X_val, y_val, label_encoder,
                          output_path=BASELINE_DIR / "lr_confusion_val.png")
    save_baseline(model, metrics, name="logistic_regression")
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from src.config import BASELINE_DIR


@dataclass
class EvaluationMetrics:
    """Container for the metrics we care about across every baseline."""

    split: str
    n_rows: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Print elapsed wall-clock time for a code block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:,.1f}s")


def evaluate(
    model,
    X: pd.DataFrame,
    y_true,
    label_encoder: LabelEncoder,
    split_name: str,
) -> EvaluationMetrics:
    """
    Run prediction and compute the standard metric battery.

    `y_true` may be either string labels or integer-encoded labels — whatever
    the model was trained on. `label_encoder` is used purely to map integer
    predictions back to class names in the per-class report.
    """
    y_pred = model.predict(X)

    sample = y_true.iloc[0] if hasattr(y_true, "iloc") else y_true[0]
    is_string_labels = isinstance(sample, str)
    target_names = list(label_encoder.classes_)
    labels = target_names if is_string_labels else list(range(len(target_names)))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {k: v for k, v in report.items() if k in target_names}

    return EvaluationMetrics(
        split=split_name,
        n_rows=int(len(y_true)),
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        per_class=per_class,
    )


def per_class_dataframe(metrics: EvaluationMetrics) -> pd.DataFrame:
    """Tidy per-class precision/recall/F1/support table sorted by F1 ascending."""
    df = pd.DataFrame(metrics.per_class).T
    df = df[["precision", "recall", "f1-score", "support"]]
    df = df.sort_values("f1-score", ascending=True)
    return df


def plot_confusion_matrix(
    model,
    X: pd.DataFrame,
    y_true,
    label_encoder: LabelEncoder,
    normalize: str | None = "true",
    title: str = "Confusion matrix (row-normalized)",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot a confusion matrix with class names on both axes.

    `normalize="true"` (default) divides each row by its sum, so the diagonal
    shows per-class **recall** directly — which is what we care about for IDS.
    Off-diagonal entries reveal which classes get confused with which.
    """
    y_pred = model.predict(X)
    classes = label_encoder.classes_
    sample = y_true.iloc[0] if hasattr(y_true, "iloc") else y_true[0]
    is_string_labels = isinstance(sample, str)
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(classes) if is_string_labels else list(range(len(classes))),
        normalize=normalize,
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else cm.max())

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


def save_baseline(
    model,
    metrics: EvaluationMetrics | list[EvaluationMetrics],
    name: str,
    output_dir: str | Path = BASELINE_DIR,
) -> Path:
    """
    Persist a trained baseline:

      models/baseline/{name}.joblib
      models/baseline/{name}_metrics.json

    `metrics` can be a single EvaluationMetrics or a list (e.g. one per split).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / f"{name}.joblib")

    payload = (
        [asdict(m) for m in metrics]
        if isinstance(metrics, list)
        else asdict(metrics)
    )
    (out / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2))
    return out


def load_baseline(name: str, input_dir: str | Path = BASELINE_DIR):
    """Mirror of save_baseline for re-loading a persisted baseline."""
    in_dir = Path(input_dir)
    model = joblib.load(in_dir / f"{name}.joblib")
    metrics_payload = json.loads((in_dir / f"{name}_metrics.json").read_text())
    return model, metrics_payload
