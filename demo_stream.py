"""
Streaming demo for the IoT intrusion detection API.

Samples records from the test split, posts them to the running API one at a
time, and prints a color-coded live feed. Mirrors what a real-time IDS looks
like on a SOC analyst's terminal. Designed for demo videos.

Prerequisites:
    1. Start the API:
           uvicorn src.api:app --port 8000
    2. Install demo deps (included in requirements.txt):
           pip install requests rich

Run:
    python demo_stream.py                         # defaults: 60 records at 4/sec
    python demo_stream.py --n 120 --rate 6        # longer, faster feed
    python demo_stream.py --attack-ratio 0.5      # more alerts for drama
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import pandas as pd
import requests
from rich.console import Console

from src.config import PROCESSED_DIR
from src.data_pipeline import clean_dataset, load_dataset, split_dataset
from src.schema import FEATURE_COLUMNS, LABEL_COLUMN

BENIGN_LABEL = "BenignTraffic"
DEMO_POOL_PATH = PROCESSED_DIR / "demo_pool.parquet"
DEMO_POOL_PER_CLASS = 200
console = Console()


def load_demo_pool() -> pd.DataFrame:
    """Load (or build once) a small stratified sample of the raw held-out test split."""
    if DEMO_POOL_PATH.exists():
        cached = pd.read_parquet(DEMO_POOL_PATH)
        if LABEL_COLUMN in cached.columns:
            return cached
        # Old cache from a buggy build — fall through and rebuild.

    with console.status("[dim]Building demo fixture — one-time, ~15s..."):
        splits = split_dataset(clean_dataset(load_dataset()))
        df = splits.X_test.copy()
        df[LABEL_COLUMN] = splits.y_test.values
        # Stratified per-class down-sample. Plain loop (groupby.apply drops
        # the group column in pandas 2.2+).
        chunks = [
            g.sample(n=min(len(g), DEMO_POOL_PER_CLASS), random_state=42)
            for _, g in df.groupby(LABEL_COLUMN)
        ]
        df = pd.concat(chunks, ignore_index=True)
        DEMO_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DEMO_POOL_PATH, index=False)
    console.print(f"[dim]Cached demo fixture: {DEMO_POOL_PATH}  ({len(df)} rows)[/]")
    return df


def sample_stream(df: pd.DataFrame, n: int, attack_ratio: float, seed: int) -> pd.DataFrame:
    n_attacks = int(round(n * attack_ratio))
    n_benign = n - n_attacks

    benign = df[df[LABEL_COLUMN] == BENIGN_LABEL]
    attacks = df[df[LABEL_COLUMN] != BENIGN_LABEL]

    benign_sample = benign.sample(n=min(n_benign, len(benign)), random_state=seed)
    attacks_sample = attacks.sample(n=min(n_attacks, len(attacks)), random_state=seed)

    combined = pd.concat([benign_sample, attacks_sample])
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def format_line(i: int, expected: str, result: dict) -> str:
    predicted = result["prediction"]
    conf = result["confidence"]
    is_benign = predicted == BENIGN_LABEL
    correct = predicted == expected
    ts = time.strftime("%H:%M:%S")

    if is_benign:
        icon = "[green]✓[/]"
        pred_style = "green"
    else:
        icon = "[bold red]⚠ ALERT[/]"
        pred_style = "red"

    wrong = "" if correct else "   [bold white on red] MISCLASSIFIED [/]"

    return (
        f"[dim]{ts}[/]  "
        f"flow [cyan]{i:>4}[/]  "
        f"[{pred_style}]{predicted:<26}[/]  "
        f"([yellow]{conf:.2f}[/])  "
        f"{icon}"
        f"{wrong}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Streaming IDS demo feed")
    ap.add_argument("--n", type=int, default=60, help="number of records to stream")
    ap.add_argument("--rate", type=float, default=4.0, help="records per second")
    ap.add_argument("--attack-ratio", type=float, default=0.35, help="fraction of attack traffic")
    ap.add_argument("--api", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        health = requests.get(f"{args.api}/health", timeout=3).json()
    except Exception as e:
        console.print(f"[red]Cannot reach API at {args.api}[/]: {e}")
        console.print("[yellow]Start it first:[/]  uvicorn src.api:app --port 8000")
        raise SystemExit(1)

    console.print("[bold]IoT Intrusion Detection — streaming feed[/]")
    console.print(
        f"[dim]model:[/] {health.get('model')}   "
        f"[dim]classes:[/] {health.get('n_classes')}   "
        f"[dim]rate:[/] {args.rate}/s   "
        f"[dim]attack ratio:[/] {args.attack_ratio:.0%}"
    )
    console.print("[dim]" + "─" * 78 + "[/]")

    df = load_demo_pool()
    stream = sample_stream(df, args.n, args.attack_ratio, seed=args.seed)

    counts: Counter[str] = Counter()
    correct = 0
    delay = 1.0 / args.rate

    with requests.Session() as session:
        for i, (_, row) in enumerate(stream.iterrows(), start=1):
            expected = str(row[LABEL_COLUMN])
            record = {k: float(v) for k, v in row[FEATURE_COLUMNS].items()}

            try:
                r = session.post(f"{args.api}/predict", json=record, timeout=5)
                r.raise_for_status()
                result = r.json()
            except Exception as e:
                console.print(f"[red]Request failed at flow {i}:[/] {e}")
                break

            console.print(format_line(i, expected, result))
            counts[result["prediction"]] += 1
            if result["prediction"] == expected:
                correct += 1
            time.sleep(delay)

    total = sum(counts.values())
    if total == 0:
        return

    alerts = sum(c for k, c in counts.items() if k != BENIGN_LABEL)
    top_attacks = [(k, v) for k, v in counts.most_common() if k != BENIGN_LABEL][:3]

    console.print("[dim]" + "─" * 78 + "[/]")
    console.print(f"[bold]Streamed:[/]     {total} records")
    console.print(f"[bold]Accuracy:[/]     {correct}/{total} ({correct / total:.1%})")
    console.print(f"[bold]Alert rate:[/]   {alerts}/{total} ({alerts / total:.1%})")
    if top_attacks:
        summary = ", ".join(f"{k} ({v})" for k, v in top_attacks)
        console.print(f"[bold]Top attacks:[/]  {summary}")


if __name__ == "__main__":
    main()
