"""SentinelBMS ML anomaly detector (Step 1 upgrade).

Adds a statistical second line of defense alongside the rule engine:
an IsolationForest model trained on a synthetic "normal fleet" baseline.
The rules stay the primary, auditable detector; the ML layer catches
subtle multi-signal patterns the fixed rules don't cover.

IMPORTANT — synthetic baseline disclaimer:
  The training data below is generated noise around healthy-looking
  ranges, NOT real fleet telemetry. In production this model would be
  trained on historical fleet data and re-trained as the fleet evolves.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

from rule_engine import compute_soc_gap

# ---------------------------------------------------------------------------
# Paths (relative to project root, resolved at import time).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"

# ---------------------------------------------------------------------------
# Model / baseline hyperparameters (named constants, easy to tune).
# ---------------------------------------------------------------------------
BASELINE_N_RECORDS = 250
BASELINE_SEED = 42
ISO_FOREST_CONTAMINATION = 0.05  # expected outlier fraction; a small
# fleet rarely exceeds a few percent anomalous readings, so 5% lets
# the model flag genuine outliers without flooding the dashboard.
ISO_FOREST_RANDOM_STATE = 42
ISO_FOREST_VERSION = "isolation_forest_v1"

# Feature order used by both training and scoring — must stay in sync.
FEATURE_NAMES = [
    "pack_avg_voltage",
    "cell_voltage_spread",
    "cell_voltage_std",
    "pack_temperature_c",
    "state_of_charge_reported_pct",
    "soc_gap_pct",
]


def _build_features(record: dict) -> np.ndarray:
    """Build the 6-D feature vector for one telemetry record."""
    cells = record.get("cell_voltages", [])
    pack_avg = record.get("pack_avg_voltage", 0.0)
    if cells:
        arr = np.array(cells, dtype=float)
        spread = float(arr.max() - arr.min())
        std = float(arr.std(ddof=0))
    else:
        spread = std = 0.0
    _, soc_gap = compute_soc_gap(record)
    return np.array(
        [
            pack_avg,
            spread,
            std,
            float(record.get("pack_temperature_c", 0.0)),
            float(record.get("state_of_charge_reported_pct", 0.0)),
            float(soc_gap),
        ],
        dtype=float,
    )


def _generate_baseline(n: int = BASELINE_N_RECORDS, seed: int = BASELINE_SEED) -> np.ndarray:
    """Generate a synthetic healthy fleet baseline for training.

    Ranges mirror the normal records in data/synthetic_telemetry.json:
    pack voltage ~3.6-3.85 V, cells within ~0.04 V of average,
    temperature ~22-32 C, SoC ~40-70%. No anomaly types injected.

    Features are derived from the generated cells the same way
    _build_features does, so training and scoring distributions match.
    """
    rng = np.random.default_rng(seed)
    n = max(n, 10)
    pack_avg = rng.normal(3.72, 0.04, size=n)
    cells = pack_avg[:, None] + rng.normal(0, 0.02, size=(n, 6))
    temps = rng.normal(27.0, 3.0, size=n)
    socs_reported = rng.normal(55.0, 8.0, size=n).clip(30, 80)
    implied_from_avg = (pack_avg - 3.20) / (4.15 - 3.20) * 100.0
    gaps = np.abs(socs_reported - implied_from_avg)
    spreads = cells.max(axis=1) - cells.min(axis=1)
    stds = cells.std(axis=1, ddof=0)
    return np.column_stack([pack_avg, spreads, stds, temps, socs_reported, gaps])


def _load_or_train() -> IsolationForest:
    """Load a saved model, or train on the synthetic baseline and cache it.

    Cold-start path: if models/isolation_forest.joblib is missing,
    train once on the synthetic baseline and serialize so the next
    call is instant. This keeps the demo runnable offline without
    requiring a training step.
    """
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)  # type: ignore[no-untyped-call]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X = _generate_baseline()
    model = IsolationForest(
        contamination=ISO_FOREST_CONTAMINATION,
        random_state=ISO_FOREST_RANDOM_STATE,
    )
    model.fit(X)
    joblib.dump(model, MODEL_PATH)  # type: ignore[no-untyped-call]
    return model


def ml_anomaly_score(record: dict) -> dict:
    """Score one telemetry record with the IsolationForest model.

    Returns {"anomaly_score": float, "is_outlier": bool, "model": version}.
    Handles cold-start transparently — trains and caches on first call
    if no model file exists yet.
    """
    model = _load_or_train()
    features = _build_features(record).reshape(1, -1)
    score = float(model.decision_function(features)[0])
    is_outlier = bool(model.predict(features)[0] == -1)
    return {
        "anomaly_score": round(score, 4),
        "is_outlier": is_outlier,
        "model": ISO_FOREST_VERSION,
    }
