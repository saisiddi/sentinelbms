"""Tests for the SentinelBMS ML anomaly detector."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_detector import ml_anomaly_score  # noqa: E402


NORMAL_RECORD = {
    "vehicle_id": "ML-TEST",
    "timestamp": "2026-09-01T00:00:00Z",
    "pack_avg_voltage": 3.71,
    "cell_voltages": [3.70, 3.71, 3.72, 3.70, 3.71, 3.72],
    "pack_temperature_c": 27.0,
    "state_of_charge_reported_pct": 53.0,
    "firmware_hash": "aaaa1111bbbb2222",
    "known_good_firmware_hash": "aaaa1111bbbb2222",
}


def test_score_shape():
    result = ml_anomaly_score(NORMAL_RECORD)
    assert set(result.keys()) == {"anomaly_score", "is_outlier", "model"}
    assert isinstance(result["anomaly_score"], float)
    assert isinstance(result["is_outlier"], bool)
    assert result["model"] == "isolation_forest_v1"


def test_baseline_records_mostly_not_outliers():
    """Records drawn from the healthy baseline distribution should
    score as non-outliers the majority of the time."""
    import numpy as np
    from ml_detector import _generate_baseline  # noqa: E402

    rng = np.random.default_rng(99)
    X = _generate_baseline(n=200, seed=42)
    rows = []
    for i in range(200):
        row = NORMAL_RECORD.copy()
        cells = X[i, 0] + rng.normal(0, 0.02, 6)
        row["pack_avg_voltage"] = float(X[i, 0])
        row["cell_voltages"] = [float(v) for v in cells]
        row["pack_temperature_c"] = float(X[i, 3])
        row["state_of_charge_reported_pct"] = float(X[i, 4])
        rows.append(row)

    outlier_count = sum(
        1 for r in rows if ml_anomaly_score(r)["is_outlier"]
    )
    assert outlier_count < 40, f"Too many outliers ({outlier_count}/200) on healthy data"


def test_extreme_record_is_outlier():
    """A physically implausible combination must be flagged."""
    extreme = NORMAL_RECORD.copy()
    extreme.update(
        {
            "pack_avg_voltage": 4.50,
            "cell_voltages": [1.0, 4.5, 4.4, 4.5, 1.0, 4.5],
            "pack_temperature_c": 95.0,
            "state_of_charge_reported_pct": 99.0,
        }
    )
    result = ml_anomaly_score(extreme)
    assert result["is_outlier"] is True


def test_cold_start_creates_model_file(tmp_path: Path, monkeypatch):
    """When no model exists, ml_anomaly_score trains and caches one."""
    monkeypatch.setenv("BASELINE_N_RECORDS", "30")
    model_dir = tmp_path / "models"
    model_path = model_dir / "isolation_forest.joblib"
    monkeypatch.chdir(tmp_path)

    # Patch paths inside ml_detector by importing fresh module attributes
    import ml_detector  # noqa: F401

    ml_detector.MODEL_PATH = model_path
    ml_detector.MODEL_DIR = model_dir

    assert not model_path.exists()
    result = ml_anomaly_score(NORMAL_RECORD)
    assert model_path.exists()
    assert set(result.keys()) == {"anomaly_score", "is_outlier", "model"}
