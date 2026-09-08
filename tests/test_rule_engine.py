"""Unit tests for the SentinelBMS rule engine.

Each test builds a small hand-crafted telemetry dict (independent of the
synthetic dataset) and asserts that classify() flags — or correctly ignores —
the record. Run with:  pytest tests/
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rule_engine import (  # noqa: E402
    CELL_DEVIATION_THRESHOLD,
    AnomalyType,
    classify,
)

GOOD_HASH = "aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa1111bbbb2222"


def make_record(**overrides) -> dict:
    """A baseline, perfectly healthy record; tests override what they need."""
    record = {
        "vehicle_id": "TEST-001",
        "timestamp": "2026-09-01T00:00:00Z",
        "pack_avg_voltage": 3.70,
        "cell_voltages": [3.70, 3.71, 3.69, 3.70],
        "pack_temperature_c": 26.0,
        # 3.70 V sits near the middle of the 3.20-4.15 V range, so a
        # reported SoC around 50% is consistent.
        "state_of_charge_reported_pct": 52.0,
        "firmware_hash": GOOD_HASH,
        "known_good_firmware_hash": GOOD_HASH,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Healthy baseline
# ---------------------------------------------------------------------------


def test_healthy_record_is_not_flagged():
    result = classify(make_record())
    assert result["is_anomalous"] is False
    assert result["anomalies"] == []


# ---------------------------------------------------------------------------
# Rule 1: cell voltage deviation
# ---------------------------------------------------------------------------


def test_cell_deviation_above_threshold_flagged_high():
    # One cell at 3.00 V vs a 3.70 V pack average -> ~19% deviation.
    result = classify(make_record(cell_voltages=[3.70, 3.71, 3.69, 3.00]))
    types = [a["type"] for a in result["anomalies"]]
    assert AnomalyType.CELL_VOLTAGE_DEVIATION.value in types
    anomaly = next(
        a for a in result["anomalies"] if a["type"] == AnomalyType.CELL_VOLTAGE_DEVIATION.value
    )
    assert anomaly["severity_hint"] == "high"


def test_cell_deviation_below_threshold_not_flagged():
    # All cells within ~1% of the pack average -> no flag.
    result = classify(make_record(cell_voltages=[3.70, 3.72, 3.68, 3.71]))
    types = [a["type"] for a in result["anomalies"]]
    assert AnomalyType.CELL_VOLTAGE_DEVIATION.value not in types


def test_cell_deviation_just_below_threshold_not_high():
    # Just inside the threshold (14.9% deviation) must not trip the HIGH rule.
    avg = 4.0
    borderline = avg * (1 - CELL_DEVIATION_THRESHOLD + 0.001)
    result = classify(make_record(pack_avg_voltage=avg, cell_voltages=[avg, avg, avg, borderline]))
    dev_flags = [
        a for a in result["anomalies"] if a["type"] == AnomalyType.CELL_VOLTAGE_DEVIATION.value
    ]
    assert all(a["severity_hint"] != "high" for a in dev_flags)


def test_cell_deviation_just_above_threshold_is_high():
    # Just past the threshold (15.1% deviation) must be HIGH.
    avg = 4.0
    borderline = avg * (1 - CELL_DEVIATION_THRESHOLD - 0.001)
    result = classify(make_record(pack_avg_voltage=avg, cell_voltages=[avg, avg, avg, borderline]))
    dev_flags = [
        a for a in result["anomalies"] if a["type"] == AnomalyType.CELL_VOLTAGE_DEVIATION.value
    ]
    assert len(dev_flags) == 1
    assert dev_flags[0]["severity_hint"] == "high"


# ---------------------------------------------------------------------------
# Rule 2: firmware integrity
# ---------------------------------------------------------------------------


def test_firmware_mismatch_flagged_high():
    result = classify(make_record(firmware_hash="ffff0000" + "0" * 56))
    types = [a["type"] for a in result["anomalies"]]
    assert AnomalyType.FIRMWARE_INTEGRITY.value in types
    anomaly = next(
        a for a in result["anomalies"] if a["type"] == AnomalyType.FIRMWARE_INTEGRITY.value
    )
    assert anomaly["severity_hint"] == "high"


def test_matching_firmware_not_flagged():
    result = classify(make_record())
    types = [a["type"] for a in result["anomalies"]]
    assert AnomalyType.FIRMWARE_INTEGRITY.value not in types


# ---------------------------------------------------------------------------
# Rule 3: SoC vs voltage consistency
# ---------------------------------------------------------------------------


def test_spoofed_soc_flagged_high():
    # Cells at ~3.30 V imply roughly 10% charge, but the BMS reports 95%.
    result = classify(
        make_record(
            pack_avg_voltage=3.30,
            cell_voltages=[3.30, 3.31, 3.29, 3.30],
            state_of_charge_reported_pct=95.0,
        )
    )
    types = [a["type"] for a in result["anomalies"]]
    assert AnomalyType.SOC_INCONSISTENCY.value in types
    anomaly = next(
        a for a in result["anomalies"] if a["type"] == AnomalyType.SOC_INCONSISTENCY.value
    )
    assert anomaly["severity_hint"] == "high"


def test_consistent_soc_not_flagged():
    result = classify(make_record())
    types = [a["type"] for a in result["anomalies"]]
    assert AnomalyType.SOC_INCONSISTENCY.value not in types


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_classify_result_shape():
    result = classify(make_record())
    assert set(result.keys()) == {"vehicle_id", "timestamp", "anomalies", "is_anomalous"}
    assert result["vehicle_id"] == "TEST-001"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
