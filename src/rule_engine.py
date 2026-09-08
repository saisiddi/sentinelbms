"""SentinelBMS rule engine (Step 1 of the pipeline).

This module checks one BMS telemetry record (a plain dict, as loaded from
data/synthetic_telemetry.json) against a small set of named, testable rules.

Background for non-EV readers:
  - An EV battery "pack" is made of many individual "cells". Healthy cells
    hold almost the same voltage; one cell drifting far from the pack
    average can mean a failing cell or a spoofed sensor.
  - "SoC" (state of charge) is the battery percentage shown to the driver.
    Cell voltages roughly map to charge level, so a reported SoC that
    disagrees with the measured voltages is suspicious.
  - The BMS runs firmware. Every legitimate release has a known
    cryptographic hash; a mismatch means the software may have been
    tampered with.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Adjustable detection thresholds (named constants, not magic numbers).
# ---------------------------------------------------------------------------

# A cell more than 15% away from the pack-average voltage is a HIGH concern.
CELL_DEVIATION_THRESHOLD = 0.15
# Between 5% and 15% deviation we still flag it, but as MEDIUM (watch-list).
CELL_DEVIATION_WARN_THRESHOLD = 0.05

# A Li-ion cell is near-empty around 3.20 V and near-full around 4.15 V.
# We use this linear approximation to estimate SoC from voltage.
CELL_V_EMPTY = 3.20
CELL_V_FULL = 4.15

# If the reported SoC differs from the voltage-implied SoC by more than
# 25 percentage points, something is wrong (spoofed report or bad sensor).
SOC_GAP_THRESHOLD_PCT = 25.0
# A smaller gap (10-25 points) is worth a MEDIUM watch-list flag.
SOC_GAP_WARN_THRESHOLD_PCT = 10.0


class AnomalyType(str, Enum):
    """Canonical identifiers for each thing the engine can detect.

    Using an enum (instead of bare strings everywhere) keeps the rule set
    easy to extend: add a member here, write a check function, register it
    in RULE_CHECKS below.
    """

    CELL_VOLTAGE_DEVIATION = "cell_voltage_deviation"
    FIRMWARE_INTEGRITY = "firmware_integrity"
    SOC_INCONSISTENCY = "soc_inconsistency"


class Severity(str, Enum):
    """Severity hints produced by rules; triage.py maps these to routing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Individual rule functions. Each takes the raw record dict and returns an
# anomaly dict ({"type", "severity_hint", "detail"}) or None when the record
# looks healthy for that rule.
# ---------------------------------------------------------------------------


def check_cell_voltage_deviation(record: dict) -> Optional[dict]:
    """Flag any single cell sitting far from the pack-average voltage."""
    cells = record.get("cell_voltages", [])
    pack_avg = record.get("pack_avg_voltage", 0.0)
    if not cells or pack_avg <= 0:
        return None

    worst_idx, worst_dev = -1, 0.0
    for idx, voltage in enumerate(cells):
        deviation = abs(voltage - pack_avg) / pack_avg
        if deviation > worst_dev:
            worst_idx, worst_dev = idx, deviation

    if worst_dev > CELL_DEVIATION_THRESHOLD:
        severity = Severity.HIGH
    elif worst_dev > CELL_DEVIATION_WARN_THRESHOLD:
        severity = Severity.MEDIUM
    else:
        return None

    return {
        "type": AnomalyType.CELL_VOLTAGE_DEVIATION.value,
        "severity_hint": severity.value,
        "detail": (
            f"Cell {worst_idx} reads {cells[worst_idx]:.2f} V, which is "
            f"{worst_dev * 100:.1f}% away from the pack average of "
            f"{pack_avg:.2f} V (threshold: {CELL_DEVIATION_THRESHOLD * 100:.0f}%)."
        ),
    }


def check_firmware_integrity(record: dict) -> Optional[dict]:
    """Flag firmware whose hash does not match the known-good release."""
    observed = record.get("firmware_hash", "")
    expected = record.get("known_good_firmware_hash", "")
    if not observed or observed == expected:
        return None

    return {
        "type": AnomalyType.FIRMWARE_INTEGRITY.value,
        "severity_hint": Severity.HIGH.value,
        "detail": (
            f"Observed firmware hash {observed[:12]}... does not match the "
            f"known-good hash {expected[:12]}... for this vehicle model."
        ),
    }


def compute_soc_gap(record: dict) -> tuple[float, float]:
    """Return (implied_soc_pct, gap_from_reported) for a record.

    Reusable by both the rule engine and the ML feature builder so the
    voltage-to-SoC mapping is defined in exactly one place.
    """
    cells = record.get("cell_voltages", [])
    reported = record.get("state_of_charge_reported_pct")
    if not cells:
        return 0.0, 0.0
    avg_cell_v = sum(cells) / len(cells)
    implied = max(0.0, min(100.0, (avg_cell_v - CELL_V_EMPTY) / (CELL_V_FULL - CELL_V_EMPTY) * 100.0))
    gap = abs((reported or 0.0) - implied) if reported is not None else 0.0
    return implied, gap


def check_soc_consistency(record: dict) -> Optional[dict]:
    """Flag a reported SoC that disagrees with the measured cell voltages."""
    cells = record.get("cell_voltages", [])
    reported = record.get("state_of_charge_reported_pct")
    if not cells or reported is None:
        return None
    implied, gap = compute_soc_gap(record)
    avg_cell_v = sum(cells) / len(cells)
    if gap <= SOC_GAP_WARN_THRESHOLD_PCT:
        return None
    severity = Severity.HIGH if gap > SOC_GAP_THRESHOLD_PCT else Severity.MEDIUM
    return {
        "type": AnomalyType.SOC_INCONSISTENCY.value,
        "severity_hint": severity.value,
        "detail": (
            f"The BMS reports {reported:.0f}% charge, but the average cell "
            f"voltage of {avg_cell_v:.2f} V implies roughly {implied:.0f}% "
            f"(a gap of {gap:.0f} percentage points)."
        ),
    }


# Registry mapping rule names to their check functions. Adding a new rule is
# a one-line change here plus a new function above.
RULE_CHECKS = {
    "check_cell_voltage_deviation": check_cell_voltage_deviation,
    "check_firmware_integrity": check_firmware_integrity,
    "check_soc_consistency": check_soc_consistency,
}


def classify(record: dict) -> dict:
    """Run every registered rule against one telemetry record.

    Returns a structured result describing which anomalies (if any) fired.
    """
    anomalies = []
    for check in RULE_CHECKS.values():
        finding = check(record)
        if finding is not None:
            anomalies.append(finding)

    return {
        "vehicle_id": record.get("vehicle_id", "UNKNOWN"),
        "timestamp": record.get("timestamp", ""),
        "anomalies": anomalies,
        "is_anomalous": len(anomalies) > 0,
    }
