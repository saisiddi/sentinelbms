"""SentinelBMS triage layer (Step 3 of the pipeline).

Decides how urgent a flagged record is and where the alert should go:
  - low / medium severity  -> "dashboard_alert" (a fleet operator reviews it)
  - high severity          -> "voice_call_notification" (the driver is called
                              directly; for this demo we only emit the routing
                              label — no real call is placed).
"""

from __future__ import annotations

# Numeric ranks so "max severity" is a simple comparison, independent of the
# order in which anomaly strings appear.
_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}

ROUTE_DASHBOARD = "dashboard_alert"
ROUTE_VOICE_CALL = "voice_call_notification"


def triage(classification_result: dict) -> dict:
    """Compute overall severity and routing for one classified record."""
    hints = [a.get("severity_hint", "low") for a in classification_result.get("anomalies", [])]
    overall = max(hints, key=lambda h: _SEVERITY_RANK.get(h, 0), default="low")

    route = ROUTE_VOICE_CALL if overall == "high" else ROUTE_DASHBOARD

    return {
        "vehicle_id": classification_result.get("vehicle_id", "UNKNOWN"),
        "overall_severity": overall,
        "route": route,
        "is_anomalous": classification_result.get("is_anomalous", False),
    }
