"""SentinelBMS triage layer (Step 3 of the pipeline).

Decides how urgent a flagged record is and where the alert should go:
  - high severity           -> "voice_call_notification" (driver called)
  - low / medium / watch    -> "dashboard_alert" (fleet operator reviews)
  - clean (no rules, no ML) -> "dashboard_alert" as well, but
                               print_summary renders it as "OK".

The new "watch" tier exists for records where the rule engine finds
nothing, but the ML model flags a statistically unusual pattern —
exactly the case the ML layer was added to catch.
"""

from __future__ import annotations

_SEVERITY_RANK = {"ok": 0, "low": 1, "medium": 2, "high": 3, "watch": 2}

ROUTE_DASHBOARD = "dashboard_alert"
ROUTE_VOICE_CALL = "voice_call_notification"


def triage(classification_result: dict) -> dict:
    """Compute overall severity and routing for one classified record,
    combining the rule-based verdict with the ML signal."""
    hints = [a.get("severity_hint", "low") for a in classification_result.get("anomalies", [])]
    rules_anomalous = classification_result.get("is_anomalous", False)
    ml_outlier = classification_result.get("ml_signal", {}).get("is_outlier", False)

    if rules_anomalous:
        overall = max(hints, key=lambda h: _SEVERITY_RANK.get(h, 0), default="low")
    elif ml_outlier:
        overall = "watch"
    else:
        overall = "ok"

    route = ROUTE_VOICE_CALL if overall == "high" else ROUTE_DASHBOARD

    return {
        "vehicle_id": classification_result.get("vehicle_id", "UNKNOWN"),
        "overall_severity": overall,
        "route": route,
        "is_anomalous": rules_anomalous or ml_outlier,
        "ml_outlier": ml_outlier,
    }