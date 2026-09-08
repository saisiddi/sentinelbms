"""SentinelBMS demo entry point.

Loads synthetic BMS telemetry, then runs every record through the full
pipeline:  classify  ->  explain (only for anomalous records)  ->  triage.

Usage:
    python src/run_demo.py           # uses an LLM if an API key is set,
                                     # otherwise falls back to mock mode
    python src/run_demo.py --mock    # force template-based explanations
                                     # (fully offline, no API key needed)

Console output is written for humans; the full structured result is saved
to output/demo_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve all paths relative to the project root so the demo works no
# matter which directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_telemetry.json"
KB_PATH = PROJECT_ROOT / "docs" / "knowledge_base.md"
OUTPUT_PATH = PROJECT_ROOT / "output" / "demo_results.json"

from explain import OpenAICompatibleClient, explain  # noqa: E402
from rule_engine import classify  # noqa: E402
from triage import triage  # noqa: E402


def run_pipeline(mock: bool) -> list[dict]:
    """Run every telemetry record through classify -> explain -> triage."""
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    client = OpenAICompatibleClient()
    results = []
    for record in records:
        classification = classify(record)
        explanation = (
            explain(classification, str(KB_PATH), mock=mock, llm_client=client)
            if classification["is_anomalous"]
            else None
        )
        routing = triage(classification)
        results.append(
            {
                "vehicle_id": record["vehicle_id"],
                "timestamp": record["timestamp"],
                "classification": classification,
                "explanation": explanation,
                "triage": routing,
            }
        )
    return results


def print_summary(results: list[dict], mock: bool) -> None:
    """Readable per-record console summary."""
    client_note = "mock (templated)" if mock else "auto (LLM if configured)"
    print("=" * 78)
    print(f"SentinelBMS demo - explanation mode: {client_note}")
    print("=" * 78)

    for item in results:
        classification = item["classification"]
        routing = item["triage"]
        print(f"\nVehicle: {item['vehicle_id']}   ({item['timestamp']})")

        if not classification["is_anomalous"]:
            print("  Status: OK - no anomalies detected; logged for the weekly summary.")
            continue

        print(f"  Status: ANOMALOUS - overall severity: {routing['overall_severity'].upper()}")
        for anomaly in classification["anomalies"]:
            print(f"    - [{anomaly['severity_hint']}] {anomaly['type']}: {anomaly['detail']}")

        for expl in (item["explanation"] or {}).get("explanations", []):
            print(f"  Explanation ({expl['anomaly_type']}, source: {expl['source_advisory']}):")
            print(f"    {expl['plain_language_summary']}")

        print(f"  Routing: {routing['route']}")

    flagged = sum(1 for i in results if i["classification"]["is_anomalous"])
    print("\n" + "-" * 78)
    print(f"Processed {len(results)} records: {flagged} anomalous, {len(results) - flagged} clean.")


def write_output(results: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Full structured output written to: {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SentinelBMS demo pipeline")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use templated explanations instead of calling an LLM",
    )
    args = parser.parse_args()

    results = run_pipeline(mock=args.mock)
    print_summary(results, mock=args.mock)
    write_output(results)


if __name__ == "__main__":
    main()
