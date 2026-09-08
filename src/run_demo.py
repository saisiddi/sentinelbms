"""SentinelBMS demo entry point (updated with ML signal).

Pipeline: telemetry JSON -> rule_engine.classify() -> ml_detector.ml_anomaly_score()
          -> triage.triage() -> explain.explain() (anomalous only)

Usage:
    python src/run_demo.py           # LLM if API key set, else mock
    python src/run_demo.py --mock    # forced templated explanations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_telemetry.json"
KB_PATH = PROJECT_ROOT / "docs" / "knowledge_base.md"
OUTPUT_PATH = PROJECT_ROOT / "output" / "demo_results.json"

from explain import OpenAICompatibleClient, explain  # noqa: E402
from ml_detector import ml_anomaly_score  # noqa: E402
from rule_engine import classify  # noqa: E402
from triage import triage  # noqa: E402


def run_pipeline(mock: bool) -> list[dict]:
    """Run every telemetry record through classify -> ML -> triage -> explain."""
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    client = OpenAICompatibleClient()
    results = []
    for record in records:
        classification = classify(record)
        ml = ml_anomaly_score(record)
        merged = {**classification, "ml_signal": ml}
        explanation = (
            explain(merged, str(KB_PATH), mock=mock, llm_client=client)
            if merged["is_anomalous"] or ml.get("is_outlier")
            else None
        )
        routing = triage(merged)
        results.append(
            {
                "vehicle_id": record["vehicle_id"],
                "timestamp": record["timestamp"],
                "classification": classification,
                "ml_signal": ml,
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
        ml = item["ml_signal"]
        print(f"\nVehicle: {item['vehicle_id']}   ({item['timestamp']})")
        print(f"  ML signal: score={ml['anomaly_score']}, outlier={ml['is_outlier']}")

        if not classification["is_anomalous"] and not ml["is_outlier"]:
            print("  Status: OK - no anomalies detected; logged for the weekly summary.")
            continue

        if routing["overall_severity"] == "watch":
            print("  Status: ML-FLAGGED - no rule violated, but pattern is statistically unusual")
        else:
            print(f"  Status: ANOMALOUS - overall severity: {routing['overall_severity'].upper()}")
            for anomaly in classification["anomalies"]:
                print(f"    - [{anomaly['severity_hint']}] {anomaly['type']}: {anomaly['detail']}")

        if item["explanation"]:
            for expl in item["explanation"].get("explanations", []):
                print(f"  Explanation ({expl['anomaly_type']}, source: {expl['source_advisory']}):")
                print(f"    {expl['plain_language_summary']}")

        print(f"  Routing: {routing['route']}")

    flagged = sum(1 for i in results if i["triage"]["is_anomalous"])
    print("\n" + "-" * 78)
    print(f"Processed {len(results)} records: {flagged} flagged ({flagged - sum(1 for i in results if i['classification']['is_anomalous'])} ML-only), {len(results) - flagged} clean.")


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
