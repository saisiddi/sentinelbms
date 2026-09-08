"""Explicit baseline-training script.

Usage:  python scripts/train_baseline.py

Generates a synthetic healthy fleet, fits the IsolationForest,
and serializes it to models/isolation_forest.joblib.

The synthetic data is a stand-in for real fleet telemetry —
in production, replace _generate_baseline() with historical
data loaded from your telemetry store.
"""

from pathlib import Path

import joblib

from ml_detector import _generate_baseline, MODEL_PATH, ISO_FOREST_CONTAMINATION, ISO_FOREST_RANDOM_STATE  # noqa: E402


def main() -> None:
    X = _generate_baseline()
    model = joblib.load(str(MODEL_PATH)) if MODEL_PATH.exists() else None  # type: ignore[no-untyped-call]
    if model is None:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(
            contamination=ISO_FOREST_CONTAMINATION,
            random_state=ISO_FOREST_RANDOM_STATE,
        )
        model.fit(X)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)  # type: ignore[no-untyped-call]
    print(f"Baseline trained on {X.shape[0]} records -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
