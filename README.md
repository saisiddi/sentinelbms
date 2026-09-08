# SentinelBMS

An AI-powered early-warning advisor for EV Battery Management System (BMS)
cybersecurity. SentinelBMS ingests battery telemetry, flags anomalous or
potentially malicious behavior with a transparent rule engine, retrieves
relevant safety advisories, and produces plain-language explanations plus a
severity-based routing decision — so fleet operators and EV owners can trust
the batteries powering clean-energy transport.

Built as a prototype for the 1M1B AI for Sustainability Virtual Internship
(IBM SkillsBuild x AICTE). Aligned with SDG 7 (Affordable and Clean Energy).

## How it works

```
telemetry JSON -> rule_engine.classify()  ->  ml_detector.ml_anomaly_score()
                  (3 named anomaly rules)    (IsolationForest, adaptive)
                                \\                    /
                                 v                  v
                           triage.triage() (combines both signals)
                                     |
                                     v
                            explain.explain() (LLM or mock)
```

- **Rule engine** (`src/rule_engine.py`): three testable rules — cell-voltage
  deviation, firmware-hash integrity, and SoC/voltage consistency — with
  named, adjustable thresholds. This stays the primary, auditable detector.
- **ML detector** (`src/ml_detector.py`): an IsolationForest trained on a
  synthetic healthy-fleet baseline adds adaptive pattern detection for
  anomalies the fixed rules don't anticipate.
- **Explanation** (`src/explain.py`): keyword retrieval over
  `docs/knowledge_base.md`, then either an LLM-written paragraph (if an API
  key is configured) or a templated paragraph in `--mock` mode.
- **Triage** (`src/triage.py`): max severity across rule anomalies decides the
  route; a new `watch` tier covers records the rules clear but the ML model
  flags — routed to `dashboard_alert` so nothing slips through silently.

## Why hybrid, not pure ML

A learned model alone would be a black box in a safety-critical domain: a
regulator, a technician, or an owner deserves to know *why* an alert fired,
and the rules give exactly that — each flag names the signal, the threshold,
and the deviation. The ML layer's job is complementary, not competitive: it
catches subtle multi-signal patterns (e.g., a mildly low pack voltage together
with a slightly elevated temperature) that no single fixed rule anticipates,
while the rules continue to cover the high-confidence, must-be-auditable cases.
The LLM layer then turns *both* signals into a plain-language explanation,
so the final advisory is always readable. This is a deliberate architecture
choice, not an afterthought.

## Install

```
pip install -r requirements.txt
```

Python 3.11+; the pipeline uses only the standard library plus
`scikit-learn` and `joblib` (the ML detector), pytest for the test suite.

## Run

```
python src/run_demo.py          # uses an LLM if SENTINELBMS_API_KEY/OPENAI_API_KEY is set
python src/run_demo.py --mock   # fully offline, templated explanations
```

Console output shows a per-vehicle summary including the ML score; full
structured results are written to `output/demo_results.json`.

### Optional LLM configuration

| Env var | Purpose | Default |
|---|---|---|
| `SENTINELBMS_API_KEY` or `OPENAI_API_KEY` | API key (never hardcoded) | none -> mock mode |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | model name | `gpt-4o-mini` |

## Test

```
pytest tests/
```

## Project layout

```
sentinelbms/
├── data/synthetic_telemetry.json   # 7 sample records (3 normal, 3 anomalous + 1 ML-only case)
├── models/.gitignore               # IsolationForest cache (generated, not committed)
├── scripts/train_baseline.py       # explicit baseline training + serialization
├── src/rule_engine.py              # anomaly classification rules
├── src/ml_detector.py              # IsolationForest anomaly scorer
├── src/explain.py                  # retrieval + explanation (LLM or mock)
├── src/triage.py                   # severity + routing decision (includes watch tier)
├── src/run_demo.py                 # CLI entry point
├── docs/knowledge_base.md          # reference safety advisories
├── tests/test_rule_engine.py       # pytest suite for rules
└── tests/test_ml_detector.py       # pytest suite for ML detector
```
