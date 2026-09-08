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
telemetry JSON -> rule_engine.classify()  ->  explain.explain()   ->  triage.triage()
                  (3 named anomaly rules)     (retrieve advisory +     (severity ->
                                               plain-language summary)  routing)
```

- **Rule engine** (`src/rule_engine.py`): three testable rules — cell-voltage
  deviation, firmware-hash integrity, and SoC/voltage consistency — with
  named, adjustable thresholds.
- **Explanation** (`src/explain.py`): keyword retrieval over
  `docs/knowledge_base.md`, then either an LLM-written paragraph (if an API
  key is configured) or a templated paragraph in `--mock` mode.
- **Triage** (`src/triage.py`): max severity across anomalies decides the
  route — `dashboard_alert` for low/medium, `voice_call_notification` for
  high (label only; no real call is placed).

## Install

```
pip install -r requirements.txt
```

(Python 3.11+; the pipeline itself uses only the standard library — pytest is
needed solely for the test suite.)

## Run

```
python src/run_demo.py          # uses an LLM if SENTINELBMS_API_KEY/OPENAI_API_KEY is set
python src/run_demo.py --mock   # fully offline, templated explanations
```

Console output shows a per-vehicle summary; full structured results are
written to `output/demo_results.json`.

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
├── data/synthetic_telemetry.json   # 6 sample records (3 normal, 3 anomalous)
├── src/rule_engine.py              # anomaly classification rules
├── src/explain.py                  # retrieval + explanation (LLM or mock)
├── src/triage.py                   # severity + routing decision
├── src/run_demo.py                 # CLI entry point
├── docs/knowledge_base.md          # reference safety advisories
└── tests/test_rule_engine.py       # pytest suite
```
