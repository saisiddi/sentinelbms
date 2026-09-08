"""SentinelBMS explanation layer (Step 2 of the pipeline).

For each anomaly flagged by the rule engine, this module:
  1. RETRIEVES the relevant safety advisory from docs/knowledge_base.md
     (simple keyword matching — no vector DB needed at demo scale).
  2. GENERATES a plain-language explanation for a vehicle owner or
     technician, combining the specific reading that triggered the flag
     with the retrieved advisory context.

Generation can use an LLM when an API key is configured via environment
variables; otherwise it falls back to a deterministic template ("mock"
mode) so the demo runs fully offline.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Retrieval: split the markdown knowledge base into sections and match on
# the anomaly-type keyword each section declares.
# ---------------------------------------------------------------------------


def load_sections(knowledge_base_path: str) -> list[dict]:
    """Parse the knowledge base into {"title", "body"} sections."""
    with open(knowledge_base_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    sections = []
    current_title, current_lines = "", []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_title:
                sections.append(
                    {"title": current_title, "body": "\n".join(current_lines).strip()}
                )
            current_title = line.lstrip("# ").strip()
            current_lines = []
        elif current_title:
            current_lines.append(line)
    if current_title:
        sections.append(
            {"title": current_title, "body": "\n".join(current_lines).strip()}
        )
    return sections


def retrieve_advisory(anomaly_type: str, sections: list[dict]) -> Optional[dict]:
    """Return the section whose text mentions this anomaly type.

    We match both the raw enum value ("cell_voltage_deviation") and its
    humanized form ("cell voltage deviation") so either writing style in
    the knowledge base will hit.
    """
    needle = anomaly_type.lower()
    humanized = needle.replace("_", " ")
    for section in sections:
        haystack = f"{section['title']}\n{section['body']}".lower()
        if needle in haystack or humanized in haystack:
            return section
    return None


# ---------------------------------------------------------------------------
# Optional LLM client (OpenAI-compatible chat-completions API over stdlib
# urllib, so no third-party SDK is required). The API key is read from the
# environment ONLY — never hardcoded.
# ---------------------------------------------------------------------------


class OpenAICompatibleClient:
    """Minimal client for any OpenAI-compatible /chat/completions endpoint.

    Configuration (all optional except the key):
      SENTINELBMS_API_KEY or OPENAI_API_KEY  -> API key
      OPENAI_BASE_URL                        -> default https://api.openai.com/v1
      OPENAI_MODEL                           -> default gpt-4o-mini
    """

    def __init__(self) -> None:
        self.api_key = os.environ.get("SENTINELBMS_API_KEY") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You write short, calm, plain-language battery-safety "
                            "advisories for EV owners and technicians. No jargon, "
                            "no headings, one paragraph."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------


def _build_prompt(vehicle_id: str, anomaly: dict, advisory_text: str) -> str:
    """Prompt handed to the LLM when one is configured."""
    return (
        f"Vehicle {vehicle_id} was flagged by an automated safety rule.\n"
        f"What the rule saw: {anomaly['detail']}\n"
        f"Reference advisory from the knowledge base:\n{advisory_text}\n\n"
        "Write one short paragraph for a non-technical vehicle owner: what "
        "was detected, why it matters, and what they should do next."
    )


def _mock_explanation(vehicle_id: str, anomaly: dict, advisory_text: str) -> str:
    """Deterministic template used in --mock mode (no API key needed).

    Still produces a readable paragraph rather than a debug dump: it quotes
    the triggering reading and pulls the recommended action out of the
    retrieved advisory.
    """
    action = ""
    marker = "Recommended action:"
    if marker in advisory_text:
        # Collapse newlines/extra spaces so the template reads as one paragraph.
        action = " ".join(advisory_text.split(marker, 1)[1].split()).rstrip(".")
    template = (
        f"Vehicle {vehicle_id} triggered a {anomaly['severity_hint']}-severity "
        f"alert of type '{anomaly['type']}'. {anomaly['detail']} "
        "This pattern is covered by the manufacturer safety advisory retrieved "
        "from the knowledge base."
    )
    if action:
        template += f" Recommended next step: {action[0].lower() + action[1:]}."
    return template


def explain(
    classification_result: dict,
    knowledge_base_path: str,
    mock: bool = False,
    llm_client: Optional[OpenAICompatibleClient] = None,
) -> dict:
    """Attach a plain-language explanation to each flagged anomaly."""
    client = llm_client if llm_client is not None else OpenAICompatibleClient()
    use_llm = client.available and not mock
    sections = load_sections(knowledge_base_path)

    explanations = []
    for anomaly in classification_result.get("anomalies", []):
        section = retrieve_advisory(anomaly["type"], sections)
        advisory_text = section["body"] if section else "No matching advisory found."
        source = section["title"] if section else "none"

        if use_llm:
            summary = client.generate(
                _build_prompt(classification_result["vehicle_id"], anomaly, advisory_text)
            )
        else:
            summary = _mock_explanation(
                classification_result["vehicle_id"], anomaly, advisory_text
            )

        explanations.append(
            {
                "anomaly_type": anomaly["type"],
                "plain_language_summary": summary,
                "source_advisory": source,
            }
        )

    return {
        "vehicle_id": classification_result.get("vehicle_id", "UNKNOWN"),
        "explanations": explanations,
        "generation_mode": "llm" if use_llm else "mock",
    }
