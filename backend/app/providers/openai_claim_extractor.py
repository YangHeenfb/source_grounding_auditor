from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ..citation_parser import extract_source_mentions
from ..claim_extractor import QUANT_RE, _normalize_clause
from ..schemas import Claim, ClaimType, ImportanceLabel
from .llm_provider import LLMProviderConfigurationError, LLMProviderError

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


SYSTEM_PROMPT = """You extract auditable atomic claims for a source-grounding auditor.

Return only JSON matching the schema. Do not fact-check and do not decide whether a
claim is true. Split compound sentences into minimal claims that can each be checked
against a source. Remove citation markers and raw URLs from normalized_claim.

Classification rules:
- factual: concrete fact claims that can be supported or contradicted.
- attribution: claims about who said, reported, found, wrote, or argued something.
- judgment: predictions, recommendations, causal interpretations, value judgments,
  or opinion-like claims.
- non_claim: headings, transitions, summaries without an auditable assertion.

Set has_quantitative_data to true when the claim contains a number, proportion,
amount, rank, year, date, or numeric comparison. Include source_mentions for vague
or explicit source phrases, such as "experts say", "sources say", "a study", or
"according to the report".
"""


CLAIMS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "original_text_span": {"type": "string"},
                    "normalized_claim": {"type": "string"},
                    "claim_type": {
                        "type": "string",
                        "enum": [claim_type.value for claim_type in ClaimType],
                    },
                    "has_quantitative_data": {"type": "boolean"},
                    "source_mentions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "importance_label": {
                        "type": "string",
                        "enum": [label.value for label in ImportanceLabel],
                    },
                },
                "required": [
                    "original_text_span",
                    "normalized_claim",
                    "claim_type",
                    "has_quantitative_data",
                    "source_mentions",
                    "importance_label",
                ],
            },
        }
    },
    "required": ["claims"],
}


class OpenAIClaimExtractor:
    """OpenAI-backed claim extractor.

    This intentionally uses the existing httpx dependency instead of adding the
    OpenAI SDK. The rest of the analyzer consumes the same Claim schema as other providers.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 45.0,
    ):
        self.api_key = api_key
        self.model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self._api_key())

    def extract_claims(self, input_text: str, original_question: str | None = None) -> list[Claim]:
        api_key = self._api_key()
        if not api_key:
            raise LLMProviderConfigurationError(
                "OPENAI_API_KEY is not set. A Codex or ChatGPT subscription is not an API key; "
                "set OPENAI_API_KEY to test OpenAI-backed extraction."
            )

        payload = self._request_payload(input_text, original_question)
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"OpenAI API request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMProviderError(f"OpenAI API request failed ({response.status_code}): {_safe_error_body(response)}")

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            model_payload = _parse_json_content(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMProviderError("OpenAI API response did not contain valid claim JSON.") from exc

        return claims_from_model_payload(model_payload)

    def _api_key(self) -> str | None:
        return self.api_key or os.environ.get("OPENAI_API_KEY")

    def _request_payload(self, input_text: str, original_question: str | None) -> dict[str, Any]:
        user_text = input_text
        if original_question:
            user_text = f"Original question:\n{original_question}\n\nText to analyze:\n{input_text}"

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "source_grounding_claim_extraction",
                    "strict": True,
                    "schema": CLAIMS_JSON_SCHEMA,
                },
            },
        }


def claims_from_model_payload(payload: dict[str, Any]) -> list[Claim]:
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        raise LLMProviderError("OpenAI claim payload is missing a claims list.")

    claims: list[Claim] = []
    for raw in raw_claims:
        if not isinstance(raw, dict):
            continue

        normalized = _normalize_clause(str(raw.get("normalized_claim") or raw.get("original_text_span") or ""))
        if not normalized:
            continue

        source_mentions = _clean_string_list(raw.get("source_mentions"))
        for mention in extract_source_mentions(normalized):
            if mention not in source_mentions:
                source_mentions.append(mention)

        claims.append(
            Claim(
                claim_id=f"c{len(claims)+1:03d}",
                original_text_span=str(raw.get("original_text_span") or normalized).strip(),
                normalized_claim=normalized,
                claim_type=_coerce_enum(ClaimType, raw.get("claim_type"), ClaimType.FACTUAL),
                has_quantitative_data=bool(raw.get("has_quantitative_data")) or bool(QUANT_RE.search(normalized)),
                source_mentions=source_mentions,
                importance_label=_coerce_enum(
                    ImportanceLabel,
                    raw.get("importance_label"),
                    ImportanceLabel.SUPPORTING,
                ),
            )
        )

    return claims


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("Expected object", content, 0)
        return parsed

    if isinstance(content, list):
        text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("Expected object", text, 0)
        return parsed

    raise json.JSONDecodeError("Unsupported content type", str(content), 0)


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_enum(enum_type: Any, value: Any, default: Any) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        return default


def _safe_error_body(response: httpx.Response) -> str:
    text = response.text.strip().replace("\n", " ")
    if len(text) > 500:
        return f"{text[:500]}..."
    return text
