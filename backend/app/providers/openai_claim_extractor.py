from __future__ import annotations

import asyncio
import re
from typing import Any

from ..citation_parser import extract_source_mentions, parse_citations
from ..schemas import Claim
from .llm_provider import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    ClaimExtractionResponse,
    OpenAILLMProvider,
    claims_from_model_payload as _claims_from_model_payload,
    structured_output_schema,
)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
SYSTEM_PROMPT = CLAIM_EXTRACTION_SYSTEM_PROMPT
CLAIMS_JSON_SCHEMA: dict[str, Any] = structured_output_schema(ClaimExtractionResponse)


class OpenAIClaimExtractor(OpenAILLMProvider):
    """Backward-compatible sync extraction wrapper around the unified provider."""

    def extract_claims(self, input_text: str, original_question: str | None = None) -> list[Claim]:
        citations = parse_citations(input_text)
        context = {"original_question": original_question}
        return _run_async(super().extract_claims(input_text, citations, context))


def claims_from_model_payload(payload: dict[str, Any]) -> list[Claim]:
    claims = _claims_from_model_payload(payload)
    for claim in claims:
        if _contains_number(claim.normalized_claim):
            claim.has_quantitative_data = True
        for mention in extract_source_mentions(claim.normalized_claim):
            if mention not in claim.source_mentions:
                claim.source_mentions.append(mention)
    return claims


def _contains_number(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def _run_async(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("OpenAIClaimExtractor cannot run inside an already running event loop.") from None
