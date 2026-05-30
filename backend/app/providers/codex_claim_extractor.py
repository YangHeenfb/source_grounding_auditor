from __future__ import annotations

import asyncio
import subprocess

from ..citation_parser import parse_citations
from ..schemas import Claim
from .llm_provider import CodexCLILLMProvider, LLMProviderError, LLMProviderTimeoutError
from .openai_claim_extractor import CLAIMS_JSON_SCHEMA, SYSTEM_PROMPT, claims_from_model_payload

DEFAULT_CODEX_MODEL = "gpt-5.3-codex-spark"
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_SERVICE_TIER = "fast"
DEFAULT_REASONING_EFFORT = "low"


class CodexCLIClaimExtractor(CodexCLILLMProvider):
    """Backward-compatible sync extraction wrapper around the unified Codex provider."""

    def extract_claims(self, input_text: str, original_question: str | None = None) -> list[Claim]:
        citations = parse_citations(input_text)
        context = {"original_question": original_question}
        return _run_async(super().extract_claims(input_text, citations, context))


def _run_async(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("CodexCLIClaimExtractor cannot run inside an already running event loop.") from None
