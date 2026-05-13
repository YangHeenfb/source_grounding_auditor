from __future__ import annotations

from typing import Protocol

from ..schemas import Claim


class LLMProviderError(RuntimeError):
    """Raised when a production LLM provider cannot return usable claims."""


class LLMProviderConfigurationError(LLMProviderError):
    """Raised when a requested LLM provider is not configured."""


class LLMProviderTimeoutError(LLMProviderError):
    """Raised when an LLM provider takes too long to return a response."""


class LLMProvider(Protocol):
    """Extension point for production LLM extraction/checking.

    Providers should return the same
    Pydantic schemas and validate model output before the analyzer consumes it.
    """

    def extract_claims(self, input_text: str, original_question: str | None = None) -> list[Claim]:
        ...
