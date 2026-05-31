from __future__ import annotations

import asyncio
import json
import os
import signal
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..schemas import (
    Claim,
    ClaimReviewCategory,
    ClaimType,
    DiscourseRole,
    FinalGroundingBucket,
    ImportanceLabel,
    ParsedCitation,
    RiskFlag,
    SourceOpacity,
    SourceRole,
    SupportRelation,
)


class LLMProviderError(RuntimeError):
    """Raised when a production LLM provider cannot return usable structured output."""


class LLMProviderConfigurationError(LLMProviderError):
    """Raised when a requested LLM provider is not configured."""


class LLMProviderTimeoutError(LLMProviderError):
    """Raised when an LLM provider takes too long to return a response."""


class AnalysisCancelledError(LLMProviderError):
    """Raised when a user cancels an in-flight analysis job."""


class CancellationToken:
    def __init__(self):
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[str]] = set()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            _terminate_process(process)

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise AnalysisCancelledError("Analysis was cancelled.")

    def register_process(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes.add(process)
            cancelled = self.is_cancelled()
        if cancelled:
            _terminate_process(process)

    def unregister_process(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes.discard(process)


class LLMProvider(Protocol):
    async def extract_claims(
        self,
        input_text: str,
        citations: list[ParsedCitation],
        context: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        ...

    async def check_claim_support(
        self,
        claim: Claim,
        source_bundle: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> Claim:
        ...

    async def check_claim_supports(
        self,
        checks: list[dict[str, Any]],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        ...

    async def classify_review_category(
        self,
        claim: Claim,
        cancellation_token: CancellationToken | None = None,
    ) -> Claim:
        ...

    async def classify_review_categories(
        self,
        claims: list[Claim],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        ...


CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are an evidence chain auditor. Your job is not to decide whether an opinion is correct. Your job is to decompose the input into atomic claims and classify each claim by its role in the argument.

Important rules:

1. Do not treat every sentence as a claim. Extract only atomic auditable claims, attribution claims, judgment claims, and relevant caveats.
2. If a phrase appears as an example of what the author says cannot be concluded, mark it as UNSUPPORTED_EXAMPLE and set not_asserted_by_author to true.
3. If a sentence says no public evidence was found, or says something is unclear, mark it as CAVEAT_OR_LIMITATION. Do not treat it as a high risk asserted fact.
4. If a claim says "Reuters reported that X", the claim being directly made is that Reuters reported X. Do not rewrite it into "X is confirmed true."
5. If the source is named but the underlying evidence is anonymous, use NAMED_SECONDARY_WITH_OPAQUE_UNDERLYING.
6. If the text says "someone said", "some discussions say", or "experts say" without a named publisher or named expert, use VAGUE_SOURCE_MENTION.
7. Distinguish factual claims, attribution claims, and judgment or analysis claims.
8. A school, company, agency, or product webpage stating facts about its own degrees, programs, reports, admissions, courses, staff, data, or announcements is a factual asserted claim when the author uses it as evidence. Do not classify this as ATTRIBUTION_REPORT merely because the sentence says "the official page says/shows".
9. ATTRIBUTION_REPORT is for reporting or attributed statements such as "Reuters reported...", "Bloomberg cited sources...", "an expert said...", or "a report argues...".
10. Dates, years, footnote numbers, and citation numbers are not material quantitative data. Money, percentage, valuation, revenue, market share, sample size, performance metric, and ranking are material quantitative data.
11. Preserve original_span exactly as much as possible.
12. Keep normalized_claim in the same language as the input claim. Do not translate Chinese claims into English or English claims into Chinese.
13. Output only JSON that matches the provided schema."""


SUPPORT_CHECK_SYSTEM_PROMPT = """You are checking whether a cited source supports a specific claim. Do not decide whether the world is actually true or false. Only decide whether the provided source text supports the claim at the level stated.

Rules:

1. A source can be accessible, relevant, and still fail to support the claim.
2. For attribution claims, check whether the source says that the named publisher, person, report, or organization made the statement. Do not convert attribution into factual confirmation.
3. For judgment or analysis claims, mark them as supported only if the source directly makes the same judgment. If the source only provides facts from which the author infers the judgment, use PARTIALLY_SUPPORTS or BACKGROUND_ONLY.
4. If the claim is stronger than the source, use SUPPORTS_WEAKER_CLAIM.
5. If the claim states causation but the source only supports correlation or sequence, mark SOURCE_ONLY_SUPPORTS_WEAKER_CLAIM and, when appropriate, CORRELATION_PRESENTED_AS_CAUSATION.
6. If the source is an opinion or analysis source used as factual proof, mark OPINION_USED_AS_FACT.
7. If the source is not available, use INACCESSIBLE. Do not call the claim high risk only because source text is missing.
8. The source_bundle contains top evidence_snippets selected from the source body. Judge only from those snippets and metadata, not from the full page.
9. If relevant snippets are not available, use INACCESSIBLE or the audit-limited relation supplied by the caller. Do not infer NO_SUPPORT from missing snippets.
10. Output only JSON that matches the provided schema."""


SUPPORT_CHECK_BATCH_SYSTEM_PROMPT = SUPPORT_CHECK_SYSTEM_PROMPT + """

Batch mode:
1. Each input item has a check_id, claim, and source_bundle.
2. Return exactly one check result for every input item.
3. Preserve each check_id exactly.
4. Evaluate each claim only against its paired source_bundle."""


REVIEW_CATEGORY_SYSTEM_PROMPT = """You are deciding how to display a claim in an evidence audit report. Your job is to avoid over flagging.

Risk flags are diagnostic hints. They are not automatically high risk.

Use HIGH_RISK only when all of these are true:
1. The author or model is actually asserting the claim, or making a core judgment.
2. The claim is important enough to affect user understanding.
3. The evidence relationship has a substantive problem, not merely missing source access.

Do not classify these as HIGH_RISK:
1. A Reuters attribution claim that preserves "Reuters reported that..."
2. A caveat such as "no public evidence was found"
3. A limitation such as "the price is unclear"
4. An example of what cannot be concluded
5. A source pointer or citation list
6. A claim where the only issue is that the source body was not available in this run
7. A claim where the source body existed but no relevant evidence snippet was retrieved

Classify attribution claims separately. Attribution support means the source supports that someone said or reported something. It does not mean the underlying statement has primary fact support.

Output only JSON matching the schema."""


class LLMClaimItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: Optional[str] = None
    original_span: str = ""
    normalized_claim: str
    claim_type: ClaimType
    discourse_role: DiscourseRole = DiscourseRole.ASSERTED_CLAIM
    source_opacity: SourceOpacity = SourceOpacity.NOT_APPLICABLE
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    has_quantitative_data: bool = False
    has_material_quantitative_data: bool = False
    importance_label: ImportanceLabel = ImportanceLabel.SUPPORTING
    attributed_to: Optional[str] = None
    source_mentions: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    evidence_needed: list[str] = Field(default_factory=list)
    not_asserted_by_author: bool = False

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_span(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("original_span") and data.get("original_text_span"):
                data["original_span"] = data["original_text_span"]
            data.pop("original_text_span", None)
            if not data.get("original_span"):
                data["original_span"] = data.get("normalized_claim", "")
            if data.get("importance_label") == "background":
                data["importance_label"] = "minor"
        return data


class ClaimExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[LLMClaimItem]


class SupportCheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    support_relation: SupportRelation
    final_bucket: FinalGroundingBucket
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    reasoning_summary: str
    evidence_quote: str = Field(default="", max_length=300)
    source_role: SourceRole = SourceRole.UNKNOWN


class SupportCheckBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    support_relation: SupportRelation
    final_bucket: FinalGroundingBucket
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    reasoning_summary: str
    evidence_quote: str = Field(default="", max_length=300)
    source_role: SourceRole = SourceRole.UNKNOWN


class SupportCheckBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checks: list[SupportCheckBatchItem]


class ReviewCategoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_category: ClaimReviewCategory
    explanation: str


class ReviewCategoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    review_category: ClaimReviewCategory
    explanation: str


class ReviewCategoryBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[ReviewCategoryItem]


T = TypeVar("T", bound=BaseModel)


class OpenAILLMProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 45.0,
    ):
        self.api_key = api_key
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self._api_key())

    async def extract_claims(
        self,
        input_text: str,
        citations: list[ParsedCitation],
        context: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        response = await self._request_structured(
            ClaimExtractionResponse,
            "source_grounding_claim_extraction",
            CLAIM_EXTRACTION_SYSTEM_PROMPT,
            {
                "input_text": input_text,
                "citations": [_dump_model(citation) for citation in citations],
                "context": context,
            },
            cancellation_token=cancellation_token,
        )
        return _claims_from_extraction_response(response)

    async def check_claim_support(
        self,
        claim: Claim,
        source_bundle: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> Claim:
        response = await self._request_structured(
            SupportCheckResponse,
            "source_grounding_support_check",
            SUPPORT_CHECK_SYSTEM_PROMPT,
            {
                "claim": _dump_model(claim),
                "source_bundle": source_bundle,
            },
            cancellation_token=cancellation_token,
        )
        return _merge_support_response(claim, response)

    async def check_claim_supports(
        self,
        checks: list[dict[str, Any]],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        if not checks:
            return []
        response = await self._request_structured(
            SupportCheckBatchResponse,
            "source_grounding_support_check_batch",
            SUPPORT_CHECK_BATCH_SYSTEM_PROMPT,
            {
                "checks": [
                    {
                        "check_id": item["check_id"],
                        "claim": _dump_model(item["claim"]),
                        "source_bundle": item["source_bundle"],
                    }
                    for item in checks
                ]
            },
            cancellation_token=cancellation_token,
        )
        return _merge_support_batch_response(checks, response)

    async def classify_review_category(
        self,
        claim: Claim,
        cancellation_token: CancellationToken | None = None,
    ) -> Claim:
        response = await self._request_structured(
            ReviewCategoryResponse,
            "source_grounding_review_category",
            REVIEW_CATEGORY_SYSTEM_PROMPT,
            {"claim": _dump_model(claim)},
            cancellation_token=cancellation_token,
        )
        updated = claim.model_copy(deep=True)
        updated.review_category = response.review_category
        updated.reasoning_summary = response.explanation
        return updated

    async def classify_review_categories(
        self,
        claims: list[Claim],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        if not claims:
            return []
        response = await self._request_structured(
            ReviewCategoryBatchResponse,
            "source_grounding_review_category_batch",
            REVIEW_CATEGORY_SYSTEM_PROMPT,
            {"claims": [_dump_model(claim) for claim in claims]},
            cancellation_token=cancellation_token,
        )
        return _merge_review_batch_response(claims, response)

    async def _request_structured(
        self,
        response_model: type[T],
        schema_name: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> T:
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        api_key = self._api_key()
        if not api_key:
            raise LLMProviderConfigurationError("OPENAI_API_KEY is not set.")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, default=str),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": structured_output_schema(response_model),
                    },
                },
            }
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
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
                raise LLMProviderError(
                    f"OpenAI API request failed ({response.status_code}): {_safe_error_body(response)}"
                )

            try:
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                raw = response.json()["choices"][0]["message"]["content"]
                parsed = _parse_json_content(raw)
                return response_model.model_validate(parsed)
            except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
                last_error = exc
                if attempt >= 2:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response failed schema validation. "
                            f"Validation error: {exc}. Return corrected JSON only."
                        ),
                    }
                )
        raise LLMProviderError(f"OpenAI response failed schema validation: {last_error}")

    def _api_key(self) -> str | None:
        return self.api_key or os.environ.get("OPENAI_API_KEY")


class CodexCLILLMProvider:
    def __init__(
        self,
        *,
        codex_bin: str | None = None,
        model: str | None = None,
        service_tier: str | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.codex_bin = codex_bin or os.environ.get("CODEX_BIN") or shutil.which("codex")
        self.model = model or os.environ.get("CODEX_MODEL") or "gpt-5.3-codex-spark"
        self.service_tier = service_tier or os.environ.get("CODEX_SERVICE_TIER") or "fast"
        self.reasoning_effort = reasoning_effort or os.environ.get("CODEX_REASONING_EFFORT") or "low"
        self.timeout_seconds = float(os.environ.get("CODEX_TIMEOUT_SECONDS") or timeout_seconds or 90.0)

    def is_configured(self) -> bool:
        if not self.codex_bin:
            return False
        try:
            result = subprocess.run(
                [self.codex_bin, "login", "status"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and "Logged in" in (result.stdout + result.stderr)

    async def extract_claims(
        self,
        input_text: str,
        citations: list[ParsedCitation],
        context: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        response = await self._request_structured(
            ClaimExtractionResponse,
            CLAIM_EXTRACTION_SYSTEM_PROMPT,
            {
                "input_text": input_text,
                "citations": [_dump_model(citation) for citation in citations],
                "context": context,
            },
            cancellation_token=cancellation_token,
        )
        return _claims_from_extraction_response(response)

    async def check_claim_support(
        self,
        claim: Claim,
        source_bundle: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> Claim:
        response = await self._request_structured(
            SupportCheckResponse,
            SUPPORT_CHECK_SYSTEM_PROMPT,
            {
                "claim": _dump_model(claim),
                "source_bundle": source_bundle,
            },
            cancellation_token=cancellation_token,
        )
        return _merge_support_response(claim, response)

    async def check_claim_supports(
        self,
        checks: list[dict[str, Any]],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        if not checks:
            return []
        response = await self._request_structured(
            SupportCheckBatchResponse,
            SUPPORT_CHECK_BATCH_SYSTEM_PROMPT,
            {
                "checks": [
                    {
                        "check_id": item["check_id"],
                        "claim": _dump_model(item["claim"]),
                        "source_bundle": item["source_bundle"],
                    }
                    for item in checks
                ]
            },
            cancellation_token=cancellation_token,
        )
        return _merge_support_batch_response(checks, response)

    async def classify_review_category(
        self,
        claim: Claim,
        cancellation_token: CancellationToken | None = None,
    ) -> Claim:
        response = await self._request_structured(
            ReviewCategoryResponse,
            REVIEW_CATEGORY_SYSTEM_PROMPT,
            {"claim": _dump_model(claim)},
            cancellation_token=cancellation_token,
        )
        updated = claim.model_copy(deep=True)
        updated.review_category = response.review_category
        updated.reasoning_summary = response.explanation
        return updated

    async def classify_review_categories(
        self,
        claims: list[Claim],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        if not claims:
            return []
        response = await self._request_structured(
            ReviewCategoryBatchResponse,
            REVIEW_CATEGORY_SYSTEM_PROMPT,
            {"claims": [_dump_model(claim) for claim in claims]},
            cancellation_token=cancellation_token,
        )
        return _merge_review_batch_response(claims, response)

    async def _request_structured(
        self,
        response_model: type[T],
        system_prompt: str,
        user_payload: dict[str, Any],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> T:
        return await asyncio.to_thread(
            self._request_structured_sync,
            response_model,
            system_prompt,
            user_payload,
            cancellation_token,
        )

    def _request_structured_sync(
        self,
        response_model: type[T],
        system_prompt: str,
        user_payload: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> T:
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        if not self.codex_bin:
            raise LLMProviderConfigurationError("codex CLI was not found. Install or expose codex on PATH.")
        if not self.is_configured():
            raise LLMProviderConfigurationError("Codex CLI is not logged in. Run `codex login` first.")

        last_error: Exception | None = None
        correction = ""
        for attempt in range(3):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            with tempfile.TemporaryDirectory(prefix="source-grounding-codex-") as tmpdir:
                tmp = Path(tmpdir)
                schema_path = tmp / "schema.json"
                output_path = tmp / "response.json"
                schema_path.write_text(json.dumps(structured_output_schema(response_model), ensure_ascii=False), encoding="utf-8")
                prompt = (
                    f"{system_prompt}\n\n"
                    "You are running as a local subprocess for a FastAPI app. Do not inspect files, do not edit files, "
                    "and do not run tools. Return only JSON matching the provided schema.\n\n"
                    f"{correction}"
                    f"Input:\n{json.dumps(user_payload, ensure_ascii=False, default=str)}"
                )
                cmd = [
                    self.codex_bin,
                    "--ask-for-approval",
                    "never",
                    "exec",
                    "-c",
                    f'service_tier="{self.service_tier}"',
                    "-c",
                    f'model_reasoning_effort="{self.reasoning_effort}"',
                    "--model",
                    self.model,
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--ignore-rules",
                    "--color",
                    "never",
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "-",
                ]
                process: subprocess.Popen[str] | None = None
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
                    if cancellation_token:
                        cancellation_token.register_process(process)
                    stdout, stderr = process.communicate(input=prompt, timeout=self.timeout_seconds)
                    result = subprocess.CompletedProcess(cmd, process.returncode, stdout=stdout, stderr=stderr)
                except subprocess.TimeoutExpired as exc:
                    if process:
                        _terminate_process(process)
                    raise LLMProviderTimeoutError(
                        f"Codex CLI timed out after {self.timeout_seconds:.0f} seconds while using model {self.model}."
                    ) from exc
                finally:
                    if process and cancellation_token:
                        cancellation_token.unregister_process(process)
                if cancellation_token and cancellation_token.is_cancelled():
                    raise AnalysisCancelledError("Analysis was cancelled.")
                if result.returncode != 0:
                    raise LLMProviderError(f"Codex CLI request failed ({result.returncode}): {_safe_process_output(result)}")
                try:
                    parsed = json.loads(output_path.read_text(encoding="utf-8"))
                    return response_model.model_validate(parsed)
                except (OSError, ValueError, ValidationError) as exc:
                    last_error = exc
                    if attempt >= 2:
                        break
                    correction = f"The previous response failed validation: {exc}. Return corrected JSON only.\n\n"
        raise LLMProviderError(f"Codex CLI response failed schema validation: {last_error}")


class MockLLMProvider:
    """Structured mock provider for tests and explicit no-key development.

    It returns caller-provided Pydantic-compatible structures. It is not a semantic
    analyzer and should not be used to claim real audit accuracy.
    """

    def __init__(
        self,
        *,
        extraction_outputs: list[Claim] | dict[str, list[Claim]] | Callable[[str], list[Claim]] | None = None,
        support_outputs: dict[str, dict[str, Any] | Claim] | Callable[[Claim, dict[str, Any]], dict[str, Any] | Claim] | None = None,
        review_outputs: dict[str, dict[str, Any] | ClaimReviewCategory] | Callable[[Claim], dict[str, Any] | ClaimReviewCategory] | None = None,
    ):
        self.extraction_outputs = extraction_outputs
        self.support_outputs = support_outputs
        self.review_outputs = review_outputs

    def is_configured(self) -> bool:
        return True

    async def extract_claims(
        self,
        input_text: str,
        citations: list[ParsedCitation],
        context: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        outputs = self.extraction_outputs
        if callable(outputs):
            return _renumber_claims(outputs(input_text))
        if isinstance(outputs, dict):
            return _renumber_claims(outputs.get(input_text, []))
        if isinstance(outputs, list):
            return _renumber_claims(outputs)
        return []

    async def check_claim_support(
        self,
        claim: Claim,
        source_bundle: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> Claim:
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        outputs = self.support_outputs
        raw: dict[str, Any] | Claim | None = None
        if callable(outputs):
            raw = outputs(claim, source_bundle)
        elif isinstance(outputs, dict):
            raw = outputs.get(claim.claim_id) or outputs.get(claim.normalized_claim)
        if isinstance(raw, Claim):
            return raw
        if isinstance(raw, dict):
            response = SupportCheckResponse.model_validate(raw)
            return _merge_support_response(claim, response)
        return claim.model_copy(deep=True)

    async def check_claim_supports(
        self,
        checks: list[dict[str, Any]],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        return [
            await self.check_claim_support(
                item["claim"],
                item["source_bundle"],
                cancellation_token=cancellation_token,
            )
            for item in checks
        ]

    async def classify_review_category(
        self,
        claim: Claim,
        cancellation_token: CancellationToken | None = None,
    ) -> Claim:
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        outputs = self.review_outputs
        raw: dict[str, Any] | ClaimReviewCategory | None = None
        if callable(outputs):
            raw = outputs(claim)
        elif isinstance(outputs, dict):
            raw = outputs.get(claim.claim_id) or outputs.get(claim.normalized_claim)
        updated = claim.model_copy(deep=True)
        if isinstance(raw, ClaimReviewCategory):
            updated.review_category = raw
            return updated
        if isinstance(raw, str):
            updated.review_category = ClaimReviewCategory(raw)
            return updated
        if isinstance(raw, dict):
            response = ReviewCategoryResponse.model_validate(raw)
            updated.review_category = response.review_category
            updated.reasoning_summary = response.explanation
            return updated
        return updated

    async def classify_review_categories(
        self,
        claims: list[Claim],
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        return [
            await self.classify_review_category(claim, cancellation_token=cancellation_token)
            for claim in claims
        ]


def _claims_from_extraction_response(response: ClaimExtractionResponse) -> list[Claim]:
    claims: list[Claim] = []
    for item in response.claims:
        claim = Claim(
            claim_id=item.claim_id or f"c{len(claims)+1:03d}",
            original_text_span=item.original_span,
            original_span=item.original_span,
            normalized_claim=item.normalized_claim,
            claim_type=item.claim_type,
            discourse_role=item.discourse_role,
            source_opacity=item.source_opacity,
            risk_flags=list(dict.fromkeys(item.risk_flags)),
            has_quantitative_data=item.has_quantitative_data,
            has_material_quantitative_data=item.has_material_quantitative_data,
            importance_label=item.importance_label,
            attributed_to=item.attributed_to,
            source_mentions=list(dict.fromkeys(item.source_mentions)),
            reasoning_summary=item.reasoning_summary,
            evidence_needed=item.evidence_needed,
            not_asserted_by_author=item.not_asserted_by_author,
            final_bucket=FinalGroundingBucket.EXCLUDED_OR_CONTEXT
            if item.claim_type == ClaimType.NON_CLAIM
            else None,
            support_relation=SupportRelation.NOT_CHECKED,
            review_category=ClaimReviewCategory.EXCLUDED_OR_CONTEXT
            if item.claim_type == ClaimType.NON_CLAIM
            else ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK,
        )
        claims.append(claim)
    return _renumber_claims(claims)


def claims_from_model_payload(payload: dict[str, Any]) -> list[Claim]:
    if "claims" not in payload or not isinstance(payload.get("claims"), list):
        raise LLMProviderError("LLM claim payload is missing a claims list.")
    response = ClaimExtractionResponse.model_validate(payload)
    return _claims_from_extraction_response(response)


def structured_output_schema(response_model: type[BaseModel]) -> dict[str, Any]:
    """Return an OpenAI/Codex strict-compatible JSON schema.

    Pydantic marks fields with defaults as optional in JSON Schema. OpenAI-style
    structured output requires every object property to be listed in `required`;
    nullable fields must still be required and use null as a valid value.
    """

    schema = response_model.model_json_schema()
    return _strict_json_schema(schema)


def _merge_support_response(claim: Claim, response: SupportCheckResponse) -> Claim:
    updated = claim.model_copy(deep=True)
    updated.support_relation = response.support_relation
    updated.final_bucket = response.final_bucket
    updated.risk_flags = list(dict.fromkeys(response.risk_flags))
    updated.reasoning_summary = response.reasoning_summary
    updated.evidence_quote = response.evidence_quote
    updated.source_role = response.source_role
    return updated


def _merge_support_batch_response(checks: list[dict[str, Any]], response: SupportCheckBatchResponse) -> list[Claim]:
    response_by_id = {item.check_id: item for item in response.checks}
    updated_claims: list[Claim] = []
    for item in checks:
        claim = item["claim"]
        support = response_by_id.get(item["check_id"])
        if support is None:
            updated_claims.append(claim.model_copy(deep=True))
            continue
        updated = claim.model_copy(deep=True)
        updated.support_relation = support.support_relation
        updated.final_bucket = support.final_bucket
        updated.risk_flags = list(dict.fromkeys(support.risk_flags))
        updated.reasoning_summary = support.reasoning_summary
        updated.evidence_quote = support.evidence_quote
        updated.source_role = support.source_role
        updated_claims.append(updated)
    return updated_claims


def _merge_review_batch_response(claims: list[Claim], response: ReviewCategoryBatchResponse) -> list[Claim]:
    review_by_id = {item.claim_id: item for item in response.reviews}
    updated_claims: list[Claim] = []
    for claim in claims:
        updated = claim.model_copy(deep=True)
        review = review_by_id.get(claim.claim_id)
        if review:
            updated.review_category = review.review_category
            updated.reasoning_summary = review.explanation
        updated_claims.append(updated)
    return updated_claims


def _renumber_claims(claims: list[Claim]) -> list[Claim]:
    renumbered: list[Claim] = []
    for idx, claim in enumerate(claims, start=1):
        updated = claim.model_copy(deep=True)
        if not updated.claim_id:
            updated.claim_id = f"c{idx:03d}"
        renumbered.append(updated)
    return renumbered


def _dump_model(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
        return parsed
    if isinstance(content, list):
        text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
        return parsed
    raise ValueError("Unsupported JSON content type")


def _strict_json_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_strict_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    cleaned: dict[str, Any] = {}
    for key, child in value.items():
        if key == "default":
            continue
        if key == "enum" and isinstance(child, list):
            deduped = []
            for item in child:
                if item not in deduped:
                    deduped.append(item)
            cleaned[key] = deduped
            continue
        cleaned[key] = _strict_json_schema(child)

    properties = cleaned.get("properties")
    if isinstance(properties, dict):
        cleaned["required"] = list(properties.keys())
        cleaned.setdefault("additionalProperties", False)
    return cleaned


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass


def _safe_error_body(response: httpx.Response) -> str:
    text = response.text.strip().replace("\n", " ")
    return f"{text[:500]}..." if len(text) > 500 else text


def _safe_process_output(result: subprocess.CompletedProcess[str]) -> str:
    text = ((result.stderr or "") + "\n" + (result.stdout or "")).strip().replace("\n", " ")
    return f"{text[:700]}..." if len(text) > 700 else text
