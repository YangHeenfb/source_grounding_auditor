from __future__ import annotations

import os
import uuid

from .citation_parser import (
    citations_near_text_span,
    parse_citations,
    parse_reference_descriptions,
    reference_descriptions_near_text_span,
)
from .claim_extractor import ClaimExtractor
from .providers.codex_claim_extractor import CodexCLIClaimExtractor
from .providers.llm_provider import LLMProviderConfigurationError
from .providers.openai_claim_extractor import OpenAIClaimExtractor
from .providers.search_provider import DuckDuckGoSearchProvider, SearchProvider
from .ratio_reporter import RatioReporter
from .schemas import (
    AnalysisRequest,
    AnalysisResult,
    Claim,
    ClaimExtractionMode,
    ClaimType,
    EdgeBasis,
    EdgeType,
    GroundingBucket,
    HighRiskClaim,
    RiskFlag,
    Source,
)
from .source_fetcher import SourceFetcher
from .support_checker import SupportChecker
from .upstream_tracer import UpstreamTracer

BUCKET_PRIORITY = {
    GroundingBucket.HARD_FACT_GROUNDING: 4,
    GroundingBucket.WEAK_FACT_GROUNDING: 3,
    GroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING: 2,
    GroundingBucket.UNVERIFIABLE_OR_MISMATCH: 1,
}


def _source_key_for_url(url: str) -> str:
    return url.rstrip("/")


def _make_high_risk_explanation(claim: Claim) -> str:
    if RiskFlag.OPINION_USED_AS_FACT in claim.risk_flags:
        return "The cited material appears to be opinion or analysis rather than direct factual evidence."
    if RiskFlag.CORRELATION_PRESENTED_AS_CAUSATION in claim.risk_flags:
        return "The claim uses causal language, while the source appears to support only association or correlation."
    if RiskFlag.SOURCE_CLAIM_MISMATCH in claim.risk_flags:
        return "The cited source does not appear to support the claim."
    if RiskFlag.ANONYMOUS_SOURCE in claim.risk_flags or RiskFlag.VAGUE_SOURCE in claim.risk_flags:
        return "The claim relies on an unnamed, vague, or opaque source mention."
    if RiskFlag.INACCESSIBLE_SOURCE in claim.risk_flags:
        return "The cited source was not publicly available to the analyzer in this run."
    if RiskFlag.QUANTITATIVE_CLAIM_WITHOUT_PRIMARY_DATA in claim.risk_flags:
        return "The claim contains a number, but no primary data source was found."
    return "This claim has risk flags that require manual review."


class SourceGroundingAnalyzer:
    def __init__(
        self,
        *,
        enable_url_fetch: bool = True,
        claim_extraction_mode: str | None = None,
        search_provider: SearchProvider | None = None,
    ):
        self.openai_extractor = OpenAIClaimExtractor()
        self.codex_extractor = CodexCLIClaimExtractor()
        self.search_provider = search_provider or DuckDuckGoSearchProvider()
        self.support_checker = SupportChecker()
        self.reporter = RatioReporter()
        self.enable_url_fetch = enable_url_fetch
        self.claim_extraction_mode = _coerce_claim_extraction_mode(
            claim_extraction_mode
            or os.environ.get("SOURCE_GROUNDING_CLAIM_EXTRACTOR")
            or ClaimExtractionMode.CODEX.value
        )

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        analysis_id = str(uuid.uuid4())
        citations = parse_citations(request.input_text)
        reference_descriptions = parse_reference_descriptions(request.input_text)
        extractor, effective_extraction_mode = self._select_claim_extractor(request.claim_extraction_mode)
        claims = extractor.extract_claims(request.input_text, request.original_question)

        fetcher = SourceFetcher(enable_url_fetch=request.enable_url_fetch or self.enable_url_fetch)
        tracer = UpstreamTracer(fetcher)
        sources_by_id: dict[str, Source] = {}
        source_id_by_url: dict[str, str] = {}
        search_cache: dict[str, list[Source]] = {}
        search_result_count = 0
        search_query_count = 0

        def get_or_fetch_source(url: str, basis: EdgeBasis) -> Source:
            key = _source_key_for_url(url)
            if key in source_id_by_url:
                return sources_by_id[source_id_by_url[key]]
            source_id = f"s{len(sources_by_id)+1:03d}"
            source = fetcher.fetch_url(url, source_id, request.provided_sources)
            sources_by_id[source_id] = source
            source_id_by_url[key] = source_id
            return source

        def make_opaque_source(mention: str) -> Source:
            key = f"opaque::{mention.lower()}"
            if key in source_id_by_url:
                return sources_by_id[source_id_by_url[key]]
            source_id = f"s{len(sources_by_id)+1:03d}"
            source = fetcher.make_opaque_source(source_id, mention)
            sources_by_id[source_id] = source
            source_id_by_url[key] = source_id
            return source

        def source_from_search_url(query: str):
            nonlocal search_result_count, search_query_count
            cache_key = " ".join(query.split()).lower()
            if cache_key in search_cache:
                return search_cache[cache_key]
            search_query_count += 1
            results = self.search_provider.search(query, max_results=request.max_search_results)
            discovered: list[Source] = []
            for result in results:
                key = _source_key_for_url(result.url)
                if key in source_id_by_url:
                    discovered.append(sources_by_id[source_id_by_url[key]])
                    continue
                source_id = f"s{len(sources_by_id)+1:03d}"
                source = fetcher.source_from_search_result(result, source_id)
                sources_by_id[source_id] = source
                source_id_by_url[key] = source_id
                discovered.append(source)
                search_result_count += 1
            search_cache[cache_key] = discovered
            return discovered

        for claim in claims:
            if claim.claim_type == ClaimType.NON_CLAIM:
                edge, flags = self.support_checker.check(claim, None, basis=EdgeBasis.NONE)
                claim.evidence_chain.append(edge)
                claim.final_bucket = None
                claim.support_relation = edge.support_relation
                claim.risk_flags = list(dict.fromkeys(claim.risk_flags + flags))
                continue

            linked_sources: list[tuple[Source, EdgeBasis]] = []
            near_citations = citations_near_text_span(request.input_text, claim.original_text_span, citations)
            for citation in near_citations:
                if citation.url:
                    linked_sources.append((get_or_fetch_source(citation.url, citation.kind), citation.kind))

            if request.enable_web_search:
                near_references = reference_descriptions_near_text_span(
                    request.input_text,
                    claim.original_text_span,
                    reference_descriptions,
                )
                for reference in near_references:
                    for source in source_from_search_url(reference.description):
                        linked_sources.append((source, EdgeBasis.DISCOVERED_SOURCE))

            # If the sentence contains vague or explicit source mentions but no URL, make the opacity auditable.
            if not linked_sources and claim.source_mentions:
                for mention in claim.source_mentions:
                    linked_sources.append((make_opaque_source(mention), EdgeBasis.SOURCE_STATEMENT))

            if not linked_sources:
                edge, flags = self.support_checker.check(claim, None, basis=EdgeBasis.NONE)
                claim.evidence_chain.append(edge)
                claim.final_bucket = edge.final_bucket
                claim.support_relation = edge.support_relation
                claim.risk_flags = list(dict.fromkeys(claim.risk_flags + flags))
                continue

            candidate_edges = []
            all_flags = []
            for source, basis in linked_sources:
                claim.linked_source_ids.append(source.source_id)
                # Trace explicit upstream links from this source body only. The claim's final edge remains the direct citation.
                upstream_sources = tracer.trace(source, existing_sources=sources_by_id, max_depth=request.max_upstream_depth)
                edge_type = EdgeType.DISCOVERED_SOURCE if basis == EdgeBasis.DISCOVERED_SOURCE else EdgeType.AUTHOR_CITED
                edge, flags = self.support_checker.check(claim, source, edge_type=edge_type, basis=basis)
                edge.upstream_source_ids = source.upstream_source_ids
                candidate_edges.append(edge)
                all_flags.extend(flags)

            # Pick the strongest usable evidence edge, but preserve all edges in the evidence chain.
            best_edge = sorted(candidate_edges, key=lambda e: BUCKET_PRIORITY[e.final_bucket], reverse=True)[0]
            claim.evidence_chain = candidate_edges
            claim.final_bucket = best_edge.final_bucket
            claim.support_relation = best_edge.support_relation
            claim.risk_flags = list(dict.fromkeys(claim.risk_flags + all_flags))

        summary = self.reporter.build_summary(claims)
        high_risk_claims = [
            HighRiskClaim(
                claim_id=claim.claim_id,
                normalized_claim=claim.normalized_claim,
                risk_flags=claim.risk_flags,
                final_bucket=claim.final_bucket,
                support_relation=claim.support_relation,
                explanation=_make_high_risk_explanation(claim),
            )
            for claim in claims
            if claim.risk_flags and claim.claim_type != ClaimType.NON_CLAIM
        ]

        return AnalysisResult(
            analysis_id=analysis_id,
            summary=summary,
            claims=claims,
            sources=list(sources_by_id.values()),
            high_risk_claims=high_risk_claims,
            metadata={
                "mode": request.mode,
                "requested_claim_extraction_mode": (
                    request.claim_extraction_mode.value
                    if request.claim_extraction_mode
                    else self.claim_extraction_mode.value
                ),
                "claim_extraction_mode": effective_extraction_mode.value,
                "openai_model": (
                    self.openai_extractor.model
                    if effective_extraction_mode == ClaimExtractionMode.OPENAI
                    else None
                ),
                "codex_model": (
                    self.codex_extractor.model
                    if effective_extraction_mode in {ClaimExtractionMode.CODEX, ClaimExtractionMode.CODEX_CLI}
                    else None
                ),
                "codex_service_tier": (
                    self.codex_extractor.service_tier
                    if effective_extraction_mode in {ClaimExtractionMode.CODEX, ClaimExtractionMode.CODEX_CLI}
                    else None
                ),
                "max_upstream_depth": request.max_upstream_depth,
                "enable_url_fetch": request.enable_url_fetch or self.enable_url_fetch,
                "enable_web_search": request.enable_web_search,
                "citation_count": len(citations),
                "reference_description_count": len(reference_descriptions),
                "search_query_count": search_query_count,
                "search_result_count": search_result_count,
                "note": "MVP mode: no truth verdict or single credibility score is produced.",
            },
        )

    def _select_claim_extractor(
        self, requested_mode: ClaimExtractionMode | None
    ) -> tuple[ClaimExtractor, ClaimExtractionMode]:
        mode = requested_mode or self.claim_extraction_mode
        if mode == ClaimExtractionMode.AUTO:
            if self.openai_extractor.is_configured():
                return self.openai_extractor, ClaimExtractionMode.OPENAI
            if self.codex_extractor.is_configured():
                return self.codex_extractor, ClaimExtractionMode.CODEX
            raise LLMProviderConfigurationError(
                "No LLM provider is configured. Set OPENAI_API_KEY or run `codex login`."
            )
        if mode == ClaimExtractionMode.OPENAI:
            if not self.openai_extractor.is_configured():
                raise LLMProviderConfigurationError(
                    "OPENAI_API_KEY is not set. Set it before using claim_extraction_mode='openai'."
                )
            return self.openai_extractor, ClaimExtractionMode.OPENAI
        if mode in {ClaimExtractionMode.CODEX, ClaimExtractionMode.CODEX_CLI}:
            if not self.codex_extractor.is_configured():
                raise LLMProviderConfigurationError(
                    "Codex CLI is not logged in. Run `codex login` before using claim_extraction_mode='codex'."
                )
            return self.codex_extractor, ClaimExtractionMode.CODEX
        if self.codex_extractor.is_configured():
            return self.codex_extractor, ClaimExtractionMode.CODEX
        raise LLMProviderConfigurationError(
            "No LLM provider is configured. Set OPENAI_API_KEY or run `codex login`."
        )


def _coerce_claim_extraction_mode(value: str) -> ClaimExtractionMode:
    try:
        return ClaimExtractionMode(value)
    except ValueError:
        return ClaimExtractionMode.CODEX
