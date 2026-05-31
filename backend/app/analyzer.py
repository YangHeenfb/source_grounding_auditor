from __future__ import annotations

import os
import uuid
import asyncio
import re
from typing import Any, Callable

from .citation_parser import (
    citations_near_text_span,
    parse_citations,
    parse_reference_descriptions,
    reference_descriptions_near_text_span,
)
from .citation_parser import FOOTNOTE_REF_RE
from .citation_units import build_citation_units, parsed_citation_for_unit
from .claim_aware_source_role_classifier import classify_claim_source_role
from .display_status_mapper import map_claims_to_display_results
from .document_evidence_graph_builder import build_document_evidence_graph
from .evidence_graph_builder import build_evidence_graphs
from .official_domain_verifier import OfficialDomainVerificationResult
from .providers.llm_provider import (
    CancellationToken,
    CodexCLILLMProvider,
    LLMProvider,
    LLMProviderConfigurationError,
    LLMProviderError,
    LLMProviderTimeoutError,
    MockLLMProvider,
    OpenAILLMProvider,
)
from .providers.search_provider import DuckDuckGoSearchProvider, SearchProvider
from .ratio_reporter import RatioReporter
from .schemas import (
    AnalysisRequest,
    AnalysisResult,
    Claim,
    ClaimReviewCategory,
    ClaimReviewItem,
    ClaimExtractionMode,
    ClaimType,
    CitationUnit,
    DiscourseRole,
    DisplayStatus,
    EdgeBasis,
    EdgeType,
    FinalGroundingBucket,
    ImportanceLabel,
    RiskFlag,
    Source,
    SourceOpacity,
    SupportRelation,
)
from .source_entity_resolver import SourceEntityResolution
from .source_fetcher import SourceFetcher
from .support_checker import SupportChecker, SupportCheckInput
from .terminal_classifier import build_document_evidence_summary, classify_citation_terminals
from .upstream_tracer import UpstreamTracer

ProgressCallback = Callable[[dict[str, Any]], None]

BUCKET_PRIORITY = {
    FinalGroundingBucket.HARD_FACT_GROUNDING: 4,
    FinalGroundingBucket.WEAK_FACT_GROUNDING: 3,
    FinalGroundingBucket.ATTRIBUTION_OR_OPINION_GROUNDING: 2,
    FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH: 1,
    FinalGroundingBucket.EXCLUDED_OR_CONTEXT: 0,
}
MAX_EXTRACTION_CHARS = int(os.environ.get("SOURCE_GROUNDING_EXTRACTION_CHUNK_CHARS") or "2500")


def _source_key_for_url(url: str) -> str:
    return url.rstrip("/")


def _source_title_is_url_like(title: str, url: str) -> bool:
    if not title:
        return True
    normalized = title.strip().lower()
    return normalized == url.strip().lower() or normalized in url.strip().lower()


def _review_item_for_claim(claim: Claim) -> ClaimReviewItem:
    explanation = claim.reasoning_summary
    if not explanation and claim.evidence_chain:
        explanation = claim.evidence_chain[0].reasoning_summary
    return ClaimReviewItem(
        claim_id=claim.claim_id,
        normalized_claim=claim.normalized_claim,
        category=claim.review_category,
        risk_flags=claim.risk_flags,
        final_bucket=claim.final_bucket,
        support_relation=claim.support_relation,
        discourse_role=claim.discourse_role,
        source_opacity=claim.source_opacity,
        importance_label=claim.importance_label,
        explanation=explanation or "",
    )


class SourceGroundingAnalyzer:
    def __init__(
        self,
        *,
        enable_url_fetch: bool = True,
        claim_extraction_mode: str | None = None,
        search_provider: SearchProvider | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        self.openai_extractor = OpenAILLMProvider()
        self.codex_extractor = CodexCLILLMProvider()
        self.llm_provider = llm_provider
        self.search_provider = search_provider or DuckDuckGoSearchProvider()
        self.reporter = RatioReporter()
        self.enable_url_fetch = enable_url_fetch
        self.claim_extraction_mode = _coerce_claim_extraction_mode(
            claim_extraction_mode
            or os.environ.get("SOURCE_GROUNDING_CLAIM_EXTRACTOR")
            or ClaimExtractionMode.CODEX.value
        )

    def analyze(
        self,
        request: AnalysisRequest,
        cancellation_token: CancellationToken | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AnalysisResult:
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        _emit_progress(progress_callback, "parsing", "Parsing citations and citation-only chunks.", 0, None)
        analysis_id = str(uuid.uuid4())
        citations = parse_citations(request.input_text)
        reference_descriptions = parse_reference_descriptions(request.input_text)
        citation_units = (
            []
            if request.uncited_claim_analysis_enabled
            else build_citation_units(request.input_text, citations, reference_descriptions)
        )
        analysis_text, cited_chunk_count = _analysis_text_for_request(
            request.input_text,
            citations,
            reference_descriptions,
            citation_units=citation_units,
            uncited_claim_analysis_enabled=request.uncited_claim_analysis_enabled,
        )
        provider, effective_extraction_mode = self._select_llm_provider(request.claim_extraction_mode)
        context = {
            "original_question": request.original_question,
            "mode": request.mode,
            "citation_only_mode": not request.uncited_claim_analysis_enabled,
            "ratios_basis": (
                "based on all extracted claims"
                if request.uncited_claim_analysis_enabled
                else "based only on cited claims"
            ),
            "paragraphs": [p.strip() for p in analysis_text.split("\n\n") if p.strip()],
        }
        _emit_progress(progress_callback, "extracting_claims", "Extracting cited atomic claims.", 0, None)
        claims = _extract_claims(
            provider,
            analysis_text,
            citations,
            context,
            citation_units=citation_units,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )
        _emit_progress(progress_callback, "extracting_claims", f"Extracted {len(claims)} cited claims.", len(claims), len(claims))
        support_checker = SupportChecker(provider)

        fetcher = SourceFetcher(enable_url_fetch=request.enable_url_fetch or self.enable_url_fetch)
        tracer = UpstreamTracer(fetcher)
        sources_by_id: dict[str, Source] = {}
        source_id_by_url: dict[str, str] = {}
        search_cache: dict[str, list[Source]] = {}
        search_result_count = 0
        search_query_count = 0

        def get_or_fetch_source(url: str, basis: EdgeBasis, source_title: str = "") -> Source:
            key = _source_key_for_url(url)
            if key in source_id_by_url:
                source = sources_by_id[source_id_by_url[key]]
                if source_title and _source_title_is_url_like(source.title, url):
                    source.title = source_title
                return source
            source_id = f"s{len(sources_by_id)+1:03d}"
            source = fetcher.fetch_url(url, source_id, request.provided_sources)
            if source_title and _source_title_is_url_like(source.title, url):
                source.title = source_title
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

        support_inputs: list[SupportCheckInput] = []
        _emit_progress(progress_callback, "resolving_sources", "Resolving cited sources.", 0, len(claims))
        for claim_index, claim in enumerate(claims, start=1):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            _emit_progress(
                progress_callback,
                "resolving_sources",
                f"Resolving sources for claim {claim_index} of {len(claims)}.",
                claim_index - 1,
                len(claims),
            )
            if claim.claim_type == ClaimType.NON_CLAIM:
                support_inputs.append(
                    SupportCheckInput(
                        claim=claim,
                        source=None,
                        basis=EdgeBasis.NONE,
                    )
                )
                continue

            linked_sources: list[tuple[Source, EdgeBasis]] = []
            if not request.uncited_claim_analysis_enabled and claim.citation_source_url:
                linked_sources.append(
                    (
                        get_or_fetch_source(
                            claim.citation_source_url,
                            EdgeBasis.FOOTNOTE if claim.citation_label else EdgeBasis.EXPLICIT_LINK,
                            claim.citation_source_title,
                        ),
                        EdgeBasis.FOOTNOTE if claim.citation_label else EdgeBasis.EXPLICIT_LINK,
                    )
                )
            elif not request.uncited_claim_analysis_enabled and request.enable_web_search and claim.source_registry_entry:
                for source in source_from_search_url(claim.source_registry_entry):
                    linked_sources.append((source, EdgeBasis.DISCOVERED_SOURCE))
            elif request.uncited_claim_analysis_enabled:
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
                support_inputs.append(
                    SupportCheckInput(
                        claim=claim,
                        source=None,
                        basis=EdgeBasis.NONE,
                    )
                )
                continue

            for source, basis in linked_sources:
                claim.linked_source_ids.append(source.source_id)
                if claim.citation_source_url and _source_key_for_url(claim.citation_source_url) == _source_key_for_url(source.url or ""):
                    claim.citation_source_id = source.source_id
                # Trace explicit upstream links from this source body only. The claim's final edge remains the direct citation.
                tracer.trace(source, existing_sources=sources_by_id, max_depth=request.max_upstream_depth)
                edge_type = EdgeType.DISCOVERED_SOURCE if basis == EdgeBasis.DISCOVERED_SOURCE else EdgeType.AUTHOR_CITED
                support_inputs.append(
                    SupportCheckInput(
                        claim=claim,
                        source=source,
                        edge_type=edge_type,
                        basis=basis,
                    )
                )

        _emit_progress(progress_callback, "resolving_sources", "Resolved cited sources.", len(claims), len(claims))
        _emit_progress(
            progress_callback,
            "checking_support",
            f"Checking support for {len(support_inputs)} claim-source pairs.",
            0,
            len(support_inputs),
        )
        support_results = support_checker.check_many(support_inputs, cancellation_token=cancellation_token)
        support_results_by_claim: dict[str, list[tuple[Any, list[RiskFlag]]]] = {}
        for support_input, (edge, flags) in zip(support_inputs, support_results):
            if support_input.source is not None:
                edge.upstream_source_ids = support_input.source.upstream_source_ids
                _apply_claim_aware_source_role(support_input.claim, support_input.source, edge)
            support_results_by_claim.setdefault(support_input.claim.claim_id, []).append((edge, flags))

        for claim in claims:
            entries = support_results_by_claim.get(claim.claim_id, [])
            if not entries:
                continue
            candidate_edges = [edge for edge, _flags in entries]
            all_flags = [flag for _edge, flags in entries for flag in flags]
            # Pick the strongest usable evidence edge, but preserve all edges in the evidence chain.
            best_edge = sorted(candidate_edges, key=lambda e: BUCKET_PRIORITY[e.final_bucket], reverse=True)[0]
            claim.evidence_chain = candidate_edges
            claim.final_bucket = best_edge.final_bucket
            claim.support_relation = best_edge.support_relation
            claim.reasoning_summary = best_edge.reasoning_summary or claim.reasoning_summary
            claim.evidence_quote = best_edge.evidence_quote or best_edge.evidence_span
            claim.source_role = best_edge.source_role
            claim.source_role_for_claim = best_edge.source_role_for_claim
            claim.source_to_claim_relation = best_edge.source_to_claim_relation
            claim.support_scope = best_edge.support_scope
            claim.source_role_basis = best_edge.source_role_basis
            claim.risk_flags = list(dict.fromkeys(claim.risk_flags + all_flags))

        _emit_progress(
            progress_callback,
            "checking_support",
            f"Checked support for {len(support_inputs)} claim-source pairs.",
            len(support_inputs),
            len(support_inputs),
        )
        _emit_progress(progress_callback, "classifying_review", "Classifying review categories.", 0, len(claims))
        claims = self.classify_claim_reviews(claims, provider, cancellation_token=cancellation_token)
        for claim in claims:
            _enforce_problematic_threshold(claim)
        _emit_progress(progress_callback, "classifying_review", "Classified review categories.", len(claims), len(claims))

        _emit_progress(progress_callback, "summarizing", "Building audit summary.", 0, None)
        display_citations = map_claims_to_display_results(claims)
        evidence_graphs = build_evidence_graphs(claims, sources_by_id)
        citation_terminal_results = classify_citation_terminals(
            claims,
            display_citations,
            sources_by_id,
            max_terminal_trace_depth=request.max_terminal_trace_depth,
        )
        document_evidence_summary = build_document_evidence_summary(citation_terminal_results)
        document_evidence_graph = build_document_evidence_graph(citation_terminal_results)
        summary = self.reporter.build_summary(claims, display_citations=display_citations)
        summary.ratios_basis = (
            "based on all extracted claims"
            if request.uncited_claim_analysis_enabled
            else "based only on cited claims"
        )
        review_items = [_review_item_for_claim(claim) for claim in claims]
        review_item_by_claim_id = {item.claim_id: item for item in review_items}
        problematic_citations = [
            review_item_by_claim_id[item.claim_id]
            for item in display_citations
            if item.should_show_in_problematic and item.claim_id in review_item_by_claim_id
        ]
        attribution_supported_citations = [
            review_item_by_claim_id[item.claim_id]
            for item in display_citations
            if item.display_status == DisplayStatus.ATTRIBUTION_SUPPORT and item.claim_id in review_item_by_claim_id
        ]
        audit_limited_citations = [
            review_item_by_claim_id[item.claim_id]
            for item in display_citations
            if item.display_status == DisplayStatus.AUDIT_LIMITED and item.claim_id in review_item_by_claim_id
        ]
        flagged_citations = [
            review_item_by_claim_id[item.claim_id]
            for item in display_citations
            if item.display_status
            in {
                DisplayStatus.PARTIAL_OR_WEAK_SUPPORT,
                DisplayStatus.ANALYSIS_FROM_SOURCED_PREMISES,
            }
            and item.claim_id in review_item_by_claim_id
        ]
        excluded_or_context_citations = [
            review_item_by_claim_id[item.claim_id]
            for item in display_citations
            if item.display_status == DisplayStatus.EXCLUDED_OR_CONTEXT and item.claim_id in review_item_by_claim_id
        ]

        return AnalysisResult(
            analysis_id=analysis_id,
            summary=summary,
            claims=claims,
            sources=list(sources_by_id.values()),
            display_citations=display_citations,
            evidence_graphs=evidence_graphs,
            citation_terminal_results=citation_terminal_results,
            document_evidence_summary=document_evidence_summary,
            document_evidence_graph=document_evidence_graph,
            problematic_citations=problematic_citations,
            audit_limited_citations=audit_limited_citations,
            attribution_supported_citations=attribution_supported_citations,
            flagged_citations=flagged_citations,
            excluded_or_context_citations=excluded_or_context_citations,
            uncited_claim_analysis_enabled=request.uncited_claim_analysis_enabled,
            high_risk_claims=problematic_citations,
            flagged_claims=flagged_citations,
            audit_limited_claims=audit_limited_citations,
            attribution_supported_claims=attribution_supported_citations,
            excluded_or_context_claims=excluded_or_context_citations,
            metadata={
                "mode": request.mode,
                "citation_only_mode": not request.uncited_claim_analysis_enabled,
                "uncited_claim_analysis_enabled": request.uncited_claim_analysis_enabled,
                "ratios_basis": summary.ratios_basis,
                "cited_chunk_count": cited_chunk_count,
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
                "codex_reasoning_effort": (
                    self.codex_extractor.reasoning_effort
                    if effective_extraction_mode in {ClaimExtractionMode.CODEX, ClaimExtractionMode.CODEX_CLI}
                    else None
                ),
                "max_upstream_depth": request.max_upstream_depth,
                "max_terminal_trace_depth": request.max_terminal_trace_depth,
                "enable_url_fetch": request.enable_url_fetch or self.enable_url_fetch,
                "enable_web_search": request.enable_web_search,
                "citation_count": len(citations),
                "reference_description_count": len(reference_descriptions),
                "citation_units": [unit.model_dump(mode="json") for unit in citation_units],
                "citation_unit_count": len(citation_units),
                "search_query_count": search_query_count,
                "search_result_count": search_result_count,
                "note": "MVP mode: no truth verdict or single credibility score is produced.",
            },
        )

    def classify_claim_review(self, claim: Claim, provider: LLMProvider) -> Claim:
        updated = _run_async(provider.classify_review_category(claim))
        claim.review_category = updated.review_category
        if updated.reasoning_summary:
            claim.reasoning_summary = updated.reasoning_summary
        _apply_audit_limited_override(claim)
        return claim

    def classify_claim_reviews(
        self,
        claims: list[Claim],
        provider: LLMProvider,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[Claim]:
        if not claims:
            return claims
        try:
            updated_claims = _run_async(
                provider.classify_review_categories(claims, cancellation_token=cancellation_token)
            )
        except LLMProviderTimeoutError as exc:
            for claim in claims:
                _apply_timeout_review_fallback(claim, exc)
            return claims
        except LLMProviderError as exc:
            for claim in claims:
                _apply_provider_error_review_fallback(claim, exc)
            return claims
        for claim, updated in zip(claims, updated_claims):
            claim.review_category = updated.review_category
            if updated.reasoning_summary:
                claim.reasoning_summary = updated.reasoning_summary
            _apply_audit_limited_override(claim)
        return claims

    def _select_llm_provider(
        self, requested_mode: ClaimExtractionMode | None
    ) -> tuple[LLMProvider, ClaimExtractionMode]:
        if self.llm_provider is not None:
            return self.llm_provider, ClaimExtractionMode.MOCK
        mode = requested_mode or self.claim_extraction_mode
        if mode == ClaimExtractionMode.MOCK:
            return MockLLMProvider(), ClaimExtractionMode.MOCK
        if mode == ClaimExtractionMode.AUTO:
            if self.openai_extractor.is_configured():
                return self.openai_extractor, ClaimExtractionMode.OPENAI
            if self.codex_extractor.is_configured():
                return self.codex_extractor, ClaimExtractionMode.CODEX
            if os.environ.get("SOURCE_GROUNDING_ALLOW_MOCK") == "1":
                return MockLLMProvider(), ClaimExtractionMode.MOCK
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


def _apply_audit_limited_override(claim: Claim) -> None:
    substantive_flags = set(claim.risk_flags) - {RiskFlag.INACCESSIBLE_SOURCE}
    if (
        RiskFlag.INACCESSIBLE_SOURCE in claim.risk_flags
        and not substantive_flags
        and claim.support_relation
        and claim.support_relation
        in {
            SupportRelation.INACCESSIBLE,
            SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
        }
    ):
        claim.review_category = ClaimReviewCategory.AUDIT_LIMITED


def _enforce_problematic_threshold(claim: Claim) -> None:
    if claim.review_category != ClaimReviewCategory.HIGH_RISK:
        _apply_audit_limited_override(claim)
        return
    if _is_true_problematic_citation(claim):
        return
    if claim.support_relation in {
        SupportRelation.INACCESSIBLE,
        SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
    } or RiskFlag.INACCESSIBLE_SOURCE in claim.risk_flags:
        claim.review_category = ClaimReviewCategory.AUDIT_LIMITED
    else:
        claim.review_category = ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK


def _is_true_problematic_citation(claim: Claim) -> bool:
    if claim.claim_type == ClaimType.NON_CLAIM or claim.not_asserted_by_author:
        return False
    if claim.discourse_role in {
        DiscourseRole.CAVEAT_OR_LIMITATION,
        DiscourseRole.UNSUPPORTED_EXAMPLE,
        DiscourseRole.SOURCE_POINTER,
        DiscourseRole.USER_QUESTION,
        DiscourseRole.SECTION_HEADING,
        DiscourseRole.CONTEXT_OR_TRANSITION,
    }:
        return False
    has_substantive_problem = (
        claim.support_relation in {SupportRelation.NO_SUPPORT, SupportRelation.CONTRADICTS}
        or RiskFlag.SOURCE_CLAIM_MISMATCH in claim.risk_flags
    )
    if not has_substantive_problem:
        return False
    return any(
        edge.source_id
        and edge.support_relation in {SupportRelation.NO_SUPPORT, SupportRelation.CONTRADICTS}
        and bool(edge.evidence_quote or edge.evidence_span)
        for edge in claim.evidence_chain
    ) or (
        RiskFlag.SOURCE_CLAIM_MISMATCH in claim.risk_flags
        and any(edge.source_id and bool(edge.evidence_quote or edge.evidence_span) for edge in claim.evidence_chain)
    )


def _apply_claim_aware_source_role(claim: Claim, source: Source, edge: Any) -> None:
    resolution = SourceEntityResolution(
        source_entity=source.source_entity,
        registrable_domain=source.registrable_domain,
        publisher_name=source.publisher_name,
        organization_type=source.organization_type,
        entity_aliases=source.entity_aliases,
        metadata_basis=source.metadata_basis,
    )
    officialness = OfficialDomainVerificationResult(
        officialness_status=source.officialness_status,
        basis=source.officialness_basis,
    )
    result = classify_claim_source_role(
        claim=claim,
        source=source,
        source_entity_resolution=resolution,
        officialness_result=officialness,
        support_relation=edge.support_relation,
    )
    edge.source_role_for_claim = result.source_role_for_claim
    edge.source_to_claim_relation = result.source_to_claim_relation
    edge.support_scope = result.support_scope
    edge.source_role_basis = result.basis
    if (
        result.support_scope.value == "premise_support_for_analysis"
        and claim.claim_type == ClaimType.JUDGMENT
        and edge.support_relation == SupportRelation.DIRECTLY_SUPPORTS
    ):
        edge.support_relation = SupportRelation.PARTIALLY_SUPPORTS
        if edge.final_bucket == FinalGroundingBucket.HARD_FACT_GROUNDING:
            edge.final_bucket = FinalGroundingBucket.WEAK_FACT_GROUNDING
        edge.reasoning_summary = (
            f"{edge.reasoning_summary} Claim-aware source role: official source provides premises "
            "for analysis/judgment, not direct fact support."
        ).strip()


def _apply_timeout_review_fallback(claim: Claim, exc: LLMProviderTimeoutError) -> None:
    if claim.support_relation in {
        SupportRelation.INACCESSIBLE,
        SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
    }:
        claim.review_category = ClaimReviewCategory.AUDIT_LIMITED
    else:
        claim.review_category = ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK
    claim.reasoning_summary = f"Review classification timed out: {exc}"
    _apply_audit_limited_override(claim)


def _apply_provider_error_review_fallback(claim: Claim, exc: LLMProviderError) -> None:
    if claim.claim_type == ClaimType.NON_CLAIM or claim.not_asserted_by_author:
        claim.review_category = ClaimReviewCategory.EXCLUDED_OR_CONTEXT
    elif claim.support_relation in {
        SupportRelation.INACCESSIBLE,
        SupportRelation.AUDIT_LIMITED_NO_RELEVANT_SNIPPET,
    }:
        claim.review_category = ClaimReviewCategory.AUDIT_LIMITED
    else:
        claim.review_category = ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK
    claim.reasoning_summary = (
        "Review classification could not be completed by the LLM provider in this run; "
        f"no problematic citation is assigned without structured review. Provider error: {exc}"
    )
    _apply_audit_limited_override(claim)


def _extract_claims(
    provider: LLMProvider,
    analysis_text: str,
    citations,
    context: dict[str, Any],
    *,
    citation_units: list[CitationUnit],
    cancellation_token: CancellationToken | None,
    progress_callback: ProgressCallback | None,
) -> list[Claim]:
    if citation_units:
        return _extract_claims_from_citation_units(
            provider,
            citation_units,
            context,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )
    if not analysis_text.strip():
        return []
    paragraphs = [p.strip() for p in analysis_text.split("\n\n") if p.strip()]
    chunks = _chunk_paragraphs(paragraphs, MAX_EXTRACTION_CHARS)
    claims: list[Claim] = []
    for index, chunk in enumerate(chunks, start=1):
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        _emit_progress(
            progress_callback,
            "extracting_claims",
            f"Extracting cited atomic claims, chunk {index} of {len(chunks)}.",
            index - 1,
            len(chunks),
        )
        chunk_context = dict(context)
        chunk_context["paragraphs"] = [p.strip() for p in chunk.split("\n\n") if p.strip()]
        chunk_citations = parse_citations(chunk)
        try:
            claims.extend(_run_async(
                provider.extract_claims(
                    chunk,
                    chunk_citations,
                    chunk_context,
                    cancellation_token=cancellation_token,
                )
            ))
        except LLMProviderTimeoutError as exc:
            claims.append(_audit_limited_extraction_timeout_claim(chunk, exc))
    return _renumber_claims(claims)


def _extract_claims_from_citation_units(
    provider: LLMProvider,
    citation_units: list[CitationUnit],
    context: dict[str, Any],
    *,
    cancellation_token: CancellationToken | None,
    progress_callback: ProgressCallback | None,
) -> list[Claim]:
    claims: list[Claim] = []
    for index, unit in enumerate(citation_units, start=1):
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        _emit_progress(
            progress_callback,
            "extracting_claims",
            f"Extracting cited atomic claims, citation unit {index} of {len(citation_units)}.",
            index - 1,
            len(citation_units),
        )
        unit_context = dict(context)
        unit_context["citation_unit"] = unit.model_dump(mode="json")
        unit_context["paragraphs"] = [unit.cited_text]
        unit_citations = [parsed_citation_for_unit(unit, f"cit_unit_{index:03d}")]
        try:
            extracted = _run_async(
                provider.extract_claims(
                    unit.cited_text,
                    unit_citations,
                    unit_context,
                    cancellation_token=cancellation_token,
                )
            )
        except LLMProviderTimeoutError as exc:
            extracted = [_audit_limited_extraction_timeout_claim(unit.cited_text, exc)]
        for claim in extracted:
            claims.append(_attach_citation_unit_to_claim(claim, unit))
    return _renumber_claims(claims)


def _attach_citation_unit_to_claim(claim: Claim, unit: CitationUnit) -> Claim:
    updated = claim.model_copy(deep=True)
    updated.citation_label = unit.citation_label
    updated.citation_source_url = unit.source_url
    updated.citation_source_title = unit.source_title
    updated.source_registry_entry = unit.source_registry_entry
    if not updated.original_text_span:
        updated.original_text_span = unit.cited_text
    if not updated.original_span:
        updated.original_span = updated.original_text_span
    return updated


def _chunk_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        parts = _split_long_text(paragraph, max_chars)
        for part in parts:
            part_len = len(part)
            if current and current_len + part_len + 2 > max_chars:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(part)
            current_len += part_len + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[。！？.!?])\s*", text):
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    return pieces


def _audit_limited_extraction_timeout_claim(text: str, exc: LLMProviderTimeoutError) -> Claim:
    excerpt = re.sub(r"\s+", " ", text).strip()
    if len(excerpt) > 280:
        excerpt = f"{excerpt[:277]}..."
    return Claim(
        claim_id="",
        original_text_span=text,
        original_span=text,
        normalized_claim=f"LLM claim extraction timed out for cited passage: {excerpt}",
        claim_type=ClaimType.NON_CLAIM,
        discourse_role=DiscourseRole.CONTEXT_OR_TRANSITION,
        source_opacity=SourceOpacity.NOT_APPLICABLE,
        has_quantitative_data=False,
        has_material_quantitative_data=False,
        importance_label=ImportanceLabel.MINOR,
        final_bucket=FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH,
        support_relation=SupportRelation.INACCESSIBLE,
        review_category=ClaimReviewCategory.AUDIT_LIMITED,
        risk_flags=[RiskFlag.INACCESSIBLE_SOURCE],
        reasoning_summary=f"Claim extraction timed out for this cited passage: {exc}",
        evidence_needed=["Retry with a shorter cited passage or provide source bodies directly."],
    )


def _renumber_claims(claims: list[Claim]) -> list[Claim]:
    renumbered: list[Claim] = []
    for index, claim in enumerate(claims, start=1):
        updated = claim.model_copy(deep=True)
        updated.claim_id = f"c{index:03d}"
        renumbered.append(updated)
    return renumbered


def _emit_progress(
    progress_callback: ProgressCallback | None,
    phase: str,
    message: str,
    current: int | None,
    total: int | None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "phase": phase,
            "message": message,
            "current": current,
            "total": total,
        }
    )


def _coerce_claim_extraction_mode(value: str) -> ClaimExtractionMode:
    try:
        return ClaimExtractionMode(value)
    except ValueError:
        return ClaimExtractionMode.CODEX


def _analysis_text_for_request(
    input_text: str,
    citations,
    reference_descriptions,
    *,
    citation_units: list[CitationUnit],
    uncited_claim_analysis_enabled: bool,
) -> tuple[str, int]:
    if uncited_claim_analysis_enabled:
        return input_text, len([p for p in input_text.split("\n\n") if p.strip()])
    if citation_units:
        return "\n\n".join(unit.cited_text for unit in citation_units), len(citation_units)

    citation_labels = {citation.label for citation in citations if citation.label}
    citation_labels.update(reference.label for reference in reference_descriptions if reference.label)

    chunks: list[str] = []
    seen: set[str] = set()
    for start, end, paragraph in _paragraph_spans(input_text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _is_source_registry_paragraph(paragraph):
            continue
        has_explicit_citation = any(
            start <= citation.span_start < end
            for citation in citations
            if citation.span_start != citation.span_end
        )
        labels_in_paragraph = set(FOOTNOTE_REF_RE.findall(paragraph))
        has_label_reference = bool(labels_in_paragraph & citation_labels)
        if has_explicit_citation or has_label_reference:
            normalized = re.sub(r"\s+", " ", paragraph).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                chunks.append(paragraph)
    return "\n\n".join(chunks), len(chunks)


def _is_source_registry_paragraph(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return False
    heading_terms = {"sources", "references", "source pointer", "来源", "来源指针", "参考资料", "资料来源"}
    content_lines = [line for line in lines if line.strip(" ：:").lower() not in heading_terms]
    if not content_lines:
        return True
    reference_like = 0
    for line in content_lines:
        if re.match(r"^(?:\[\d+\]|\d+\.|-)\s+", line) or re.match(r"^\[?\^?\d+\]?\s*[:.]\s+", line):
            reference_like += 1
    return bool(reference_like and reference_like == len(content_lines))


def _paragraph_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        spans.append((match.start(), match.end(), match.group(0)))
    return spans


def _run_async(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("SourceGroundingAnalyzer.analyze cannot run inside an already running event loop.") from None
