from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClaimType(str, Enum):
    FACTUAL = "factual"
    ATTRIBUTION = "attribution"
    JUDGMENT = "judgment"
    NON_CLAIM = "non_claim"


class ImportanceLabel(str, Enum):
    CORE = "core"
    SUPPORTING = "supporting"
    MINOR = "minor"
    # Backward-compatible alias for older tests and callers.
    BACKGROUND = "minor"


class DiscourseRole(str, Enum):
    ASSERTED_CLAIM = "asserted_claim"
    ATTRIBUTION_REPORT = "attribution_report"
    JUDGMENT_OR_ANALYSIS = "judgment_or_analysis"
    CAVEAT_OR_LIMITATION = "caveat_or_limitation"
    UNSUPPORTED_EXAMPLE = "unsupported_example"
    SOURCE_POINTER = "source_pointer"
    USER_QUESTION = "user_question"
    SECTION_HEADING = "section_heading"
    CONTEXT_OR_TRANSITION = "context_or_transition"


class SourceOpacity(str, Enum):
    CLEAR_NAMED_SOURCE = "clear_named_source"
    NAMED_SECONDARY_WITH_OPAQUE_UNDERLYING = "named_secondary_with_opaque_underlying"
    VAGUE_SOURCE_MENTION = "vague_source_mention"
    ANONYMOUS_SOURCE = "anonymous_source"
    NOT_APPLICABLE = "not_applicable"


class ClaimExtractionMode(str, Enum):
    OPENAI = "openai"
    CODEX = "codex"
    CODEX_CLI = "codex_cli"
    MOCK = "mock"
    AUTO = "auto"


class AccessStatus(str, Enum):
    ACCESSIBLE = "accessible"
    PAYWALLED = "paywalled"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class SourceType(str, Enum):
    PRIMARY_FACT_SOURCE = "primary_fact_source"
    EVIDENCE_SYNTHESIS = "evidence_synthesis"
    SECONDARY_REPORTING = "secondary_reporting"
    OPINION_ANALYSIS = "opinion_analysis"
    ANONYMOUS_OR_OPAQUE = "anonymous_or_opaque"
    UNKNOWN = "unknown"


class SupportRelation(str, Enum):
    DIRECTLY_SUPPORTS = "directly_supports"
    PARTIALLY_SUPPORTS = "partially_supports"
    SUPPORTS_WEAKER_CLAIM = "supports_weaker_claim"
    ATTRIBUTION_ONLY = "attribution_only"
    OPINION_ONLY = "opinion_only"
    BACKGROUND_ONLY = "background_only"
    NO_SUPPORT = "no_support"
    CONTRADICTS = "contradicts"
    INACCESSIBLE = "inaccessible"
    AUDIT_LIMITED_NO_RELEVANT_SNIPPET = "audit_limited_no_relevant_snippet"
    NOT_CHECKED = "not_checked"


class FinalGroundingBucket(str, Enum):
    HARD_FACT_GROUNDING = "hard_fact_grounding"
    WEAK_FACT_GROUNDING = "weak_fact_grounding"
    ATTRIBUTION_OR_OPINION_GROUNDING = "attribution_or_opinion_grounding"
    UNVERIFIABLE_OR_MISMATCH = "unverifiable_or_mismatch"
    EXCLUDED_OR_CONTEXT = "excluded_or_context"


# Backward-compatible public name used by the original MVP.
GroundingBucket = FinalGroundingBucket


class ClaimReviewCategory(str, Enum):
    HIGH_RISK = "high_risk"
    ATTRIBUTION_SUPPORTED = "attribution_supported"
    AUDIT_LIMITED = "audit_limited"
    FLAGGED_BUT_NOT_HIGH_RISK = "flagged_but_not_high_risk"
    EXCLUDED_OR_CONTEXT = "excluded_or_context"


class DisplayStatus(str, Enum):
    VERIFIED_FACT_SUPPORT = "verified_fact_support"
    PARTIAL_OR_WEAK_SUPPORT = "partial_or_weak_support"
    ATTRIBUTION_SUPPORT = "attribution_support"
    ANALYSIS_FROM_SOURCED_PREMISES = "analysis_from_sourced_premises"
    AUDIT_LIMITED = "audit_limited"
    TRUE_CITATION_PROBLEM = "true_citation_problem"
    EXCLUDED_OR_CONTEXT = "excluded_or_context"


class TerminalClass(str, Enum):
    FACT = "fact"
    OPINION = "opinion"
    UNRESOLVED = "unresolved"
    MISMATCH = "mismatch"


class SourceRole(str, Enum):
    PRIMARY_FACT_SOURCE = "primary_fact_source"
    SECONDARY_REPORT = "secondary_report"
    OPINION_OR_ANALYSIS = "opinion_or_analysis"
    OFFICIAL_ANNOUNCEMENT = "official_announcement"
    ANONYMOUS_REPORTING = "anonymous_reporting"
    UNKNOWN = "unknown"


class OrganizationType(str, Enum):
    COMPANY = "company"
    INSTITUTION = "institution"
    GOVERNMENT = "government"
    NEWS_OR_MEDIA = "news_or_media"
    SCHOLARLY_OR_RESEARCH = "scholarly_or_research"
    NONPROFIT = "nonprofit"
    UNKNOWN = "unknown"


class OfficialnessStatus(str, Enum):
    VERIFIED_FIRST_PARTY = "verified_first_party"
    PROBABLE_FIRST_PARTY = "probable_first_party"
    VERIFIED_AFFILIATED_SOURCE = "verified_affiliated_source"
    THIRD_PARTY = "third_party"
    UNKNOWN = "unknown"


class SourceRoleForClaim(str, Enum):
    OFFICIAL_INSTITUTION_SOURCE = "official_institution_source"
    OFFICIAL_COMPANY_SOURCE = "official_company_source"
    REGULATORY_OR_FILING_SOURCE = "regulatory_or_filing_source"
    SCHOLARLY_PRIMARY_SOURCE = "scholarly_primary_source"
    EVIDENCE_SYNTHESIS_SOURCE = "evidence_synthesis_source"
    SECONDARY_REPORTING_SOURCE = "secondary_reporting_source"
    OPINION_OR_ANALYSIS_SOURCE = "opinion_or_analysis_source"
    ANONYMOUS_OR_OPAQUE_SOURCE = "anonymous_or_opaque_source"
    UNKNOWN_SOURCE = "unknown_source"


class SourceToClaimRelation(str, Enum):
    SAME_ENTITY = "same_entity"
    PARENT_CHILD_ENTITY = "parent_child_entity"
    AFFILIATED_ENTITY = "affiliated_entity"
    OFFICIAL_PARTNER = "official_partner"
    THIRD_PARTY = "third_party"
    UNKNOWN = "unknown"


class SupportScope(str, Enum):
    OWN_INSTITUTIONAL_FACT = "own_institutional_fact"
    OWN_PRODUCT_OR_PROGRAM_FACT = "own_product_or_program_fact"
    OWN_REPORTED_DATA = "own_reported_data"
    OFFICIAL_ANNOUNCEMENT = "official_announcement"
    ATTRIBUTION_ONLY = "attribution_only"
    PREMISE_SUPPORT_FOR_ANALYSIS = "premise_support_for_analysis"
    NOT_SUFFICIENT_FOR_CLAIM = "not_sufficient_for_claim"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    AUTHOR_CITED = "author_cited"
    DISCOVERED_SOURCE = "discovered_source"
    UPSTREAM_SOURCE = "upstream_source"


class EdgeBasis(str, Enum):
    EXPLICIT_LINK = "explicit_link"
    MARKDOWN_CITATION = "markdown_citation"
    FOOTNOTE = "footnote"
    REFERENCE_LIST = "reference_list"
    SOURCE_STATEMENT = "source_statement"
    DISCOVERED_SOURCE = "discovered_source"
    NONE = "none"


class RiskFlag(str, Enum):
    INACCESSIBLE_SOURCE = "inaccessible_source"
    AUDIT_LIMITED_NO_RELEVANT_SNIPPET = "audit_limited_no_relevant_snippet"
    SOURCE_FETCH_FAILED = "source_fetch_failed"
    SOURCE_BODY_MISSING = "source_body_missing"
    VAGUE_SOURCE = "vague_source"
    ANONYMOUS_SOURCE = "anonymous_source"
    NAMED_SECONDARY_WITH_OPAQUE_UNDERLYING = "named_secondary_with_opaque_underlying"
    SOURCE_CLAIM_MISMATCH = "source_claim_mismatch"
    CORRELATION_PRESENTED_AS_CAUSATION = "correlation_presented_as_causation"
    SOURCE_ONLY_SUPPORTS_WEAKER_CLAIM = "source_only_supports_weaker_claim"
    OPINION_USED_AS_FACT = "opinion_used_as_fact"
    QUANTITATIVE_CLAIM_WITHOUT_PRIMARY_DATA = "quantitative_claim_without_primary_data"
    ATTRIBUTION_DROPPED = "attribution_dropped"
    UNSUPPORTED_CAUSAL_OR_STRATEGIC_INFERENCE = "unsupported_causal_or_strategic_inference"
    NOT_ASSERTED_BY_AUTHOR = "not_asserted_by_author"
    # Legacy diagnostic flags kept so older API payloads still validate.
    CAUSAL_OVERCLAIM = "causal_overclaim"
    OUTDATED_SOURCE = "outdated_source"
    SECONDARY_SOURCE_ONLY = "secondary_source_only"
    OVERGENERALIZATION = "overgeneralization"


class ParsedCitation(BaseModel):
    citation_id: str
    raw_text: str
    url: Optional[str] = None
    label: Optional[str] = None
    kind: EdgeBasis = EdgeBasis.EXPLICIT_LINK
    span_start: int = 0
    span_end: int = 0


class CitationUnit(BaseModel):
    cited_text: str
    citation_label: Optional[str] = None
    source_url: Optional[str] = None
    source_title: str = ""
    source_id: Optional[str] = None
    source_registry_entry: str = ""
    char_start: int = 0
    char_end: int = 0


class ProvidedSource(BaseModel):
    """Optional source body supplied by the caller.

    Useful for tests, demos, and private documents where the backend should not fetch a URL.
    If `url` matches a parsed citation URL, this content is used as the source body.
    """

    url: Optional[str] = None
    title: Optional[str] = None
    publisher_or_author: Optional[str] = None
    publication_date: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    extracted_text: str = ""
    access_status: AccessStatus = AccessStatus.ACCESSIBLE


class Source(BaseModel):
    source_id: str
    url: Optional[str] = None
    title: str = ""
    publisher_or_author: str = ""
    publication_date: Optional[str] = None
    access_status: AccessStatus = AccessStatus.UNAVAILABLE
    source_type: SourceType = SourceType.UNKNOWN
    extracted_text_preview: str = ""
    extracted_text: str = Field(default="", exclude=True)
    upstream_source_ids: List[str] = Field(default_factory=list)
    source_entity: str = ""
    registrable_domain: str = ""
    publisher_name: str = ""
    organization_type: OrganizationType = OrganizationType.UNKNOWN
    entity_aliases: List[str] = Field(default_factory=list)
    metadata_basis: List[str] = Field(default_factory=list)
    officialness_status: OfficialnessStatus = OfficialnessStatus.UNKNOWN
    officialness_basis: List[str] = Field(default_factory=list)


class EvidenceEdge(BaseModel):
    claim_id: str
    source_id: Optional[str] = None
    edge_type: EdgeType = EdgeType.AUTHOR_CITED
    basis: EdgeBasis = EdgeBasis.NONE
    support_relation: SupportRelation = SupportRelation.NOT_CHECKED
    evidence_span: str = ""
    evidence_quote: str = ""
    reasoning_summary: str = ""
    final_bucket: FinalGroundingBucket = FinalGroundingBucket.UNVERIFIABLE_OR_MISMATCH
    source_role: SourceRole = SourceRole.UNKNOWN
    upstream_source_ids: List[str] = Field(default_factory=list)
    source_role_for_claim: SourceRoleForClaim = SourceRoleForClaim.UNKNOWN_SOURCE
    source_to_claim_relation: SourceToClaimRelation = SourceToClaimRelation.UNKNOWN
    support_scope: SupportScope = SupportScope.UNKNOWN
    source_role_basis: List[str] = Field(default_factory=list)


class Claim(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    claim_id: str
    original_text_span: str
    original_span: str = ""
    normalized_claim: str
    claim_type: ClaimType
    discourse_role: DiscourseRole = DiscourseRole.ASSERTED_CLAIM
    source_opacity: SourceOpacity = SourceOpacity.NOT_APPLICABLE
    has_quantitative_data: bool = False
    has_material_quantitative_data: bool = False
    source_mentions: List[str] = Field(default_factory=list)
    importance_label: ImportanceLabel = ImportanceLabel.SUPPORTING
    attributed_to: Optional[str] = None
    linked_source_ids: List[str] = Field(default_factory=list)
    final_bucket: Optional[FinalGroundingBucket] = None
    support_relation: Optional[SupportRelation] = SupportRelation.NOT_CHECKED
    review_category: ClaimReviewCategory = ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    evidence_chain: List[EvidenceEdge] = Field(default_factory=list)
    reasoning_summary: str = ""
    evidence_needed: List[str] = Field(default_factory=list)
    not_asserted_by_author: bool = False
    evidence_quote: str = ""
    source_role: SourceRole = SourceRole.UNKNOWN
    source_role_for_claim: SourceRoleForClaim = SourceRoleForClaim.UNKNOWN_SOURCE
    source_to_claim_relation: SourceToClaimRelation = SourceToClaimRelation.UNKNOWN
    support_scope: SupportScope = SupportScope.UNKNOWN
    source_role_basis: List[str] = Field(default_factory=list)
    citation_label: Optional[str] = None
    citation_source_url: Optional[str] = None
    citation_source_title: str = ""
    citation_source_id: Optional[str] = None
    source_registry_entry: str = ""

    @model_validator(mode="before")
    @classmethod
    def _sync_span_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if not data.get("original_text_span") and data.get("original_span"):
            data["original_text_span"] = data["original_span"]
        if not data.get("original_span") and data.get("original_text_span"):
            data["original_span"] = data["original_text_span"]
        if data.get("importance_label") == "background":
            data["importance_label"] = "minor"
        return data

    @model_validator(mode="after")
    def _default_contextual_fields(self) -> "Claim":
        if not self.original_span:
            self.original_span = self.original_text_span
        if self.not_asserted_by_author and RiskFlag.NOT_ASSERTED_BY_AUTHOR not in self.risk_flags:
            self.risk_flags.append(RiskFlag.NOT_ASSERTED_BY_AUTHOR)
        return self


class ContentMix(BaseModel):
    factual: float = 0.0
    attribution: float = 0.0
    judgment: float = 0.0
    has_quantitative_data: float = 0.0


class GroundingMix(BaseModel):
    hard_fact_grounding: float = 0.0
    weak_fact_grounding: float = 0.0
    attribution_or_opinion_grounding: float = 0.0
    unverifiable_or_mismatch: float = 0.0
    excluded_or_context: float = 0.0


class SupportRelationMix(BaseModel):
    directly_supports: float = 0.0
    partially_supports: float = 0.0
    supports_weaker_claim: float = 0.0
    attribution_only: float = 0.0
    opinion_only: float = 0.0
    background_only: float = 0.0
    no_support: float = 0.0
    contradicts: float = 0.0
    inaccessible: float = 0.0
    audit_limited_no_relevant_snippet: float = 0.0
    not_checked: float = 0.0


class KeyRates(BaseModel):
    verified_fact_support_rate: float = 0.0
    partial_or_weak_support_rate: float = 0.0
    attribution_support_rate: float = 0.0
    analysis_from_sourced_premises_rate: float = 0.0
    audit_limited_rate: float = 0.0
    true_mismatch_rate: float = 0.0
    public_fact_support_rate: float = 0.0
    loose_fact_support_rate: float = 0.0
    opinion_packaging_rate: float = 0.0
    source_opacity_rate: float = 0.0
    citation_mismatch_rate: float = 0.0
    premise_support_for_analysis_rate: float = 0.0
    official_fact_support_rate: float = 0.0


class AnalysisSummary(BaseModel):
    total_claims: int = 0
    auditable_claims: int = 0
    non_claim_items: int = 0
    ratios_basis: str = "based only on cited claims"
    content_mix: ContentMix = Field(default_factory=ContentMix)
    grounding_mix: GroundingMix = Field(default_factory=GroundingMix)
    support_relation_mix: SupportRelationMix = Field(default_factory=SupportRelationMix)
    key_rates: KeyRates = Field(default_factory=KeyRates)


class AnalysisRequest(BaseModel):
    input_text: str = Field(..., min_length=1)
    original_question: Optional[str] = None
    mode: str = "ai_answer_or_article"
    uncited_claim_analysis_enabled: bool = False
    claim_extraction_mode: Optional[ClaimExtractionMode] = None
    max_upstream_depth: int = Field(default=2, ge=0, le=3)
    enable_url_fetch: bool = True
    enable_web_search: bool = True
    max_search_results: int = Field(default=2, ge=1, le=5)
    max_terminal_trace_depth: int = Field(default=2, ge=0, le=5)
    provided_sources: List[ProvidedSource] = Field(default_factory=list)


class ClaimReviewItem(BaseModel):
    claim_id: str
    normalized_claim: str
    category: ClaimReviewCategory = ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK
    risk_flags: List[RiskFlag]
    final_bucket: Optional[FinalGroundingBucket]
    support_relation: Optional[SupportRelation]
    discourse_role: DiscourseRole = DiscourseRole.ASSERTED_CLAIM
    source_opacity: SourceOpacity = SourceOpacity.NOT_APPLICABLE
    importance_label: ImportanceLabel = ImportanceLabel.SUPPORTING
    explanation: str = ""


class DisplayCitationResult(BaseModel):
    claim_id: str
    display_claim_text: str = ""
    display_status: DisplayStatus = DisplayStatus.AUDIT_LIMITED
    display_label: str = ""
    display_explanation: str = ""
    severity: str = "info"
    primary_reason: str = ""
    debug_tags: List[str] = Field(default_factory=list)
    should_show_in_problematic: bool = False
    should_count_as_true_mismatch: bool = False


class EvidenceGraphNode(BaseModel):
    id: str
    type: str
    label: str
    subtitle: str = ""
    status: str = ""
    source_id: Optional[str] = None
    claim_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceGraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str = ""
    relation: str = ""
    status: str = ""
    basis: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceGraph(BaseModel):
    graph_id: str
    claim_id: str
    nodes: List[EvidenceGraphNode] = Field(default_factory=list)
    edges: List[EvidenceGraphEdge] = Field(default_factory=list)


class CitationTerminalResult(BaseModel):
    citation_id: str
    cited_text: str = ""
    citation_label: Optional[str] = None
    source_title: str = ""
    source_url: Optional[str] = None
    terminal_class: TerminalClass = TerminalClass.UNRESOLVED
    terminal_reason: str = ""
    path_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    path_edges: List[Dict[str, Any]] = Field(default_factory=list)
    depth: int = 0
    short_explanation: str = ""
    debug_claim_ids: List[str] = Field(default_factory=list)
    debug_tags: List[str] = Field(default_factory=list)


class DocumentEvidenceSummary(BaseModel):
    total_cited_statements: int = 0
    fact_terminal_count: int = 0
    opinion_terminal_count: int = 0
    unresolved_terminal_count: int = 0
    mismatch_count: int = 0
    fact_terminal_ratio: float = 0.0
    opinion_terminal_ratio: float = 0.0
    unresolved_ratio: float = 0.0


class DocumentEvidenceGraphNode(BaseModel):
    id: str
    type: str
    label: str
    count: int = 0
    terminal_class: Optional[TerminalClass] = None
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentEvidenceGraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str
    label: str = ""
    count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentEvidenceGraph(BaseModel):
    graph_id: str = "document-evidence-graph"
    nodes: List[DocumentEvidenceGraphNode] = Field(default_factory=list)
    edges: List[DocumentEvidenceGraphEdge] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    analysis_id: str
    summary: AnalysisSummary
    claims: List[Claim]
    sources: List[Source] = Field(default_factory=list)
    display_citations: List[DisplayCitationResult] = Field(default_factory=list)
    evidence_graphs: List[EvidenceGraph] = Field(default_factory=list)
    citation_terminal_results: List[CitationTerminalResult] = Field(default_factory=list)
    document_evidence_summary: DocumentEvidenceSummary = Field(default_factory=DocumentEvidenceSummary)
    document_evidence_graph: DocumentEvidenceGraph = Field(default_factory=DocumentEvidenceGraph)
    problematic_citations: List[ClaimReviewItem] = Field(default_factory=list)
    audit_limited_citations: List[ClaimReviewItem] = Field(default_factory=list)
    attribution_supported_citations: List[ClaimReviewItem] = Field(default_factory=list)
    flagged_citations: List[ClaimReviewItem] = Field(default_factory=list)
    excluded_or_context_citations: List[ClaimReviewItem] = Field(default_factory=list)
    uncited_claim_analysis_enabled: bool = False
    # Deprecated compatibility mirrors. New clients should use the *_citations fields above.
    high_risk_claims: List[ClaimReviewItem] = Field(default_factory=list)
    flagged_claims: List[ClaimReviewItem] = Field(default_factory=list)
    audit_limited_claims: List[ClaimReviewItem] = Field(default_factory=list)
    attribution_supported_claims: List[ClaimReviewItem] = Field(default_factory=list)
    excluded_or_context_claims: List[ClaimReviewItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Backward-compatible import name.
HighRiskClaim = ClaimReviewItem
