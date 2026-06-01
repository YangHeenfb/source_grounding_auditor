from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .official_domain_verifier import OfficialDomainVerificationResult
from .schemas import (
    Claim,
    ClaimType,
    DiscourseRole,
    OfficialnessStatus,
    OrganizationType,
    Source,
    SourceRoleForClaim,
    SourceToClaimRelation,
    SourceType,
    SupportRelation,
    SupportScope,
)
from .source_entity_resolver import SourceEntityResolution

ANALYSIS_TERMS = [
    "更适合",
    "更强",
    "硬实力",
    "适合",
    "建议",
    "优先",
    "偏好",
    "值得",
    "战略",
    "说明",
    "意味着",
    "影响",
    "长期持有",
    "会让",
    "以为",
    "看起来",
    "集中度",
    "风险",
    "买太多",
    "整体",
    "better",
    "stronger",
    "suitable",
    "recommend",
    "should",
    "prefer",
    "strategy",
    "impact",
    "risk",
    "concentration",
]
INSTITUTION_FACT_TERMS = [
    "师生比",
    "学生教师比",
    "class size",
    "班级",
    "课程",
    "course",
    "program",
    "专业",
    "学位",
    "degree",
    "招生",
    "admission",
    "就业",
    "employment",
    "career",
    "质量报告",
    "report",
    "airs",
    "institute",
    "学院",
    "school",
]
REPORTED_DATA_TERMS = ["%", "％", "数据", "报告", "就业", "employment", "quality report", "survey"]
PROGRAM_TERMS = ["课程", "course", "program", "专业", "major", "学位", "degree", "airs", "institute", "学院", "school"]
ANNOUNCEMENT_TERMS = ["公告", "announcement", "press release", "officially announced", "发布"]
OFFICIAL_CONFIRMATION_TERMS = ["官方确认", "officially confirmed", "confirmed by", "正式确认"]


class ClaimAwareSourceRoleResult(BaseModel):
    source_role_for_claim: SourceRoleForClaim = SourceRoleForClaim.UNKNOWN_SOURCE
    source_to_claim_relation: SourceToClaimRelation = SourceToClaimRelation.UNKNOWN
    support_scope: SupportScope = SupportScope.UNKNOWN
    basis: list[str] = Field(default_factory=list)


def classify_claim_source_role(
    *,
    claim: Claim,
    source: Source | None,
    source_entity_resolution: SourceEntityResolution,
    officialness_result: OfficialDomainVerificationResult,
    support_relation: SupportRelation,
) -> ClaimAwareSourceRoleResult:
    if source is None:
        return ClaimAwareSourceRoleResult(basis=["no source available for claim-aware source role classification"])

    claim_text = _norm(claim.normalized_claim or claim.original_text_span)
    domain = source_entity_resolution.registrable_domain
    entity = source_entity_resolution.source_entity
    aliases = [entity, *source_entity_resolution.entity_aliases]
    basis = [*source_entity_resolution.metadata_basis[:3], *officialness_result.basis[:3]]

    if source.source_type == SourceType.ANONYMOUS_OR_OPAQUE:
        return ClaimAwareSourceRoleResult(
            source_role_for_claim=SourceRoleForClaim.ANONYMOUS_OR_OPAQUE_SOURCE,
            source_to_claim_relation=SourceToClaimRelation.UNKNOWN,
            support_scope=SupportScope.UNKNOWN,
            basis=[*basis, "source_type is anonymous_or_opaque"],
        )

    if source.source_type == SourceType.OPINION_ANALYSIS:
        return ClaimAwareSourceRoleResult(
            source_role_for_claim=SourceRoleForClaim.OPINION_OR_ANALYSIS_SOURCE,
            source_to_claim_relation=SourceToClaimRelation.THIRD_PARTY,
            support_scope=SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS
            if _is_analysis_claim(claim, claim_text)
            else SupportScope.NOT_SUFFICIENT_FOR_CLAIM,
            basis=[*basis, "source_type is opinion_analysis"],
        )

    if source.source_type == SourceType.EVIDENCE_SYNTHESIS:
        return ClaimAwareSourceRoleResult(
            source_role_for_claim=SourceRoleForClaim.EVIDENCE_SYNTHESIS_SOURCE,
            source_to_claim_relation=SourceToClaimRelation.THIRD_PARTY,
            support_scope=SupportScope.OWN_REPORTED_DATA,
            basis=[*basis, "source_type is evidence_synthesis"],
        )

    if _is_reuters_source(domain, entity, aliases):
        if claim.claim_type == ClaimType.ATTRIBUTION or "reuters" in claim_text or "路透" in claim_text:
            return ClaimAwareSourceRoleResult(
                source_role_for_claim=SourceRoleForClaim.SECONDARY_REPORTING_SOURCE,
                source_to_claim_relation=SourceToClaimRelation.SAME_ENTITY,
                support_scope=SupportScope.ATTRIBUTION_ONLY,
                basis=[*basis, "Reuters source is evaluated as support for Reuters attribution only"],
            )
        if _contains_any(claim_text, OFFICIAL_CONFIRMATION_TERMS):
            return ClaimAwareSourceRoleResult(
                source_role_for_claim=SourceRoleForClaim.SECONDARY_REPORTING_SOURCE,
                source_to_claim_relation=SourceToClaimRelation.THIRD_PARTY,
                support_scope=SupportScope.NOT_SUFFICIENT_FOR_CLAIM,
                basis=[*basis, "Reuters is not first-party confirmation for another entity's official fact"],
            )
        return ClaimAwareSourceRoleResult(
            source_role_for_claim=SourceRoleForClaim.SECONDARY_REPORTING_SOURCE,
            source_to_claim_relation=SourceToClaimRelation.THIRD_PARTY,
            support_scope=SupportScope.UNKNOWN,
            basis=[*basis, "Reuters is secondary reporting for non-attribution claims"],
        )

    if _is_regulatory_or_filing_source(source, domain, claim_text):
        return ClaimAwareSourceRoleResult(
            source_role_for_claim=SourceRoleForClaim.REGULATORY_OR_FILING_SOURCE,
            source_to_claim_relation=SourceToClaimRelation.THIRD_PARTY,
            support_scope=SupportScope.OWN_REPORTED_DATA,
            basis=[*basis, "domain or text indicates regulatory/filing source"],
        )

    if _is_scholarly_source(source, domain):
        return ClaimAwareSourceRoleResult(
            source_role_for_claim=SourceRoleForClaim.SCHOLARLY_PRIMARY_SOURCE,
            source_to_claim_relation=SourceToClaimRelation.THIRD_PARTY,
            support_scope=SupportScope.OWN_REPORTED_DATA,
            basis=[*basis, "domain/source_type indicates scholarly primary source"],
        )

    if officialness_result.officialness_status in {
        OfficialnessStatus.VERIFIED_FIRST_PARTY,
        OfficialnessStatus.PROBABLE_FIRST_PARTY,
        OfficialnessStatus.VERIFIED_AFFILIATED_SOURCE,
    }:
        role = (
            SourceRoleForClaim.OFFICIAL_INSTITUTION_SOURCE
            if source_entity_resolution.organization_type == OrganizationType.INSTITUTION
            else SourceRoleForClaim.OFFICIAL_COMPANY_SOURCE
        )
        relation = SourceToClaimRelation.SAME_ENTITY if _claim_mentions_alias(claim_text, aliases) else SourceToClaimRelation.UNKNOWN

        if _is_analysis_claim(claim, claim_text):
            return ClaimAwareSourceRoleResult(
                source_role_for_claim=role,
                source_to_claim_relation=relation,
                support_scope=SupportScope.PREMISE_SUPPORT_FOR_ANALYSIS,
                basis=[*basis, "official source supplies premises for an analysis/judgment claim, not direct fact support"],
            )

        if _claim_mentions_alias(claim_text, aliases) or _contains_any(claim_text, INSTITUTION_FACT_TERMS):
            return ClaimAwareSourceRoleResult(
                source_role_for_claim=role,
                source_to_claim_relation=relation,
                support_scope=_official_fact_scope(claim_text),
                basis=[*basis, "official source is first-party for its own institutional/company facts"],
            )

        return ClaimAwareSourceRoleResult(
            source_role_for_claim=role,
            source_to_claim_relation=SourceToClaimRelation.THIRD_PARTY,
            support_scope=SupportScope.NOT_SUFFICIENT_FOR_CLAIM,
            basis=[*basis, "official source does not appear to be first-party for this claim subject"],
        )

    if source.source_type == SourceType.SECONDARY_REPORTING:
        return ClaimAwareSourceRoleResult(
            source_role_for_claim=SourceRoleForClaim.SECONDARY_REPORTING_SOURCE,
            source_to_claim_relation=SourceToClaimRelation.THIRD_PARTY,
            support_scope=SupportScope.ATTRIBUTION_ONLY if claim.claim_type == ClaimType.ATTRIBUTION else SupportScope.UNKNOWN,
            basis=[*basis, "source_type is secondary_reporting"],
        )

    return ClaimAwareSourceRoleResult(
        source_role_for_claim=SourceRoleForClaim.UNKNOWN_SOURCE,
        source_to_claim_relation=SourceToClaimRelation.UNKNOWN,
        support_scope=SupportScope.UNKNOWN,
        basis=basis or ["no claim-aware source role rule matched"],
    )


def _official_fact_scope(claim_text: str) -> SupportScope:
    if _contains_any(claim_text, ANNOUNCEMENT_TERMS):
        return SupportScope.OFFICIAL_ANNOUNCEMENT
    if _contains_any(claim_text, REPORTED_DATA_TERMS) or re.search(r"\d+(?:\.\d+)?\s*[%％]", claim_text):
        return SupportScope.OWN_REPORTED_DATA
    if _contains_any(claim_text, ["学位", "degree", "校友", "alumni", "毕业", "graduation"]):
        return SupportScope.OWN_INSTITUTIONAL_FACT
    if _contains_any(claim_text, PROGRAM_TERMS):
        return SupportScope.OWN_PRODUCT_OR_PROGRAM_FACT
    return SupportScope.OWN_INSTITUTIONAL_FACT


def _is_analysis_claim(claim: Claim, claim_text: str) -> bool:
    return (
        claim.claim_type == ClaimType.JUDGMENT
        or claim.discourse_role == DiscourseRole.JUDGMENT_OR_ANALYSIS
        or _contains_any(claim_text, ANALYSIS_TERMS)
    )


def _claim_mentions_alias(claim_text: str, aliases: list[str]) -> bool:
    return any(_norm(alias) and _norm(alias) in claim_text for alias in aliases)


def _is_reuters_source(domain: str, entity: str, aliases: list[str]) -> bool:
    haystack = _norm(" ".join([domain, entity, *aliases]))
    return "reuters" in haystack or "路透" in haystack


def _is_regulatory_or_filing_source(source: Source, domain: str, claim_text: str) -> bool:
    haystack = _norm(" ".join([domain, source.title, claim_text]))
    return any(term in haystack for term in ["sec.gov", ".gov", "filing", "10-k", "10-q", "regulation"])


def _is_scholarly_source(source: Source, domain: str) -> bool:
    haystack = _norm(" ".join([domain, source.title]))
    return source.source_type == SourceType.PRIMARY_FACT_SOURCE and any(
        term in haystack for term in ["doi.org", "pubmed", "ncbi", "journal", "nature.com", "science.org", "arxiv"]
    )


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(_norm(term) in text for term in terms)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").lower()).strip()
