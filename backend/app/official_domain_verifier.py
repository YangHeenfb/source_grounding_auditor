from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .schemas import OfficialnessStatus, SourceType
from .source_entity_resolver import SourceEntityResolution
from .source_overrides import override_for_domain


class OfficialDomainVerificationResult(BaseModel):
    officialness_status: OfficialnessStatus = OfficialnessStatus.UNKNOWN
    basis: list[str] = Field(default_factory=list)


def verify_official_domain(
    resolution: SourceEntityResolution,
    metadata: dict[str, Any] | None = None,
) -> OfficialDomainVerificationResult:
    metadata = metadata or {}
    domain = resolution.registrable_domain
    basis: list[str] = []

    override = override_for_domain(domain)
    if override:
        return OfficialDomainVerificationResult(
            officialness_status=override.officialness_status,
            basis=[f"known_source_overrides matched {domain} as {override.officialness_status.value}"],
        )

    provided_type = metadata.get("provided_source_type")
    if provided_type in {
        SourceType.PRIMARY_FACT_SOURCE.value,
        SourceType.EVIDENCE_SYNTHESIS.value,
        "official",
        "official_source",
        "primary",
    } and resolution.source_entity:
        return OfficialDomainVerificationResult(
            officialness_status=OfficialnessStatus.VERIFIED_FIRST_PARTY,
            basis=[f"provided_sources marked source_type={provided_type} for {resolution.source_entity}"],
        )

    if resolution.source_entity and domain:
        aliases = [resolution.source_entity, *resolution.entity_aliases]
        if any(_domain_matches_alias(domain, alias) for alias in aliases):
            basis.append("source entity was extracted from page metadata/title and matches registrable domain")
            if resolution.metadata_basis:
                basis.extend(resolution.metadata_basis[:3])
            return OfficialDomainVerificationResult(
                officialness_status=OfficialnessStatus.PROBABLE_FIRST_PARTY,
                basis=basis,
            )

    if domain:
        basis.append(f"registrable_domain={domain} did not verify against extracted source entity")
    else:
        basis.append("no registrable domain available")
    return OfficialDomainVerificationResult(
        officialness_status=OfficialnessStatus.UNKNOWN,
        basis=basis,
    )


def _domain_matches_alias(domain: str, alias: str) -> bool:
    if not alias:
        return False
    domain_label = domain.split(".")[0].lower()
    alias_tokens = re.findall(r"[a-z0-9]+", alias.lower())
    if not alias_tokens:
        return False
    joined = "".join(alias_tokens)
    acronym = "".join(token[0] for token in alias_tokens if token)
    return domain_label == joined or domain_label == acronym or domain_label in alias_tokens or joined in domain_label
