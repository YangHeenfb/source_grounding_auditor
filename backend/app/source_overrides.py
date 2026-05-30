from __future__ import annotations

from dataclasses import dataclass

from .schemas import OfficialnessStatus, OrganizationType, SourceType


@dataclass(frozen=True)
class KnownSourceOverride:
    entity: str
    aliases: tuple[str, ...]
    organization_type: OrganizationType
    source_type: SourceType
    officialness_status: OfficialnessStatus = OfficialnessStatus.VERIFIED_FIRST_PARTY


KNOWN_SOURCE_OVERRIDES: dict[str, KnownSourceOverride] = {
    "dukekunshan.edu.cn": KnownSourceOverride(
        entity="Duke Kunshan University",
        aliases=("DKU", "Duke Kunshan", "Duke Kunshan University", "昆山杜克", "昆山杜克大学"),
        organization_type=OrganizationType.INSTITUTION,
        source_type=SourceType.PRIMARY_FACT_SOURCE,
    ),
    "cuhk.edu.cn": KnownSourceOverride(
        entity="The Chinese University of Hong Kong, Shenzhen",
        aliases=("CUHK-Shenzhen", "CUHK Shenzhen", "港中深", "香港中文大学（深圳）", "香港中文大学深圳"),
        organization_type=OrganizationType.INSTITUTION,
        source_type=SourceType.PRIMARY_FACT_SOURCE,
    ),
    "reuters.com": KnownSourceOverride(
        entity="Reuters",
        aliases=("Reuters", "路透", "路透社"),
        organization_type=OrganizationType.NEWS_OR_MEDIA,
        source_type=SourceType.SECONDARY_REPORTING,
    ),
}


def override_for_domain(registrable_domain: str) -> KnownSourceOverride | None:
    domain = (registrable_domain or "").lower().strip()
    return KNOWN_SOURCE_OVERRIDES.get(domain)
