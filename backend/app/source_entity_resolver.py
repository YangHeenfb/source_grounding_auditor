from __future__ import annotations

import html
import json
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel, Field

from .schemas import OrganizationType
from .source_overrides import override_for_domain

TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "msclkid",
    "ref",
    "ref_src",
    "spm",
}


class SourceEntityResolution(BaseModel):
    source_entity: str = ""
    registrable_domain: str = ""
    publisher_name: str = ""
    organization_type: OrganizationType = OrganizationType.UNKNOWN
    entity_aliases: list[str] = Field(default_factory=list)
    metadata_basis: list[str] = Field(default_factory=list)


def clean_tracking_query(url: str | None) -> str | None:
    if not url:
        return url
    parsed = urlparse(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_QUERY_KEYS
    ]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def registrable_domain_for_url(url: str | None) -> str:
    if not url:
        return ""
    hostname = urlparse(clean_tracking_query(url) or url).hostname or ""
    hostname = hostname.lower().strip(".")
    if not hostname:
        return ""
    try:
        import tldextract

        extracted = tldextract.TLDExtract(suffix_list_urls=())(hostname)
        return ".".join(part for part in [extracted.domain, extracted.suffix] if part)
    except Exception:
        labels = hostname.split(".")
        if len(labels) >= 3 and ".".join(labels[-2:]) in {"edu.cn", "ac.cn", "gov.cn", "com.cn", "org.cn"}:
            return ".".join(labels[-3:])
        return ".".join(labels[-2:]) if len(labels) >= 2 else hostname


def extract_html_metadata(raw_html: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not raw_html:
        return metadata

    for match in re.finditer(r"<meta\s+[^>]*>", raw_html, flags=re.IGNORECASE | re.DOTALL):
        tag = match.group(0)
        key_match = re.search(r'(?:property|name)=["\']([^"\']+)["\']', tag, flags=re.IGNORECASE)
        content_match = re.search(r'content=["\']([^"\']*)["\']', tag, flags=re.IGNORECASE | re.DOTALL)
        if key_match and content_match:
            metadata[key_match.group(1).lower()] = html.unescape(content_match.group(1)).strip()

    organizations: list[str] = []
    publishers: list[str] = []
    for match in re.finditer(
        r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        raw = html.unescape(match.group(1)).strip()
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        _collect_jsonld_entities(parsed, organizations, publishers)

    if organizations:
        metadata["jsonld_organizations"] = organizations
    if publishers:
        metadata["jsonld_publishers"] = publishers
    return metadata


def resolve_source_entity(
    *,
    url: str | None,
    title: str = "",
    html_metadata: dict[str, Any] | None = None,
    source_text: str = "",
    publisher_or_author: str = "",
) -> SourceEntityResolution:
    metadata = html_metadata or {}
    clean_url = clean_tracking_query(url)
    registrable_domain = registrable_domain_for_url(clean_url)
    basis: list[str] = []
    candidates: list[tuple[str, str]] = []

    override = override_for_domain(registrable_domain)
    if override:
        candidates.append((override.entity, f"known_source_overrides matched {registrable_domain}"))

    for key in ["og:site_name", "application-name", "twitter:site", "publisher", "author"]:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append((value.strip().lstrip("@"), f"html metadata {key}"))

    for key in ["jsonld_organizations", "jsonld_publishers"]:
        values = metadata.get(key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value.strip():
                    candidates.append((value.strip(), f"JSON-LD {key}"))

    if publisher_or_author.strip():
        candidates.append((publisher_or_author.strip(), "provided publisher_or_author"))

    title_candidate = _entity_from_title(title)
    if title_candidate:
        candidates.append((title_candidate, "page title pattern"))

    copyright_candidate = _entity_from_copyright(source_text)
    if copyright_candidate:
        candidates.append((copyright_candidate, "footer copyright text"))

    domain_candidate = _entity_from_domain(registrable_domain)
    if domain_candidate:
        candidates.append((domain_candidate, "registrable domain label"))

    source_entity = ""
    for candidate, candidate_basis in candidates:
        cleaned = _clean_entity_name(candidate)
        if cleaned:
            source_entity = cleaned
            basis.append(candidate_basis)
            break

    if not source_entity and domain_candidate:
        source_entity = domain_candidate

    aliases = _aliases_for_entity(source_entity, registrable_domain)
    if override:
        aliases.extend(override.aliases)
    aliases = _dedupe([alias for alias in aliases if alias])
    organization_type = override.organization_type if override else _infer_organization_type(source_entity, registrable_domain)
    publisher_name = source_entity

    if not basis and source_entity:
        basis.append("inferred from domain/title fallback")
    if registrable_domain:
        basis.append(f"registrable_domain={registrable_domain}")

    return SourceEntityResolution(
        source_entity=source_entity,
        registrable_domain=registrable_domain,
        publisher_name=publisher_name,
        organization_type=organization_type,
        entity_aliases=aliases,
        metadata_basis=basis,
    )


def _collect_jsonld_entities(value: Any, organizations: list[str], publishers: list[str]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_jsonld_entities(item, organizations, publishers)
        return
    if not isinstance(value, dict):
        return
    value_type = value.get("@type")
    types = value_type if isinstance(value_type, list) else [value_type]
    name = value.get("name")
    if isinstance(name, str) and any(t in {"Organization", "CollegeOrUniversity", "NewsMediaOrganization"} for t in types):
        organizations.append(name)
    publisher = value.get("publisher")
    if isinstance(publisher, dict) and isinstance(publisher.get("name"), str):
        publishers.append(publisher["name"])
    elif isinstance(publisher, str):
        publishers.append(publisher)
    for child in value.values():
        if isinstance(child, (dict, list)):
            _collect_jsonld_entities(child, organizations, publishers)


def _entity_from_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title or "").strip()
    if not title:
        return ""
    for sep in [" | ", " - ", " — ", " – ", "::"]:
        if sep in title:
            pieces = [piece.strip() for piece in title.split(sep) if piece.strip()]
            if pieces:
                return pieces[-1] if len(pieces[-1]) >= 3 else pieces[0]
    return title if len(title) <= 80 else ""


def _entity_from_copyright(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"(?:©|Copyright)\s*(?:\d{4})?\s*([^.|,;]{3,80})", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _entity_from_domain(domain: str) -> str:
    if not domain:
        return ""
    label = domain.split(".")[0]
    return " ".join(part.capitalize() for part in re.split(r"[-_]+", label) if part)


def _clean_entity_name(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"\s+", " ", value).strip(" -–—|:·")
    value = re.sub(r"^welcome to\s+", "", value, flags=re.IGNORECASE)
    return value[:120]


def _aliases_for_entity(entity: str, domain: str) -> list[str]:
    aliases = [entity]
    if domain:
        aliases.append(domain.split(".")[0])
    if entity:
        acronym = "".join(word[0] for word in re.findall(r"[A-Z][A-Za-z]+", entity))
        if len(acronym) >= 2:
            aliases.append(acronym)
    return aliases


def _infer_organization_type(entity: str, domain: str) -> OrganizationType:
    haystack = f"{entity} {domain}".lower()
    if any(term in haystack for term in ["university", "college", "school", "institute", ".edu", "edu."]):
        return OrganizationType.INSTITUTION
    if any(term in haystack for term in ["reuters", "news", "times", "post", "journal", "media"]):
        return OrganizationType.NEWS_OR_MEDIA
    if any(term in haystack for term in [".gov", "government", "ministry", "department"]):
        return OrganizationType.GOVERNMENT
    if any(term in haystack for term in ["doi", "pubmed", "journal", "nature", "science"]):
        return OrganizationType.SCHOLARLY_OR_RESEARCH
    if any(term in haystack for term in ["foundation", "nonprofit", "ngo"]):
        return OrganizationType.NONPROFIT
    if entity:
        return OrganizationType.COMPANY
    return OrganizationType.UNKNOWN


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped
