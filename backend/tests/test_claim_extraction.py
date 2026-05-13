from app.claim_extractor import RuleBasedClaimExtractor
from app.schemas import ClaimType


def test_extracts_atomic_support_claims():
    extractor = RuleBasedClaimExtractor()
    claims = extractor.extract_claims("OpenAI released GPT-4o in 2024, and GPT-4o supports text, audio, and image inputs.")
    texts = [c.normalized_claim for c in claims]
    assert any("released GPT-4o" in t for t in texts)
    assert any("supports text inputs" in t for t in texts)
    assert any("supports audio inputs" in t for t in texts)
    assert any("supports image inputs" in t for t in texts)
    assert any(c.has_quantitative_data for c in claims)


def test_experts_say_is_judgment_with_vague_source():
    extractor = RuleBasedClaimExtractor()
    claim = extractor.extract_claims("Experts say this policy will damage the middle class.")[0]
    assert claim.claim_type == ClaimType.ATTRIBUTION or claim.claim_type == ClaimType.JUDGMENT
    assert claim.source_mentions


def test_market_commentary_is_judgment_or_attribution():
    extractor = RuleBasedClaimExtractor()
    claim = extractor.extract_claims("A market commentary article shows that the company is guaranteed to dominate AI infrastructure.")[0]
    assert claim.claim_type in {ClaimType.ATTRIBUTION, ClaimType.JUDGMENT}
    assert claim.importance_label.value in {"core", "supporting"}
