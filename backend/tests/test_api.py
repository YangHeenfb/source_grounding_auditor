import asyncio
import time

from fastapi.testclient import TestClient

import app.main as main_module
from app.analyzer import SourceGroundingAnalyzer
from app.providers.llm_provider import AnalysisCancelledError, MockLLMProvider
from app.providers.search_provider import SearchResult
from app.schemas import (
    AnalysisRequest,
    Claim,
    ClaimReviewCategory,
    ClaimType,
    DiscourseRole,
    FinalGroundingBucket,
    ImportanceLabel,
    SourceOpacity,
    SourceRole,
    SupportRelation,
)

app = main_module.app


def fake_provider_for_revenue_claim(span):
    claim = Claim(
        claim_id="c001",
        original_text_span=span,
        normalized_claim="The company reported revenue of $10 billion in its 2024 annual report",
        claim_type=ClaimType.FACTUAL,
        discourse_role=DiscourseRole.ASSERTED_CLAIM,
        source_opacity=SourceOpacity.CLEAR_NAMED_SOURCE,
        has_quantitative_data=True,
        has_material_quantitative_data=True,
        importance_label=ImportanceLabel.SUPPORTING,
        review_category=ClaimReviewCategory.FLAGGED_BUT_NOT_HIGH_RISK,
    )
    return MockLLMProvider(
        extraction_outputs=[claim],
        support_outputs={
            "c001": {
                "support_relation": SupportRelation.DIRECTLY_SUPPORTS.value,
                "final_bucket": FinalGroundingBucket.HARD_FACT_GROUNDING.value,
                "risk_flags": [],
                "reasoning_summary": "The source directly states the revenue claim.",
                "evidence_quote": "The company reported revenue of $10 billion in its 2024 annual report.",
                "source_role": SourceRole.PRIMARY_FACT_SOURCE.value,
            }
        },
    )


def test_health():
    client = TestClient(app)
    assert client.get('/health').json()['status'] == 'ok'


def test_analyze_with_provided_source(monkeypatch):
    span = "The company reported revenue of $10 billion in its 2024 annual report [source](https://example.com/ar)."
    analyzer = SourceGroundingAnalyzer(llm_provider=fake_provider_for_revenue_claim(span))
    monkeypatch.setattr(main_module, "analyzer", analyzer)
    client = TestClient(app)
    payload = {
        "input_text": "The company reported revenue of $10 billion in its 2024 annual report [source](https://example.com/ar).",
        "provided_sources": [
            {
                "url": "https://example.com/ar",
                "title": "2024 annual report",
                "source_type": "primary_fact_source",
                "extracted_text": "The company reported revenue of $10 billion in its 2024 annual report.",
                "access_status": "accessible"
            }
        ]
    }
    data = client.post('/analyze', json=payload).json()
    assert data['summary']['auditable_claims'] >= 1
    assert data['summary']['ratios_basis'] == 'based only on cited claims'
    assert data['uncited_claim_analysis_enabled'] is False
    assert 'problematic_citations' in data
    assert 'audit_limited_citations' in data
    assert 'attribution_supported_citations' in data
    assert data['summary']['grounding_mix']['hard_fact_grounding'] > 0
    assert 'claims' in data


def test_analyze_with_web_search_reference_description():
    class FakeSearchProvider:
        def search(self, query, max_results=5):
            self.calls = getattr(self, "calls", 0) + 1
            return [
                SearchResult(
                    title="2024 annual report",
                    url="https://example.com/annual-report",
                    snippet="The company reported revenue of $10 billion in its 2024 annual report.",
                )
            ]

    search_provider = FakeSearchProvider()
    span = "The company reported revenue of $10 billion in its 2024 annual report [1]."
    analyzer = SourceGroundingAnalyzer(
        enable_url_fetch=False,
        search_provider=search_provider,
        llm_provider=fake_provider_for_revenue_claim(span),
    )
    result = analyzer.analyze(
        AnalysisRequest(
            input_text=(
                "The company reported revenue of $10 billion in its 2024 annual report [1].\n\n"
                "来源指针\n[1] 2024 annual report revenue $10 billion"
            ),
            enable_url_fetch=False,
            enable_web_search=True,
        )
    )

    assert result.metadata["enable_web_search"] is True
    assert result.metadata["search_result_count"] == 1
    assert result.metadata["search_query_count"] == 1
    assert search_provider.calls == 1
    assert result.sources[0].url == "https://example.com/annual-report"
    assert result.summary.key_rates.source_opacity_rate < 1
    assert result.summary.grounding_mix.hard_fact_grounding > 0


def test_background_analyze_job_completes(monkeypatch):
    span = "The company reported revenue of $10 billion in its 2024 annual report [source](https://example.com/ar)."
    analyzer = SourceGroundingAnalyzer(llm_provider=fake_provider_for_revenue_claim(span))
    monkeypatch.setattr(main_module, "analyzer", analyzer)
    client = TestClient(app)

    response = client.post(
        "/analyze/start",
        json={
            "input_text": span,
            "enable_url_fetch": False,
            "enable_web_search": False,
            "provided_sources": [
                {
                    "url": "https://example.com/ar",
                    "title": "2024 annual report",
                    "source_type": "primary_fact_source",
                    "extracted_text": "The company reported revenue of $10 billion in its 2024 annual report.",
                    "access_status": "accessible",
                }
            ],
        },
    )

    job_id = response.json()["job_id"]
    payload = _wait_for_job(client, job_id)
    assert payload["status"] == "completed"
    assert payload["progress"]["phase"] == "completed"
    assert payload["result"]["summary"]["ratios_basis"] == "based only on cited claims"


def test_background_analyze_job_can_be_cancelled(monkeypatch):
    class SlowProvider(MockLLMProvider):
        async def extract_claims(self, input_text, citations, context, cancellation_token=None):
            for _ in range(100):
                if cancellation_token and cancellation_token.is_cancelled():
                    raise AnalysisCancelledError("Analysis was cancelled.")
                await asyncio.sleep(0.01)
            return []

    analyzer = SourceGroundingAnalyzer(llm_provider=SlowProvider())
    monkeypatch.setattr(main_module, "analyzer", analyzer)
    client = TestClient(app)

    response = client.post(
        "/analyze/start",
        json={
            "input_text": "Reuters reported that Nvidia contacted Cerebras [1].\n\n[1] Reuters report",
            "enable_url_fetch": False,
            "enable_web_search": False,
        },
    )
    job_id = response.json()["job_id"]
    cancel = client.post(f"/analyze/jobs/{job_id}/cancel").json()
    assert cancel["status"] in {"cancelling", "cancelled"}
    payload = _wait_for_job(client, job_id)
    assert payload["status"] == "cancelled"
    assert payload["progress"]["phase"] == "cancelled"


def _wait_for_job(client, job_id):
    payload = {}
    for _ in range(100):
        payload = client.get(f"/analyze/jobs/{job_id}").json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.02)
    return payload
