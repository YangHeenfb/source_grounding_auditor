from fastapi.testclient import TestClient

import app.main as main_module
from app.analyzer import SourceGroundingAnalyzer
from app.providers.search_provider import SearchResult
from app.schemas import AnalysisRequest, Claim, ClaimType, ImportanceLabel

app = main_module.app


class FakeCodexExtractor:
    model = "gpt-5.5"
    service_tier = "fast"

    def is_configured(self):
        return True

    def extract_claims(self, input_text, original_question=None):
        span = input_text.split("\n", 1)[0].strip()
        normalized = span.replace("[1]", "").strip(" .")
        if "https://example.com/ar" in input_text:
            span = (
                "The company reported revenue of $10 billion in its 2024 annual report "
                "[source](https://example.com/ar)."
            )
            normalized = "The company reported revenue of $10 billion in its 2024 annual report"
        return [
            Claim(
                claim_id="c001",
                original_text_span=span,
                normalized_claim=normalized,
                claim_type=ClaimType.FACTUAL,
                has_quantitative_data=True,
                importance_label=ImportanceLabel.SUPPORTING,
            )
        ]


def test_health():
    client = TestClient(app)
    assert client.get('/health').json()['status'] == 'ok'


def test_analyze_with_provided_source(monkeypatch):
    analyzer = SourceGroundingAnalyzer()
    analyzer.codex_extractor = FakeCodexExtractor()
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
    analyzer = SourceGroundingAnalyzer(enable_url_fetch=False, search_provider=search_provider)
    analyzer.codex_extractor = FakeCodexExtractor()
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
