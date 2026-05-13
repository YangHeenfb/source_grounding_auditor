import subprocess

import pytest
from fastapi.testclient import TestClient

from app.analyzer import SourceGroundingAnalyzer
from app.main import app
from app.providers.codex_claim_extractor import CodexCLIClaimExtractor
from app.providers.llm_provider import LLMProviderError, LLMProviderTimeoutError
from app.providers.openai_claim_extractor import claims_from_model_payload
from app.schemas import AnalysisRequest, Claim, ClaimExtractionMode, ClaimType, ImportanceLabel


def test_openai_mode_without_api_key_returns_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={
            "input_text": "The company reported revenue of $10 billion in 2024.",
            "claim_extraction_mode": "openai",
        },
    )

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_claims_from_model_payload_assigns_schema_fields():
    claims = claims_from_model_payload(
        {
            "claims": [
                {
                    "original_text_span": "Experts say revenue rose 10% in 2024 [source](https://example.com).",
                    "normalized_claim": "Experts say revenue rose 10% in 2024",
                    "claim_type": "attribution",
                    "has_quantitative_data": False,
                    "source_mentions": [],
                    "importance_label": "supporting",
                }
            ]
        }
    )

    assert len(claims) == 1
    assert claims[0].claim_id == "c001"
    assert claims[0].claim_type == ClaimType.ATTRIBUTION
    assert claims[0].has_quantitative_data is True
    assert claims[0].source_mentions


def test_claims_from_model_payload_requires_claims_list():
    with pytest.raises(LLMProviderError, match="claims list"):
        claims_from_model_payload({"items": []})


def test_codex_mode_uses_configured_codex_extractor():
    class FakeCodexExtractor:
        model = "gpt-5.5"
        service_tier = "fast"

        def is_configured(self):
            return True

        def extract_claims(self, input_text, original_question=None):
            return [
                Claim(
                    claim_id="c001",
                    original_text_span=input_text,
                    normalized_claim=input_text,
                    claim_type=ClaimType.FACTUAL,
                    has_quantitative_data=False,
                    importance_label=ImportanceLabel.SUPPORTING,
                )
            ]

    analyzer = SourceGroundingAnalyzer()
    analyzer.codex_extractor = FakeCodexExtractor()

    result = analyzer.analyze(
        AnalysisRequest(input_text="GPT-5.5 is available in Codex.", claim_extraction_mode=ClaimExtractionMode.CODEX)
    )

    assert result.metadata["claim_extraction_mode"] == "codex"
    assert result.metadata["codex_model"] == "gpt-5.5"
    assert result.metadata["codex_service_tier"] == "fast"


def test_codex_command_uses_fast_service_tier(monkeypatch, tmp_path):
    extractor = CodexCLIClaimExtractor(codex_bin="/tmp/fake-codex")
    monkeypatch.setattr(extractor, "is_configured", lambda: True)
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                '{"claims":[{"original_text_span":"OpenAI released GPT-4o in 2024",'
                '"normalized_claim":"OpenAI released GPT-4o in 2024","claim_type":"factual",'
                '"has_quantitative_data":true,"source_mentions":[],"importance_label":"core"}]}'
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("app.providers.codex_claim_extractor.subprocess.run", fake_run)
    claims = extractor.extract_claims("OpenAI released GPT-4o in 2024.")

    assert claims
    assert "-c" in captured["cmd"]
    assert 'service_tier="fast"' in captured["cmd"]


def test_codex_timeout_becomes_provider_timeout(monkeypatch):
    extractor = CodexCLIClaimExtractor(codex_bin="/tmp/fake-codex", timeout_seconds=0.01)
    monkeypatch.setattr(extractor, "is_configured", lambda: True)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

    monkeypatch.setattr("app.providers.codex_claim_extractor.subprocess.run", fake_run)

    with pytest.raises(LLMProviderTimeoutError, match="timed out"):
        extractor.extract_claims("A long input")
