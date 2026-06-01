import subprocess

import pytest
from fastapi.testclient import TestClient

from app.analyzer import SourceGroundingAnalyzer
from app.main import app
from app.providers.codex_claim_extractor import CodexCLIClaimExtractor
from app.providers.llm_provider import LLMProviderError, LLMProviderTimeoutError, MockLLMProvider
from app.providers.openai_claim_extractor import claims_from_model_payload
from app.schemas import AnalysisRequest, Claim, ClaimExtractionMode, ClaimType, DiscourseRole, ImportanceLabel


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
    provider = MockLLMProvider(
        extraction_outputs=[
            Claim(
                claim_id="c001",
                original_text_span="GPT-5.5 is available in Codex.",
                normalized_claim="GPT-5.5 is available in Codex",
                claim_type=ClaimType.FACTUAL,
                discourse_role=DiscourseRole.ASSERTED_CLAIM,
                has_quantitative_data=True,
                importance_label=ImportanceLabel.SUPPORTING,
            )
        ]
    )
    analyzer = SourceGroundingAnalyzer(llm_provider=provider)

    result = analyzer.analyze(
        AnalysisRequest(
            input_text="GPT-5.5 is available in Codex [1].\n\n[1] Codex release note",
            claim_extraction_mode=ClaimExtractionMode.CODEX,
            enable_web_search=False,
            enable_url_fetch=False,
            split_atomic_claims=True,
        )
    )

    assert result.metadata["claim_extraction_mode"] == "mock"
    assert result.claims[0].normalized_claim == "GPT-5.5 is available in Codex"


def test_codex_command_uses_fast_service_tier(monkeypatch, tmp_path):
    extractor = CodexCLIClaimExtractor(codex_bin="/tmp/fake-codex")
    monkeypatch.setattr(extractor, "is_configured", lambda: True)
    captured = {}

    class FakePopen:
        returncode = 0

        def __init__(self, cmd, *args, **kwargs):
            captured["cmd"] = cmd
            self.cmd = cmd

        def communicate(self, input=None, timeout=None):
            output_path = self.cmd[self.cmd.index("--output-last-message") + 1]
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(
                    '{"claims":[{"original_text_span":"OpenAI released GPT-4o in 2024",'
                    '"normalized_claim":"OpenAI released GPT-4o in 2024","claim_type":"factual",'
                    '"has_quantitative_data":true,"source_mentions":[],"importance_label":"core"}]}'
                )
            return "", ""

        def poll(self):
            return self.returncode

    monkeypatch.setattr("app.providers.llm_provider.subprocess.Popen", FakePopen)
    claims = extractor.extract_claims("OpenAI released GPT-4o in 2024.")

    assert claims
    assert "-c" in captured["cmd"]
    assert 'service_tier="fast"' in captured["cmd"]
    assert 'model_reasoning_effort="low"' in captured["cmd"]


def test_codex_default_timeout_is_short(monkeypatch):
    monkeypatch.delenv("CODEX_TIMEOUT_SECONDS", raising=False)
    extractor = CodexCLIClaimExtractor(codex_bin="/tmp/fake-codex")
    assert extractor.timeout_seconds == 90.0


def test_codex_timeout_becomes_provider_timeout(monkeypatch):
    extractor = CodexCLIClaimExtractor(codex_bin="/tmp/fake-codex", timeout_seconds=0.01)
    monkeypatch.setattr(extractor, "is_configured", lambda: True)

    class TimeoutPopen:
        returncode = None

        def __init__(self, cmd, *args, **kwargs):
            self.cmd = cmd

        def communicate(self, input=None, timeout=None):
            raise subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr("app.providers.llm_provider.subprocess.Popen", TimeoutPopen)

    with pytest.raises(LLMProviderTimeoutError, match="timed out"):
        extractor.extract_claims("A long input")
