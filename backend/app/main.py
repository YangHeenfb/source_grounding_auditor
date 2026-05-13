from __future__ import annotations

from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analyzer import SourceGroundingAnalyzer
from .providers.llm_provider import LLMProviderConfigurationError, LLMProviderError, LLMProviderTimeoutError
from .schemas import AnalysisRequest, AnalysisResult

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Source Grounding Auditor", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SourceGroundingAnalyzer(enable_url_fetch=True)
STORE: dict[str, AnalysisResult] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "default_claim_extraction_mode": analyzer.claim_extraction_mode.value,
        "openai_configured": str(analyzer.openai_extractor.is_configured()).lower(),
        "codex_configured": str(analyzer.codex_extractor.is_configured()).lower(),
        "codex_model": analyzer.codex_extractor.model,
        "codex_service_tier": analyzer.codex_extractor.service_tier,
        "codex_timeout_seconds": str(int(analyzer.codex_extractor.timeout_seconds)),
    }


@app.post("/analyze", response_model=AnalysisResult)
def analyze(request: AnalysisRequest) -> AnalysisResult:
    try:
        result = analyzer.analyze(request)
    except LLMProviderConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    STORE[result.analysis_id] = result
    return result


@app.get("/analysis/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str) -> AnalysisResult:
    try:
        return STORE[analysis_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="analysis_id not found") from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
