from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analyzer import SourceGroundingAnalyzer
from .providers.llm_provider import (
    AnalysisCancelledError,
    CancellationToken,
    LLMProviderConfigurationError,
    LLMProviderError,
    LLMProviderTimeoutError,
)
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
EXECUTOR = ThreadPoolExecutor(max_workers=2)
JOBS: dict[str, "AnalysisJob"] = {}
JOBS_LOCK = threading.Lock()


@dataclass
class AnalysisJob:
    job_id: str
    request: AnalysisRequest
    token: CancellationToken
    future: Future | None = None
    status: str = "pending"
    result: AnalysisResult | None = None
    error: str | None = None
    progress: dict[str, Any] = field(
        default_factory=lambda: {
            "phase": "pending",
            "message": "Waiting to start.",
            "current": None,
            "total": None,
        }
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "default_claim_extraction_mode": analyzer.claim_extraction_mode.value,
        "openai_configured": str(analyzer.openai_extractor.is_configured()).lower(),
        "codex_configured": str(analyzer.codex_extractor.is_configured()).lower(),
        "codex_model": analyzer.codex_extractor.model,
        "codex_service_tier": analyzer.codex_extractor.service_tier,
        "codex_reasoning_effort": analyzer.codex_extractor.reasoning_effort,
        "codex_timeout_seconds": str(int(analyzer.codex_extractor.timeout_seconds)),
    }


@app.post("/analyze", response_model=AnalysisResult)
def analyze(request: AnalysisRequest) -> AnalysisResult:
    try:
        result = analyzer.analyze(request, cancellation_token=CancellationToken())
    except LLMProviderConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except AnalysisCancelledError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    STORE[result.analysis_id] = result
    return result


@app.post("/analyze/start")
def start_analysis(request: AnalysisRequest) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    job = AnalysisJob(job_id=job_id, request=request, token=CancellationToken())
    with JOBS_LOCK:
        JOBS[job_id] = job
    job.future = EXECUTOR.submit(_run_analysis_job, job)
    return {"job_id": job_id, "status": job.status, "progress": job.progress}


@app.get("/analyze/jobs/{job_id}")
def get_analysis_job(job_id: str) -> dict[str, Any]:
    job = _get_job(job_id)
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
        "progress": job.progress,
    }
    if job.result is not None:
        payload["result"] = job.result.model_dump(mode="json")
    return payload


@app.post("/analyze/jobs/{job_id}/cancel")
def cancel_analysis_job(job_id: str) -> dict[str, str]:
    job = _get_job(job_id)
    job.token.cancel()
    if job.status in {"pending", "running"}:
        job.status = "cancelling"
    return {"job_id": job.job_id, "status": job.status}


@app.get("/analysis/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str) -> AnalysisResult:
    try:
        return STORE[analysis_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="analysis_id not found") from exc


@app.get("/")
def index() -> FileResponse:
    response = FileResponse(STATIC_DIR / "index.html")
    response.headers["Cache-Control"] = "no-store"
    return response

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _get_job(job_id: str) -> AnalysisJob:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job


def _run_analysis_job(job: AnalysisJob) -> None:
    job.status = "running"

    def update_progress(progress: dict[str, Any]) -> None:
        job.progress = progress

    try:
        result = analyzer.analyze(
            job.request,
            cancellation_token=job.token,
            progress_callback=update_progress,
        )
        if job.token.is_cancelled():
            job.status = "cancelled"
            return
        job.result = result
        STORE[result.analysis_id] = result
        job.progress = {
            "phase": "completed",
            "message": "Analysis completed.",
            "current": None,
            "total": None,
        }
        job.status = "completed"
    except AnalysisCancelledError:
        job.status = "cancelled"
        job.error = "Analysis was cancelled."
        job.progress = _terminal_progress(job.progress, "cancelled", "Analysis was cancelled.")
    except LLMProviderConfigurationError as exc:
        job.status = "failed"
        job.error = str(exc)
        job.progress = _terminal_progress(job.progress, "failed", str(exc))
    except LLMProviderTimeoutError as exc:
        job.status = "failed"
        job.error = str(exc)
        job.progress = _terminal_progress(job.progress, "failed", str(exc))
    except LLMProviderError as exc:
        job.status = "failed"
        job.error = str(exc)
        job.progress = _terminal_progress(job.progress, "failed", str(exc))
    except Exception as exc:  # pragma: no cover - defensive job boundary
        job.status = "failed"
        job.error = str(exc)
        job.progress = _terminal_progress(job.progress, "failed", str(exc))


def _terminal_progress(previous: dict[str, Any], phase: str, message: str) -> dict[str, Any]:
    previous_phase = previous.get("phase")
    if previous_phase and previous_phase not in {phase, "pending"}:
        message = f"{message} Last phase: {previous_phase}."
    return {
        "phase": phase,
        "message": message,
        "current": previous.get("current"),
        "total": previous.get("total"),
        "last_phase": previous_phase,
    }
