from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..schemas import Claim
from .llm_provider import LLMProviderConfigurationError, LLMProviderError, LLMProviderTimeoutError
from .openai_claim_extractor import CLAIMS_JSON_SCHEMA, SYSTEM_PROMPT, claims_from_model_payload

DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_SERVICE_TIER = "fast"


class CodexCLIClaimExtractor:
    """Claim extractor that uses the locally logged-in Codex subscription.

    This is intentionally separate from the OpenAI API provider. It shells out to
    `codex exec`, so it uses the user's local Codex login instead of OPENAI_API_KEY.
    It is suitable for local MVP testing, not high-throughput production serving.
    """

    def __init__(
        self,
        *,
        codex_bin: str | None = None,
        model: str | None = None,
        service_tier: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.codex_bin = codex_bin or os.environ.get("CODEX_BIN") or shutil.which("codex")
        self.model = model or os.environ.get("CODEX_MODEL") or DEFAULT_CODEX_MODEL
        self.service_tier = service_tier or os.environ.get("CODEX_SERVICE_TIER") or DEFAULT_SERVICE_TIER
        self.timeout_seconds = float(os.environ.get("CODEX_TIMEOUT_SECONDS") or timeout_seconds or DEFAULT_TIMEOUT_SECONDS)

    def is_configured(self) -> bool:
        if not self.codex_bin:
            return False
        try:
            result = subprocess.run(
                [self.codex_bin, "login", "status"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and "Logged in" in (result.stdout + result.stderr)

    def extract_claims(self, input_text: str, original_question: str | None = None) -> list[Claim]:
        if not self.codex_bin:
            raise LLMProviderConfigurationError("codex CLI was not found. Install or expose codex on PATH.")
        if not self.is_configured():
            raise LLMProviderConfigurationError(
                "Codex CLI is not logged in. Run `codex login` before using claim_extraction_mode='codex'."
            )

        with tempfile.TemporaryDirectory(prefix="source-grounding-codex-") as tmpdir:
            tmp = Path(tmpdir)
            schema_path = tmp / "claim_schema.json"
            output_path = tmp / "claims.json"
            schema_path.write_text(_schema_json(), encoding="utf-8")

            cmd = [
                self.codex_bin,
                "--ask-for-approval",
                "never",
                "exec",
                "-c",
                f'service_tier="{self.service_tier}"',
                "--model",
                self.model,
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "--color",
                "never",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "-",
            ]
            try:
                result = subprocess.run(
                    cmd,
                    input=_prompt(input_text, original_question),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMProviderTimeoutError(
                    f"Codex CLI timed out after {self.timeout_seconds:.0f} seconds while using model "
                    f"{self.model}. Try a shorter input or set CODEX_TIMEOUT_SECONDS higher."
                ) from exc

            if result.returncode != 0:
                raise LLMProviderError(
                    f"Codex CLI request failed ({result.returncode}): {_safe_process_output(result)}"
                )

            if not output_path.exists() or not output_path.read_text(encoding="utf-8").strip():
                raise LLMProviderError("Codex CLI did not write a claim extraction response.")

            try:
                import json

                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise LLMProviderError("Codex CLI response was not valid JSON.") from exc

        return claims_from_model_payload(payload)


def _schema_json() -> str:
    import json

    return json.dumps(CLAIMS_JSON_SCHEMA, ensure_ascii=False)


def _prompt(input_text: str, original_question: str | None) -> str:
    question_block = f"Original question:\n{original_question}\n\n" if original_question else ""
    return f"""{SYSTEM_PROMPT}

You are running as a local subprocess for a FastAPI app. Do not inspect files, do not
edit files, and do not run tools. Only extract claims from the text below.

{question_block}Text to analyze:
{input_text}
"""


def _safe_process_output(result: subprocess.CompletedProcess[str]) -> str:
    text = ((result.stderr or "") + "\n" + (result.stdout or "")).strip().replace("\n", " ")
    if len(text) > 700:
        return f"{text[:700]}..."
    return text
