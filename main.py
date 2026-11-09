# main.py - E3A Gemini gateway (Cloud Run). Vertex AI via the service account; callers authenticate with x-api-key.
import hmac
import logging
import os
import time

from fastapi import FastAPI, Header, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("VERTEX_LOCATION", "global")
API_KEY = os.environ.get("PROXY_KEY", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gemini-3.5-flash")
ALLOWED_MODELS = {m.strip() for m in os.environ.get(
    "ALLOWED_MODELS", "gemini-3.5-flash,gemini-3.1-flash-lite").split(",") if m.strip()}
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "200000"))

log = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class ChatIn(BaseModel):          # legacy shape used by AI-Prediction-Project
    text: str


class GenerateIn(BaseModel):      # used by the MPE portal (MPE_GEMINI_BACKEND=proxy)
    contents: str
    model: str | None = None
    system_instruction: str | None = None
    json_mode: bool = False
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=65536)


def _auth(key: str | None) -> None:
    # Refuse everything if PROXY_KEY is not configured - never run as an open relay.
    if not API_KEY or not key or not hmac.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid api key")


def _generate(model: str, contents: str, cfg: types.GenerateContentConfig) -> dict:
    if model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"model not allowed: {model}")
    if len(contents) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail="prompt too long")
    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    except Exception as e:
        log.warning("vertex_error model=%s err=%s", model, e.__class__.__name__)
        raise HTTPException(status_code=502, detail=f"vertex_error: {e.__class__.__name__}: {e}")
    ms = round((time.perf_counter() - t0) * 1000)
    u = resp.usage_metadata
    usage = {"prompt": getattr(u, "prompt_token_count", None),
             "output": getattr(u, "candidates_token_count", None),
             "total": getattr(u, "total_token_count", None)}
    # Metadata only - never log prompt or response text (may contain patient data).
    log.info("ok model=%s latency_ms=%s tokens=%s", model, ms, usage["total"])
    return {"text": resp.text or "", "model": model, "usage": usage, "latency_ms": ms}


@app.get("/")
@app.get("/health")       # not /healthz: Cloud Run reserves paths ending in "z"
def health():
    return {"ok": True}


@app.post("/chat")
def chat(payload: ChatIn, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    out = _generate(DEFAULT_MODEL, payload.text, types.GenerateContentConfig())
    return {"text": out["text"]}


@app.post("/v1/generate")
def generate(p: GenerateIn, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    cfg = types.GenerateContentConfig(
        system_instruction=p.system_instruction or None,
        temperature=p.temperature,
        max_output_tokens=p.max_output_tokens,
        response_mime_type="application/json" if p.json_mode else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    return _generate(p.model or DEFAULT_MODEL, p.contents, cfg)
