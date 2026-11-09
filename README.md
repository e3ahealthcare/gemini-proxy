# gemini-proxy

Gemini gateway on Cloud Run (E3A GCP). Vertex AI through the service account - no API key to Google.

- `GET /health` - liveness (not `/healthz`: Cloud Run reserves paths ending in `z`)
- `POST /chat` `{"text": "..."}` -> `{"text": "..."}` - legacy, uses `DEFAULT_MODEL`
- `POST /v1/generate` `{"contents", "model", "system_instruction", "json_mode", "temperature", "max_output_tokens"}`
  -> `{"text", "model", "usage": {"prompt","output","total"}, "latency_ms"}`

All POSTs need header `x-api-key: $PROXY_KEY`. With no `PROXY_KEY` set, every request is refused.

Env: `PROJECT_ID`, `VERTEX_LOCATION` (default `global`), `PROXY_KEY` (from Secret Manager), `DEFAULT_MODEL`,
`ALLOWED_MODELS` (comma list), `MAX_PROMPT_CHARS`. Logs carry model, latency and token counts only, never prompt text.
