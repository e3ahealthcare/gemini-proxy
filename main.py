# main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = os.environ["PROJECT_ID"]
LOCATION = os.environ.get("VERTEX_LOCATION", "global")
API_KEY = os.environ.get("PROXY_KEY")

class ChatIn(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # init AFTER server boots, before first request
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    app.state.model = GenerativeModel("gemini-2.5-flash")
    yield  # no teardown

app = FastAPI(lifespan=lifespan)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/chat")
def chat(payload: ChatIn, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")
    try:
        resp = app.state.model.generate_content(payload.text)
        return {"text": resp.text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vertex_error: {e.__class__.__name__}: {e}")

