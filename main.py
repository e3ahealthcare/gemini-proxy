import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel
import uvicorn

PROJECT_ID = os.environ["PROJECT_ID"]
LOCATION = os.environ.get("VERTEX_LOCATION", "global")  # "global" recommended for Gemini 2.x
API_KEY = os.environ.get("PROXY_KEY")  # optional shared secret

app = FastAPI()

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-flash")

class ChatIn(BaseModel):
    text: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/chat")
def chat(payload: ChatIn, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")
    try:
        resp = model.generate_content(payload.text)
        return {"text": resp.text}
    except Exception as e:
        # surface a clear message to the caller
        raise HTTPException(status_code=502, detail=f"vertex_error: {e.__class__.__name__}: {e}") from e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
