# app/main.py (Phase 2 — naive version)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.models import ModerateRequest, ModerateResponse
from app.model_loader import load_model, run_inference
import time, hashlib, os

app = FastAPI(title="FluxServe")
security = HTTPBearer()
model, tokenizer = load_model()

API_KEY = os.getenv("API_KEY", "dev-key")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/moderate", response_model=ModerateResponse)
async def moderate(request: ModerateRequest, _=Depends(verify_token)):
    start = time.time()
    result = run_inference(model, tokenizer, request.text, request.max_new_tokens)
    latency = (time.time() - start) * 1000
    return ModerateResponse(
        label=result["label"],
        flagged=result["flagged"],
        latency_ms=latency,
        cached=False,
        model="qwen2.5-0.5b"
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": "qwen2.5-0.5b"}