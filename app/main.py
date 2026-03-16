# app/main.py (Phase 3 — with Redis cache + Prometheus)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.models import ModerateRequest, ModerateResponse
from app.model_loader import load_model, run_inference
from app.cache import ResponseCache
from app.metrics import (
    metrics_endpoint,
    requests_total,
    inference_latency,
)
import time
import os

app = FastAPI(title="FluxServe")
security = HTTPBearer()
model, tokenizer = load_model()
cache = ResponseCache()

API_KEY = os.getenv("API_KEY", "dev-key")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/moderate", response_model=ModerateResponse)
async def moderate(request: ModerateRequest, _=Depends(verify_token)):
    start = time.time()

    # Redis cache check
    cached = cache.get(request.text)
    if cached:
        requests_total.labels(status="cached").inc()
        cached["cached"] = True
        cached["latency_ms"] = (time.time() - start) * 1000
        return ModerateResponse(**cached)

    result = run_inference(model, tokenizer, request.text, request.max_new_tokens)
    latency = (time.time() - start) * 1000

    response = ModerateResponse(
        label=result["label"],
        flagged=result["flagged"],
        latency_ms=latency,
        cached=False,
        model="qwen2.5-0.5b",
    )

    # Store in cache and record metrics
    cache.set(request.text, response.model_dump())
    requests_total.labels(status="ok").inc()
    inference_latency.observe(latency / 1000)

    return response


@app.get("/health")
async def health():
    return {"status": "ok", "model": "qwen2.5-0.5b", "precision": "fp16"}


@app.get("/metrics")
async def metrics():
    return metrics_endpoint()
