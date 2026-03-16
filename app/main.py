from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from app.models import ModerateRequest, ModerateResponse
from app.model_loader import load_model
from app.queue_manager import QueueManager, InferenceRequest
from app.batch_worker import BatchWorker
from app.cache import ResponseCache
from app.metrics import metrics_endpoint, requests_total, inference_latency
from app.kv_cache import KVCachePool
import asyncio
import time
import os
import uuid

queue_manager = QueueManager(max_depth=int(os.getenv("QUEUE_MAX_DEPTH", "256")))
cache = ResponseCache()
batch_worker: BatchWorker | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batch_worker
    model, tokenizer = load_model()
    kv_cache = KVCachePool(max_sequences=64)
    batch_worker = BatchWorker(
        model=model,
        tokenizer=tokenizer,
        queue_manager=queue_manager,
        kv_cache_pool=kv_cache,
        max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "8")),
        max_wait_ms=float(os.getenv("MAX_WAIT_MS", "20")),
    )
    asyncio.create_task(batch_worker.run())
    yield

app = FastAPI(title="FluxServe", lifespan=lifespan)
security = HTTPBearer()
API_KEY = os.getenv("API_KEY", "dev-key")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/moderate", response_model=ModerateResponse)
async def moderate(request: ModerateRequest, _=Depends(verify_token)):
    start = time.time()
    request_id = request.request_id or str(uuid.uuid4())

    # Check Redis cache first
    cached = cache.get(request.text)
    if cached:
        requests_total.labels(status="cached").inc()
        return ModerateResponse(
            label=cached["label"],
            flagged=cached["flagged"],
            tokens_generated=cached.get("tokens_generated"),
            latency_ms=(time.time() - start) * 1000,
            cached=True,
            model="qwen2.5-0.5b",
            request_id=request_id,
        )

    # Check queue capacity
    if queue_manager.is_full():
        requests_total.labels(status="rate_limited").inc()
        raise HTTPException(
            status_code=429,
            detail={"error": "Server at capacity. Retry after 1s.", "retry_after": 1}
        )

    # Enqueue request and await result
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    inference_request = InferenceRequest(
        text=request.text,
        max_new_tokens=request.max_new_tokens or 10,
        future=future,
        request_id=request_id,
    )

    await queue_manager.enqueue(inference_request)
    result = await future  # Blocks here until batch worker resolves it

    # Cache the result
    cache.set(request.text, result)

    latency = (time.time() - start) * 1000
    inference_latency.observe(latency / 1000)
    requests_total.labels(status="ok").inc()

    return ModerateResponse(**result, cached=False, request_id=request_id, model="qwen2.5-0.5b")

@app.get("/health")
async def health():
    import torch
    return {
        "status": "ok",
        "queue_depth": queue_manager.depth(),
        "active_sequences": len(batch_worker.active) if batch_worker else 0,
        "gpu_memory_used_gb": torch.cuda.memory_allocated() / 1e9,
    }

@app.get("/metrics")
async def metrics():
    return metrics_endpoint()