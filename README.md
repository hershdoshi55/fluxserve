# FluxServe

A production-grade LLM inference server built from scratch for content moderation. Serves **Qwen2.5-0.5B-Instruct** to classify text as toxic or safe, with continuous batching, KV cache management, Redis response caching, backpressure, and full Prometheus/Grafana observability — all containerized with Docker Compose.

**Hardware tested on:** NVIDIA GTX 1080 (8GB VRAM) · **Model:** Qwen/Qwen2.5-0.5B-Instruct (FP16)

---

## Results at a Glance

| Optimization | Throughput | vs Baseline |
|---|---|---|
| Naive sequential server (baseline) | 1.94 req/s | — |
| + Continuous batching (4 users) | 6.24 req/s | **3.2x** |
| + Continuous batching (16 users) | 9.02 req/s | **4.6x** |
| + Redis cache (warm, 16 users) | 44.5 req/s | **23x** |

---

## Architecture

```
HTTP POST /moderate
  → Auth middleware (Bearer token)
  → Backpressure check (429 if queue full)
  → Redis exact-match cache (MD5 key, return immediately if hit)
  → Enqueue (text, asyncio.Future) → asyncio.Queue
  → HTTP handler awaits Future
        ↓
  Continuous Batch Worker (runs forever in background)
    → Pull up to max_batch_size requests from queue
    → KV cache check per request (skip prefill if hit)
    → Run one decode step across all active sequences
    → Remove finished sequences, resolve their Futures
    → Update Prometheus metrics
    → Loop
        ↓
  GPU: Qwen2.5-0.5B (FP16)
```

### Key design decisions

**Continuous batching** — the batch worker runs a tight decode loop. Every step, finished sequences leave and queued requests enter. GPU utilization stays near 100% vs static batching which idles waiting for slow sequences.

**asyncio.Future pattern** — each HTTP handler creates a Future and awaits it. The single-threaded GPU worker resolves Futures when sequences complete. This fans hundreds of concurrent HTTP requests into one GPU thread without threads or locks.

**KV cache pool** — fixed VRAM budget with LRU eviction. Stores the attention key-value tensors from the prefill step. On a cache hit, the expensive prefill is skipped entirely — only the cheap decode step runs. Achieved **66% hit rate** in testing.

**Redis exact-match cache** — MD5-keyed response cache. Repeated identical inputs (e.g. the same viral tweet being moderated 10,000 times) never hit the GPU at all. Served in ~2ms vs ~150ms for GPU inference.

**Backpressure** — queue depth check before enqueue. Returns 429 with `retry_after` immediately when the server is saturated rather than letting latency spiral unboundedly.

---

## Performance Deep Dive

### Baseline: naive sequential server
Single request at a time, no batching, no caching. Benchmarked with Locust (1 user, 60s).

| Metric | Value |
|---|---|
| Throughput | 1.94 req/s |
| p50 latency | 120ms |
| p95 latency | 250ms |

### Continuous batching improvement
Tested with unique text per request (cache disabled) to measure true GPU batching gains.

| Concurrent Users | RPS | p50 | p95 | p99 | vs Baseline |
|---|---|---|---|---|---|
| 1 | 1.96 | 150ms | 260ms | 360ms | 1.0x |
| 4 | 6.24 | 240ms | 440ms | 740ms | **3.2x** |
| 8 | 7.87 | 620ms | 970ms | 1100ms | **4.1x** |
| 16 | 9.02 | 1400ms | 1800ms | 2000ms | **4.6x** |

The sweet spot is **4 concurrent users**: 3.2x throughput with only 2x latency increase. Beyond 8 users the GPU is saturated (~9 RPS ceiling on GTX 1080) and additional users only deepen the queue.

### KV cache impact
The KV cache stores prefill attention tensors (the expensive part of autoregressive inference). On a hit, only the decode step runs.

- **Hit rate:** 66% across all sequences
- **Effect on latency:** ~50–80ms prefill eliminated on cache hits, reducing per-request GPU time to decode-only

### Redis cache impact
Exact-match caching for repeated inputs. The same text is never sent to the GPU twice.

| Condition | p50 | Req/s |
|---|---|---|
| GPU inference (cold) | 150ms | 1.96 |
| Redis cache hit (warm) | 2ms | 44.5 @ 16 users |

Redis serves cached results at **~2ms** regardless of concurrency, enabling near-linear throughput scaling for repeated inputs.

### Precision: why FP16 over INT8

| Precision | Avg latency | VRAM |
|---|---|---|
| FP16 | 146ms | 1.00 GB |
| INT8 | 499ms | 0.64 GB |

INT8 is **3.4x slower** on the GTX 1080 (Pascal architecture). Pascal has no native INT8 tensor cores — bitsandbytes falls back to software emulation. INT8 speed gains only materialize on Turing (RTX 20xx) and newer. FP16 was selected for all benchmarks.

---

## Stack

- **Inference:** PyTorch · HuggingFace Transformers · Qwen2.5-0.5B-Instruct
- **Server:** FastAPI · Uvicorn (asyncio)
- **Caching:** Redis 7
- **Observability:** Prometheus · Grafana
- **Containerization:** Docker · Docker Compose

---

## Setup Guide

This guide assumes a fresh clone with nothing pre-installed. There are two ways to run FluxServe:

- **Option A — Local** (steps 1–6): run the server directly on your machine. Faster iteration, GPU used directly. Requires Python and NVIDIA drivers.
- **Option B — Docker Compose** (jump to the Docker section): one command spins up the full stack. Dependencies are handled inside the container automatically.

### Prerequisites

- **OS:** Linux or WSL2 (Ubuntu 22.04/24.04 recommended)
- **GPU:** NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
  - Install the [NVIDIA driver](https://www.nvidia.com/drivers) for your GPU
  - Verify: `nvidia-smi`
- **Python:** 3.11 or 3.12
  - `python3 --version`
- **Docker:** for Redis and Docker Compose workflow
  - Install: https://docs.docker.com/engine/install/ubuntu/
  - Verify: `docker --version`

### Option A — Local Dev

#### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd fluxserve

python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install PyTorch

The correct PyTorch build depends on your CUDA version. Check with `nvidia-smi` (top-right corner shows CUDA version).

**CUDA 11.8 (GTX 10xx, Pascal, or if nvidia-smi shows CUDA 11.x):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.x (RTX 20xx/30xx/40xx, or if nvidia-smi shows CUDA 12.x):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**No GPU / CPU only:**
```bash
pip install torch
```

#### 3. Install remaining dependencies

```bash
pip install transformers accelerate fastapi "uvicorn[standard]" redis prometheus-client
```

#### 4. Validate the model works

This downloads Qwen2.5-0.5B-Instruct (~1GB) on first run and runs a smoke test.

```bash
python scripts/validate_model.py
```

Expected output: 5 or 6 out of 6 classifications correct, latency under 2000ms per request.

#### 5. Start Redis

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

#### 6. Start the server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for the model to load (~10–30 seconds), then test:

```bash
# Health check
curl http://localhost:8000/health

# Classify text
curl -X POST http://localhost:8000/moderate \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this community!"}'
```

### Option B — Docker Compose (recommended)

Runs the inference server, Redis, Prometheus, and Grafana together.

```bash
docker compose up --build -d
```

The first build takes ~15 minutes (downloads PyTorch CUDA wheels). Subsequent builds use the layer cache and finish in seconds.

Once running:
- **API:** http://localhost:8000
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (login: admin / admin → FluxServe dashboard)

---

## API Reference

All endpoints except `/health` and `/metrics` require `Authorization: Bearer <API_KEY>` (default: `dev-key`).

### `POST /moderate`

Classify text as toxic or safe.

**Request:**
```json
{
  "text": "your text here",
  "max_new_tokens": 10
}
```

**Response:**
```json
{
  "label": "toxic",
  "flagged": true,
  "latency_ms": 107.4,
  "cached": false,
  "model": "qwen2.5-0.5b",
  "tokens_generated": 3,
  "request_id": "uuid"
}
```

**Status codes:**
- `200` — classified successfully
- `401` — missing or invalid Bearer token
- `429` — server at capacity (queue full), retry after `retry_after` seconds

### `GET /health`

Returns current server state.

```json
{
  "status": "ok",
  "queue_depth": 0,
  "active_sequences": 3,
  "gpu_memory_used_gb": 0.99
}
```

### `GET /metrics`

Prometheus scrape endpoint. Exposes all `fluxserve_*` metrics.

### `POST /admin/config`

Hot-update server configuration without restart.

```json
{
  "max_batch_size": 16,
  "max_wait_ms": 10,
  "queue_max_depth": 512
}
```

---

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | `dev-key` | Bearer token for auth |
| `PRECISION` | `fp16` | `fp16` or `int8` |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `MAX_BATCH_SIZE` | `8` | Max sequences per decode step |
| `MAX_WAIT_MS` | `20` | Max wait before dispatching a batch |
| `QUEUE_MAX_DEPTH` | `256` | Queue depth before returning 429 |
| `KV_CACHE_MAX_SEQS` | `64` | Max sequences in KV cache pool |

---

## Load Testing

```bash
pip install locust

# Cache-enabled test (measures Redis + batching together)
locust -f load_tests/locustfile.py --headless -u 8 -r 2 --run-time 60s \
  --host http://localhost:8000

# Cache-disabled test (measures true GPU batching performance)
locust -f load_tests/locustfile_nocache.py --headless -u 8 -r 2 --run-time 60s \
  --host http://localhost:8000 --csv results/run_8
```

---

## Project Structure

```
app/
  main.py           # FastAPI app, routes, lifespan, middleware
  model_loader.py   # Load Qwen2.5-0.5B, run_inference(), prompt template
  batch_worker.py   # Continuous batching loop — core of the project
  queue_manager.py  # asyncio.Queue wrapper, backpressure enforcement
  kv_cache.py       # KV cache pool, LRU eviction, VRAM budget
  cache.py          # Redis response cache (MD5-keyed, JSON-serialized)
  metrics.py        # All Prometheus metrics
  models.py         # Pydantic request/response schemas
scripts/
  validate_model.py # Smoke test — run before any server work
  benchmark.py      # Single-threaded throughput/latency benchmark
load_tests/
  locustfile.py          # Locust test (fixed texts, cache-enabled)
  locustfile_nocache.py  # Locust test (unique texts, cache-disabled)
benchmarks/            # Markdown benchmark results per phase
results/               # Locust CSV output
grafana/               # Grafana provisioning and dashboard JSON
docker-compose.yml
Dockerfile
prometheus.yml
```
