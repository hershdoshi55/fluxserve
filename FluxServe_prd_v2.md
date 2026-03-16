# FluxServe — LLM Inference Server
## Complete Implementation Guide — PRD v2.0

> **Who this document is for:** Someone who has built APIs and distributed systems before, has ML exposure, but has never built a production LLM serving layer. Every inference concept is explained from first principles. Every file is described. Every phase tells you exactly what to build and why it matters.

**Stack:** Python 3.11 · FastAPI · PyTorch · Transformers (HuggingFace) · Redis · Docker Compose · Prometheus · Grafana · Kubernetes (minikube)  
**Model:** Qwen 0.5B (autoregressive, decoder-only transformer)  
**Use case:** Content moderation — classify user-generated text as toxic or safe  
**Hardware:** GTX 1080 (8GB VRAM)  
**Estimated build time:** 1–2 weeks full-time  
**Lines of code (approximate):** ~2,000

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Why This Project Exists — Portfolio Framing](#2-why-this-project-exists)
3. [Every Inference Concept Explained From Scratch](#3-every-inference-concept-explained-from-scratch)
4. [The Model — Qwen 0.5B for Content Moderation](#4-the-model)
5. [System Architecture](#5-system-architecture)
6. [API Specification](#6-api-specification)
7. [Complete File Structure](#7-complete-file-structure)
8. [Phase-by-Phase Build Guide](#8-phase-by-phase-build-guide)
9. [GPU Setup (GTX 1080)](#9-gpu-setup)
10. [Benchmarking and Load Testing](#10-benchmarking-and-load-testing)
11. [Docker and Kubernetes](#11-docker-and-kubernetes)
12. [Design Tradeoffs](#12-design-tradeoffs)
13. [Interview Preparation](#13-interview-preparation)
14. [Resume Bullets](#14-resume-bullets)

---

## 1. What This Project Is

### The one-sentence version

A production-grade LLM inference server that serves Qwen 0.5B for content moderation, implementing continuous batching, KV cache management, INT8 quantization, backpressure, and Prometheus observability — benchmarked end-to-end with real throughput and latency numbers.

### The longer version

Every platform that allows user-generated content needs a moderation layer. Comments, messages, and LLM outputs must be screened before they reach other users or downstream systems. The challenge is not building a classifier. The challenge is **serving one reliably under concurrent load with acceptable latency and cost.**

This is an LLM inference serving problem, not a classification problem. The model is Qwen 0.5B — an autoregressive, decoder-only transformer that generates text token by token. Serving it naively (one request at a time, full precision, no batching) wastes GPU capacity and produces unacceptable latency at scale.

This project builds the serving infrastructure that makes it production-viable:

**Continuous batching** is how production LLM servers achieve throughput. Unlike static batching (wait for N requests, run them together, respond to all), continuous batching allows new requests to join an in-progress batch between decode steps. This is how vLLM, SGLang, and every production LLM serving system works. You will implement a simplified version of this from scratch.

**KV cache management** is the memory optimization that makes transformer inference fast. During autoregressive generation, the model recomputes attention over all previous tokens at every step — this is quadratic in sequence length. KV caching stores the key and value tensors from previous steps so they don't need recomputation. You will implement a KV cache with a fixed memory budget and an eviction policy.

**INT8 quantization** reduces the model's VRAM footprint by representing weights in 8-bit integers instead of 32-bit floats. On a GTX 1080 with 8GB VRAM, this is not optional — it's what makes Qwen 0.5B fit comfortably and leaves room for activations and the KV cache.

**Backpressure** is how servers degrade gracefully instead of crashing. When the inference queue is full, you return HTTP 429 instead of accepting more work than you can handle.

**Prometheus observability** gives you the numbers that make the project credible: tokens/sec throughput, p50/p95/p99 latency, KV cache hit rate, batch size distribution, queue depth. Without these, you have a demo. With them, you have a benchmark.

### What you ARE building

- A FastAPI inference server with a `POST /moderate` endpoint
- Continuous batching engine using asyncio.Queue + batch worker
- KV cache with configurable memory budget and LRU eviction
- INT8 quantization via `bitsandbytes`
- Backpressure via HTTP 429 when queue depth exceeds threshold
- Prometheus metrics: tokens/sec, TTFT, TPOT, p50/p95/p99 latency, KV cache hit rate, batch size distribution, queue depth
- Grafana dashboard wired to Prometheus for live visualization
- Docker Compose for local multi-service orchestration
- Kubernetes deployment on minikube with HPA autoscaling

### What you are NOT building

- A training pipeline (Qwen 0.5B is used as-is via HuggingFace)
- A UI (curl and Locust load tests are the interface)
- Full PagedAttention (vLLM's production implementation — too complex for this scope; you will implement the conceptual equivalent)
- Multi-GPU tensor parallelism (single GTX 1080)
- Streaming responses (synchronous request/response only)
- Cloud deployment (minikube covers the Kubernetes story; cloud infra is not the point of this project)

---

## 2. Why This Project Exists

### Portfolio framing

This project demonstrates that you understand the serving layer of ML — not just how to call a model, but how models run in production under real concurrent load. Specifically it shows:

- You understand the difference between **encoder-only models (BERT)** and **decoder-only autoregressive models (Qwen, GPT, Llama)** and why serving them requires different infrastructure
- You know what **continuous batching** is and why static batching wastes GPU capacity
- You understand **KV caching** at the implementation level, not just as a buzzword
- You have measured **real throughput and latency numbers** under load — not estimated, not calculated, but benchmarked
- You understand the **latency/throughput tradeoff** and can speak to how batch size and sequence length affect both

### Who will be impressed

**Fireworks AI, Baseten, Anyscale, Modal, Together AI, Groq:** These companies build exactly this infrastructure at scale. Your project is a miniature version of their core product. They will ask you to walk through the continuous batching implementation in detail — be ready.

**Google (Cloud AI, DeepMind infra), xAI:** The JDs explicitly call out inference optimization, batching, and latency/throughput tradeoffs. This project hits every one of those keywords with real implementation behind them.

**Scale AI, any AI startup:** Content moderation is a core trust and safety problem. The infrastructure you are building is what every platform with user-generated content or LLM outputs needs.

### What distinguishes this from a BERT classifier project

The v4 PRD used DistilBERT — an encoder-only model. Serving it is simple: tokenize, forward pass, softmax, done. No autoregressive generation, no KV cache, no continuous batching. That's why this project replaces it with Qwen 0.5B. The infrastructure challenges of autoregressive inference are qualitatively different and are exactly what AI infra companies care about.

---

## 3. Every Inference Concept Explained From Scratch

Read this section fully before writing any code. Every concept here maps directly to something you will implement.

### 3.1 Autoregressive Generation

Qwen 0.5B is a decoder-only transformer. Unlike BERT which processes an entire input sequence in one forward pass, Qwen generates output **one token at a time**, autoregressively.

Given an input prompt, the model:
1. Runs a forward pass over the entire prompt (the **prefill** step)
2. Produces a probability distribution over the vocabulary
3. Samples the next token from that distribution
4. Appends the new token to the sequence
5. Runs another forward pass over the entire extended sequence
6. Repeats until it generates a stop token or hits max_new_tokens

For content moderation, you prompt it with something like:
```
Is the following text toxic? Reply with only "toxic" or "safe".
Text: {user_input}
Answer:
```

Then you decode one or two tokens ("toxic" or "safe") and parse the result. The generation is short, which keeps latency low.

**Why this matters for serving:** Each decode step requires a forward pass over the full sequence so far. As sequences get longer, each step gets slower. KV caching solves this.

### 3.2 KV Cache — The Core Optimization

In a transformer's attention mechanism, every token attends to every previous token. This requires computing **key (K)** and **value (V)** tensors for every token at every layer. In naive inference, these are recomputed from scratch at every decode step — even for tokens you've already processed.

**KV caching stores these tensors after the first computation.** On subsequent decode steps, you only compute K and V for the new token. The stored K and V tensors for all previous tokens are reused directly.

Without KV cache: decode step N requires computing K and V for N tokens — O(N) work per step, O(N²) total.
With KV cache: decode step N only computes K and V for 1 new token — O(1) work per step, O(N) total.

**Memory cost:** The KV cache for a single sequence takes roughly:
```
2 × num_layers × num_heads × head_dim × seq_len × bytes_per_element
```
For Qwen 0.5B (24 layers, 16 heads, 64 head_dim, FP16):
```
2 × 24 × 16 × 64 × seq_len × 2 bytes = 98,304 × seq_len bytes
```
At seq_len=512, that's ~50MB per sequence. On 8GB VRAM with the model taking ~1GB (INT8), you have ~7GB for KV cache — enough for ~140 concurrent sequences at seq_len=512.

**Your implementation:** You will maintain a fixed-size KV cache pool. When it's full and a new sequence needs space, you evict the oldest sequence (LRU). If a sequence was evicted and it comes back (cache miss), you rerun the prefill step to repopulate its KV cache.

### 3.3 Continuous Batching

**Static batching (naive):** Wait for N requests to arrive. Run them through the model together as a batch. Return all results. Wait for the next N requests.

The problem: sequences in a batch finish at different times. If you have a batch of 8 and 7 requests finish quickly but 1 is still generating, the GPU sits idle waiting for the slow one to finish before you can start the next batch. GPU utilization is terrible.

**Continuous batching (production):** At every decode step, the batch worker checks if any sequences have finished. Finished sequences are removed from the batch immediately. New waiting requests are inserted into the batch to fill the empty slots. The GPU is always working on the maximum number of sequences.

This is the key insight behind vLLM's performance. You will implement a simplified version:

```
LOOP every decode step:
    1. Run one decode step for all active sequences in the batch
    2. Remove any sequences that generated a stop token or hit max_new_tokens
    3. Pull new requests from the queue to fill empty batch slots (up to max_batch_size)
    4. Send responses back to waiting HTTP handlers for completed sequences
    5. Go to 1
```

**Why this matters for throughput:** Continuous batching keeps GPU utilization near 100%. Static batching with mismatched sequence lengths can drop utilization below 50%.

### 3.4 INT8 Quantization

Qwen 0.5B in full float32 precision takes ~2GB of VRAM. In float16 (FP16), ~1GB. In INT8, ~500MB.

Quantization represents model weights using fewer bits. INT8 uses 8-bit integers instead of 32-bit floats. The tradeoff: slight accuracy loss in exchange for 2-4x memory reduction and faster matrix multiplications on supported hardware.

**Why it matters on a GTX 1080:** 8GB VRAM is tight. You need room for the model weights, the KV cache, activations during the forward pass, and PyTorch overhead. INT8 quantization is what makes this all fit.

**Implementation options:**
1. `bitsandbytes` library — the easiest path, wraps HuggingFace models with `load_in_8bit=True`
2. `torch.quantization.quantize_dynamic` — PyTorch native, slightly more control

Use `bitsandbytes` — it's one line and HuggingFace integrates with it natively.

**What to benchmark:** Compare tokens/sec throughput and p95 latency between FP32, FP16, and INT8. This becomes a resume bullet.

### 3.5 Prefill vs Decode — Two Phases of Inference

Every LLM inference request has two distinct phases:

**Prefill:** The model processes the entire input prompt in one forward pass. This is compute-bound and fast (the entire prompt is processed in parallel, like BERT). For a 100-token prompt, prefill runs one forward pass over 100 tokens simultaneously.

**Decode:** The model generates output tokens one at a time. Each step is a forward pass over 1 new token (plus reading the KV cache for all previous tokens). This is memory-bandwidth-bound — the GPU spends most of its time reading the KV cache, not doing compute.

**Why this matters for batching:** During prefill, different requests have different prompt lengths — you can't easily batch them together without padding (which wastes compute). During decode, every request in the batch adds exactly 1 new token per step — perfect for batching. Production systems handle this by separating prefill and decode or by using chunked prefill (breaking long prompts into fixed-size chunks).

For your implementation: keep it simple. Run prefill for new requests individually as they enter the batch, then join them into the continuous decode loop.

### 3.6 Throughput vs Latency — The Core Tradeoff

These are inversely related and you must understand both:

**Latency** = time from request submission to response. Measured as p50, p95, p99 in milliseconds.

**Throughput** = requests processed per second (or tokens generated per second). Measured as RPS or tokens/sec.

**The tradeoff:** Larger batches increase throughput (the GPU is doing more useful work per clock cycle) but increase latency for individual requests (a request that arrives alone must wait for other requests to fill the batch before processing begins). Smaller batches decrease throughput but decrease latency.

**Your configurable parameters:**
- `max_batch_size`: Maximum number of sequences in the continuous batch. Higher = more throughput, higher latency.
- `max_wait_ms`: How long to wait for the batch to fill before running a step. Lower = lower latency, less throughput.
- `max_new_tokens`: Maximum tokens to generate per request. Lower = faster turnaround, less GPU time per request.

**What to benchmark:** Run Locust with varying concurrent users and measure throughput vs p95 latency curves. This is the core result of the project.

### 3.7 Backpressure

When the inference queue is full and new requests keep arriving, you have two options:
1. Accept them all and let everything get slower (unbounded queue — bad, eventually crashes)
2. Reject new requests with HTTP 429 (Too Many Requests) and let the client retry

Option 2 is backpressure. It keeps the server stable and predictable. The client (or upstream load balancer) knows to back off and retry.

**Implementation:** Maintain a queue with a fixed max depth (e.g., 256). Before enqueuing a new request, check if the queue is full. If it is, immediately return HTTP 429 without touching the queue.

### 3.8 Prometheus Metrics

Prometheus is a time-series metrics database. Your server exposes a `/metrics` endpoint that Prometheus scrapes every 15 seconds. You instrument your code with counters and histograms.

**Metrics to implement:**

| Metric | Type | What it measures |
|--------|------|-----------------|
| `inference_latency_seconds` | Histogram | Per-request end-to-end latency, gives p50/p95/p99 |
| `time_to_first_token_seconds` | Histogram | TTFT — time from request arrival to first generated token (prefill latency) |
| `time_per_output_token_seconds` | Histogram | TPOT — average time between each successive generated token (decode step latency) |
| `tokens_per_second` | Gauge | Current throughput |
| `batch_size` | Histogram | Distribution of batch sizes at each decode step |
| `queue_depth` | Gauge | Current number of waiting requests |
| `kv_cache_hit_rate` | Gauge | Fraction of prefill steps that hit the KV cache |
| `requests_total` | Counter | Total requests, labeled by status (ok, 429, error) |

**Why TTFT and TPOT matter:** These are the two metrics inference companies actually measure and optimize separately. TTFT is dominated by prefill latency — how fast you can process the input prompt. TPOT is dominated by decode step latency and memory bandwidth. They have different optimization levers and different user impact. TTFT determines how long the user waits before seeing any output. TPOT determines how fast text streams. Every inference company (Fireworks, Baseten, Together AI, Groq) reports both. You should too.

**Why this matters:** These numbers are the proof. "Continuous batching improved throughput by 3x" means nothing without the data. With Prometheus, you have the data.

---

## 4. The Model — Qwen 0.5B for Content Moderation

### Why Qwen 0.5B

- **Fits on a GTX 1080:** ~500MB in INT8, leaving 7.5GB for KV cache and activations
- **Fast decode:** Small model = fast forward pass = good latency numbers
- **Capable enough for the task:** Binary classification (toxic/safe) is well within its capabilities
- **Open weights:** Available on HuggingFace, no API key required
- **No fine-tuning required:** Prompt engineering is sufficient for this use case

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# FP16 — fits easily, good balance of speed and accuracy
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# INT8 — requires bitsandbytes
# pip install bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="cuda"
)
```

### Prompt Template

```python
SYSTEM_PROMPT = (
    "You are a content moderation classifier. "
    "Given a piece of text, determine if it is toxic or safe. "
    "Toxic content includes hate speech, threats, harassment, and explicit harmful content. "
    "Reply with exactly one word: 'toxic' or 'safe'."
)

def build_prompt(text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

### Parsing the Response

FluxServe uses **first-word substring matching** to extract the label from Qwen's output. This is the simplest approach and works reliably when the prompt instructs the model to reply with exactly one word.

```python
def parse_response(generated_text: str) -> dict:
    """
    Extract label from Qwen's generated text.

    Strategy: substring match on the first 20 characters of output.
    The prompt instructs Qwen to reply with exactly 'toxic' or 'safe'.
    We check for those substrings rather than exact equality to handle
    edge cases like 'toxic.' or 'safe\n'.

    Fallback: 'unknown' if neither word appears. In production you would
    alert on unknown rate — high unknown rate means the prompt needs tuning.
    """
    text = generated_text.strip().lower()[:20]  # only check first 20 chars
    if "toxic" in text:
        return {"label": "toxic", "flagged": True}
    elif "safe" in text:
        return {"label": "safe", "flagged": False}
    else:
        return {"label": "unknown", "flagged": False}
```

**Why not logit masking?** Constraining generation to only emit "toxic" or "safe" tokens would guarantee single-token output and eliminate unknown responses. The implementation complexity is higher — you'd need to identify the token IDs for "toxic" and "safe" in Qwen's vocabulary and apply a logit mask before sampling. For this project, prompt engineering achieves the same result at lower complexity. If your unknown rate is above 5% during validation, switch to logit masking.

**Why not regex?** Substring match is sufficient and faster. The output space is constrained enough that false positives (e.g., "unsafe" matching "safe") are handled by checking "toxic" first.

### Validation

Before building the server, validate the model works for the task:

```python
# validate_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")

test_cases = [
    ("I love this product, it's amazing!", "safe"),
    ("I will hurt you if you don't comply", "toxic"),
    ("The weather is nice today", "safe"),
    ("You are worthless and should disappear", "toxic"),
]

for text, expected in test_cases:
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    result = parse_response(response)
    status = "✓" if result["label"] == expected else "✗"
    print(f"{status} Expected: {expected}, Got: {result['label']} | Input: {text[:50]}")
```

Run this before anything else. If the model is getting basic cases wrong, fix the prompt before building the server.

---

## 5. System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           FastAPI Server                 │
                    │                                          │
  HTTP Request ────▶│  POST /moderate                          │
                    │    │                                     │
                    │    ▼                                     │
                    │  Auth Middleware (Bearer token)          │
                    │    │                                     │
                    │    ▼                                     │
                    │  Rate Limit Check (queue depth)          │
                    │    │ full → 429                          │
                    │    │                                     │
                    │    ▼                                     │
                    │  Redis Cache Check (exact match)         │
                    │    │ hit → return cached response        │
                    │    │                                     │
                    │    ▼                                     │
                    │  Enqueue Request (asyncio.Queue)         │
                    │    │                                     │
                    │    ▼                                     │
                    │  Await Future (HTTP handler blocks here) │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │        Continuous Batch Worker           │
                    │  (separate asyncio task, runs forever)   │
                    │                                          │
                    │  LOOP:                                   │
                    │    1. Pull new requests from queue       │
                    │    2. Add to active batch (up to max)    │
                    │    3. Check KV cache for each new req    │
                    │    4. Run prefill for cache misses       │
                    │    5. Run one decode step for all active │
                    │    6. Check for finished sequences       │
                    │    7. Resolve Futures for finished reqs  │
                    │    8. Update Prometheus metrics          │
                    │    9. Go to 1                            │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │            GPU Layer                     │
                    │                                          │
                    │  Qwen 0.5B (FP16 or INT8)               │
                    │  KV Cache (fixed VRAM budget)            │
                    │  PyTorch CUDA kernels                    │
                    └─────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────┐
              │                        │                     │
    ┌─────────▼──────┐      ┌──────────▼──────┐   ┌────────▼──────┐
    │     Redis       │      │   Prometheus    │   │    Grafana    │
    │  Exact-match   │      │   /metrics      │   │  dashboard    │
    │  response cache│      │   endpoint      │   │               │
    └────────────────┘      └─────────────────┘   └───────────────┘
```

### Request Lifecycle

1. HTTP request hits `/moderate` with text payload
2. Auth middleware checks Bearer token
3. Queue depth check — if full, return 429 immediately
4. Redis exact-match check (MD5 hash of prompt) — if hit, return cached response
5. Request is wrapped in a `(prompt, asyncio.Future)` tuple and pushed to `asyncio.Queue`
6. HTTP handler awaits the Future (non-blocking — event loop continues)
7. Batch worker picks up the request on its next iteration
8. Batch worker checks KV cache — prefill if miss, skip prefill if hit
9. Request joins the active batch for decode steps
10. When generation finishes, batch worker resolves the Future with the result
11. HTTP handler wakes up, gets the result, writes to Redis cache, returns response

---

## 6. API Specification

### POST /moderate

**Request:**
```json
{
  "text": "string (required) — the content to moderate",
  "max_new_tokens": "integer (optional, default: 10) — max tokens to generate",
  "request_id": "string (optional) — for tracing"
}
```

**Response (200 OK):**
```json
{
  "label": "toxic | safe | unknown",
  "flagged": true,
  "tokens_generated": 3,
  "latency_ms": 142.3,
  "cached": false,
  "model": "qwen2.5-0.5b",
  "request_id": "req_abc123"
}
```

**Response (429 Too Many Requests):**
```json
{
  "error": "Server at capacity. Retry after 1s.",
  "queue_depth": 256,
  "retry_after": 1
}
```

**Response (401 Unauthorized):**
```json
{
  "error": "Invalid or missing Bearer token"
}
```

### GET /health

```json
{
  "status": "ok",
  "model": "qwen2.5-0.5b",
  "precision": "int8",
  "queue_depth": 12,
  "active_sequences": 8,
  "kv_cache_utilization": 0.43,
  "gpu_memory_used_gb": 3.2,
  "gpu_memory_total_gb": 8.0
}
```

### GET /metrics

Standard Prometheus text format. Scraped by Prometheus every 15 seconds.

### POST /admin/config (optional)

Hot-update server config without restart:
```json
{
  "max_batch_size": 16,
  "max_wait_ms": 20,
  "queue_max_depth": 256
}
```

---

## 7. Complete File Structure

```
fluxserve/
│
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app, routes, middleware registration
│   ├── models.py             # Pydantic request/response schemas
│   ├── model_loader.py       # Load Qwen 0.5B, handle FP16 vs INT8
│   ├── kv_cache.py           # KV cache pool, LRU eviction, hit/miss tracking
│   ├── batch_worker.py       # Continuous batching loop, decode steps
│   ├── queue_manager.py      # asyncio.Queue wrapper, backpressure logic
│   ├── cache.py              # Redis exact-match cache (MD5-keyed)
│   └── metrics.py            # Prometheus counters, histograms, gauges
│   └── middleware/
│       ├── auth.py           # Bearer token validation
│       └── rate_limit.py     # Queue depth check → 429
│
├── scripts/
│   ├── validate_model.py     # Test model output on known inputs before serving
│   └── benchmark.py          # Local throughput/latency benchmark (no Locust)
│
├── load_tests/
│   └── locustfile.py         # Locust load test — concurrent users, throughput, latency
│
├── k8s/
│   ├── deployment.yaml       # Kubernetes deployment spec
│   ├── service.yaml          # ClusterIP service
│   ├── hpa.yaml              # HPA autoscaling on queue_depth metric
│   └── prometheus-adapter.yaml  # Custom metric for HPA
│
├── Dockerfile
├── docker-compose.yml        # server + redis + prometheus + grafana
├── prometheus.yml            # Prometheus scrape config
├── grafana/
│   └── dashboard.json        # Pre-built Grafana dashboard
├── requirements.txt
└── README.md
```

---

## 8. Phase-by-Phase Build Guide

Build in this exact order. Do not skip ahead. Each phase produces something runnable that you can test before moving to the next.

---

### Phase 1: Model Validation (Day 1)

**Goal:** Qwen 0.5B runs on your GPU and produces correct moderation outputs.

**Do not build any server yet.** Just run the model directly.

```bash
pip install torch transformers accelerate bitsandbytes
```

```python
# scripts/validate_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model (FP16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda"
)
model.eval()

print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

SYSTEM_PROMPT = (
    "You are a content moderation classifier. "
    "Given a piece of text, determine if it is toxic or safe. "
    "Toxic content includes hate speech, threats, harassment, and explicit harmful content. "
    "Reply with exactly one word: 'toxic' or 'safe'."
)

def classify(text: str) -> tuple[str, float]:
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    latency = (time.time() - start) * 1000
    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip().lower()
    label = "toxic" if "toxic" in generated else "safe" if "safe" in generated else "unknown"
    return label, latency

test_cases = [
    ("I love this product, it is amazing!", "safe"),
    ("I will hurt you if you do not comply", "toxic"),
    ("The weather is nice today", "safe"),
    ("You are worthless and should disappear", "toxic"),
    ("This is a great community!", "safe"),
    ("Go kill yourself", "toxic"),
]

print("\nRunning validation...")
correct = 0
for text, expected in test_cases:
    label, latency = classify(text)
    ok = label == expected
    correct += ok
    status = "✓" if ok else "✗"
    print(f"  {status} [{latency:.0f}ms] Expected: {expected:5s} Got: {label:7s} | {text[:60]}")

print(f"\nAccuracy: {correct}/{len(test_cases)}")
print(f"VRAM after inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

**Success criteria:** At least 5/6 correct, latency under 2000ms per request.

**Also run with INT8:**
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="cuda"
)
```
Compare accuracy and latency. Record both numbers — they become your benchmark baseline.

**✅ How to verify Phase 1 is done:**
```
Expected output when you run validate_model.py:

  ✓ [450ms] Expected: safe    Got: safe    | I love this product, it is amazing!
  ✓ [480ms] Expected: toxic   Got: toxic   | I will hurt you if you do not comply
  ✓ [430ms] Expected: safe    Got: safe    | The weather is nice today
  ✓ [460ms] Expected: toxic   Got: toxic   | You are worthless and should disappear
  ✓ [440ms] Expected: safe    Got: safe    | This is a great community!
  ✓ [455ms] Expected: toxic   Got: toxic   | Go kill yourself

  Accuracy: 6/6
  VRAM after inference: 1.02 GB

If you see less than 5/6 correct — the prompt needs tuning before you touch any server code.
If VRAM is above 2GB — something is wrong with the dtype, recheck torch_dtype=torch.float16.
If latency is above 5000ms per request — CUDA is not being used, check device_map="cuda".
```

---

### Phase 2: Naive Sequential Server (Day 1-2)

**Goal:** A working FastAPI server that handles one request at a time. No batching, no KV cache management, no queue. Just a working endpoint.

This is your baseline. You will benchmark it and then compare against the continuous batching version in Phase 4.

```python
# app/main.py (Phase 2 — naive version)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.models import ModerateRequest, ModerateResponse
from app.model_loader import load_model
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
```

```python
# app/model_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PRECISION = os.getenv("PRECISION", "fp16")  # "fp16" or "int8"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if PRECISION == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, load_in_8bit=True, device_map="cuda"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
        )
    model.eval()
    return model, tokenizer

SYSTEM_PROMPT = (
    "You are a content moderation classifier. "
    "Reply with exactly one word: 'toxic' or 'safe'."
)

def run_inference(model, tokenizer, text: str, max_new_tokens: int = 10) -> dict:
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip().lower()
    label = "toxic" if "toxic" in generated else "safe" if "safe" in generated else "unknown"
    return {"label": label, "flagged": label == "toxic"}
```

**Benchmark Phase 2 before moving on:**
```bash
# Run Locust with 1 user, measure baseline latency
locust -f load_tests/locustfile.py --headless -u 1 -r 1 --run-time 60s
```
Record: p50 latency, p95 latency, requests/sec. This is your "before" number.

**✅ How to verify Phase 2 is done:**
```bash
# 1. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Health check — should return immediately
curl http://localhost:8000/health
# Expected: {"status":"ok","model":"qwen2.5-0.5b"}

# 3. Test a safe input
curl -X POST http://localhost:8000/moderate \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "The weather is nice today"}'
# Expected: {"label":"safe","flagged":false,"latency_ms":~500,...}

# 4. Test a toxic input
curl -X POST http://localhost:8000/moderate \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "I will hurt you"}'
# Expected: {"label":"toxic","flagged":true,"latency_ms":~500,...}

# 5. Test auth rejection
curl -X POST http://localhost:8000/moderate \
  -H "Authorization: Bearer wrong-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'
# Expected: 401 Unauthorized
```

**If label comes back "unknown":** The prompt isn't constraining Qwen tightly enough.
Go back to `model_loader.py` and make the system prompt more explicit.

**If latency is above 5000ms:** CUDA is not being used — check `device_map="cuda"` in `load_model()`.

---

### Phase 3: Redis Caching + Prometheus (Day 2-3)

**Goal:** Add exact-match Redis caching and Prometheus instrumentation to the naive server.

**Why before continuous batching:** Cache and metrics are independent of the batching architecture. Adding them now means you have clean baselines for both.

```python
# app/cache.py
import redis
import hashlib
import json
from typing import Optional

class ResponseCache:
    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 3600):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl

    def _key(self, text: str) -> str:
        return f"fluxserve:response:{hashlib.md5(text.encode()).hexdigest()}"

    def get(self, text: str) -> Optional[dict]:
        key = self._key(text)
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, text: str, response: dict):
        key = self._key(text)
        self.client.setex(key, self.ttl, json.dumps(response))
```

```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Counters
requests_total = Counter(
    "fluxserve_requests_total",
    "Total requests",
    ["status"]  # ok, cached, rate_limited, error
)

# Histograms — these give you p50/p95/p99 automatically
inference_latency = Histogram(
    "fluxserve_inference_latency_seconds",
    "Per-request end-to-end latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# TTFT: time from request arrival to first generated token
# Dominated by prefill — how fast you process the input prompt
time_to_first_token = Histogram(
    "fluxserve_time_to_first_token_seconds",
    "Time from request arrival to first generated token (prefill latency)",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

# TPOT: average time per output token during the decode phase
# Dominated by memory bandwidth — how fast you read the KV cache
time_per_output_token = Histogram(
    "fluxserve_time_per_output_token_seconds",
    "Average time per generated token during decode",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

batch_size_hist = Histogram(
    "fluxserve_batch_size",
    "Batch size at each decode step",
    buckets=[1, 2, 4, 8, 12, 16, 24, 32]
)

tokens_generated_hist = Histogram(
    "fluxserve_tokens_generated",
    "Tokens generated per request",
    buckets=[1, 2, 3, 5, 10, 20, 50]
)

# Gauges — current values
queue_depth_gauge = Gauge("fluxserve_queue_depth", "Current queue depth")
active_sequences_gauge = Gauge("fluxserve_active_sequences", "Active sequences in batch")
kv_cache_utilization_gauge = Gauge("fluxserve_kv_cache_utilization", "KV cache utilization 0-1")
tokens_per_second_gauge = Gauge("fluxserve_tokens_per_second", "Current tokens/sec throughput")
gpu_memory_used_gauge = Gauge("fluxserve_gpu_memory_used_gb", "GPU memory used in GB")

def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Add to `main.py`:
```python
@app.get("/metrics")
async def metrics():
    return metrics_endpoint()
```

**✅ How to verify Phase 3 is done:**
```bash
# 1. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 2. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Verify Prometheus metrics endpoint returns data
curl http://localhost:8000/metrics
# Expected: text output starting with "# HELP fluxserve_requests_total..."
# You should see all your metric names in the output

# 4. Test Redis cache hit
# Send the same request twice — second should return cached:true
curl -X POST http://localhost:8000/moderate \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "The weather is nice today"}'
# First call: {"label":"safe","cached":false,...}

curl -X POST http://localhost:8000/moderate \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "The weather is nice today"}'
# Second call: {"label":"safe","cached":true,...}  ← must be true and much faster

# 5. Verify Redis actually stored it
redis-cli get "fluxserve:response:$(echo -n 'The weather is nice today' | md5sum | cut -d' ' -f1)"
# Expected: JSON string with the cached response

# 6. Check metric counters incremented
curl http://localhost:8000/metrics | grep fluxserve_requests_total
# Expected: fluxserve_requests_total{status="ok"} 1.0
#           fluxserve_requests_total{status="cached"} 1.0
```

**If Redis connection fails:** Make sure Redis is running on localhost:6379.
Check with `redis-cli ping` — should return `PONG`.

**If metrics endpoint returns empty:** Make sure you added the `/metrics` route to `main.py` and imported `metrics_endpoint`.

---

### Phase 4: Continuous Batching Engine (Day 3-5)

**Goal:** Replace the naive sequential inference with a continuous batching worker. This is the most complex and most important phase.

**Concept recap:** The HTTP handler no longer runs inference directly. Instead, it:
1. Enqueues the request as a `(prompt, asyncio.Future)` tuple
2. Awaits the Future
3. Returns the result when the Future is resolved

The batch worker runs in an infinite loop:
1. Pulls requests from the queue
2. Adds them to the active batch
3. Runs one decode step for all active sequences
4. Removes finished sequences and resolves their Futures
5. Goes back to 1

```python
# app/queue_manager.py
import asyncio
from dataclasses import dataclass
from typing import Any

@dataclass
class InferenceRequest:
    text: str
    max_new_tokens: int
    future: asyncio.Future
    request_id: str

class QueueManager:
    def __init__(self, max_depth: int = 256):
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=max_depth)
        self.max_depth = max_depth

    def is_full(self) -> bool:
        return self.queue.qsize() >= self.max_depth

    async def enqueue(self, request: InferenceRequest) -> bool:
        """Returns False if queue is full (caller should return 429)."""
        if self.is_full():
            return False
        await self.queue.put(request)
        return True

    async def dequeue(self) -> InferenceRequest:
        return await self.queue.get()

    def depth(self) -> int:
        return self.queue.qsize()
```

```python
# app/batch_worker.py
import asyncio
import torch
import time
from typing import List, Optional
from dataclasses import dataclass, field
from app.queue_manager import QueueManager, InferenceRequest
from app.kv_cache import KVCachePool
from app.metrics import (
    batch_size_hist, active_sequences_gauge, 
    tokens_per_second_gauge, queue_depth_gauge
)

@dataclass
class ActiveSequence:
    request: InferenceRequest
    input_ids: torch.Tensor          # Current token sequence on GPU
    attention_mask: torch.Tensor     # Attention mask
    past_key_values: Optional[tuple] # KV cache for this sequence
    tokens_generated: int = 0
    start_time: float = field(default_factory=time.time)

class BatchWorker:
    def __init__(
        self,
        model,
        tokenizer,
        queue_manager: QueueManager,
        kv_cache_pool: KVCachePool,
        max_batch_size: int = 8,
        max_wait_ms: float = 20.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.queue = queue_manager
        self.kv_cache = kv_cache_pool
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.active: List[ActiveSequence] = []
        self._running = False
        self._token_count = 0
        self._last_throughput_calc = time.time()

    async def run(self):
        """Main loop — runs forever as an asyncio task."""
        self._running = True
        while self._running:
            # Pull new requests into the batch
            await self._fill_batch()

            if not self.active:
                # Nothing to do — yield control and try again
                await asyncio.sleep(0.001)
                continue

            # Run one decode step for all active sequences
            await asyncio.get_event_loop().run_in_executor(
                None, self._decode_step
            )

            # Check for finished sequences and resolve their futures
            self._resolve_finished()

            # Update metrics
            self._update_metrics()

    async def _fill_batch(self):
        """Pull waiting requests into the active batch."""
        slots_available = self.max_batch_size - len(self.active)
        if slots_available == 0:
            return

        deadline = time.time() + (self.max_wait_ms / 1000)

        while len(self.active) < self.max_batch_size:
            try:
                timeout = max(0, deadline - time.time())
                request = await asyncio.wait_for(
                    self.queue.dequeue(), timeout=timeout
                )
                # Run prefill for this new sequence
                seq = await asyncio.get_event_loop().run_in_executor(
                    None, self._prefill, request
                )
                self.active.append(seq)
            except asyncio.TimeoutError:
                break

    def _prefill(self, request: InferenceRequest) -> ActiveSequence:
        """
        Run the prefill step for a new request.
        This processes the entire prompt and populates the KV cache.
        """
        from app.model_loader import SYSTEM_PROMPT
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{request.text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            # Forward pass with use_cache=True to get past_key_values
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True
            )

        # The model's last token logit tells us the first generated token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Build the sequence with the first generated token appended
        new_input_ids = torch.cat([inputs.input_ids, next_token], dim=-1)
        new_attention_mask = torch.cat([
            inputs.attention_mask,
            torch.ones((1, 1), device="cuda")
        ], dim=-1)

        return ActiveSequence(
            request=request,
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            past_key_values=outputs.past_key_values,
            tokens_generated=1,
        )

    def _decode_step(self):
        """
        Run one decode step for ALL active sequences simultaneously.
        This is the continuous batching core — all sequences advance together.

        Note: True continuous batching requires padding handling and variable-length
        sequences. This implementation runs sequences independently per step for
        clarity. A production implementation would pad and batch the decode calls.
        """
        for seq in self.active:
            with torch.no_grad():
                # Only pass the LAST token — KV cache handles the rest
                last_token = seq.input_ids[:, -1:]
                last_mask = seq.attention_mask

                outputs = self.model(
                    input_ids=last_token,
                    attention_mask=last_mask,
                    past_key_values=seq.past_key_values,
                    use_cache=True,
                    return_dict=True
                )

            # Sample next token (greedy)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            # Update sequence state
            seq.input_ids = torch.cat([seq.input_ids, next_token], dim=-1)
            seq.attention_mask = torch.cat([
                seq.attention_mask,
                torch.ones((1, 1), device="cuda")
            ], dim=-1)
            seq.past_key_values = outputs.past_key_values
            seq.tokens_generated += 1
            self._token_count += 1

    def _resolve_finished(self):
        """
        Check which sequences are done and resolve their HTTP futures.
        A sequence is done if it generated a stop token or hit max_new_tokens.
        """
        still_active = []
        for seq in self.active:
            last_token_id = seq.input_ids[0, -1].item()
            is_eos = last_token_id == self.tokenizer.eos_token_id
            is_max = seq.tokens_generated >= seq.request.max_new_tokens

            if is_eos or is_max:
                # Decode the generated portion
                prompt_len = len(self.tokenizer.encode(
                    seq.request.text, add_special_tokens=False
                ))
                generated_ids = seq.input_ids[0, -(seq.tokens_generated):]
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip().lower()

                label = "toxic" if "toxic" in generated_text else \
                        "safe" if "safe" in generated_text else "unknown"

                latency_ms = (time.time() - seq.start_time) * 1000
                ttft_ms = seq.first_token_time * 1000  # set during prefill
                tpot_ms = (latency_ms - ttft_ms) / max(seq.tokens_generated - 1, 1)

                result = {
                    "label": label,
                    "flagged": label == "toxic",
                    "tokens_generated": seq.tokens_generated,
                    "latency_ms": latency_ms,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                }

                # Resolve the Future — this wakes up the HTTP handler
                if not seq.request.future.done():
                    seq.request.future.get_event_loop().call_soon_threadsafe(
                        seq.request.future.set_result, result
                    )
            else:
                still_active.append(seq)

        self.active = still_active

    def _update_metrics(self):
        batch_size_hist.observe(len(self.active))
        active_sequences_gauge.set(len(self.active))
        queue_depth_gauge.set(self.queue.depth())

        # Calculate tokens/sec
        now = time.time()
        elapsed = now - self._last_throughput_calc
        if elapsed >= 1.0:
            tokens_per_second_gauge.set(self._token_count / elapsed)
            self._token_count = 0
            self._last_throughput_calc = now

        # GPU memory
        try:
            import torch
            gpu_memory_used_gauge.set(torch.cuda.memory_allocated() / 1e9)
        except Exception:
            pass
```

**Update main.py to use the batch worker:**
```python
# app/main.py (Phase 4 — updated)
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
import asyncio, time, os, uuid

queue_manager = QueueManager(max_depth=int(os.getenv("QUEUE_MAX_DEPTH", "256")))
cache = ResponseCache()
batch_worker: BatchWorker = None

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
        return ModerateResponse(**cached, cached=True, request_id=request_id)

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
```

**Benchmark Phase 4:**
```bash
locust -f load_tests/locustfile.py --headless -u 16 -r 2 --run-time 120s
```
Record: p50/p95/p99 latency, requests/sec, tokens/sec. Compare against Phase 2 numbers. This is your "continuous batching improved throughput by Xx" claim.

**✅ How to verify Phase 4 is done:**
```bash
# 1. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Send a single request — should still work exactly like Phase 2
curl -X POST http://localhost:8000/moderate \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "I will hurt you"}'
# Expected: {"label":"toxic","flagged":true,...}

# 3. Check health — active_sequences should show batch worker is running
curl http://localhost:8000/health
# Expected: {"status":"ok","queue_depth":0,"active_sequences":0,"gpu_memory_used_gb":1.02}

# 4. Send 4 concurrent requests to verify batching is happening
# Open 4 terminal tabs and fire simultaneously, or use this one-liner:
for i in {1..4}; do
  curl -s -X POST http://localhost:8000/moderate \
    -H "Authorization: Bearer dev-key" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"test message $i\"}" &
done
wait
# All 4 should return results — if any hang indefinitely, the batch worker loop has a bug

# 5. Verify batch_size metric shows values > 1
curl http://localhost:8000/metrics | grep fluxserve_batch_size
# Expected: bucket counts showing batch sizes > 1 when under load

# 6. Test backpressure — temporarily set QUEUE_MAX_DEPTH=1 and flood with requests
# Expected: some requests return HTTP 429 with {"error":"Server at capacity",...}

# 7. Run the comparative benchmark
# Phase 2 (naive) RPS vs Phase 4 (batched) RPS at 8 concurrent users
# You should see at least 2-3x improvement
locust -f load_tests/locustfile.py --headless -u 8 -r 2 --run-time 60s --host http://localhost:8000
```

**If requests hang and never return:** The batch worker task isn't starting.
Check that `asyncio.create_task(batch_worker.run())` is called inside the lifespan context.

**If you see no batching improvement:** The `_decode_step` is running sequences sequentially in a Python loop instead of a batched tensor op. This is expected for the simplified implementation — the throughput gain comes from keeping the GPU busy across requests, not true tensor batching.

**If futures never resolve:** Check `_resolve_finished` — the `call_soon_threadsafe` pattern is required because the decode step runs in a thread executor, not the event loop thread.

---

### Phase 5: KV Cache Management (Day 5-6)

**Goal:** Add explicit KV cache tracking, memory budget enforcement, and LRU eviction.

The batch worker already uses `past_key_values` from PyTorch — this phase adds a management layer on top that tracks memory usage and evicts sequences when VRAM is tight.

```python
# app/kv_cache.py
import time
from collections import OrderedDict
from typing import Optional, Tuple
import torch
from app.metrics import kv_cache_utilization_gauge

class KVCacheEntry:
    def __init__(self, sequence_id: str, past_key_values: tuple, size_bytes: int):
        self.sequence_id = sequence_id
        self.past_key_values = past_key_values
        self.size_bytes = size_bytes
        self.last_accessed = time.time()
        self.hits = 0

class KVCachePool:
    def __init__(self, max_sequences: int = 64, max_bytes: Optional[int] = None):
        """
        max_sequences: Maximum number of concurrent sequences in cache.
        max_bytes: Maximum VRAM for KV cache (default: 4GB).
        """
        self.max_sequences = max_sequences
        self.max_bytes = max_bytes or (4 * 1024 * 1024 * 1024)  # 4GB default
        self.cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self.total_bytes = 0
        self._hits = 0
        self._misses = 0

    def get(self, sequence_id: str) -> Optional[tuple]:
        """Return past_key_values for sequence_id, or None if not cached."""
        if sequence_id in self.cache:
            entry = self.cache[sequence_id]
            entry.last_accessed = time.time()
            entry.hits += 1
            self.cache.move_to_end(sequence_id)  # LRU update
            self._hits += 1
            self._update_metrics()
            return entry.past_key_values
        self._misses += 1
        self._update_metrics()
        return None

    def put(self, sequence_id: str, past_key_values: tuple):
        """Store past_key_values, evicting LRU entries if necessary."""
        size_bytes = self._estimate_size(past_key_values)

        # Evict if necessary
        while (
            len(self.cache) >= self.max_sequences or
            self.total_bytes + size_bytes > self.max_bytes
        ) and self.cache:
            evicted_id, evicted_entry = self.cache.popitem(last=False)  # LRU = first
            self.total_bytes -= evicted_entry.size_bytes
            # Free GPU memory
            del evicted_entry.past_key_values
            torch.cuda.empty_cache()

        entry = KVCacheEntry(sequence_id, past_key_values, size_bytes)
        self.cache[sequence_id] = entry
        self.total_bytes += size_bytes
        self._update_metrics()

    def evict(self, sequence_id: str):
        """Explicitly remove a sequence from the cache (when request completes)."""
        if sequence_id in self.cache:
            entry = self.cache.pop(sequence_id)
            self.total_bytes -= entry.size_bytes
            del entry.past_key_values
            torch.cuda.empty_cache()
            self._update_metrics()

    def _estimate_size(self, past_key_values: tuple) -> int:
        """Estimate VRAM usage of past_key_values in bytes."""
        total = 0
        for layer_kv in past_key_values:
            for tensor in layer_kv:
                total += tensor.nelement() * tensor.element_size()
        return total

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def utilization(self) -> float:
        return len(self.cache) / self.max_sequences

    def _update_metrics(self):
        kv_cache_utilization_gauge.set(self.utilization())
```

**✅ How to verify Phase 5 is done:**
```bash
# 1. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Send a request and check KV cache utilization appears in metrics
curl http://localhost:8000/metrics | grep kv_cache
# Expected:
# fluxserve_kv_cache_utilization 0.015625  (1 sequence / 64 max)

# 3. Check health endpoint shows KV cache stats
curl http://localhost:8000/health
# Expected: includes kv_cache_utilization field > 0 after a request

# 4. Verify hit rate tracking — send the same prompt 3 times
# (Not the same as Redis cache — Redis caches the final response,
#  KV cache stores the internal attention tensors during generation)
# After 3 requests check:
curl http://localhost:8000/metrics | grep kv_cache_hit_rate
# Expected: fluxserve_kv_cache_hit_rate > 0 after repeated identical prompts

# 5. Verify LRU eviction works — temporarily set max_sequences=2 in KVCachePool
# Then send 5 different requests. The cache should never grow beyond 2 entries.
# Check VRAM doesn't grow unboundedly:
curl http://localhost:8000/metrics | grep gpu_memory
# Expected: fluxserve_gpu_memory_used_gb stays stable, not growing each request

# 6. Verify size estimation is working
python3 -c "
from app.kv_cache import KVCachePool
import torch
pool = KVCachePool(max_sequences=4)
# Create fake past_key_values (24 layers, each with K and V tensors)
fake_pkv = tuple(
    (torch.zeros(1, 2, 10, 64, device='cuda', dtype=torch.float16),
     torch.zeros(1, 2, 10, 64, device='cuda', dtype=torch.float16))
    for _ in range(24)
)
size = pool._estimate_size(fake_pkv)
print(f'KV size for seq_len=10: {size / 1024:.1f} KB')
# Expected: ~240 KB (24 layers * 2 tensors * 1*2*10*64 * 2 bytes)
"
```

**If VRAM grows without bound:** The `del evicted_entry.past_key_values` + `torch.cuda.empty_cache()` in the eviction loop isn't actually freeing memory. This is a common PyTorch gotcha — tensors aren't freed until all references are dropped. Make sure no other code is holding a reference to the evicted tensors.

**If hit rate is always 0:** The sequence_id used as cache key needs to be consistent for the same prompt. Check that `batch_worker._prefill` uses the full prompt string (including system prompt) as the key, not just `request.text`.

---

### Phase 6: INT8 Quantization Benchmark (Day 6)

**Goal:** Quantify the throughput and latency impact of INT8 quantization.

You already have the `load_in_8bit=True` option in `model_loader.py`. Now benchmark it properly.

```python
# scripts/benchmark.py
"""
Run this script to compare FP16 vs INT8 performance.
No Locust needed — direct inference loop.
"""
import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_TEXT = "I hate everything about this. You should all be ashamed."
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50

def benchmark(model, tokenizer, text: str, max_new_tokens: int = 10) -> list[float]:
    prompt = (
        f"<|im_start|>system\nYou are a content moderation classifier. "
        f"Reply with exactly one word: toxic or safe.<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    latencies = []

    # Warmup
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tokenizer.eos_token_id)

    # Benchmark
    for _ in range(BENCHMARK_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        end = time.perf_counter()
        tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        latencies.append((end - start) * 1000)  # ms

    return latencies

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

for precision in ["fp16", "int8"]:
    print(f"\n{'='*50}")
    print(f"Benchmarking {precision.upper()}")
    print(f"{'='*50}")

    if precision == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, load_in_8bit=True, device_map="cuda"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
        )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM used: {vram:.2f} GB")

    latencies = benchmark(model, tokenizer, TEST_TEXT)
    latencies.sort()
    p50 = latencies[len(latencies)//2]
    p95 = latencies[int(len(latencies)*0.95)]
    p99 = latencies[int(len(latencies)*0.99)]
    avg = sum(latencies)/len(latencies)

    print(f"Latency (ms): p50={p50:.1f} p95={p95:.1f} p99={p99:.1f} avg={avg:.1f}")
    print(f"Throughput: {1000/avg:.1f} requests/sec")

    del model
    torch.cuda.empty_cache()
```

Record these numbers. They become your resume bullets.

**✅ How to verify Phase 6 is done:**
```bash
# 1. Run the benchmark script directly
python scripts/benchmark.py

# Expected output format:
# ==================================================
# Benchmarking FP16
# ==================================================
# VRAM used: 1.02 GB
# Latency (ms): p50=480.2 p95=510.1 p99=521.3 avg=485.0
# Throughput: 2.1 requests/sec
#
# ==================================================
# Benchmarking INT8
# ==================================================
# VRAM used: 0.54 GB
# Latency (ms): p50=495.1 p95=525.3 p99=538.7 avg=500.2
# Throughput: 2.0 requests/sec

# 2. Verify INT8 accuracy hasn't degraded
# Re-run validate_model.py with INT8 model — should still be 5/6 or 6/6
PRECISION=int8 python scripts/validate_model.py

# 3. Record the VRAM delta — this is your resume number
# "Reduced VRAM from X GB to Y GB" — should be roughly 2x reduction

# 4. Verify the server still works end-to-end with INT8
PRECISION=int8 uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/moderate \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "You are terrible"}'
# Expected: {"label":"toxic","flagged":true,...}
```

**If INT8 accuracy drops below 4/6:** The GTX 1080 may have issues with bitsandbytes INT8 on Pascal architecture. Fall back to `torch.quantization.quantize_dynamic` as an alternative, or use FP16 only and note INT8 as a known limitation.

**If VRAM doesn't decrease with INT8:** `load_in_8bit=True` requires bitsandbytes to be correctly installed with CUDA support. Run `python -c "import bitsandbytes; print(bitsandbytes.__version__)"` to verify. If it fails, reinstall: `pip install bitsandbytes --upgrade`.

---

### Phase 7: Locust Load Test (Day 7)

**Goal:** Measure the server under realistic concurrent load.

```python
# load_tests/locustfile.py
from locust import HttpUser, task, between
import random

SAMPLE_TEXTS = [
    "I love this community and everyone in it!",
    "You should all go to hell, I hate you",
    "The weather is great today",
    "I will find you and make you regret this",
    "Thanks for your help, really appreciate it",
    "This is absolute garbage, worst product ever",
    "Anyone want to grab lunch today?",
    "Kill yourself you worthless piece of trash",
    "Just finished a great run!",
    "Why do you exist? You ruin everything",
]

class ModerateUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.headers = {"Authorization": "Bearer dev-key"}

    @task
    def moderate(self):
        text = random.choice(SAMPLE_TEXTS)
        self.client.post(
            "/moderate",
            json={"text": text, "max_new_tokens": 10},
            headers=self.headers,
        )
```

**Run at increasing concurrency and record results:**
```bash
# 1 user — baseline
locust -f load_tests/locustfile.py --headless -u 1 -r 1 --run-time 60s --host http://localhost:8000

# 4 users
locust -f load_tests/locustfile.py --headless -u 4 -r 1 --run-time 60s --host http://localhost:8000

# 8 users
locust -f load_tests/locustfile.py --headless -u 8 -r 2 --run-time 60s --host http://localhost:8000

# 16 users — watch for 429s and latency degradation
locust -f load_tests/locustfile.py --headless -u 16 -r 2 --run-time 60s --host http://localhost:8000
```

**Build a results table:**

| Concurrent Users | RPS | p50 (ms) | p95 (ms) | p99 (ms) | 429 Rate |
|-----------------|-----|----------|----------|----------|----------|
| 1 (naive)       | ?   | ?        | ?        | ?        | 0%       |
| 1 (batched)     | ?   | ?        | ?        | ?        | 0%       |
| 4 (batched)     | ?   | ?        | ?        | ?        | 0%       |
| 8 (batched)     | ?   | ?        | ?        | ?        | 0%       |
| 16 (batched)    | ?   | ?        | ?        | ?        | ?        |

Fill this in with real numbers. "Continuous batching improved throughput by Xx vs naive sequential inference at N concurrent users" — that's your headline resume bullet.

**✅ How to verify Phase 7 is done:**
```bash
# 1. Make sure the server is running with continuous batching (Phase 4+)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Run the full benchmark suite and check output looks sensible
locust -f load_tests/locustfile.py --headless -u 1 -r 1 --run-time 60s \
  --host http://localhost:8000 --csv results/run_1
# Expected: p50 < 1000ms, 0 failures, RPS > 0.5

# 3. Ramp up and verify throughput increases with more users (up to a point)
locust -f load_tests/locustfile.py --headless -u 8 -r 2 --run-time 60s \
  --host http://localhost:8000 --csv results/run_8
# Expected: RPS higher than run_1, latency somewhat higher but not proportional

# 4. Find your breaking point — where 429s start appearing
locust -f load_tests/locustfile.py --headless -u 16 -r 2 --run-time 60s \
  --host http://localhost:8000 --csv results/run_16
# Expected: some 429s appear, Failures% > 0

# 5. Verify CSV output was created and has real numbers
cat results/run_8_stats.csv
# Expected: rows with non-zero RPS, latency percentiles

# 6. Cross-check with Prometheus — tokens/sec in metrics should match Locust RPS * avg_tokens
curl http://localhost:8000/metrics | grep tokens_per_second
# Expected: value roughly equal to (Locust RPS * 2) since avg output is ~2 tokens

# 7. Confirm your headline number: Phase 2 RPS at 1 user vs Phase 4 RPS at 8 users
# Write it down: "X req/sec naive → Y req/sec batched = Zx improvement"
# This goes directly into your resume bullet.
```

**If all requests fail immediately:** The server isn't running or the host URL is wrong. Check `--host http://localhost:8000` matches where uvicorn is listening.

**If RPS doesn't improve with more users:** The batch worker may not be filling the batch. Add a `print(f"Batch size: {len(self.active)}")` in `_decode_step` to see what batch sizes you're actually hitting.

---

### Phase 8: Docker Compose (Day 7-8)

```yaml
# docker-compose.yml
version: '3.8'

services:
  inference-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=dev-key
      - REDIS_HOST=redis
      - PRECISION=fp16
      - MAX_BATCH_SIZE=8
      - MAX_WAIT_MS=20
      - QUEUE_MAX_DEPTH=256
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/dashboard.json:/etc/grafana/provisioning/dashboards/dashboard.json
    depends_on:
      - prometheus
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fluxserve'
    static_configs:
      - targets: ['inference-server:8000']
```

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**✅ How to verify Phase 8 is done:**
```bash
# 1. Build and start all services
docker compose up --build -d

# 2. Wait ~60 seconds for model to load, then check all containers are running
docker compose ps
# Expected: all 4 services show status "Up" or "running"
# inference-server   Up
# redis              Up
# prometheus         Up
# grafana            Up

# 3. Verify the server is responding inside Docker
curl http://localhost:8000/health
# Expected: {"status":"ok",...}

# 4. Send a test request through Docker
curl -X POST http://localhost:8000/moderate \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is terrible"}'
# Expected: {"label":"toxic","flagged":true,...}

# 5. Verify Prometheus is scraping — open http://localhost:9090
# Go to Status > Targets — fluxserve should show state=UP and no errors

# 6. Verify Grafana is accessible — open http://localhost:3000
# Login: admin / admin — dashboard should load

# 7. Check for errors in server logs
docker compose logs inference-server --tail=50
# Expected: no ERROR or EXCEPTION lines

# 8. Run Locust against the Dockerized server to confirm perf is the same
locust -f load_tests/locustfile.py --headless -u 8 -r 2 --run-time 60s \
  --host http://localhost:8000
# Expected: similar numbers to Phase 7 — Docker overhead is minimal
```

**If container exits immediately:** `docker compose logs inference-server` — usually a Python import error or missing env var.

**If GPU not found in Docker:** Run `docker exec -it $(docker compose ps -q inference-server) nvidia-smi` — if this fails, install `nvidia-container-toolkit` on the host.

**If Prometheus shows target DOWN:** The service name in `prometheus.yml` (`inference-server`) must exactly match the Docker Compose service name.

```bash
# Install minikube
minikube start --cpus 4 --memory 8192

# Enable metrics server for HPA
minikube addons enable metrics-server
```

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fluxserve
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fluxserve
  template:
    metadata:
      labels:
        app: fluxserve
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: fluxserve
          image: fluxserve:latest
          imagePullPolicy: Never
          ports:
            - containerPort: 8000
          env:
            - name: API_KEY
              value: "dev-key"
            - name: MAX_BATCH_SIZE
              value: "8"
            - name: REDIS_HOST
              value: redis
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
```

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fluxserve-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fluxserve
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

**✅ How to verify Phase 9 is done:**
```bash
# 1. Start minikube
minikube start --cpus 4 --memory 8192
minikube addons enable metrics-server

# 2. Load your Docker image into minikube's registry
eval $(minikube docker-env)
docker build -t fluxserve:latest .

# 3. Apply all manifests
kubectl apply -f k8s/

# 4. Wait for pod to be running
kubectl get pods -w
# Expected: fluxserve-xxxxxxxxx-xxxxx   1/1   Running   0   60s

# 5. Check pod logs to verify model loaded
kubectl logs -l app=fluxserve --tail=20
# Expected: "Model loaded. VRAM used: X.XX GB" and no ERROR lines

# 6. Port-forward to test the endpoint
kubectl port-forward service/fluxserve 8000:8000 &
curl http://localhost:8000/health
# Expected: {"status":"ok",...}

# 7. Send a real request through k8s
curl -X POST http://localhost:8000/moderate \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "You are worthless"}'
# Expected: {"label":"toxic","flagged":true,...}

# 8. Verify HPA is active
kubectl get hpa
# Expected:
# NAME           REFERENCE             TARGETS   MINPODS   MAXPODS   REPLICAS
# fluxserve-hpa  Deployment/fluxserve  12%/70%   1         4         1

# 9. Trigger autoscaling by running Locust against the k8s service
locust -f load_tests/locustfile.py --headless -u 16 -r 2 --run-time 120s \
  --host http://localhost:8000
# While running, watch replicas scale up:
kubectl get hpa -w
# Expected: REPLICAS increases from 1 toward 4 under load
```

**If pod stays in Pending:** `kubectl describe pod <pod-name>` — usually insufficient CPU/memory resources in minikube. Try `minikube start --cpus 4 --memory 10240`.

**If HPA never scales:** The metrics-server addon must be enabled. Check `kubectl top pods` works first — if it returns an error, the metrics server isn't running.

**If image not found:** Make sure you ran `eval $(minikube docker-env)` before `docker build` so the image lands in minikube's Docker daemon, not your host's.

### The Numbers You Need

Before you write the resume bullets, you need these numbers from real runs:

1. **Naive sequential vs continuous batching throughput ratio** (e.g., "3x improvement")
2. **p50/p95/p99 latency under 8 concurrent users**
3. **TTFT (Time To First Token)** — p50 and p95, measures prefill performance
4. **TPOT (Time Per Output Token)** — p50 and p95, measures decode step performance
5. **FP16 vs INT8 latency and VRAM difference**
6. **KV cache hit rate** (what percentage of prefill steps were skipped)
7. **Tokens/sec throughput** at peak load
8. **Maximum sustainable RPS** before 429s start appearing

### How to Run a Clean Benchmark

1. Start the server fresh: `docker compose up -d`
2. Wait 30 seconds for model to load
3. Warm up: `locust -u 1 -r 1 --run-time 30s` (don't record these)
4. Run benchmark: `locust -u N -r 1 --run-time 120s --csv results/run_N`
5. Repeat for N = 1, 2, 4, 8, 16
6. Check Prometheus for tokens/sec and batch size distribution

### What Good Numbers Look Like

These are approximate targets for Qwen 0.5B on a GTX 1080. Your numbers may differ.

| Metric | Target |
|--------|--------|
| Naive sequential RPS | 1-3 req/sec |
| Continuous batching RPS | 5-15 req/sec |
| Throughput improvement | 3-8x |
| TTFT p50 (single user) | 100-400ms |
| TTFT p95 (single user) | 200-800ms |
| TPOT p50 (decode step) | 10-50ms per token |
| p50 latency (8 users) | 200-800ms |
| p95 latency (8 users) | 500-2000ms |
| VRAM (FP16) | ~1.0 GB |
| VRAM (INT8) | ~0.5 GB |
| KV cache hit rate | 5-20% (depends on repeat queries) |

---

## 11. Design Tradeoffs

**Continuous batching vs static batching:** Static batching is simpler but wastes GPU capacity when sequences have different lengths. Continuous batching keeps utilization high at the cost of implementation complexity. For a content moderation use case where responses are short (1-2 tokens), the benefit is mainly in keeping the GPU busy across many concurrent requests rather than handling variable-length outputs.

**Greedy decoding vs sampling:** For binary classification (toxic/safe), you want deterministic outputs. Greedy decoding (argmax) is used instead of temperature sampling. This also makes benchmarks reproducible.

**Exact-match Redis cache vs semantic cache:** Exact match is appropriate here because moderation requests for the same text should always return the same result. Semantic caching (like NeuralGate's pgvector approach) would be overkill and could return cached results for semantically similar but meaningfully different inputs.

**FP16 vs INT8:** FP16 is the default — good balance of speed and accuracy, fits on 8GB VRAM with room to spare. INT8 halves memory usage and is faster on hardware with INT8 acceleration (the GTX 1080 does not have Tensor cores, so the speedup is limited). Benchmark both and let the numbers decide.

**In-process KV cache vs external cache:** Storing past_key_values in GPU memory (in-process) is the standard approach. An external KV cache would require serializing tensors to CPU/disk and deserializing them — slower than just rerunning prefill for cache misses. In-process is correct here.

**asyncio.Queue vs multiprocessing:** GPU inference is not truly async — it blocks the thread. Using `run_in_executor` pushes it to a thread pool so the event loop stays responsive. A production system would use a separate process with IPC. For this project, the thread executor approach is sufficient and simpler.

---

## 12. Interview Preparation

### Questions you will definitely be asked

**"Walk me through your continuous batching implementation."**

"The HTTP handler enqueues each request as a `(prompt, asyncio.Future)` tuple and awaits the future. A separate asyncio task runs the batch worker loop — on each iteration it pulls waiting requests from the queue up to `max_batch_size`, runs the prefill step for new sequences to populate their KV cache, then runs one decode step for all active sequences simultaneously. When a sequence generates a stop token or hits `max_new_tokens`, its future is resolved with the result and the HTTP handler wakes up. This keeps GPU utilization high because new requests fill slots immediately when old ones finish, rather than waiting for the entire batch to drain."

**"What's a KV cache and why does it matter?"**

"In transformer attention, computing the attention output for the current token requires the key and value tensors for all previous tokens in the sequence. Without caching, you recompute K and V for all previous tokens at every decode step — that's O(N²) work for a sequence of length N. KV caching stores those tensors after the first computation so each decode step only computes K and V for the one new token — O(N) total. The memory tradeoff is that you're holding O(N) tensors in VRAM per active sequence. On 8GB VRAM, I allocated a fixed budget and used LRU eviction when it was full."

**"What were your throughput numbers and what drove the improvement?"**

"Naive sequential inference handled about X req/sec at Y ms p95 latency. With continuous batching, throughput improved to Z req/sec — roughly Ax improvement — because the GPU was processing multiple sequences simultaneously instead of waiting for one to finish before starting the next. The batch worker runs decode steps for all active sequences in the same PyTorch call, so the GPU is never idle between requests."

**"How does INT8 quantization affect accuracy?"**

"I benchmarked FP16 vs INT8 on a validation set of known toxic/safe examples. Accuracy dropped from X% to Y% — less than 1 percentage point in my tests. The memory reduction was from ~1GB to ~500MB VRAM for the model weights. On a GTX 1080 without Tensor cores, the latency improvement from INT8 was modest — about Z ms per request — but the memory savings matter more: the freed VRAM goes to the KV cache, allowing more concurrent sequences."

**"How do you handle backpressure?"**

"The asyncio.Queue has a fixed maximum depth of 256. Before enqueuing a new request, the HTTP handler checks if the queue is full. If it is, it immediately returns HTTP 429 without touching the queue. This keeps the server stable under overload — requests are rejected cleanly rather than piling up until memory is exhausted or latency becomes unusable. The client can use the `retry_after` field in the 429 response to implement exponential backoff."

---

## 13. Resume Bullets

Fill in the [X] values after running your benchmarks.

**FluxServe** | Python, FastAPI, PyTorch, HuggingFace, Redis, Docker, Kubernetes

- Built a production LLM inference server serving Qwen 0.5B, implementing continuous batching via asyncio.Queue to achieve [X]x throughput improvement over naive sequential inference at [N] concurrent users, reaching [Y] tokens/sec on a GTX 1080
- Designed a KV cache with LRU eviction and [Z]GB VRAM budget, reducing per-decode-step compute from O(N²) to O(N) and achieving a [W]% cache hit rate across [N] concurrent sequences
- Quantized the model to INT8 using bitsandbytes, reducing VRAM from [A]GB to [B]GB and improving p95 latency from [C]ms to [D]ms with less than 1% accuracy degradation on validation set
- Instrumented TTFT (p50: [X]ms), TPOT (p50: [Y]ms/token), p50/p95/p99 end-to-end latency, and tokens/sec via Prometheus with Grafana dashboards; implemented backpressure at 256-item queue depth (HTTP 429); containerized with Docker Compose and deployed on Kubernetes (minikube) with HPA autoscaling 1→4 replicas

---

*End of FluxServe PRD v2.0*
