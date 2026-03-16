# Phase 7 — Locust Load Test Results (Continuous Batching Server)

**Date:** 2026-03-16
**Hardware:** NVIDIA GeForce GTX 1080 (8GB VRAM, Pascal sm_61)
**Model:** Qwen/Qwen2.5-0.5B-Instruct (FP16)
**Server:** FastAPI + continuous batching + Redis cache + KV cache pool
**Test:** `locust --headless -u <N> -r <N> --run-time 60s`
**Sample pool:** 10 fixed texts (see `load_tests/locustfile.py`)
**Phase 6 baseline (no server, no cache):** p50=118ms, 7.95 req/s

## Results

| Users | p50 (ms) | p95 (ms) | Avg (ms) | Req/s  | Total Reqs | Failures |
|-------|----------|----------|----------|--------|------------|----------|
| 1     | 2        | 87       | 12       | 2.82   | 187        | 0        |
| 4     | 2        | 5        | 2        | 11.41  | 758        | 0        |
| 8     | 2        | 4        | 2        | 23.14  | 1537       | 0        |
| 16    | 2        | 4        | 2        | 44.50  | 2959       | 0        |

## Analysis

**Results are dominated by Redis cache hits.** The 10-text sample pool means nearly all requests after warmup hit the Redis exact-match cache (~2ms), not the GPU (~118ms). This explains:

- **2ms median across all concurrency levels** — Redis serving cached results
- **1-user p95 of 87ms** — cold cache misses on the first pass before warmup completes; higher relative share with only 1 user
- **4–16 user p95 drops to 4–5ms** — cache warms up faster with parallel users, cold misses are diluted
- **Near-linear scaling (2.82 → 11.41 → 23.14 → 44.50 req/s)** — Redis handles concurrent reads well; no 429 backpressure triggered at any level

## What This Measures vs. What It Doesn't

- **Measured:** End-to-end system latency, Redis cache throughput, zero errors under concurrent load, backpressure not needed (cache absorbs all load)
- **Not measured:** GPU batching efficiency — to test that, use a large/unique text corpus to force cache misses

## Comparison vs. Phase 6 Baseline

| Condition              | p50    | Req/s  |
|------------------------|--------|--------|
| Phase 6 (no server, sequential) | 118ms | 7.95   |
| Phase 7 @ 1 user (cold cache)  | 2ms   | 2.82   |
| Phase 7 @ 16 users (warm cache) | 2ms  | 44.50  |

The ~44x throughput improvement at 16 users vs. Phase 6 sequential is primarily cache-driven, not batching-driven. A cache-miss-only test would isolate the batching gain.
