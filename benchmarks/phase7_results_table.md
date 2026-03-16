# Phase 7 — Load Test Results Table

**Date:** 2026-03-16
**Hardware:** NVIDIA GeForce GTX 1080 (8GB VRAM, Pascal sm_61)
**Model:** Qwen/Qwen2.5-0.5B-Instruct (FP16)

## Cache-Dominated Runs (locustfile.py — 10 fixed texts, warms up fast)

| Concurrent Users | RPS   | p50 (ms) | p95 (ms) | p99 (ms) | 429 Rate |
|-----------------|-------|----------|----------|----------|----------|
| 1 (naive)       | 1.94  | 120      | 250      | 300      | 0%       |
| 1 (batched)     | 2.82  | 2        | 87       | —        | 0%       |
| 4 (batched)     | 11.41 | 2        | 5        | —        | 0%       |
| 8 (batched)     | 23.14 | 2        | 4        | —        | 0%       |
| 16 (batched)    | 44.50 | 2        | 4        | —        | 0%       |

> **Note:** p50/p95 at 2ms reflects Redis cache dominance — the 10-text sample pool causes nearly all requests after warmup to hit the cache rather than the GPU. These numbers measure cache throughput, not batching throughput.

## True GPU Batching Runs (locustfile_nocache.py — unique suffix per request, all cache misses)

| Concurrent Users | RPS  | p50 (ms) | p95 (ms) | p99 (ms) | 429 Rate | vs Naive |
|-----------------|------|----------|----------|----------|----------|----------|
| 1 (naive, Ph2)  | 1.94 | 120      | 250      | 300      | 0%       | baseline |
| 1 (batched)     | 1.96 | 150      | 260      | 360      | 0%       | 1.0x     |
| 4 (batched)     | 6.24 | 240      | 440      | 740      | 0%       | 3.2x     |
| 8 (batched)     | 7.87 | 620      | 970      | 1100     | 0%       | 4.1x     |
| 16 (batched)    | 9.02 | 1400     | 1800     | 2000     | 0%       | 4.6x     |

### Analysis

- **1u nocache ≈ Phase 2 naive** (1.96 vs 1.94 RPS): confirms single-user has no batching benefit and the baselines are apples-to-apples.
- **4u is the sweet spot**: 3.2x throughput with only 2x latency increase — batching is actively merging concurrent requests.
- **Throughput plateaus at ~8–9 RPS** between 8u and 16u: the GTX 1080 GPU is saturated; additional users only deepen the queue.
- **No 429s at any concurrency**: the asyncio queue absorbs all requests, but latency climbs sharply past 8 users (620ms → 1400ms p50).
- **GPU saturation ceiling**: ~9 RPS is the practical throughput limit for FP16 Qwen2.5-0.5B on the GTX 1080 with max_batch_size=8.
- Continuous batching delivers a real **3–4x throughput improvement** over the naive sequential server at moderate concurrency.
