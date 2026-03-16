# Phase 2 Baseline — Naive Sequential Server

**Date:** 2026-03-16
**Hardware:** NVIDIA GeForce GTX 1080 (8GB VRAM, Pascal sm_61)
**Model:** Qwen/Qwen2.5-0.5B-Instruct (FP16)
**Server:** Naive sequential — one request at a time, no batching, no queue
**Test:** Locust, 1 concurrent user, 60s, `wait_time = between(0.1, 0.5)`

## Results

| Metric        | Value |
|---------------|------:|
| Total requests|   130 |
| Failures      |     0 |
| Req/sec       |  1.94 |
| Avg (ms)      |   136 |
| Min (ms)      |    77 |
| Max (ms)      |   412 |

## Latency Percentiles

| p50 | p75 | p90 | p95 | p99  | p100 |
|----:|----:|----:|----:|-----:|-----:|
| 120 | 150 | 190 | 250 |  300 |  410 |

## Notes

- 1 concurrent user → no queuing, server handles requests one at a time
- These are the "before continuous batching" numbers
- p95 = 250ms at 1 user; will degrade significantly under concurrent load (Phase 7)
- Throughput is GPU-bound at ~2 req/sec — each request occupies the GPU for its full duration
