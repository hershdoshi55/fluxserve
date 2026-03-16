# Phase 6 Baseline — FP16 Single-Request Benchmark (No Server)

**Date:** 2026-03-16
**Hardware:** NVIDIA GeForce GTX 1080 (8GB VRAM, Pascal sm_61)
**Model:** Qwen/Qwen2.5-0.5B-Instruct (FP16)
**Script:** `scripts/benchmark.py`
**Condition:** Sequential single requests, direct model inference, no server, no batching
**Warmup runs:** 5 | **Benchmark runs:** 50

## Results

| Metric          |   Value |
|-----------------|--------:|
| VRAM used       | 1.00 GB |
| Avg tokens/req  |     3.0 |
| Latency p50     | 118.1ms |
| Latency p95     | 168.8ms |
| Latency p99     | 189.5ms |
| Latency avg     | 125.8ms |
| Throughput      | 7.95 req/sec |
| Tokens/sec      |   23.85 |

## Notes

- This is the pre-batching, pre-server baseline — raw GPU inference speed with no overhead
- Compare against Phase 7 (Locust load test with continuous batching server) to measure improvement
- Throughput of 7.95 req/sec here is single-threaded sequential — no concurrency
- p95/p50 spread is tight (168ms vs 118ms), indicating stable inference time on this hardware
