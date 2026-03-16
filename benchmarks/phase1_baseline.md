# Phase 1 Baseline — Single-Request Inference (No Server)

**Date:** 2026-03-16
**Hardware:** NVIDIA GeForce GTX 1080 (8GB VRAM, Pascal sm_61)
**Model:** Qwen/Qwen2.5-0.5B-Instruct
**Script:** `scripts/validate_model.py`
**Condition:** Single sequential requests, no batching, no server overhead

## Results

| Metric           |   FP16 |   INT8 |
|------------------|-------:|-------:|
| Accuracy         |   6/6  |   6/6  |
| Avg latency (ms) |    146 |    499 |
| Min latency (ms) |     75 |    373 |
| Max latency (ms) |    329 |    638 |
| VRAM (GB)        |   1.00 |   0.64 |

## Notes

- INT8 is **3.4x slower** than FP16 on this hardware. Expected: GTX 1080 (Pascal/sm_61) has no
  native INT8 tensor cores. `bitsandbytes` falls back to a software emulation path. INT8 speed
  gains only materialize on Turing (RTX 20xx) and newer.
- INT8 saves ~360MB VRAM (36% reduction) but at a significant latency cost on this GPU.
- **Decision: use FP16 for all subsequent phases.** Better latency, negligible VRAM difference
  on 8GB. The PRECISION env var will default to `fp16`.
- Both precisions hit 6/6 accuracy — no quality loss from quantization on this task.
