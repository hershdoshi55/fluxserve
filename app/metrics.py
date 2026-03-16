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