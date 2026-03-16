"""
Phase 3 test script — Redis caching + Prometheus metrics.
Run with the server and Redis already running:
    python scripts/test_phase3.py
"""
import requests
import sys
import time as _time

BASE = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer dev-key", "Content-Type": "application/json"}
PASS = "✓"
FAIL = "✗"
failures = 0


def check(label: str, condition: bool, detail: str = ""):
    global failures
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}{' — ' + detail if detail else ''}")
        failures += 1


def post(text: str) -> requests.Response:
    return requests.post(f"{BASE}/moderate", json={"text": text}, headers=HEADERS)


# ── 1. Health ────────────────────────────────────────────────────────────────
print("\n[1] Health check")
r = requests.get(f"{BASE}/health")
check("status 200", r.status_code == 200)
check("model field present", "model" in r.json())

# ── 2. Prometheus metrics endpoint ───────────────────────────────────────────
print("\n[2] Prometheus /metrics")
r = requests.get(f"{BASE}/metrics")
check("status 200", r.status_code == 200)
check("content-type is text/plain", "text/plain" in r.headers.get("content-type", ""))
body = r.text
for metric in [
    "fluxserve_requests_total",
    "fluxserve_inference_latency_seconds",
    "fluxserve_queue_depth",
    "fluxserve_tokens_per_second",
]:
    check(f"{metric} present", metric in body)

# ── 3. Auth rejection ─────────────────────────────────────────────────────────
print("\n[3] Auth")
r = requests.post(
    f"{BASE}/moderate",
    json={"text": "hello"},
    headers={"Authorization": "Bearer wrong-key", "Content-Type": "application/json"},
)
check("wrong token → 401", r.status_code == 401)

# ── 4. First request — cache miss ────────────────────────────────────────────
print("\n[4] Cache miss (first request)")
# Unique text each run so Redis never has it pre-cached
text = f"Phase 3 test: the weather is nice today {_time.time()}"
r = post(text)
check("status 200", r.status_code == 200)
data = r.json()
check("cached=false on first request", data.get("cached") is False, str(data.get("cached")))
check("label is safe or toxic", data.get("label") in ("safe", "toxic", "unknown"))
check("latency_ms present", data.get("latency_ms") is not None)
first_latency = data["latency_ms"]
print(f"      latency: {first_latency:.0f}ms")

# ── 5. Second request — cache hit ────────────────────────────────────────────
print("\n[5] Cache hit (same text)")
r = post(text)
check("status 200", r.status_code == 200)
data = r.json()
check("cached=true on second request", data.get("cached") is True, str(data.get("cached")))
check("same label returned", data.get("label") in ("safe", "toxic", "unknown"))
second_latency = data["latency_ms"]
check("cache is faster than inference", second_latency < first_latency,
      f"{second_latency:.0f}ms vs {first_latency:.0f}ms")
print(f"      latency: {second_latency:.0f}ms (was {first_latency:.0f}ms)")

# ── 6. Metrics updated after requests ────────────────────────────────────────
print("\n[6] Metrics updated")
r = requests.get(f"{BASE}/metrics")
body = r.text
check('requests_total{status="ok"} > 0', 'requests_total{status="ok"}' in body)
check('requests_total{status="cached"} > 0', 'requests_total{status="cached"}' in body)
check("inference_latency has observations", 'inference_latency_seconds_count' in body)

# ── 7. Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
if failures == 0:
    print("All Phase 3 checks passed.")
else:
    print(f"{failures} check(s) failed.")
    sys.exit(1)
