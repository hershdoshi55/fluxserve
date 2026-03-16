"""
Phase 4 test script — continuous batching engine.
Start the server first, then run:
    python scripts/test_phase4.py
"""
import requests
import sys
import time
import threading

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


# ── 1. Health has batch worker fields ────────────────────────────────────────
print("\n[1] Health check — batch worker fields")
r = requests.get(f"{BASE}/health")
check("status 200", r.status_code == 200)
data = r.json()
check("queue_depth present", "queue_depth" in data, str(data))
check("active_sequences present", "active_sequences" in data, str(data))
check("gpu_memory_used_gb present", "gpu_memory_used_gb" in data, str(data))
print(f"      queue_depth={data.get('queue_depth')}  active_sequences={data.get('active_sequences')}")

# ── 2. Single request works end-to-end through batch worker ──────────────────
print("\n[2] Single request through batch worker")
text = f"Single request test {time.time()}"
r = post(text)
check("status 200", r.status_code == 200)
data = r.json()
check("label present", data.get("label") in ("safe", "toxic", "unknown"))
check("cached=false (unique text)", data.get("cached") is False)
check("tokens_generated present", data.get("tokens_generated") is not None)
print(f"      label={data.get('label')}  tokens={data.get('tokens_generated')}  latency={data.get('latency_ms', 0):.0f}ms")

# ── 3. Concurrent requests — core continuous batching test ───────────────────
print("\n[3] Concurrent requests (8 simultaneous)")
texts = [f"Concurrent test {i} {time.time()}: {'I will hurt you' if i % 2 else 'The weather is nice'}" for i in range(8)]
results = [None] * 8
errors = []

def send(i, text):
    try:
        results[i] = post(text)
    except Exception as e:
        errors.append(str(e))

threads = [threading.Thread(target=send, args=(i, t)) for i, t in enumerate(texts)]
start = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.time() - start

check("no request errors", len(errors) == 0, str(errors))
check("all 8 returned 200", all(r is not None and r.status_code == 200 for r in results))
valid_labels = all(r.json().get("label") in ("safe", "toxic", "unknown") for r in results if r and r.status_code == 200)
check("all responses have valid labels", valid_labels)
print(f"      8 concurrent requests completed in {elapsed:.1f}s")

# ── 4. Cache still works through new stack ───────────────────────────────────
print("\n[4] Cache still works")
cache_text = f"Cache test through batch worker {time.time()}"
r1 = post(cache_text)
r2 = post(cache_text)
check("first request cached=false", r1.json().get("cached") is False)
check("second request cached=true", r2.json().get("cached") is True)
check("cache hit is faster", r2.json().get("latency_ms", 999) < r1.json().get("latency_ms", 0))
print(f"      miss={r1.json().get('latency_ms', 0):.0f}ms  hit={r2.json().get('latency_ms', 0):.0f}ms")

# ── 5. Backpressure — 429 when queue flooded ─────────────────────────────────
print("\n[5] Backpressure (429)")
# Flood with requests beyond queue depth using many threads
flood_results = []
flood_lock = threading.Lock()

def flood(_):
    r = post(f"Flood test {time.time()}")
    with flood_lock:
        flood_results.append(r.status_code)

flood_threads = [threading.Thread(target=flood, args=(i,)) for i in range(300)]
for t in flood_threads:
    t.start()
for t in flood_threads:
    t.join()

got_429 = flood_results.count(429)
got_200 = flood_results.count(200)
check("received at least one 429", got_429 > 0, f"got {got_429} 429s and {got_200} 200s")
print(f"      {got_200} succeeded, {got_429} rejected with 429")

# ── 6. Metrics include batch worker gauges ───────────────────────────────────
print("\n[6] Metrics — batch worker gauges")
r = requests.get(f"{BASE}/metrics")
body = r.text
check("active_sequences gauge present", "fluxserve_active_sequences" in body)
check("queue_depth gauge present", "fluxserve_queue_depth" in body)
check("batch_size histogram present", "fluxserve_batch_size" in body)
check("tokens_per_second gauge present", "fluxserve_tokens_per_second" in body)

# ── 7. Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
if failures == 0:
    print("All Phase 4 checks passed.")
else:
    print(f"{failures} check(s) failed.")
    sys.exit(1)
