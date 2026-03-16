"""
Phase 5 test script — KV cache pool.
Tests the KVCachePool class directly, then verifies integration through the server.
Start the server first, then run:
    python scripts/test_phase5.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import requests
import threading
import torch

# ── Direct unit tests on KVCachePool ─────────────────────────────────────────
from app.kv_cache import KVCachePool

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


def make_fake_kv(num_layers: int = 2, seq_len: int = 16) -> tuple:
    """Create a fake past_key_values tensor tuple on CPU for testing."""
    return tuple(
        (torch.randn(1, 4, seq_len, 16), torch.randn(1, 4, seq_len, 16))
        for _ in range(num_layers)
    )


def post(text: str) -> requests.Response:
    return requests.post(f"{BASE}/moderate", json={"text": text}, headers=HEADERS)


# ── 1. Basic get/put ──────────────────────────────────────────────────────────
print("\n[1] KVCachePool — basic get/put")
pool = KVCachePool(max_sequences=4)
kv = make_fake_kv()

check("get miss returns None", pool.get("seq1") is None)
pool.put("seq1", kv)
check("get hit returns tensors", pool.get("seq1") is not None)
check("hit_rate > 0 after hit", pool.hit_rate() > 0)
check("utilization = 0.25 with 1/4 entries", abs(pool.utilization() - 0.25) < 0.01)

# ── 2. LRU eviction ───────────────────────────────────────────────────────────
print("\n[2] KVCachePool — LRU eviction")
pool = KVCachePool(max_sequences=3)
pool.put("a", make_fake_kv())
pool.put("b", make_fake_kv())
pool.put("c", make_fake_kv())
check("3 entries fill the pool", pool.utilization() == 1.0)

# Access "a" to make it recently used, so "b" becomes LRU
pool.get("a")
pool.get("c")

# Adding "d" should evict "b" (LRU)
pool.put("d", make_fake_kv())
check("pool still at max after eviction", len(pool.cache) == 3)
check("LRU entry (b) was evicted", pool.get("b") is None)
check("recently used (a) still present", pool.get("a") is not None)
check("recently used (c) still present", pool.get("c") is not None)
check("new entry (d) present", pool.get("d") is not None)

# ── 3. Explicit eviction ──────────────────────────────────────────────────────
print("\n[3] KVCachePool — explicit evict")
pool = KVCachePool(max_sequences=4)
pool.put("seq1", make_fake_kv())
pool.evict("seq1")
check("entry removed after evict", pool.get("seq1") is None)
check("total_bytes back to 0", pool.total_bytes == 0)

# ── 4. Size estimation ────────────────────────────────────────────────────────
print("\n[4] KVCachePool — size estimation")
pool = KVCachePool(max_sequences=4)
kv = make_fake_kv(num_layers=2, seq_len=16)
pool.put("seq1", kv)
check("total_bytes > 0 after put", pool.total_bytes > 0)
expected = sum(t.nelement() * t.element_size() for layer in kv for t in layer)
check("size estimation is correct", pool.total_bytes == expected,
      f"got {pool.total_bytes}, expected {expected}")

# ── 5. Byte budget eviction ───────────────────────────────────────────────────
print("\n[5] KVCachePool — byte budget eviction")
kv_small = make_fake_kv(num_layers=1, seq_len=8)
entry_size = sum(t.nelement() * t.element_size() for layer in kv_small for t in layer)
# Set budget to fit exactly 2 entries
pool = KVCachePool(max_sequences=100, max_bytes=entry_size * 2)
pool.put("x1", kv_small)
pool.put("x2", kv_small)
check("2 entries fit within budget", len(pool.cache) == 2)
pool.put("x3", kv_small)  # should evict x1
check("3rd entry triggers eviction", len(pool.cache) == 2)
check("oldest entry evicted", pool.get("x1") is None)
check("newest entry present", pool.get("x3") is not None)

# ── 6. Hit rate tracking ──────────────────────────────────────────────────────
print("\n[6] KVCachePool — hit rate")
pool = KVCachePool(max_sequences=4)
pool.put("s1", make_fake_kv())
pool.get("s1")   # hit
pool.get("s1")   # hit
pool.get("s99")  # miss
check("hit rate = 2/3", abs(pool.hit_rate() - 2/3) < 0.01,
      f"got {pool.hit_rate():.2f}")

# ── 7. Metrics gauge updated ──────────────────────────────────────────────────
print("\n[7] KV cache utilization in /metrics")
r = requests.get(f"{BASE}/metrics")
check("kv_cache_utilization gauge present", "fluxserve_kv_cache_utilization" in r.text)

# ── 8. Server still handles requests correctly ────────────────────────────────
print("\n[8] Server integration — requests still work")
texts = [f"Phase 5 integration test {i} {time.time()}" for i in range(4)]
results = [None] * 4

def send(i, text):
    results[i] = post(text)

threads = [threading.Thread(target=send, args=(i, t)) for i, t in enumerate(texts)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check("all 4 requests returned 200", all(r is not None and r.status_code == 200 for r in results))
check("all labels valid", all(r.json().get("label") in ("safe", "toxic", "unknown") for r in results if r))

# ── 9. Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
if failures == 0:
    print("All Phase 5 checks passed.")
else:
    print(f"{failures} check(s) failed.")
    sys.exit(1)
