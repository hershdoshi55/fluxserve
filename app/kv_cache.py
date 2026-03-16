# app/kv_cache.py
import time
from collections import OrderedDict
from typing import Optional
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