# app/cache.py
import redis
import hashlib
import json
from typing import Optional

class ResponseCache:
    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 3600):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl

    def _key(self, text: str) -> str:
        return f"fluxserve:response:{hashlib.md5(text.encode()).hexdigest()}"

    def get(self, text: str) -> Optional[dict]:
        key = self._key(text)
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, text: str, response: dict):
        key = self._key(text)
        self.client.setex(key, self.ttl, json.dumps(response))