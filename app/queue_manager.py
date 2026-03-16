# app/queue_manager.py
import asyncio
from dataclasses import dataclass
from typing import Any

@dataclass
class InferenceRequest:
    text: str
    max_new_tokens: int
    future: asyncio.Future
    request_id: str

class QueueManager:
    def __init__(self, max_depth: int = 256):
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=max_depth)
        self.max_depth = max_depth

    def is_full(self) -> bool:
        return self.queue.qsize() >= self.max_depth

    async def enqueue(self, request: InferenceRequest) -> bool:
        """Returns False if queue is full (caller should return 429)."""
        if self.is_full():
            return False
        await self.queue.put(request)
        return True

    async def dequeue(self) -> InferenceRequest:
        return await self.queue.get()

    def depth(self) -> int:
        return self.queue.qsize()