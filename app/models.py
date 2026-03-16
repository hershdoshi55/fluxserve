from pydantic import BaseModel
from typing import Optional


class ModerateRequest(BaseModel):
    text: str
    max_new_tokens: int = 10
    request_id: Optional[str] = None


class ModerateResponse(BaseModel):
    label: str
    flagged: bool
    latency_ms: float
    cached: bool
    model: str
    tokens_generated: Optional[int] = None
    request_id: Optional[str] = None
