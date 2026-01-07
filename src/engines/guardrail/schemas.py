from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class GuardrailStatus(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"

class GuardrailRequestDTO(BaseModel):
    prompt: str = Field(..., max_length=2000)
    image_bytes: Optional[str] = Field(None)  # base64

class GuardrailResponseDTO(BaseModel):
    status: GuardrailStatus
    reasons: List[str] = []
    scores: Dict[str, float]
    metadata: Dict[str, Any]
