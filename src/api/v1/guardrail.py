from fastapi import APIRouter, Depends, HTTPException
from src.engines.guardrail.schemas import GuardrailRequestDTO, GuardrailResponseDTO
from src.engines.guardrail.services import GuardrailService
from src.api.dependencies import get_guardrail_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/guardrail", tags=["guardrail"])

@router.post("/validate", response_model=GuardrailResponseDTO)
async def validate_guardrail(
    request: GuardrailRequestDTO,
    service: GuardrailService = Depends(get_guardrail_service)
):
    try:
        result = await service.validate(request)
        return result
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
