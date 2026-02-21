"""
Health check routes for monitoring service status.
"""

from fastapi import APIRouter, status
from datetime import datetime

from ...models.schema import HealthResponse
from ...services.prediction_service import get_prediction_service
from ...core.config import settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the health status of the API and model loading status."
)
async def health_check() -> HealthResponse:
    """
    Perform a health check on the service.

    Returns:
        HealthResponse with service status information
    """
    prediction_service = get_prediction_service()

    return HealthResponse(
        status="healthy" if prediction_service.is_ready else "degraded",
        version=settings.APP_VERSION,
        model_loaded=prediction_service.model_loaded,
        vectorizer_loaded=prediction_service.vectorizer_loaded,
        timestamp=datetime.utcnow()
    )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Check if the service is ready to handle requests."
)
async def readiness_check():
    """
    Check if the service is ready to handle prediction requests.

    Returns:
        Simple ready status
    """
    prediction_service = get_prediction_service()

    if prediction_service.is_ready:
        return {"ready": True, "message": "Service is ready"}
    else:
        return {
            "ready": False,
            "message": "Service is not ready - models not loaded",
            "model_loaded": prediction_service.model_loaded,
            "vectorizer_loaded": prediction_service.vectorizer_loaded
        }


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Check",
    description="Simple liveness probe for container orchestration."
)
async def liveness_check():
    """
    Simple liveness check - returns OK if the service is running.

    Returns:
        Simple alive status
    """
    return {"alive": True}

