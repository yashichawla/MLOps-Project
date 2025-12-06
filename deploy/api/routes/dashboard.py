"""Dashboard endpoints."""
import logging
from fastapi import APIRouter, HTTPException
from datetime import datetime

from ..models import ModelsListResponse, ModelInfo
from ..metrics_service import MetricsService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/dashboard/models", response_model=ModelsListResponse)
async def list_models():
    """List all evaluated models with metadata."""
    try:
        service = MetricsService()
        models_data = service.list_available_models()
        
        models = [
            ModelInfo(
                name=model["name"],
                has_metrics=model["has_metrics"],
                has_bias_report=model["has_bias_report"],
                last_updated=model["last_updated"]
            )
            for model in models_data
        ]
        
        return ModelsListResponse(
            models=models,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

