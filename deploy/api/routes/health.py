"""Health check endpoints."""
import logging
from fastapi import APIRouter, HTTPException
from datetime import datetime

from ..models import HealthResponse
from ..config import config
from ..gcs_client import GCSClient

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify service and GCS connectivity."""
    try:
        # Test GCS connectivity
        gcs_client = GCSClient()
        # Try to list blobs to verify connection
        test_blobs = gcs_client.list_blobs(config.METRICS_PATH_PREFIX)
        gcs_connected = True
    except Exception as e:
        logger.error(f"GCS connectivity check failed: {e}")
        gcs_connected = False
    
    return HealthResponse(
        status="healthy" if gcs_connected else "degraded",
        gcs_connected=gcs_connected,
        bucket=config.GCS_BUCKET,
        timestamp=datetime.utcnow()
    )

