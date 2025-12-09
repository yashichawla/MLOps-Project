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
    gcs_connected = False
    error_details = None
    
    try:
        # Test GCS connectivity
        gcs_client = GCSClient()
        
        # Try to list blobs to verify connection
        test_blobs = gcs_client.list_blobs(config.METRICS_PATH_PREFIX)
        
        # Also check if bucket is accessible
        try:
            bucket = gcs_client.bucket
            bucket.reload()
            gcs_connected = True
            logger.info(f"Health check: GCS connected, found {len(test_blobs)} blob(s) in {config.METRICS_PATH_PREFIX}")
        except Exception as e:
            logger.warning(f"Bucket reload failed: {e}")
            # Still consider connected if we can list blobs
            gcs_connected = True
            
    except Exception as e:
        logger.error(f"GCS connectivity check failed: {type(e).__name__}: {e}")
        gcs_connected = False
        error_details = str(e)
        import traceback
        logger.debug(traceback.format_exc())
    
    status = "healthy" if gcs_connected else "degraded"
    
    return HealthResponse(
        status=status,
        gcs_connected=gcs_connected,
        bucket=config.GCS_BUCKET,
        timestamp=datetime.utcnow()
    )

