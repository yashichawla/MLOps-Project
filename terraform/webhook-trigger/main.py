"""
GitHub Webhook Trigger Service for Cloud Composer
Receives GitHub webhooks and triggers Airflow DAGs when config files change.
"""
import os
import hmac
import hashlib
import json
import logging
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Header, status
from fastapi.responses import JSONResponse
import google.auth
from google.auth.transport.requests import Request as AuthRequest
from google.oauth2 import service_account
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Composer Webhook Trigger")

# Configuration from environment variables
COMPOSER_ENVIRONMENT = os.getenv("COMPOSER_ENVIRONMENT", "mlops-airflow-composer")
COMPOSER_LOCATION = os.getenv("COMPOSER_LOCATION", "us-central1")
COMPOSER_DAG_ID = os.getenv("COMPOSER_DAG_ID", "salad_ml_evaluation_pipeline_v1")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")

# Composer Airflow API base URL (will be validated on startup)
COMPOSER_API_BASE = None


@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup."""
    global COMPOSER_API_BASE
    
    if not GCP_PROJECT_ID:
        logger.error("GCP_PROJECT_ID environment variable is required")
        raise ValueError("GCP_PROJECT_ID environment variable is required")
    
    COMPOSER_API_BASE = f"https://{COMPOSER_LOCATION}-{COMPOSER_ENVIRONMENT}.p.{GCP_PROJECT_ID}.appspot.com/api/v1"
    logger.info(f"Webhook service initialized:")
    logger.info(f"  Composer Environment: {COMPOSER_ENVIRONMENT}")
    logger.info(f"  Composer Location: {COMPOSER_LOCATION}")
    logger.info(f"  DAG ID: {COMPOSER_DAG_ID}")
    logger.info(f"  GCP Project: {GCP_PROJECT_ID}")
    logger.info(f"  Composer API Base: {COMPOSER_API_BASE}")


def verify_github_signature(payload_body: bytes, signature: Optional[str]) -> bool:
    """Verify GitHub webhook signature."""
    if not GITHUB_WEBHOOK_SECRET or not signature:
        logger.warning("Webhook secret not configured, skipping signature verification")
        return True  # Allow if secret not configured
    
    try:
        # GitHub sends signature as "sha256=<hash>"
        if not signature.startswith("sha256="):
            return False
        
        expected_signature = signature[7:]  # Remove "sha256=" prefix
        mac = hmac.new(
            GITHUB_WEBHOOK_SECRET.encode(),
            msg=payload_body,
            digestmod=hashlib.sha256
        )
        computed_signature = mac.hexdigest()
        
        return hmac.compare_digest(computed_signature, expected_signature)
    except Exception as e:
        logger.error(f"Error verifying signature: {e}")
        return False


def get_composer_access_token() -> str:
    """Get access token for Composer Airflow API using default credentials."""
    try:
        credentials, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(AuthRequest())
        return credentials.token
    except Exception as e:
        logger.error(f"Error getting access token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to authenticate with Composer"
        )


def check_config_files_changed(payload: dict) -> bool:
    """Check if config files were changed in the push event."""
    commits = payload.get("commits", [])
    # Check for config files in both root-level and dags-level directories
    config_files = [
        "config/data_sources.json",
        "config/attack_llm_config.json",
        "dags/config/data_sources.json",
        "dags/config/attack_llm_config.json"
    ]
    
    for commit in commits:
        added = commit.get("added", [])
        modified = commit.get("modified", [])
        removed = commit.get("removed", [])
        
        all_changed = added + modified + removed
        
        for file_path in all_changed:
            # Normalize path separators (GitHub uses forward slashes)
            normalized_path = file_path.replace("\\", "/")
            # Check if any config file path matches (endswith for exact match or contains for subdirectory)
            if any(normalized_path.endswith(config_file) or normalized_path == config_file 
                   for config_file in config_files):
                logger.info(f"Config file changed: {file_path}")
                return True
    
    return False


def trigger_composer_dag(access_token: str) -> dict:
    """Trigger DAG in Cloud Composer via Airflow REST API."""
    if not COMPOSER_API_BASE:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service not properly initialized"
        )
    url = f"{COMPOSER_API_BASE}/dags/{COMPOSER_DAG_ID}/dagRuns"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "conf": {
            "triggered_by": "github_webhook",
            "reason": "config_files_changed"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error triggering DAG: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"Response: {e.response.text}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger DAG: {str(e)}"
        )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Composer Webhook Trigger",
        "composer_environment": COMPOSER_ENVIRONMENT,
        "composer_location": COMPOSER_LOCATION,
        "dag_id": COMPOSER_DAG_ID,
        "gcp_project_id": GCP_PROJECT_ID if GCP_PROJECT_ID else "NOT SET",
        "initialized": COMPOSER_API_BASE is not None
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/webhook")
async def github_webhook(
    request: Request,
    x_github_event: Optional[str] = Header(None),
    x_hub_signature_256: Optional[str] = Header(None)
):
    """
    Handle GitHub webhook events.
    Triggers Composer DAG if config files are changed.
    """
    try:
        # Read request body
        body = await request.body()
        
        # Verify signature if provided
        if x_hub_signature_256:
            if not verify_github_signature(body, x_hub_signature_256):
                logger.warning("Invalid webhook signature")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )
        
        # Parse payload
        try:
            payload = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON payload"
            )
        
        # Only process push events
        if x_github_event != "push":
            logger.info(f"Ignoring event type: {x_github_event}")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "ignored", "reason": f"Event type {x_github_event} not processed"}
            )
        
        # Check if config files changed
        if not check_config_files_changed(payload):
            logger.info("No config files changed, skipping DAG trigger")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "skipped", "reason": "No config files changed"}
            )
        
        # Get access token and trigger DAG
        logger.info(f"Config files changed, triggering DAG: {COMPOSER_DAG_ID}")
        access_token = get_composer_access_token()
        dag_run = trigger_composer_dag(access_token)
        
        logger.info(f"DAG triggered successfully: {dag_run.get('dag_run_id')}")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "dag_run_id": dag_run.get("dag_run_id"),
                "message": f"DAG {COMPOSER_DAG_ID} triggered successfully"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

