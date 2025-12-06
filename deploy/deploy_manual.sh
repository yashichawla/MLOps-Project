#!/bin/bash
# Manual deployment script for Metrics API to Google Cloud Run
# Use this if Cloud Build is not available

set -e

echo "=========================================="
echo "Manual Deployment to Google Cloud Run"
echo "=========================================="
echo ""

# Configuration
PROJECT_ID="break-the-bot-480422"
SERVICE_NAME="metrics-api"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Service Name: $SERVICE_NAME"
echo "  Region: $REGION"
echo "  Image: $IMAGE_NAME"
echo ""

# Check prerequisites
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    exit 1
fi

# Authenticate Docker with GCR
echo "Authenticating Docker with Google Container Registry..."
gcloud auth configure-docker

# Build Docker image
echo ""
echo "Building Docker image..."
cd "$SCRIPT_DIR"
docker build -t $IMAGE_NAME:latest .

# Push to Container Registry
echo ""
echo "Pushing image to Container Registry..."
docker push $IMAGE_NAME:latest

# Deploy to Cloud Run
echo ""
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET=mlops-project-dvc-480422,GCP_PROJECT_ID=$PROJECT_ID \
  --service-account=mlops-850@break-the-bot-480422.iam.gserviceaccount.com \
  --memory=512Mi \
  --cpu=1 \
  --max-instances=10 \
  --timeout=300 \
  --project $PROJECT_ID

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Getting service URL..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)" --project $PROJECT_ID)
echo ""
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "Test the deployment:"
echo "  curl $SERVICE_URL/health"
echo "  curl $SERVICE_URL/metrics/all"
echo "  curl $SERVICE_URL/dashboard/models"

