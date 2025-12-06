#!/bin/bash
# Deployment script for Metrics API to Google Cloud Run

set -e

echo "=========================================="
echo "Deploying Metrics API to Google Cloud Run"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
PROJECT_ID="break-the-bot-480422"
SERVICE_NAME="metrics-api"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Service Name: $SERVICE_NAME"
echo "  Region: $REGION"
echo "  Image: $IMAGE_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "⚠️  Not authenticated with gcloud. Please run:"
    echo "   gcloud auth login"
    exit 1
fi

# Set the project
echo "Setting GCP project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Check if Cloud Run API is enabled
echo "Checking if Cloud Run API is enabled..."
if ! gcloud services list --enabled --filter="name:run.googleapis.com" | grep -q run.googleapis.com; then
    echo "Enabling Cloud Run API..."
    gcloud services enable run.googleapis.com
fi

# Check if Container Registry API is enabled
echo "Checking if Container Registry API is enabled..."
if ! gcloud services list --enabled --filter="name:containerregistry.googleapis.com" | grep -q containerregistry.googleapis.com; then
    echo "Enabling Container Registry API..."
    gcloud services enable containerregistry.googleapis.com
fi

# Check if Cloud Build API is enabled
echo "Checking if Cloud Build API is enabled..."
if ! gcloud services list --enabled --filter="name:cloudbuild.googleapis.com" | grep -q cloudbuild.googleapis.com; then
    echo "Enabling Cloud Build API..."
    gcloud services enable cloudbuild.googleapis.com
fi

echo ""
echo "=========================================="
echo "Option 1: Deploy using Cloud Build (Recommended)"
echo "=========================================="
echo ""
echo "This will build, push, and deploy automatically."
echo ""

read -p "Deploy using Cloud Build? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Submitting build to Cloud Build..."
    cd "$PROJECT_ROOT"
    gcloud builds submit --config cloudbuild.yaml
    
    echo ""
    echo "✅ Deployment complete!"
    echo ""
    echo "Getting service URL..."
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    echo ""
    echo "Service URL: $SERVICE_URL"
    echo ""
    echo "Test the deployment:"
    echo "  curl $SERVICE_URL/health"
    echo "  curl $SERVICE_URL/metrics/all"
    exit 0
fi

echo ""
echo "=========================================="
echo "Option 2: Manual Deployment"
echo "=========================================="
echo ""

# Build Docker image
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
  --memory=512Mi \
  --cpu=1 \
  --max-instances=10 \
  --timeout=300

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Getting service URL..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test the deployment:"
echo "  curl $SERVICE_URL/health"
echo "  curl $SERVICE_URL/metrics/all"

