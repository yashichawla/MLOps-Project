#!/bin/bash
# Script to grant GCS bucket permissions to the service account

set -e

PROJECT_ID="break-the-bot-480422"
SERVICE_ACCOUNT="mlops-850@break-the-bot-480422.iam.gserviceaccount.com"
BUCKET_NAME="mlops-project-dvc"

echo "=========================================="
echo "Setting up GCS Bucket Permissions"
echo "=========================================="
echo ""
echo "Project ID: $PROJECT_ID"
echo "Service Account: $SERVICE_ACCOUNT"
echo "Bucket: gs://$BUCKET_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed"
    exit 1
fi

# Set the project
echo "Setting GCP project..."
gcloud config set project $PROJECT_ID

echo ""
echo "Granting Storage Object Viewer role to service account..."
echo "This allows the service account to read objects from the bucket."
echo ""

# Grant storage.objectViewer role on the bucket
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:roles/storage.objectViewer gs://$BUCKET_NAME

echo ""
echo "✅ Permissions granted successfully!"
echo ""
echo "Verifying access..."
echo ""

# Test access
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/../.secrets/gcp-key.json"
if gsutil ls gs://$BUCKET_NAME/ > /dev/null 2>&1; then
    echo "✅ Service account can now access the bucket!"
else
    echo "⚠️  Warning: Could not verify access. Please check manually."
fi

echo ""
echo "Done!"

