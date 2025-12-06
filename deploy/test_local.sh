#!/bin/bash
# Local testing script for API and Dashboard

echo "=========================================="
echo "Break-The-Bot Local Testing"
echo "=========================================="

# Check if GCP credentials exist
if [ ! -f "../.secrets/gcp-key.json" ]; then
    echo "❌ Error: GCP credentials not found at ../.secrets/gcp-key.json"
    exit 1
fi

echo "✅ GCP credentials found"

# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="../.secrets/gcp-key.json"
export GCS_BUCKET="mlops-project-dvc-480422"
export GCP_PROJECT_ID="break-the-bot-480422"
export PORT="8080"

echo "✅ Environment variables set"
echo "   GCS_BUCKET: $GCS_BUCKET"
echo "   GCP_PROJECT_ID: $GCP_PROJECT_ID"
echo "   PORT: $PORT"
echo ""

# Check if API dependencies are installed
echo "Checking API dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing API dependencies..."
    pip install -r requirements-api.txt
else
    echo "✅ API dependencies already installed"
fi

echo ""
echo "=========================================="
echo "Starting API server..."
echo "=========================================="
echo "API will be available at: http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""

# Run API (from deploy directory, not api directory)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

