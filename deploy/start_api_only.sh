#!/bin/bash
# Start API server only

set -e

echo "=========================================="
echo "Starting Metrics API Server"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="$PROJECT_ROOT/.secrets/gcp-key.json"
export GCS_BUCKET="mlops-project-dvc-480422"
export GCP_PROJECT_ID="break-the-bot-480422"
export PORT="8080"

echo "Environment:"
echo "  GCS_BUCKET: $GCS_BUCKET"
echo "  GCP_PROJECT_ID: $GCP_PROJECT_ID"
echo "  PORT: $PORT"
echo ""

# Check credentials
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "❌ Error: GCP credentials not found at $GOOGLE_APPLICATION_CREDENTIALS"
    exit 1
fi

# Start API
cd "$SCRIPT_DIR"
echo "Starting API server on http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""

python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

