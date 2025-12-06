#!/bin/bash
# Start API and Dashboard together

set -e  # Exit on error

echo "=========================================="
echo "Break-The-Bot: Starting API and Dashboard"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="$PROJECT_ROOT/.secrets/gcp-key.json"
export GCS_BUCKET="mlops-project-dvc-480422"
export GCP_PROJECT_ID="break-the-bot-480422"
export PORT="8080"
export METRICS_API_URL="http://localhost:8080"

echo "Environment variables:"
echo "  GOOGLE_APPLICATION_CREDENTIALS: $GOOGLE_APPLICATION_CREDENTIALS"
echo "  GCS_BUCKET: $GCS_BUCKET"
echo "  GCP_PROJECT_ID: $GCP_PROJECT_ID"
echo "  PORT: $PORT"
echo "  METRICS_API_URL: $METRICS_API_URL"
echo ""

# Check if GCP credentials exist
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "❌ Error: GCP credentials not found at $GOOGLE_APPLICATION_CREDENTIALS"
    exit 1
fi
echo "✅ GCP credentials found"
echo ""

# Check if API dependencies are installed
echo "Checking API dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing API dependencies..."
    pip install -r "$SCRIPT_DIR/requirements-api.txt"
else
    echo "✅ API dependencies installed"
fi
echo ""

# Check if dashboard dependencies are installed
echo "Checking dashboard dependencies..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dashboard dependencies..."
    pip install -r "$SCRIPT_DIR/dashboard/requirements-dashboard.txt"
else
    echo "✅ Dashboard dependencies installed"
fi
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "=========================================="
    echo "Shutting down services..."
    echo "=========================================="
    if [ ! -z "$API_PID" ]; then
        echo "Stopping API server (PID: $API_PID)..."
        kill $API_PID 2>/dev/null || true
    fi
    if [ ! -z "$DASHBOARD_PID" ]; then
        echo "Stopping dashboard (PID: $DASHBOARD_PID)..."
        kill $DASHBOARD_PID 2>/dev/null || true
    fi
    echo "✅ Services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start API server in background
echo "=========================================="
echo "Starting API server..."
echo "=========================================="
cd "$SCRIPT_DIR"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 > "$SCRIPT_DIR/api.log" 2>&1 &
API_PID=$!
echo "API server started (PID: $API_PID)"
echo "Logs: $SCRIPT_DIR/api.log"
echo ""

# Wait for API to be ready
echo "Waiting for API to be ready..."
MAX_WAIT=30
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    echo -n "."
done
echo ""

if [ $WAIT_COUNT -eq $MAX_WAIT ]; then
    echo "❌ Error: API did not start within $MAX_WAIT seconds"
    echo "Check logs: $SCRIPT_DIR/api.log"
    exit 1
fi

# Test API health
echo "Testing API health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8080/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "✅ API health check passed"
else
    echo "⚠️  API health check warning: $HEALTH_RESPONSE"
fi
echo ""

# Start dashboard
echo "=========================================="
echo "Starting Streamlit Dashboard..."
echo "=========================================="
cd "$SCRIPT_DIR/dashboard"
echo "Dashboard will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop both services"
echo ""

# Run dashboard in foreground (so script stays alive)
streamlit run app.py

