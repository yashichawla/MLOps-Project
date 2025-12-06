#!/bin/bash
# Start Dashboard only (assumes API is already running)

set -e

echo "=========================================="
echo "Starting Streamlit Dashboard"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set API URL
export METRICS_API_URL="${METRICS_API_URL:-http://localhost:8080}"

echo "API URL: $METRICS_API_URL"
echo ""

# Check if API is running
if ! curl -s "$METRICS_API_URL/health" > /dev/null 2>&1; then
    echo "⚠️  Warning: API does not appear to be running at $METRICS_API_URL"
    echo "   Start the API first with: ./start_api_only.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start dashboard
cd "$SCRIPT_DIR/dashboard"
echo "Starting dashboard on http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run app.py

