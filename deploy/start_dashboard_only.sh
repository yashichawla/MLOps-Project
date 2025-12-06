#!/bin/bash
# Start Dashboard only (connects to deployed Cloud Run API by default)

set -e

echo "=========================================="
echo "Starting Streamlit Dashboard"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set API URL (defaults to deployed Cloud Run service)
export METRICS_API_URL="${METRICS_API_URL:-https://metrics-api-hel7hrgq5q-uc.a.run.app}"

echo "API URL: $METRICS_API_URL"
echo ""

# Check if API is accessible
if ! curl -s "$METRICS_API_URL/health" > /dev/null 2>&1; then
    echo "⚠️  Warning: API does not appear to be accessible at $METRICS_API_URL"
    echo "   This may be a network issue or the service may be down."
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

