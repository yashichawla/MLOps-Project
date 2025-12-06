#!/bin/bash
# Test API endpoints

API_URL="http://localhost:8080"

echo "=========================================="
echo "Testing Metrics API Endpoints"
echo "=========================================="
echo "API URL: $API_URL"
echo ""

# Test health endpoint
echo "1. Testing /health endpoint..."
curl -s "$API_URL/health" | python -m json.tool
echo ""
echo ""

# Test list models
echo "2. Testing /dashboard/models endpoint..."
curl -s "$API_URL/dashboard/models" | python -m json.tool
echo ""
echo ""

# Test all metrics
echo "3. Testing /metrics/all endpoint..."
curl -s "$API_URL/metrics/all" | python -m json.tool
echo ""
echo ""

# Test specific model (if available)
echo "4. Testing /metrics/llama-3-8b endpoint..."
curl -s "$API_URL/metrics/llama-3-8b" | python -m json.tool
echo ""
echo ""

# Test bias report
echo "5. Testing /metrics/llama-3-8b/bias endpoint..."
curl -s "$API_URL/metrics/llama-3-8b/bias" | python -m json.tool
echo ""
echo ""

# Test summary
echo "6. Testing /metrics/llama-3-8b/summary endpoint..."
curl -s "$API_URL/metrics/llama-3-8b/summary" | python -m json.tool
echo ""
echo ""

echo "=========================================="
echo "API Testing Complete"
echo "=========================================="

