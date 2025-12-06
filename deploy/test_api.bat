@echo off
echo ==========================================
echo Testing Metrics API Endpoints
echo ==========================================
echo.

set API_URL=http://localhost:8080

echo Testing /health endpoint...
curl -s %API_URL%/health
echo.
echo.

echo Testing /dashboard/models endpoint...
curl -s %API_URL%/dashboard/models
echo.
echo.

echo Testing /metrics/all endpoint...
curl -s %API_URL%/metrics/all
echo.
echo.

echo Testing /metrics/llama-3-8b endpoint...
curl -s %API_URL%/metrics/llama-3-8b
echo.
echo.

echo ==========================================
echo Testing Complete
echo ==========================================
pause

