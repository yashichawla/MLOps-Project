@echo off
echo ==========================================
echo Starting Metrics API Server
echo ==========================================

REM Set environment variables
set GOOGLE_APPLICATION_CREDENTIALS=..\.secrets\gcp-key.json
set GCS_BUCKET=mlops-project-dvc-480422
set GCP_PROJECT_ID=break-the-bot-480422
set PORT=8080

echo Environment variables set:
echo   GOOGLE_APPLICATION_CREDENTIALS=%GOOGLE_APPLICATION_CREDENTIALS%
echo   GCS_BUCKET=%GCS_BUCKET%
echo   GCP_PROJECT_ID=%GCP_PROJECT_ID%
echo   PORT=%PORT%
echo.

echo Starting API server on http://localhost:8080
echo Press Ctrl+C to stop
echo.

python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

pause

