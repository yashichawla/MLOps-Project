@echo off
echo ==========================================
echo Starting Streamlit Dashboard
echo ==========================================

REM Set API URL (defaults to deployed Cloud Run service)
set METRICS_API_URL=https://metrics-api-hel7hrgq5q-uc.a.run.app

echo API URL: %METRICS_API_URL%
echo.
echo Starting dashboard on http://localhost:8501
echo Press Ctrl+C to stop
echo.

cd dashboard
streamlit run app.py

pause

