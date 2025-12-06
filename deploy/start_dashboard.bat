@echo off
echo ==========================================
echo Starting Streamlit Dashboard
echo ==========================================

REM Set API URL (default to localhost)
set METRICS_API_URL=http://localhost:8080

echo API URL: %METRICS_API_URL%
echo.
echo Starting dashboard on http://localhost:8501
echo Press Ctrl+C to stop
echo.

cd dashboard
streamlit run app.py

pause

