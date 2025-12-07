# Quick Start - Local Testing

## Prerequisites Check

✅ All dependencies are already installed:
- FastAPI, Uvicorn, Google Cloud Storage (API)
- Streamlit, Plotly, Pandas, Requests (Dashboard)

## Step 1: Start the API Server

**Option A: Using batch file (Windows)**
```bash
# Double-click or run:
deploy\start_api.bat
```

**Option B: Manual start**
```bash
cd MLOps-Project/deploy/api

# Set environment variables (Windows CMD):
set GOOGLE_APPLICATION_CREDENTIALS=..\.secrets\gcp-key.json
set GCS_BUCKET=mlops-project-dvc
set GCP_PROJECT_ID=break-the-bot
set PORT=8080

# Start server:
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

**Option B: Manual start (PowerShell)**
```powershell
cd MLOps-Project/deploy/api

$env:GOOGLE_APPLICATION_CREDENTIALS="..\.secrets\gcp-key.json"
$env:GCS_BUCKET="mlops-project-dvc"
$env:GCP_PROJECT_ID="break-the-bot"
$env:PORT="8080"

python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

The API will start at: **http://localhost:8080**

## Step 2: Test API (Optional)

Open a **new terminal** and run:

**Windows:**
```bash
deploy\test_api.bat
```

**Or manually test:**
```bash
curl http://localhost:8080/health
curl http://localhost:8080/dashboard/models
curl http://localhost:8080/metrics/all
```

## Step 3: Start the Dashboard

Open a **new terminal** and run:

**Option A: Using batch file (Windows)**
```bash
deploy\start_dashboard.bat
```

**Option B: Manual start**
```bash
# From repository root:
streamlit run deploy/dashboard/app.py

# Or from dashboard directory:
cd MLOps-Project/deploy/dashboard
streamlit run app.py
```

The dashboard will start at: **http://localhost:8501**

## Step 4: Use the Dashboard

1. Open browser to: **http://localhost:8501**
2. Check sidebar for API connection status
3. Select a model from dropdown (or "All Models" for overview)
4. Explore metrics, charts, and bias detection

## Troubleshooting

### API won't start
- ✅ Check GCP credentials exist at `.secrets/gcp-key.json`
- ✅ Verify bucket name: `mlops-project-dvc`
- ✅ Check port 8080 is not in use: `netstat -an | findstr ":8080"`

### API returns errors
- Check API terminal for error messages
- Verify GCS bucket is accessible
- Ensure metrics files exist in GCS

### Dashboard can't connect
- Verify API is running (check http://localhost:8080/health)
- Check API URL in dashboard sidebar (should be `http://localhost:8080`)
- Look for connection errors in dashboard terminal

### Import errors
- Reinstall dependencies: `pip install -r requirements-api.txt` or `requirements-dashboard.txt`
- Check Python version: `python --version` (should be 3.11+)

## Expected Output

### API Terminal:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080
```

### Dashboard Terminal:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

## Next Steps

Once both are running:
1. ✅ API accessible at http://localhost:8080
2. ✅ Dashboard accessible at http://localhost:8501
3. ✅ Test all endpoints
4. ✅ Explore dashboard features
5. ✅ Verify data visualization

