# Local Testing Guide

## Step 1: Install API Dependencies

```bash
cd MLOps-Project/deploy
pip install -r requirements-api.txt
```

## Step 2: Set Environment Variables

**Windows (PowerShell):**
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS=".secrets/gcp-key.json"
$env:GCS_BUCKET="mlops-project-dvc"
$env:GCP_PROJECT_ID="break-the-bot"
$env:PORT="8080"
```

**Windows (CMD):**
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=.secrets\gcp-key.json
set GCS_BUCKET=mlops-project-dvc
set GCP_PROJECT_ID=break-the-bot
set PORT=8080
```

**Linux/Mac:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=".secrets/gcp-key.json"
export GCS_BUCKET="mlops-project-dvc"
export GCP_PROJECT_ID="break-the-bot"
export PORT="8080"
```

## Step 3: Start the API Server

```bash
cd MLOps-Project/deploy/api
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at: `http://localhost:8080`

## Step 4: Test API Endpoints (in a new terminal)

### Test Health Endpoint
```bash
curl http://localhost:8080/health
```

### Test List Models
```bash
curl http://localhost:8080/dashboard/models
```

### Test All Metrics
```bash
curl http://localhost:8080/metrics/all
```

### Test Specific Model
```bash
curl http://localhost:8080/metrics/llama-3-8b
curl http://localhost:8080/metrics/llama-3-8b/bias
curl http://localhost:8080/metrics/llama-3-8b/summary
```

## Step 5: Install Dashboard Dependencies

In a new terminal:
```bash
cd MLOps-Project/deploy/dashboard
pip install -r requirements-dashboard.txt
```

## Step 6: Start the Dashboard

```bash
# From repository root
streamlit run deploy/dashboard/app.py
```

Or from dashboard directory:
```bash
cd MLOps-Project/deploy/dashboard
streamlit run app.py
```

The dashboard will be available at: `http://localhost:8501`

## Step 7: Test Dashboard

1. Open browser to `http://localhost:8501`
2. Check API connection status in sidebar
3. Select a model from dropdown
4. View metrics, charts, and bias detection
5. Test auto-refresh functionality

## Troubleshooting

### API won't start
- Check GCP credentials path is correct
- Verify GCS bucket name matches
- Check Python dependencies are installed

### API returns 404
- Verify metrics exist in GCS bucket
- Check model names match exactly
- Review API logs for errors

### Dashboard can't connect
- Verify API is running on port 8080
- Check API URL in dashboard sidebar
- Test API health endpoint manually

### Import errors in dashboard
- Ensure all dependencies are installed
- Check Python path includes dashboard directory
- Verify file structure is correct

