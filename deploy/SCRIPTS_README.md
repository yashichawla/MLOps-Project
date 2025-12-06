# Bash Scripts for Running API and Dashboard

## Available Scripts

### 1. `start_all.sh` - Start Both API and Dashboard

Starts the API server in the background, waits for it to be ready, then launches the dashboard.

**Usage:**
```bash
cd MLOps-Project/deploy
./start_all.sh
```

**What it does:**
- Sets up environment variables
- Checks and installs dependencies if needed
- Starts API server in background
- Waits for API to be ready (health check)
- Launches Streamlit dashboard
- Handles cleanup on Ctrl+C

**Output:**
- API: http://localhost:8080
- Dashboard: http://localhost:8501
- API logs: `deploy/api.log`

### 2. `start_api_only.sh` - Start API Server Only

Starts only the API server.

**Usage:**
```bash
cd MLOps-Project/deploy
./start_api_only.sh
```

**Use when:**
- You want to test API separately
- You want to run dashboard in a different terminal
- You need API logs in the same terminal

### 3. `start_dashboard_only.sh` - Start Dashboard Only

Starts only the dashboard (assumes API is already running).

**Usage:**
```bash
cd MLOps-Project/deploy
./start_dashboard_only.sh
```

**Use when:**
- API is already running
- You want to restart just the dashboard
- You're running API in a different terminal

## Quick Start

**Easiest way (one command):**
```bash
cd MLOps-Project/deploy
./start_all.sh
```

This will:
1. ✅ Start API server
2. ✅ Wait for API to be ready
3. ✅ Launch dashboard
4. ✅ Open browser to http://localhost:8501

Press **Ctrl+C** to stop both services.

## Manual Two-Terminal Approach

**Terminal 1 - API:**
```bash
cd MLOps-Project/deploy
./start_api_only.sh
```

**Terminal 2 - Dashboard:**
```bash
cd MLOps-Project/deploy
./start_dashboard_only.sh
```

## Troubleshooting

### Script won't run
```bash
# Make scripts executable
chmod +x start_all.sh start_api_only.sh start_dashboard_only.sh
```

### API won't start
- Check GCP credentials: `ls -la ../.secrets/gcp-key.json`
- Verify bucket name matches your GCS bucket
- Check port 8080 is free: `netstat -an | grep 8080`

### Dashboard can't connect
- Verify API is running: `curl http://localhost:8080/health`
- Check API URL in dashboard sidebar
- Ensure API started successfully (check api.log)

### Permission denied
```bash
# On Windows Git Bash, you might need:
bash start_all.sh
```

## Environment Variables

Scripts automatically set:
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account key
- `GCS_BUCKET` - GCS bucket name (default: mlops-project-dvc)
- `GCP_PROJECT_ID` - GCP project ID (default: break-the-bot)
- `PORT` - API port (default: 8080)
- `METRICS_API_URL` - Dashboard API URL (default: http://localhost:8080)

To override, export before running:
```bash
export GCS_BUCKET=your-bucket-name
export METRICS_API_URL=http://your-api-url
./start_all.sh
```

