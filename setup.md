# Setup Guide - Break The Bot

Complete setup guide for deploying the Break The Bot MLOps project on a clean system.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
5. [Configuration](#configuration)
6. [Running Services](#running-services)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)
9. [Cloud Deployment](#cloud-deployment)
10. [Next Steps](#next-steps)

---

## Introduction

Break The Bot is an automated MLOps pipeline for continuous safety evaluation of Large Language Models (LLMs). The system:

- Preprocesses and validates adversarial prompts
- Runs prompts through victim LLMs to generate responses
- Uses LLM-as-a-Judge for automated safety scoring
- Computes metrics including Attack Success Rate (ASR) and bias detection
- Provides a dashboard for visualization and monitoring

The project consists of:
- **Airflow DAGs**: Data preprocessing, validation, and model evaluation pipeline
- **Metrics API**: FastAPI service for accessing evaluation metrics
- **Dashboard**: Streamlit web interface for visualization
- **Infrastructure**: Terraform configurations for GCP deployment

---

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **Git**: For cloning the repository
- **gcloud CLI**: (Optional) For cloud deployment

### Required Accounts and API Keys

Before starting, ensure you have:

1. **Google Cloud Platform Account**
   - GCP project with billing enabled
   - Service account key JSON file with GCS access
   - **DVC GCS bucket created** (default: `mlops-project-dvc-480422`) - see [GCP Configuration](#gcp-configuration) for creation instructions

2. **HuggingFace Account**
   - Read-only token from https://huggingface.co/settings/tokens
   - Used for running prompts through victim LLMs

3. **Groq API Key**
   - API key from https://console.groq.com/
   - Used for LLM-as-a-Judge evaluation

4. **Gmail Account**
   - Gmail App Password (16-digit password)
   - Generate from https://myaccount.google.com/apppasswords
   - Used for email notifications from Airflow

### Verify Prerequisites

```bash
# Check Python version
python --version  # Should be 3.11+

# Check Docker
docker --version
docker compose version

# Check Git
git --version

# (Optional) Check gcloud
gcloud --version
```

---

## Quick Start

For the fastest path to running the project locally:

```bash
# 1. Clone the repository
git clone https://github.com/yashichawla/MLOps-Project
cd MLOps-Project

# 2. Run the automated setup script
chmod +x setup.sh
./setup.sh

# 3. Follow the interactive prompts to:
#    - Set up environment variables
#    - Configure GCP credentials
#    - Initialize Airflow

# 4. Start Airflow
docker compose up -d webserver scheduler

# 5. Start API and Dashboard
cd deploy
./start_all.sh
```

**Access Points:**
- Airflow UI: http://localhost:8080 (admin/admin)
- API: http://localhost:8080 (if running separately)
- Dashboard: http://localhost:8501

---

## Detailed Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/yashichawla/MLOps-Project
cd MLOps-Project
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows (Git Bash):
source venv/Scripts/activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install API dependencies (if running API locally)
pip install -r deploy/requirements-api.txt

# Install dashboard dependencies (if running dashboard locally)
pip install -r deploy/dashboard/requirements-dashboard.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy template if available, or create new file
touch .env
```

Add the following variables to `.env`:

```bash
# SMTP Configuration (for Airflow email notifications)
AIRFLOW_SMTP_USER=your_email@gmail.com
AIRFLOW_SMTP_PASSWORD=your_16_digit_gmail_app_password

# HuggingFace Token (for victim LLM API calls)
HF_TOKEN=your_huggingface_token

# Groq API Key (for LLM-as-a-Judge)
GROQ_API_KEY=your_groq_api_key

# Airflow Web Port (optional, defaults to 8080)
AIRFLOW_WEB_PORT=8080
```

**Important Notes:**
- Use a Gmail App Password, not your regular Gmail password
- The App Password is a 16-character string (format: `xxxx xxxx xxxx xxxx`)
- Keep `.env` file secure and never commit it to version control

### Step 5: Set Up GCP Credentials

1. Obtain a GCP service account key JSON file
2. Create the `.secrets` directory:

```bash
mkdir -p .secrets
```

3. Place the service account key file at:

```bash
.secrets/gcp-key.json
```

4. Ensure the service account has the following permissions:
   - `roles/storage.objectViewer` on the GCS bucket
   - `roles/storage.objectAdmin` (if using DVC push)

### Step 6: Create Required Directories

```bash
# Create logs directory for Airflow
mkdir -p airflow_artifacts/logs

# Create data directories (if not using DVC)
mkdir -p data/processed
mkdir -p data/metrics
mkdir -p data/responses
mkdir -p data/judge
mkdir -p data/bias
```

### Step 7: Initialize Airflow

```bash
# Initialize Airflow database and create admin user
docker compose run --rm airflow-init

# This will:
# - Create the Airflow database schema
# - Create an admin user (username: admin, password: admin)
# - Set up initial configuration
```

### Step 8: Configure SMTP Connection in Airflow

After Airflow is initialized, set up the SMTP connection:

1. **Start Airflow services:**
   ```bash
   docker compose up -d webserver scheduler
   ```

2. **Open Airflow UI:** http://localhost:8080
   - Username: `admin`
   - Password: `admin`

3. **Navigate to Connections:**
   - Go to: Admin → Connections
   - Click the "+" button (Add a new record)

4. **Configure SMTP Connection:**
   - **Connection Id**: `gmail_smtp`
   - **Connection Type**: `Email`
   - **Host**: `smtp.gmail.com`
   - **Schema**: (leave empty)
   - **Login**: Your Gmail address (e.g., `your.email@gmail.com`)
   - **Password**: Your 16-character Gmail App Password
   - **Port**: `587`
   - **Extra** (JSON):
     ```json
     {
       "starttls": true,
       "ssl": false
     }
     ```

5. **Click Save**

**Alternative:** Use the provided script:
```bash
./scripts/setup_smtp_connection.sh your.email@gmail.com "xxxx xxxx xxxx xxxx"
```

---

## Configuration

### Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AIRFLOW_SMTP_USER` | Gmail address for SMTP | Yes | - |
| `AIRFLOW_SMTP_PASSWORD` | Gmail App Password | Yes | - |
| `HF_TOKEN` | HuggingFace API token | Yes | - |
| `GROQ_API_KEY` | Groq API key | Yes | - |
| `AIRFLOW_WEB_PORT` | Airflow web server port | No | 8080 |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account key | Yes | `.secrets/gcp-key.json` |
| `GCS_BUCKET` | GCS bucket name for DVC | Yes | `mlops-project-dvc-480422` |
| `GCP_PROJECT_ID` | GCP project ID | Yes | `break-the-bot-480422` |

### GCP Configuration

The project uses Google Cloud Storage (GCS) for:
- DVC remote storage
- Metrics and bias reports
- Model responses and judgements

**Bucket Configuration:**
- Default bucket: `mlops-project-dvc-480422`
- Project ID: `break-the-bot-480422`

**Important:** The DVC bucket (`mlops-project-dvc-480422` by default) is **not created by Terraform**. You must create it manually before deploying or running the pipeline:

```bash
# Create the DVC bucket
gsutil mb -p break-the-bot-480422 -l us-central1 gs://mlops-project-dvc-480422

# Or using gcloud
gcloud storage buckets create gs://mlops-project-dvc-480422 \
  --project=break-the-bot-480422 \
  --location=us-central1
```

Ensure the service account has appropriate permissions (`roles/storage.objectViewer` or `roles/storage.objectAdmin`) on this bucket.

To use different buckets, update:
- `.env` file (for local development)
- `deploy/api/config.py` (for API service)
- `dags/dvc_project/.dvc/config` (for DVC remote)

### DVC Configuration

DVC is configured to use GCS as remote storage. The configuration is in:
- `dags/dvc_project/.dvc/config`

To verify DVC remote:
```bash
cd dags/dvc_project
dvc remote list
```

---

## Running Services

### Airflow Services

**Start Airflow:**
```bash
docker compose up -d webserver scheduler
```

**Stop Airflow:**
```bash
docker compose down
```

**View Logs:**
```bash
docker compose logs -f webserver
docker compose logs -f scheduler
```

**Access Airflow UI:**
- URL: http://localhost:8080
- Username: `admin`
- Password: `admin`

### API Service

**Option 1: Using the start script (Recommended)**
```bash
cd deploy
./start_api_only.sh
```

**Option 2: Manual start**
```bash
cd deploy
export GOOGLE_APPLICATION_CREDENTIALS="../.secrets/gcp-key.json"
export GCS_BUCKET="mlops-project-dvc-480422"
export GCP_PROJECT_ID="break-the-bot-480422"
export PORT="8080"

python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

**Access API:**
- URL: http://localhost:8080
- Health check: http://localhost:8080/health
- API docs: http://localhost:8080/docs

### Dashboard Service

**Option 1: Using the start script (Recommended)**
```bash
cd deploy
./start_dashboard_only.sh
```

**Option 2: Manual start**
```bash
# From project root
streamlit run deploy/dashboard/app.py

# Or from dashboard directory
cd deploy/dashboard
streamlit run app.py
```

**Access Dashboard:**
- URL: http://localhost:8501

### Start All Services Together

```bash
cd deploy
./start_all.sh
```

This script:
- Starts the API server in the background
- Waits for API to be ready
- Starts the dashboard in the foreground
- Handles cleanup on exit (Ctrl+C)

---

## Verification

### Verify Airflow Setup

1. **Check Airflow UI:**
   - Open http://localhost:8080
   - Login with admin/admin
   - Verify DAG `salad_ml_evaluation_pipeline_v1` is visible

2. **Check SMTP Connection:**
   - Go to Admin → Connections
   - Verify `gmail_smtp` connection exists and is configured correctly

3. **Test DAG:**
   - In Airflow UI, find the DAG
   - Click "Play" button to trigger a test run
   - Monitor task execution in the Graph view

### Verify API Service

```bash
# Health check
curl http://localhost:8080/health

# Expected response:
# {
#   "status": "healthy",
#   "gcs_connected": true,
#   "bucket": "mlops-project-dvc-480422",
#   "timestamp": "..."
# }

# List all models
curl http://localhost:8080/metrics/all

# Get specific model metrics
curl http://localhost:8080/metrics/llama-3-8b
```

### Verify Dashboard

1. Open http://localhost:8501
2. Check sidebar for API connection status
3. Verify model list is populated (if data exists)
4. Test model selection and metric visualization

### Verify GCP Access

```bash
# Test GCS access
gsutil ls gs://mlops-project-dvc-480422/

# Test with service account
export GOOGLE_APPLICATION_CREDENTIALS=".secrets/gcp-key.json"
gsutil ls gs://mlops-project-dvc-480422/
```

---

## Troubleshooting

### Common Issues

#### 1. Docker Not Running

**Error:** `Cannot connect to the Docker daemon`

**Solution:**
```bash
# Start Docker service
# On Linux:
sudo systemctl start docker

# On Mac/Windows: Start Docker Desktop application
```

#### 2. Port Already in Use

**Error:** `Port 8080 is already in use`

**Solution:**
```bash
# Find process using port 8080
# On Linux/Mac:
lsof -i :8080

# On Windows:
netstat -ano | findstr :8080

# Kill the process or change AIRFLOW_WEB_PORT in .env
```

#### 3. GCP Credentials Not Found

**Error:** `FileNotFoundError: .secrets/gcp-key.json`

**Solution:**
- Verify the file exists at `.secrets/gcp-key.json`
- Check file permissions (should be readable)
- Verify the path in environment variables

#### 4. SMTP Connection Failed

**Error:** `SMTPAuthenticationError` or connection timeout

**Solution:**
- Verify you're using a Gmail App Password, not regular password
- Check that 2-Step Verification is enabled on Google Account
- Verify connection settings in Airflow UI match the guide
- Check firewall/network settings

#### 5. DVC Remote Access Denied

**Error:** `AccessDeniedException` when running DVC commands

**Solution:**
- Verify service account has `storage.objectViewer` permission
- Check bucket name is correct
- Verify GCP project ID matches

#### 6. API Can't Connect to GCS

**Error:** API health check shows `gcs_connected: false`

**Solution:**
```bash
# Verify credentials
export GOOGLE_APPLICATION_CREDENTIALS=".secrets/gcp-key.json"
gsutil ls gs://mlops-project-dvc-480422/

# Check API logs
tail -f deploy/api.log
```

#### 7. Dashboard Can't Connect to API

**Error:** Dashboard shows "API connection failed"

**Solution:**
- Verify API is running: `curl http://localhost:8080/health`
- Check API URL in dashboard sidebar (should be `http://localhost:8080`)
- Check for CORS issues in API logs

### Getting Help

If issues persist:

1. Check service logs:
   ```bash
   # Airflow logs
   docker compose logs webserver
   docker compose logs scheduler
   
   # API logs
   tail -f deploy/api.log
   
   # Dashboard logs (in terminal where it's running)
   ```

2. Verify environment variables:
   ```bash
   # Check .env file
   cat .env
   
   # Check environment variables
   env | grep AIRFLOW
   env | grep GCS
   ```

3. Review configuration files:
   - `docker-compose.yml`
   - `deploy/api/config.py`
   - `deploy/dashboard/config.py`

---

## Cloud Deployment

### Deploy API to Cloud Run

**Prerequisites:**
- gcloud CLI installed and authenticated
- GCP project with Cloud Run API enabled
- Service account with GCS read permissions

**Deploy using Cloud Build (Recommended):**
```bash
cd MLOps-Project
gcloud builds submit --config cloudbuild.yaml
```

**Deploy manually:**
```bash
cd deploy
./deploy.sh
```

**Verify deployment:**
```bash
# Get service URL
gcloud run services describe metrics-api \
  --region us-central1 \
  --format "value(status.url)"

# Test health endpoint
curl https://your-service-url.run.app/health
```

For detailed deployment instructions, see [deployment_guide.md](./deployment_guide.md).

### Deploy Infrastructure with Terraform

**Prerequisites:**
- Terraform installed
- GCP project with billing enabled
- Appropriate IAM permissions
- **DVC GCS bucket must exist** (default: `mlops-project-dvc-480422`)

**Note:** The DVC bucket is **not created by Terraform**. It must be created manually before deploying infrastructure. The Terraform configuration references an existing bucket via the `dvc_bucket_name` variable.

**Create DVC Bucket (if not exists):**
```bash
# Create the DVC bucket before running Terraform
gsutil mb -p break-the-bot-480422 -l us-central1 gs://mlops-project-dvc-480422

# Or using gcloud
gcloud storage buckets create gs://mlops-project-dvc-480422 \
  --project=break-the-bot-480422 \
  --location=us-central1
```

**Deploy:**
```bash
cd terraform

# Initialize Terraform
terraform init

# Review plan
terraform plan

# Apply configuration
terraform apply
```

For detailed Terraform setup, see [terraform/README.md](./terraform/README.md).

### Deploy to Cloud Composer

The Terraform configuration includes Cloud Composer setup. After deploying:

1. Get Composer Airflow URI:
   ```bash
   gcloud composer environments describe mlops-airflow-composer \
     --location us-central1 \
     --format "value(config.airflowUri)"
   ```

2. Set up secrets:
   ```bash
   cd terraform
   ./setup-secrets.sh
   ```

3. Upload DAGs:
   ```bash
   ./upload_to_composer.sh
   ```

4. Deploy Webhook Trigger Service:
   ```bash
   cd terraform/webhook-trigger
   gcloud builds submit --config cloudbuild.yaml
   ```
   
   The webhook service automatically:
   - Fetches the Composer Airflow URI
   - Deploys to Cloud Run
   - Configures environment variables
   
   Get the webhook URL:
   ```bash
   gcloud run services describe composer-webhook-trigger \
     --region us-central1 \
     --format "value(status.url)"
   ```
   
   Configure in GitHub:
   - Go to repository Settings → Webhooks
   - Add webhook with the URL from above + `/webhook`
   - Select "Just the push event"
   - Use the same secret from `GITHUB_WEBHOOK_SECRET`

---

## Next Steps

After successful setup:

1. **Run the DAG:**
   - Open Airflow UI
   - Trigger `salad_ml_evaluation_pipeline_v1`
   - Monitor execution

2. **Explore the Dashboard:**
   - View metrics and bias detection results
   - Compare model performance
   - Analyze trends

3. **Review Documentation:**
   - [README.md](./README.md) - Project overview and usage
   - [deployment_guide.md](./deployment_guide.md) - Cloud deployment details
   - [quick_start.md](./quick_start.md) - Quick reference guide
   - [terraform/README.md](./terraform/README.md) - Infrastructure setup

4. **Customize Configuration:**
   - Update model configurations in `config/attack_llm_config.json`
   - Modify data sources in `config/data_sources.json`
   - Adjust validation rules in DAG scripts

5. **Run Tests:**
   ```bash
   pytest -q
   ```

---

## Additional Resources

- **Project Repository:** https://github.com/yashichawla/MLOps-Project
- **Airflow Documentation:** https://airflow.apache.org/docs/
- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **Streamlit Documentation:** https://docs.streamlit.io/
- **DVC Documentation:** https://dvc.org/doc
- **Google Cloud Documentation:** https://cloud.google.com/docs

---

**Need Help?** Check the troubleshooting section or review the logs for specific error messages.

