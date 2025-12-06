# Deployment Guide

This guide explains how to deploy the Metrics API to Google Cloud Run.

## Prerequisites

1. **Google Cloud SDK (gcloud CLI)** installed and configured
2. **Docker** installed and running
3. **GCP Project** with the following APIs enabled:
   - Cloud Run API
   - Container Registry API
   - Cloud Build API (for automated deployment)

4. **Service Account** with permissions:
   - `roles/storage.objectViewer` on GCS bucket `gs://mlops-project-dvc`
   - The service account `dvc-airflow@break-the-bot.iam.gserviceaccount.com` is already configured

## Quick Deployment

### Option 1: Automated Deployment (Cloud Build)

If you have Cloud Build permissions:

```bash
cd MLOps-Project
gcloud builds submit --config cloudbuild.yaml
```

### Option 2: Manual Deployment

If you don't have Cloud Build permissions:

```bash
cd MLOps-Project/deploy
./deploy_manual.sh
```

Or step by step:

```bash
# 1. Set project
gcloud config set project break-the-bot

# 2. Authenticate Docker
gcloud auth configure-docker

# 3. Build and push image
cd MLOps-Project/deploy
docker build -t gcr.io/break-the-bot/metrics-api:latest .
docker push gcr.io/break-the-bot/metrics-api:latest

# 4. Deploy to Cloud Run
gcloud run deploy metrics-api \
  --image gcr.io/break-the-bot/metrics-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET=mlops-project-dvc,GCP_PROJECT_ID=break-the-bot \
  --service-account=dvc-airflow@break-the-bot.iam.gserviceaccount.com \
  --memory=512Mi \
  --cpu=1 \
  --max-instances=10 \
  --timeout=300
```

## Verify Deployment

After deployment, get the service URL:

```bash
gcloud run services describe metrics-api \
  --region us-central1 \
  --format "value(status.url)"
```

Test the endpoints:

```bash
# Health check
curl https://metrics-api-xxxxx.run.app/health

# List all models
curl https://metrics-api-xxxxx.run.app/metrics/all

# Get model metrics
curl https://metrics-api-xxxxx.run.app/metrics/llama-3-8b
```

## Environment Variables

The service uses these environment variables (set automatically in Cloud Run):

- `GCS_BUCKET`: GCS bucket name (default: `mlops-project-dvc`)
- `GCP_PROJECT_ID`: GCP project ID (default: `break-the-bot`)

## Service Account

The Cloud Run service uses the service account `dvc-airflow@break-the-bot.iam.gserviceaccount.com` which has:
- Read access to GCS bucket `gs://mlops-project-dvc`
- This allows the API to read metrics and bias reports from GCS

## Troubleshooting

### Permission Errors

If you get permission errors:

1. **Cloud Build**: Ensure you have `Cloud Build Service Account` role or `Service Usage Admin` role
2. **GCS Access**: The service account needs `storage.objectViewer` on the bucket
3. **Container Registry**: Ensure Container Registry API is enabled

### Service Not Starting

Check logs:

```bash
gcloud run services logs read metrics-api --region us-central1
```

### Missing Data

Ensure data is in GCS:
- Metrics: `gs://mlops-project-dvc/data/metrics/additional_metrics_{model_name}.json`
- Bias reports: `gs://mlops-project-dvc/data/bias/{model_name}/bias_report.json`

You can check with:

```bash
gsutil ls gs://mlops-project-dvc/data/metrics/
gsutil ls gs://mlops-project-dvc/data/bias/
```

## Updating the Service

To update after making changes:

1. Rebuild the Docker image
2. Push to Container Registry
3. Deploy to Cloud Run (it will automatically use the new image)

Or use the deployment scripts again.

## Monitoring

- **Logs**: Available in Cloud Logging
- **Metrics**: Cloud Run provides built-in metrics (requests, latency, errors)
- **Health Check**: Use `/health` endpoint for monitoring

