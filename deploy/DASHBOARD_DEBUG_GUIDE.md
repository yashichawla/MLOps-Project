# Dashboard Data Pull Debugging Guide

## Problem
Dashboard cannot pull data after DAG runs on GCP Composer.

## Architecture Overview

The data flow is:
1. **DAG runs on GCP Composer** → Writes files to Composer bucket
2. **DVC Push** → Syncs files to DVC remote bucket (by hash)
3. **copy_to_api_paths task** → Syncs files to API-accessible paths
4. **API reads from GCS** → Serves data to dashboard

### Buckets and Paths

- **Composer Bucket**: `us-central1-mlops-airflow-c-47afa20c-bucket`
  - DAG writes to: `dags/dvc_project/data/metrics/additional/`
  - DAG writes to: `dags/dvc_project/data/bias/`

- **DVC Remote Bucket**: `mlops-project-dvc-480422`
  - DVC stores files by hash at: `mlops-project/{hash}`
  - API-accessible paths (after copy_to_api_paths):
    - `data/metrics/additional/additional_metrics_{model_name}.json`
    - `data/bias/{model_name}/bias_report.json`

## Common Issues and Solutions

### Issue 1: Files Not Synced to API Bucket

**Symptoms:**
- Dashboard shows "No models available"
- API `/health` endpoint shows `gcs_connected: true` but `/metrics/all` returns empty list
- Files exist in Composer bucket but not in API bucket

**Diagnosis:**
1. Check if `copy_to_api_paths` task ran successfully:
   ```bash
   # In Airflow UI, check the task logs for "copy_to_api_paths"
   # Look for errors or warnings
   ```

2. Run diagnostic script:
   ```bash
   cd MLOps-Project/deploy
   python diagnose_dashboard_issue.py
   ```

3. Manually check bucket contents:
   ```bash
   # Check Composer bucket (where DAG writes)
   gsutil ls gs://us-central1-mlops-airflow-c-47afa20c-bucket/dags/dvc_project/data/metrics/additional/
   
   # Check API bucket (where API reads)
   gsutil ls gs://mlops-project-dvc-480422/data/metrics/additional/
   ```

**Solution:**
If files exist in Composer bucket but not in API bucket:

1. **Manual sync** (temporary fix):
   ```bash
   gcloud storage rsync -r \
     gs://us-central1-mlops-airflow-c-47afa20c-bucket/dags/dvc_project/data/metrics/additional \
     gs://mlops-project-dvc-480422/data/metrics/additional/
   
   gcloud storage rsync -r \
     gs://us-central1-mlops-airflow-c-47afa20c-bucket/dags/dvc_project/data/bias \
     gs://mlops-project-dvc-480422/data/bias/
   ```

2. **Fix the DAG task** (permanent fix):
   - Check `copy_to_api_paths` task logs in Airflow
   - Verify task has proper permissions
   - Ensure task runs after `dvc_push_final` completes

### Issue 2: Permission Errors

**Symptoms:**
- API `/health` endpoint shows `gcs_connected: false`
- API logs show "Permission denied" errors
- Diagnostic script shows "Forbidden" errors

**Diagnosis:**
Check service account permissions:
```bash
# Get service account email
gcloud iam service-accounts list

# Check bucket IAM
gsutil iam get gs://mlops-project-dvc-480422
```

**Solution:**
Grant necessary permissions to service account:
```bash
# Grant read access to API bucket
gsutil iam ch serviceAccount:YOUR_SERVICE_ACCOUNT@PROJECT.iam.gserviceaccount.com:roles/storage.objectViewer \
  gs://mlops-project-dvc-480422

# Grant write access for copy_to_api_paths task (if needed)
gsutil iam ch serviceAccount:YOUR_SERVICE_ACCOUNT@PROJECT.iam.gserviceaccount.com:roles/storage.objectAdmin \
  gs://mlops-project-dvc-480422
```

### Issue 3: Path Mismatch

**Symptoms:**
- Files exist in bucket but API can't find them
- Different path structure than expected

**Diagnosis:**
1. Check actual file paths in bucket:
   ```bash
   gsutil ls -r gs://mlops-project-dvc-480422/data/metrics/
   ```

2. Compare with API expected paths:
   - API expects: `data/metrics/additional/additional_metrics_{model_name}.json`
   - Check if files are at different location

**Solution:**
Update API config or fix DAG sync task to match expected paths.

### Issue 4: Bucket Name Mismatch

**Symptoms:**
- API can't connect to bucket
- "Bucket not found" errors

**Diagnosis:**
Check API configuration:
```bash
# Check environment variable
echo $GCS_BUCKET

# Or check API config default
# Should be: mlops-project-dvc-480422
```

**Solution:**
Set correct bucket name:
```bash
export GCS_BUCKET="mlops-project-dvc-480422"
```

## Step-by-Step Debugging

### Step 1: Run Diagnostic Script

```bash
cd MLOps-Project/deploy
python diagnose_dashboard_issue.py
```

This will show:
- Current configuration
- GCS connectivity status
- Files found in buckets
- API client test results

### Step 2: Check API Health

```bash
# If API is running locally
curl http://localhost:8080/health

# Expected response:
# {
#   "status": "healthy",
#   "gcs_connected": true,
#   "bucket": "mlops-project-dvc-480422",
#   "timestamp": "..."
# }
```

### Step 3: Check Airflow DAG Logs

1. Open Airflow UI
2. Find the latest DAG run
3. Check `copy_to_api_paths` task:
   - Did it run?
   - Did it succeed?
   - Any errors in logs?

### Step 4: Verify File Locations

```bash
# Check Composer bucket (source)
gsutil ls -r gs://us-central1-mlops-airflow-c-47afa20c-bucket/dags/dvc_project/data/metrics/additional/

# Check API bucket (destination)
gsutil ls -r gs://mlops-project-dvc-480422/data/metrics/additional/
```

### Step 5: Test API Endpoints

```bash
# List all models
curl http://localhost:8080/metrics/all

# Get specific model metrics
curl http://localhost:8080/metrics/{model_name}

# Get bias report
curl http://localhost:8080/metrics/{model_name}/bias
```

## Quick Fixes

### Quick Fix 1: Manual Sync

If `copy_to_api_paths` task failed, manually sync files:

```bash
# Sync metrics
gcloud storage rsync -r \
  gs://us-central1-mlops-airflow-c-47afa20c-bucket/dags/dvc_project/data/metrics/additional \
  gs://mlops-project-dvc-480422/data/metrics/additional/

# Sync bias reports
gcloud storage rsync -r \
  gs://us-central1-mlops-airflow-c-47afa20c-bucket/dags/dvc_project/data/bias \
  gs://mlops-project-dvc-480422/data/bias/
```

### Quick Fix 2: Re-run copy_to_api_paths Task

In Airflow UI:
1. Find the DAG run
2. Clear the `copy_to_api_paths` task
3. Re-run it

### Quick Fix 3: Check API Logs

```bash
# If running locally
tail -f deploy/api.log

# Or check container logs if using Docker
docker logs <api-container-name>
```

## Prevention

To prevent this issue in the future:

1. **Monitor copy_to_api_paths task**: Set up alerts if task fails
2. **Add verification step**: Add a task after `copy_to_api_paths` that verifies files exist
3. **Improve error handling**: The enhanced GCS client now has better error logging
4. **Regular health checks**: Monitor API `/health` endpoint

## Related Files

- **DAG**: `MLOps-Project/dags/salad_preprocess_dag.py` (line ~2851: `copy_to_api_paths` task)
- **API Config**: `MLOps-Project/deploy/api/config.py`
- **GCS Client**: `MLOps-Project/deploy/api/gcs_client.py`
- **Diagnostic Script**: `MLOps-Project/deploy/diagnose_dashboard_issue.py`

## Getting Help

If issues persist:
1. Check Airflow DAG logs for detailed error messages
2. Check API logs for GCS access errors
3. Verify service account permissions
4. Run diagnostic script and share output

