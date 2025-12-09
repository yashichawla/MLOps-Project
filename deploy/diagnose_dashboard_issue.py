#!/usr/bin/env python3
"""Diagnostic script to debug why dashboard cannot pull data from GCS.

This script checks:
1. GCS bucket configuration
2. File existence in expected paths
3. API configuration
4. GCS client connectivity
"""

import os
import sys
from pathlib import Path

# Add deploy/api to path
sys.path.insert(0, str(Path(__file__).parent / "api"))

from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden
from api.config import config
from api.gcs_client import GCSClient

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def check_config():
    """Check API configuration."""
    print_section("API Configuration")
    print(f"GCS_BUCKET: {config.GCS_BUCKET}")
    print(f"GCP_PROJECT_ID: {config.GCP_PROJECT_ID}")
    print(f"METRICS_PATH_PREFIX: {config.METRICS_PATH_PREFIX}")
    print(f"BIAS_PATH_PREFIX: {config.BIAS_PATH_PREFIX}")
    print(f"\nExpected metrics path pattern: {config.get_metrics_path('MODEL_NAME')}")
    print(f"Expected bias path pattern: {config.get_bias_report_path('MODEL_NAME')}")

def check_gcs_connectivity():
    """Check GCS client connectivity."""
    print_section("GCS Connectivity Check")
    try:
        client = storage.Client(project=config.GCP_PROJECT_ID)
        bucket = client.bucket(config.GCS_BUCKET)
        
        # Check if bucket exists
        try:
            bucket.reload()
            print(f"✅ Bucket exists: gs://{config.GCS_BUCKET}")
            print(f"   Location: {bucket.location}")
            print(f"   Storage class: {bucket.storage_class}")
        except NotFound:
            print(f"❌ Bucket not found: gs://{config.GCS_BUCKET}")
            return False
        except Forbidden:
            print(f"❌ Permission denied accessing bucket: gs://{config.GCS_BUCKET}")
            print("   Check service account permissions")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error connecting to GCS: {e}")
        return False

def check_files_in_bucket():
    """Check if expected files exist in bucket."""
    print_section("Checking Files in Bucket")
    
    try:
        client = storage.Client(project=config.GCP_PROJECT_ID)
        bucket = client.bucket(config.GCS_BUCKET)
        
        # Check metrics directory
        metrics_prefix = f"{config.METRICS_PATH_PREFIX}/additional/"
        print(f"\n📊 Checking metrics files at: gs://{config.GCS_BUCKET}/{metrics_prefix}")
        blobs = list(client.list_blobs(bucket, prefix=metrics_prefix))
        
        if blobs:
            print(f"✅ Found {len(blobs)} file(s) in metrics directory:")
            for blob in blobs[:10]:  # Show first 10
                print(f"   - {blob.name} ({blob.size} bytes, updated: {blob.updated})")
            if len(blobs) > 10:
                print(f"   ... and {len(blobs) - 10} more files")
        else:
            print("❌ No files found in metrics directory")
            print(f"   Expected path: gs://{config.GCS_BUCKET}/{metrics_prefix}")
            print("\n   Possible issues:")
            print("   1. DAG has not run yet")
            print("   2. copy_to_api_paths task failed")
            print("   3. Files are in a different location")
        
        # Check bias directory
        bias_prefix = f"{config.BIAS_PATH_PREFIX}/"
        print(f"\n🎯 Checking bias reports at: gs://{config.GCS_BUCKET}/{bias_prefix}")
        blobs = list(client.list_blobs(bucket, prefix=bias_prefix))
        
        if blobs:
            print(f"✅ Found {len(blobs)} file(s) in bias directory:")
            for blob in blobs[:10]:  # Show first 10
                print(f"   - {blob.name} ({blob.size} bytes, updated: {blob.updated})")
            if len(blobs) > 10:
                print(f"   ... and {len(blobs) - 10} more files")
        else:
            print("❌ No files found in bias directory")
            print(f"   Expected path: gs://{config.GCS_BUCKET}/{bias_prefix}")
        
        # Check for files in Composer bucket (where DAG writes)
        composer_bucket_name = "us-central1-mlops-airflow-c-47afa20c-bucket"
        composer_prefix = "dags/dvc_project/data/metrics/additional/"
        print(f"\n🔍 Checking Composer bucket (where DAG writes):")
        print(f"   gs://{composer_bucket_name}/{composer_prefix}")
        try:
            composer_bucket = client.bucket(composer_bucket_name)
            composer_bucket.reload()
            composer_blobs = list(client.list_blobs(composer_bucket, prefix=composer_prefix))
            if composer_blobs:
                print(f"✅ Found {len(composer_blobs)} file(s) in Composer bucket:")
                for blob in composer_blobs[:5]:
                    print(f"   - {blob.name} ({blob.size} bytes)")
                print("\n   ⚠️  Files exist in Composer bucket but may not be synced to API bucket")
                print("   Check if copy_to_api_paths task ran successfully")
            else:
                print("❌ No files found in Composer bucket either")
        except NotFound:
            print(f"⚠️  Composer bucket not accessible: {composer_bucket_name}")
        except Forbidden:
            print(f"⚠️  Permission denied accessing Composer bucket: {composer_bucket_name}")
        
    except Exception as e:
        print(f"❌ Error checking files: {e}")
        import traceback
        traceback.print_exc()

def check_api_client():
    """Test API client functionality."""
    print_section("Testing API Client")
    
    try:
        gcs_client = GCSClient()
        
        # Test listing models
        print("\n📋 Testing list_available_models()...")
        service = __import__("api.metrics_service", fromlist=["MetricsService"])
        MetricsService = service.MetricsService
        metrics_service = MetricsService(gcs_client)
        
        models = metrics_service.list_available_models()
        if models:
            print(f"✅ Found {len(models)} model(s):")
            for model in models:
                print(f"   - {model['name']}")
                print(f"     Has metrics: {model.get('has_metrics', False)}")
                print(f"     Has bias report: {model.get('has_bias_report', False)}")
        else:
            print("❌ No models found")
            print("   This means the API cannot find any metrics files")
        
        # Test reading a specific file if models exist
        if models:
            test_model = models[0]['name']
            print(f"\n📖 Testing read for model: {test_model}")
            
            metrics_path = config.get_metrics_path(test_model)
            print(f"   Metrics path: {metrics_path}")
            metrics = gcs_client.read_json(metrics_path)
            if metrics:
                print(f"   ✅ Successfully read metrics file")
            else:
                print(f"   ❌ Failed to read metrics file")
            
            bias_path = config.get_bias_report_path(test_model)
            print(f"   Bias path: {bias_path}")
            bias = gcs_client.read_json(bias_path)
            if bias:
                print(f"   ✅ Successfully read bias report")
            else:
                print(f"   ❌ Failed to read bias report")
        
    except Exception as e:
        print(f"❌ Error testing API client: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all diagnostic checks."""
    print("\n" + "=" * 80)
    print("  Dashboard Data Pull Diagnostic Tool")
    print("=" * 80)
    
    check_config()
    
    if not check_gcs_connectivity():
        print("\n❌ Cannot proceed - GCS connectivity check failed")
        return
    
    check_files_in_bucket()
    check_api_client()
    
    print_section("Summary and Recommendations")
    print("""
If files are missing:
1. Check Airflow DAG logs for 'copy_to_api_paths' task
2. Verify the task completed successfully
3. Check GCS bucket permissions for service account
4. Verify bucket names match between DAG and API config

If files exist but API can't read them:
1. Check service account has storage.objects.get permission
2. Verify file paths match exactly (case-sensitive)
3. Check for authentication issues (GOOGLE_APPLICATION_CREDENTIALS)

To manually sync files from Composer bucket:
  gcloud storage rsync -r gs://us-central1-mlops-airflow-c-47afa20c-bucket/dags/dvc_project/data/metrics/additional \\
    gs://mlops-project-dvc-480422/data/metrics/additional/
    """)

if __name__ == "__main__":
    main()

