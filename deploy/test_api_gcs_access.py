#!/usr/bin/env python3
"""Test script to verify API can access GCS files."""
import sys
import os
from pathlib import Path

# Add deploy/api to path
sys.path.insert(0, str(Path(__file__).parent / "api"))

from api.config import config
from api.gcs_client import GCSClient
from api.metrics_service import MetricsService

def test_file_access():
    """Test if API can access files."""
    print("=" * 80)
    print("Testing GCS File Access")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Bucket: {config.GCS_BUCKET}")
    print(f"  Project: {config.GCP_PROJECT_ID}")
    
    # Test GCS client
    print(f"\n1. Testing GCS Client...")
    gcs_client = GCSClient()
    
    # Test listing
    print(f"\n2. Testing list_blobs...")
    metrics_prefix = f"{config.METRICS_PATH_PREFIX}/additional/"
    blobs = gcs_client.list_blobs(metrics_prefix)
    print(f"   Found {len(blobs)} blob(s) with prefix: {metrics_prefix}")
    if blobs:
        print(f"   First few blobs:")
        for blob in blobs[:3]:
            print(f"     - {blob}")
    
    # Test blob_exists
    if blobs:
        test_blob = blobs[0]
        print(f"\n3. Testing blob_exists for: {test_blob}")
        exists = gcs_client.blob_exists(test_blob)
        print(f"   Result: {exists}")
        
        # Test read_json
        print(f"\n4. Testing read_json for: {test_blob}")
        data = gcs_client.read_json(test_blob)
        if data:
            print(f"   ✅ Successfully read file!")
            print(f"   Keys in data: {list(data.keys())}")
        else:
            print(f"   ❌ Failed to read file")
    
    # Test metrics service
    print(f"\n5. Testing MetricsService...")
    service = MetricsService(gcs_client)
    
    # Test get_model_metrics
    test_model = "llama-3-8b"
    print(f"\n6. Testing get_model_metrics for: {test_model}")
    metrics_path = config.get_metrics_path(test_model)
    print(f"   Expected path: {metrics_path}")
    metrics = service.get_model_metrics(test_model)
    if metrics:
        print(f"   ✅ Successfully retrieved metrics!")
        print(f"   Keys: {list(metrics.keys())}")
    else:
        print(f"   ❌ Failed to retrieve metrics")
        print(f"   This is the issue - file exists but can't be read")
    
    # Test list_available_models
    print(f"\n7. Testing list_available_models...")
    models = service.list_available_models()
    print(f"   Found {len(models)} model(s)")
    for model in models:
        print(f"   - {model['name']}: metrics={model.get('has_metrics')}, bias={model.get('has_bias_report')}")

if __name__ == "__main__":
    test_file_access()

