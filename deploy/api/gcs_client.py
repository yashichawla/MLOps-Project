"""GCS client for reading metrics files from Google Cloud Storage."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from google.cloud import storage
from google.cloud.exceptions import NotFound

from .config import config

logger = logging.getLogger(__name__)


class GCSClient:
    """Client for reading files from GCS bucket with local file fallback."""
    
    def __init__(self, bucket_name: Optional[str] = None, local_data_root: Optional[str] = None):
        """Initialize GCS client.
        
        Args:
            bucket_name: GCS bucket name. If None, uses config.GCS_BUCKET.
            local_data_root: Path to project root for local file fallback. If None, uses config.LOCAL_DATA_ROOT.
        """
        self.bucket_name = bucket_name or config.GCS_BUCKET
        self.local_data_root = Path(local_data_root) if local_data_root else (Path(config.LOCAL_DATA_ROOT) if config.LOCAL_DATA_ROOT else None)
        self.client = storage.Client(project=config.GCP_PROJECT_ID)
        self.bucket = self.client.bucket(self.bucket_name)
    
    def read_json(self, blob_path: str) -> Optional[Dict]:
        """Read JSON file from GCS or local filesystem (fallback).
        
        Args:
            blob_path: Path to the blob in GCS (e.g., 'data/metrics/file.json').
        
        Returns:
            Parsed JSON as dictionary, or None if file not found or error.
        """
        # Try local file first if local_data_root is configured
        if self.local_data_root:
            local_path = self.local_data_root / blob_path
            if local_path.exists():
                try:
                    logger.info(f"Reading from local file: {local_path}")
                    with open(local_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from {local_path}: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Error reading local file {local_path}: {e}")
                    # Fall through to try GCS
        
        # Try GCS
        try:
            blob = self.bucket.blob(blob_path)
            if not blob.exists():
                logger.warning(f"Blob not found: {blob_path}")
                return None
            
            content = blob.download_as_text()
            return json.loads(content)
        except NotFound:
            logger.warning(f"Blob not found: {blob_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {blob_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading blob {blob_path}: {e}")
            return None
    
    def list_blobs(self, prefix: str) -> List[str]:
        """List all blobs with given prefix (from local filesystem or GCS).
        
        Args:
            prefix: Prefix to filter blobs (e.g., 'data/metrics/').
        
        Returns:
            List of blob paths.
        """
        # Try local filesystem first if local_data_root is configured
        if self.local_data_root:
            local_prefix = self.local_data_root / prefix
            if local_prefix.exists():
                try:
                    files = []
                    for file_path in local_prefix.rglob('*'):
                        if file_path.is_file():
                            # Get relative path from local_data_root
                            rel_path = file_path.relative_to(self.local_data_root)
                            files.append(str(rel_path).replace('\\', '/'))
                    if files:
                        logger.info(f"Found {len(files)} local files with prefix {prefix}")
                        return files
                except Exception as e:
                    logger.warning(f"Error listing local files with prefix {prefix}: {e}")
                    # Fall through to try GCS
        
        # Try GCS
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing blobs with prefix {prefix}: {e}")
            return []
    
    def get_blob_metadata(self, blob_path: str) -> Optional[Dict]:
        """Get blob metadata including last updated timestamp.
        
        Args:
            blob_path: Path to the blob in GCS.
        
        Returns:
            Dictionary with metadata, or None if blob not found.
        """
        try:
            blob = self.bucket.blob(blob_path)
            if not blob.exists():
                return None
            
            blob.reload()
            return {
                "name": blob.name,
                "size": blob.size,
                "updated": blob.updated,
                "content_type": blob.content_type
            }
        except Exception as e:
            logger.error(f"Error getting blob metadata for {blob_path}: {e}")
            return None
    
    def blob_exists(self, blob_path: str) -> bool:
        """Check if blob exists in GCS or local filesystem.
        
        Args:
            blob_path: Path to the blob in GCS.
        
        Returns:
            True if blob exists, False otherwise.
        """
        # Try local file first if local_data_root is configured
        if self.local_data_root:
            local_path = self.local_data_root / blob_path
            if local_path.exists():
                return True
        
        # Try GCS
        try:
            blob = self.bucket.blob(blob_path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking blob existence for {blob_path}: {e}")
            return False

