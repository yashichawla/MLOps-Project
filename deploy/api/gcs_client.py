"""GCS client for reading metrics files from Google Cloud Storage."""
import json
import logging
from typing import Dict, List, Optional
from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden

from .config import config

logger = logging.getLogger(__name__)


class GCSClient:
    """Client for reading files from GCS bucket with local file fallback."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        """Initialize GCS client.
        
        Args:
            bucket_name: GCS bucket name. If None, uses config.GCS_BUCKET.
        """
        self.bucket_name = bucket_name or config.GCS_BUCKET
        try:
            self.client = storage.Client(project=config.GCP_PROJECT_ID)
            self.bucket = self.client.bucket(self.bucket_name)
            logger.info(f"Initialized GCS client for bucket: {self.bucket_name}, project: {config.GCP_PROJECT_ID}")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    def read_json(self, blob_path: str) -> Optional[Dict]:
        """Read JSON file from GCS.
        
        Args:
            blob_path: Path to the blob in GCS (e.g., 'data/metrics/file.json').
        
        Returns:
            Parsed JSON as dictionary, or None if file not found or error.
        """
        full_path = f"gs://{self.bucket_name}/{blob_path}"
        try:
            blob = self.bucket.blob(blob_path)
            
            # Try to reload blob metadata first to check existence and permissions
            try:
                blob.reload()
                logger.debug(f"Blob exists and accessible: {full_path} (size: {blob.size} bytes)")
            except NotFound:
                logger.warning(f"Blob not found: {full_path}")
                return None
            except Forbidden:
                logger.error(f"Permission denied accessing blob: {full_path}")
                logger.error("Check service account has storage.objects.get permission")
                return None
            
            logger.debug(f"Reading blob: {full_path}")
            content = blob.download_as_text()
            data = json.loads(content)
            logger.debug(f"Successfully read and parsed JSON from {full_path}")
            return data
        except NotFound:
            logger.warning(f"Blob not found (NotFound exception): {full_path}")
            return None
        except Forbidden:
            logger.error(f"Permission denied accessing blob: {full_path}")
            logger.error("Check service account has storage.objects.get permission")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {full_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading blob {full_path}: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def list_blobs(self, prefix: str) -> List[str]:
        """List all blobs with given prefix from GCS.
        
        Args:
            prefix: Prefix to filter blobs (e.g., 'data/metrics/').
        
        Returns:
            List of blob paths.
        """
        try:
            logger.debug(f"Listing blobs with prefix: gs://{self.bucket_name}/{prefix}")
            blobs = list(self.client.list_blobs(self.bucket_name, prefix=prefix))
            logger.debug(f"Found {len(blobs)} blob(s) with prefix {prefix}")
            return [blob.name for blob in blobs]
        except Forbidden:
            logger.error(f"Permission denied listing blobs with prefix: gs://{self.bucket_name}/{prefix}")
            logger.error("Check service account has storage.objects.list permission")
            return []
        except Exception as e:
            logger.error(f"Error listing blobs with prefix gs://{self.bucket_name}/{prefix}: {type(e).__name__}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
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
        """Check if blob exists in GCS.
        
        Args:
            blob_path: Path to the blob in GCS.
        
        Returns:
            True if blob exists, False otherwise.
        """
        try:
            blob = self.bucket.blob(blob_path)
            # Use reload() instead of exists() for more reliable check
            try:
                blob.reload()
                return True
            except NotFound:
                return False
            except Forbidden:
                logger.warning(f"Permission denied checking blob: gs://{self.bucket_name}/{blob_path}")
                # If we can't check due to permissions, assume it doesn't exist
                return False
        except Exception as e:
            logger.error(f"Error checking blob existence for {blob_path}: {type(e).__name__}: {e}")
            return False

