"""API client for interacting with the Metrics API service."""
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime

from config import config

logger = logging.getLogger(__name__)


class MetricsAPIClient:
    """Client for fetching data from the Metrics API service."""
    
    def __init__(self, base_url: Optional[str] = None):
        """Initialize API client.
        
        Args:
            base_url: Base URL of the Metrics API. If None, uses config.API_BASE_URL.
        """
        self.base_url = (base_url or config.API_BASE_URL).rstrip('/')
        self.timeout = config.API_TIMEOUT
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make HTTP GET request to API endpoint.
        
        Args:
            endpoint: API endpoint path (e.g., '/health').
        
        Returns:
            JSON response as dictionary, or None if error.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to API at {url}")
            return None
        except requests.exceptions.Timeout:
            logger.error(f"Request to {url} timed out")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            return None
    
    def get_health(self) -> Optional[Dict]:
        """Check API health and GCS connectivity.
        
        Returns:
            Health check response, or None if error.
        """
        return self._make_request("/health")
    
    def get_all_models(self) -> Optional[Dict]:
        """Fetch summary for all available models.
        
        Returns:
            All models summary response, or None if error.
        """
        return self._make_request("/metrics/all")
    
    def get_model_metrics(self, model_name: str) -> Optional[Dict]:
        """Get full metrics for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'llama-3-8b').
        
        Returns:
            Model metrics response, or None if error.
        """
        return self._make_request(f"/metrics/{model_name}")
    
    def get_bias_report(self, model_name: str) -> Optional[Dict]:
        """Get bias report for a specific model.
        
        Args:
            model_name: Name of the model.
        
        Returns:
            Bias report response, or None if error.
        """
        return self._make_request(f"/metrics/{model_name}/bias")
    
    def get_model_summary(self, model_name: str) -> Optional[Dict]:
        """Get combined summary (metrics + bias) for a specific model.
        
        Args:
            model_name: Name of the model.
        
        Returns:
            Model summary response, or None if error.
        """
        return self._make_request(f"/metrics/{model_name}/summary")
    
    def list_models(self) -> Optional[Dict]:
        """List all evaluated models with metadata.
        
        Returns:
            Models list response, or None if error.
        """
        return self._make_request("/dashboard/models")
    
    def is_connected(self) -> bool:
        """Check if API is reachable.
        
        Returns:
            True if API is reachable, False otherwise.
        """
        health = self.get_health()
        return health is not None and health.get("status") == "healthy"

