"""Service layer for metrics aggregation and business logic."""
import logging
import re
from typing import Dict, List, Optional
from datetime import datetime

from .gcs_client import GCSClient
from .config import config

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for fetching and aggregating metrics from GCS."""
    
    def __init__(self, gcs_client: Optional[GCSClient] = None):
        """Initialize metrics service.
        
        Args:
            gcs_client: GCSClient instance. If None, creates a new one.
        """
        self.gcs_client = gcs_client or GCSClient()
    
    def get_model_metrics(self, model_name: str) -> Optional[Dict]:
        """Get metrics for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'llama-3-8b').
        
        Returns:
            Dictionary with model metrics, or None if not found.
        """
        blob_path = config.get_metrics_path(model_name)
        logger.info(f"Fetching metrics for {model_name} from {blob_path}")
        return self.gcs_client.read_json(blob_path)
    
    def get_bias_report(self, model_name: str) -> Optional[Dict]:
        """Get bias report for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'llama-3-8b').
        
        Returns:
            Dictionary with bias report, or None if not found.
        """
        blob_path = config.get_bias_report_path(model_name)
        logger.info(f"Fetching bias report for {model_name} from {blob_path}")
        bias_data = self.gcs_client.read_json(blob_path)
        if bias_data:
            bias_data["model"] = model_name
        return bias_data
    
    def get_model_summary(self, model_name: str) -> Optional[Dict]:
        """Get combined summary for a model (metrics + bias).
        
        Args:
            model_name: Name of the model.
        
        Returns:
            Dictionary with combined summary, or None if model not found.
        """
        metrics = self.get_model_metrics(model_name)
        bias = self.get_bias_report(model_name)
        
        if not metrics:
            logger.warning(f"Metrics not found for model {model_name}")
            return None
        
        # Build summary
        coverage = metrics.get("coverage_metrics", {})
        over_refusal = metrics.get("over_refusal_metrics", {})
        
        # Get bias data
        global_asr = None
        biased_categories_count = 0
        biased_sizes_count = 0
        
        if bias:
            global_metrics = bias.get("global", {})
            global_asr = global_metrics.get("asr")
            
            biased_slices = bias.get("biased_slices", {})
            biased_categories_count = len(biased_slices.get("category", []))
            biased_sizes_count = len(biased_slices.get("size_label", []))
        
        summary = {
            "model": model_name,
            "summary": {
                "total_prompts": coverage.get("total_prompts", 0),
                "global_asr": global_asr,
                "over_refusal_rate": over_refusal.get("over_refusal_rate", 0.0),
                "biased_categories_count": biased_categories_count,
                "biased_sizes_count": biased_sizes_count
            },
            "coverage": coverage,
            "bias": bias.get("global", {}) if bias else {}
        }
        
        return summary
    
    def list_available_models(self) -> List[Dict]:
        """List all available models with metadata.
        
        Returns:
            List of dictionaries with model information.
        """
        models = []
        
        # Scan for metrics files in the additional/ subdirectory
        # Updated to match new location: data/metrics/additional/
        metrics_prefix = f"{config.METRICS_PATH_PREFIX}/additional/"
        blobs = self.gcs_client.list_blobs(metrics_prefix)
        
        # Extract model names from blob paths
        pattern = re.compile(r"additional_metrics_(.+)\.json$")
        
        for blob_path in blobs:
            match = pattern.search(blob_path)
            if match:
                model_name = match.group(1)
                
                # Check if metrics and bias report exist
                metrics_path = config.get_metrics_path(model_name)
                bias_path = config.get_bias_report_path(model_name)
                
                has_metrics = self.gcs_client.blob_exists(metrics_path)
                has_bias = self.gcs_client.blob_exists(bias_path)
                
                # Get last updated timestamp from metrics file
                metadata = self.gcs_client.get_blob_metadata(metrics_path)
                last_updated = metadata.get("updated") if metadata else None
                
                models.append({
                    "name": model_name,
                    "has_metrics": has_metrics,
                    "has_bias_report": has_bias,
                    "last_updated": last_updated
                })
        
        logger.info(f"Found {len(models)} available models")
        return models
    
    def get_all_models_summary(self) -> List[Dict]:
        """Get summary for all available models.
        
        Returns:
            List of model summaries.
        """
        models = self.list_available_models()
        summaries = []
        
        for model_info in models:
            model_name = model_info["name"]
            metrics = self.get_model_metrics(model_name)
            bias = self.get_bias_report(model_name)
            
            summary_item = {
                "model_name": model_name,
                "coverage": metrics.get("coverage_metrics") if metrics else None,
                "global_asr": None,
                "over_refusal_rate": None
            }
            
            if metrics:
                over_refusal = metrics.get("over_refusal_metrics", {})
                summary_item["over_refusal_rate"] = over_refusal.get("over_refusal_rate")
            
            if bias:
                global_metrics = bias.get("global", {})
                summary_item["global_asr"] = global_metrics.get("asr")
            
            summaries.append(summary_item)
        
        return summaries

