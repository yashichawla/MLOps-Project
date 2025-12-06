"""Configuration management for the Metrics API service."""
import os
from pathlib import Path
from typing import List


class Config:
    """Application configuration loaded from environment variables."""
    
    # GCS Configuration
    GCS_BUCKET: str = os.getenv("GCS_BUCKET", "mlops-project-dvc")
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "break-the-bot")
    
    # Server Configuration
    PORT: int = int(os.getenv("PORT", "8080"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Local Development - Path to project root for local file fallback
    LOCAL_DATA_ROOT: str = os.getenv("LOCAL_DATA_ROOT", "")
    
    # GCS Paths
    METRICS_PATH_PREFIX: str = "data/metrics"
    BIAS_PATH_PREFIX: str = "data/bias"
    
    @classmethod
    def get_metrics_path(cls, model_name: str) -> str:
        """Get GCS path for model metrics file."""
        return f"{cls.METRICS_PATH_PREFIX}/additional_metrics_{model_name}.json"
    
    @classmethod
    def get_bias_report_path(cls, model_name: str) -> str:
        """Get GCS path for model bias report."""
        return f"{cls.BIAS_PATH_PREFIX}/{model_name}/bias_report.json"


# Global config instance
config = Config()

