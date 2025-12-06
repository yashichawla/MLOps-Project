"""Configuration management for the Streamlit dashboard."""
import os
from typing import List


class DashboardConfig:
    """Dashboard configuration loaded from environment variables or defaults."""
    
    # API Configuration
    API_BASE_URL: str = os.getenv(
        "METRICS_API_URL",
        os.getenv("STREAMLIT_SECRETS_METRICS_API_URL", "http://localhost:8080")
    )
    
    # Refresh Configuration
    AUTO_REFRESH_INTERVAL: int = int(os.getenv("AUTO_REFRESH_INTERVAL", "30"))  # seconds
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))  # seconds
    
    # Chart Configuration
    CHART_COLORS: dict = {
        "normal": "#1f77b4",
        "biased": "#ef4444",
        "success": "#10b981",
        "warning": "#f59e0b",
        "info": "#3b82f6"
    }
    
    # Thresholds for color coding
    ASR_WARNING_THRESHOLD: float = 0.7
    ASR_CRITICAL_THRESHOLD: float = 0.9
    OVER_REFUSAL_WARNING_THRESHOLD: float = 0.2
    
    # API Timeout
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "10"))  # seconds


# Global config instance
config = DashboardConfig()

