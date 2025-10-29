"""Pytest configuration and fixtures for test isolation.

This module ensures tests don't modify the real baseline files by redirecting
all metrics/baseline writes to temporary directories.
"""
import pytest
from pathlib import Path
from scripts import ge_runner


@pytest.fixture(autouse=True)
def use_temp_metrics_dir(tmp_path, monkeypatch):
    """Redirect all metrics/baseline writes to temporary directory.
    
    This fixture automatically runs for every test (autouse=True) and ensures
    that baseline files are created in temporary directories instead of the
    actual project directory. This prevents tests from modifying the real
    baseline files.
    
    Args:
        tmp_path: Pytest fixture providing a temporary directory
        monkeypatch: Pytest fixture for patching module attributes
    """
    temp_metrics = tmp_path / "data" / "metrics"
    temp_metrics.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch the baseline paths to use temporary directory
    monkeypatch.setattr(ge_runner, "METRICS_DIR", temp_metrics)
    monkeypatch.setattr(
        ge_runner, 
        "SCHEMA_BASELINE", 
        temp_metrics / "schema" / "baseline" / "schema.json"
    )
    monkeypatch.setattr(
        ge_runner,
        "STATS_BASELINE",
        temp_metrics / "stats" / "baseline" / "stats.json"
    )
    
    yield temp_metrics

