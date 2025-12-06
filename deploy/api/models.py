"""Pydantic models for API request/response schemas."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class CoverageMetrics(BaseModel):
    """Coverage metrics model."""
    total_prompts: int
    num_categories: Optional[int] = None
    prompts_per_category: Optional[Dict[str, int]] = None
    prompts_per_size: Optional[Dict[str, int]] = None
    unique_prompt_ids: Optional[int] = None


class OverRefusalMetrics(BaseModel):
    """Over-refusal metrics model."""
    threshold: float
    safe_count: int
    over_refusal_count: int
    over_refusal_rate: float


class ModelMetricsResponse(BaseModel):
    """Response model for model metrics endpoint."""
    model: str
    coverage_metrics: CoverageMetrics
    over_refusal_metrics: OverRefusalMetrics
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class GlobalBiasMetrics(BaseModel):
    """Global bias metrics."""
    asr: float
    mean_refusal: float
    count: int


class BiasThresholds(BaseModel):
    """Bias detection thresholds."""
    relative_asr_deviation: float
    min_count: int


class CategorySlice(BaseModel):
    """Category slice metrics."""
    category: str
    asr: float
    mean_refusal: float
    count: int
    asr_deviation: float
    biased_flag: bool


class SizeSlice(BaseModel):
    """Size label slice metrics."""
    size_label: str
    asr: float
    mean_refusal: float
    count: int
    asr_deviation: float
    biased_flag: bool


class BiasedSlices(BaseModel):
    """Biased slices summary."""
    category: List[CategorySlice]
    size_label: List[SizeSlice]


class BiasReportResponse(BaseModel):
    """Response model for bias report endpoint."""
    model: Optional[str] = None
    global_metrics: GlobalBiasMetrics = Field(..., alias="global")
    thresholds: BiasThresholds
    by_category: List[CategorySlice]
    by_size_label: List[SizeSlice]
    biased_slices: BiasedSlices
    notes: Optional[List[str]] = None
    
    class Config:
        populate_by_name = True


class ModelSummary(BaseModel):
    """Summary statistics for a model."""
    total_prompts: int
    global_asr: float
    over_refusal_rate: float
    biased_categories_count: int
    biased_sizes_count: int


class ModelSummaryResponse(BaseModel):
    """Response model for model summary endpoint."""
    model: str
    summary: ModelSummary
    coverage: CoverageMetrics
    bias: GlobalBiasMetrics
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class ModelInfo(BaseModel):
    """Model information for listing."""
    name: str
    has_metrics: bool
    has_bias_report: bool
    last_updated: Optional[datetime] = None


class ModelsListResponse(BaseModel):
    """Response model for models list endpoint."""
    models: List[ModelInfo]
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class ModelSummaryItem(BaseModel):
    """Summary item for all models endpoint."""
    model_name: str
    coverage: Optional[Dict[str, Any]] = None
    global_asr: Optional[float] = None
    over_refusal_rate: Optional[float] = None


class AllMetricsResponse(BaseModel):
    """Response model for all metrics endpoint."""
    models: List[ModelSummaryItem]
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    gcs_connected: bool
    bucket: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)

