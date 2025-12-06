"""Metrics endpoints."""
import logging
from fastapi import APIRouter, HTTPException
from datetime import datetime

from ..models import (
    ModelMetricsResponse,
    BiasReportResponse,
    ModelSummaryResponse,
    AllMetricsResponse,
    ModelSummaryItem
)
from ..metrics_service import MetricsService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/metrics/all", response_model=AllMetricsResponse)
async def get_all_metrics():
    """Get summary for all available models."""
    try:
        service = MetricsService()
        summaries = service.get_all_models_summary()
        
        models = [
            ModelSummaryItem(
                model_name=item["model_name"],
                coverage=item["coverage"],
                global_asr=item["global_asr"],
                over_refusal_rate=item["over_refusal_rate"]
            )
            for item in summaries
        ]
        
        return AllMetricsResponse(
            models=models,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error fetching all metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch all metrics: {str(e)}")


@router.get("/metrics/{model_name}", response_model=ModelMetricsResponse)
async def get_model_metrics(model_name: str):
    """Get full metrics for a specific model."""
    try:
        service = MetricsService()
        metrics = service.get_model_metrics(model_name)
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"Metrics not found for model: {model_name}"
            )
        
        return ModelMetricsResponse(
            model=metrics.get("model", model_name),
            coverage_metrics=metrics.get("coverage_metrics", {}),
            over_refusal_metrics=metrics.get("over_refusal_metrics", {}),
            timestamp=datetime.utcnow()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch metrics for {model_name}: {str(e)}"
        )


@router.get("/metrics/{model_name}/bias", response_model=BiasReportResponse)
async def get_bias_report(model_name: str):
    """Get bias report for a specific model."""
    try:
        service = MetricsService()
        bias_data = service.get_bias_report(model_name)
        
        if not bias_data:
            raise HTTPException(
                status_code=404,
                detail=f"Bias report not found for model: {model_name}"
            )
        
        return BiasReportResponse(**bias_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching bias report for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch bias report for {model_name}: {str(e)}"
        )


@router.get("/metrics/{model_name}/summary", response_model=ModelSummaryResponse)
async def get_model_summary(model_name: str):
    """Get combined summary (metrics + bias) for a specific model."""
    try:
        service = MetricsService()
        summary = service.get_model_summary(model_name)
        
        if not summary:
            raise HTTPException(
                status_code=404,
                detail=f"Summary not found for model: {model_name}"
            )
        
        return ModelSummaryResponse(
            model=summary["model"],
            summary=summary["summary"],
            coverage=summary["coverage"],
            bias=summary["bias"],
            timestamp=datetime.utcnow()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching summary for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch summary for {model_name}: {str(e)}"
        )

