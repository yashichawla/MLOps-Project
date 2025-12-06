"""Chart rendering functions for the dashboard."""
import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional

# Add dashboard directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


def render_asr_by_category_chart(bias_data: Dict) -> None:
    """Render ASR by category bar chart with bias highlighting.
    
    Args:
        bias_data: Bias report data from API.
    """
    if not bias_data or "by_category" not in bias_data:
        st.warning("No category data available for chart.")
        return
    
    categories = bias_data["by_category"]
    if not categories:
        st.info("No category data to display.")
        return
    
    # Extract data
    category_names = [cat["category"] for cat in categories]
    asr_values = [cat["asr"] for cat in categories]
    biased_flags = [cat.get("biased_flag", False) for cat in categories]
    
    # Create colors based on bias flags
    colors = [config.CHART_COLORS["biased"] if biased else config.CHART_COLORS["normal"] 
              for biased in biased_flags]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=category_names,
            y=asr_values,
            marker_color=colors,
            text=[f"{asr:.1%}" for asr in asr_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>ASR: %{y:.3f}<br>Biased: %{customdata}<extra></extra>',
            customdata=["Yes" if b else "No" for b in biased_flags]
        )
    ])
    
    fig.update_layout(
        title="Attack Success Rate (ASR) by Category",
        xaxis_title="Category",
        yaxis_title="ASR",
        yaxis=dict(range=[0, 1.1]),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add legend
    st.caption("🔴 Red bars indicate biased categories")


def render_coverage_pie_chart(coverage_data: Dict) -> None:
    """Render coverage distribution pie chart.
    
    Args:
        coverage_data: Coverage metrics data from API.
    """
    if not coverage_data or "prompts_per_category" not in coverage_data:
        st.warning("No coverage data available for chart.")
        return
    
    prompts_per_category = coverage_data.get("prompts_per_category", {})
    if not prompts_per_category:
        st.info("No category distribution data to display.")
        return
    
    # Extract data
    categories = list(prompts_per_category.keys())
    values = list(prompts_per_category.values())
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=values,
        hole=0.3,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Coverage Distribution by Category",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison_chart(all_models_data: Dict) -> None:
    """Render model comparison chart (ASR and over-refusal).
    
    Args:
        all_models_data: All models summary data from API.
    """
    if not all_models_data or "models" not in all_models_data:
        st.warning("No model comparison data available.")
        return
    
    models = all_models_data["models"]
    if not models:
        st.info("No models to compare.")
        return
    
    # Extract data
    model_names = [m["model_name"] for m in models]
    asr_values = [m.get("global_asr", 0) if m.get("global_asr") is not None else 0 for m in models]
    over_refusal_values = [m.get("over_refusal_rate", 0) if m.get("over_refusal_rate") is not None else 0 for m in models]
    
    # Create grouped bar chart
    fig = go.Figure(data=[
        go.Bar(
            name='ASR',
            x=model_names,
            y=asr_values,
            marker_color=config.CHART_COLORS["normal"],
            text=[f"{asr:.1%}" for asr in asr_values],
            textposition='outside'
        ),
        go.Bar(
            name='Over-Refusal Rate',
            x=model_names,
            y=over_refusal_values,
            marker_color=config.CHART_COLORS["warning"],
            text=[f"{rate:.1%}" for rate in over_refusal_values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Model Comparison: ASR vs Over-Refusal Rate",
        xaxis_title="Model",
        yaxis_title="Rate",
        yaxis=dict(range=[0, max(max(asr_values), max(over_refusal_values), 0.1) * 1.2]),
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

