"""Summary metric cards component."""
import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Optional

# Add dashboard directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


def get_asr_color(asr: Optional[float]) -> str:
    """Get color for ASR value based on thresholds.
    
    Args:
        asr: Attack Success Rate value.
    
    Returns:
        Color code.
    """
    if asr is None:
        return config.CHART_COLORS["info"]
    if asr >= config.ASR_CRITICAL_THRESHOLD:
        return config.CHART_COLORS["biased"]
    elif asr >= config.ASR_WARNING_THRESHOLD:
        return config.CHART_COLORS["warning"]
    else:
        return config.CHART_COLORS["success"]


def get_over_refusal_color(rate: Optional[float]) -> str:
    """Get color for over-refusal rate.
    
    Args:
        rate: Over-refusal rate value.
    
    Returns:
        Color code.
    """
    if rate is None:
        return config.CHART_COLORS["info"]
    if rate >= config.OVER_REFUSAL_WARNING_THRESHOLD:
        return config.CHART_COLORS["warning"]
    else:
        return config.CHART_COLORS["success"]


def render_summary_cards(summary_data: Dict) -> None:
    """Render summary metric cards.
    
    Args:
        summary_data: Model summary data from API.
    """
    if not summary_data:
        st.error("No summary data available.")
        return
    
    summary = summary_data.get("summary", {})
    
    # Create columns for cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Total Prompts
    with col1:
        total_prompts = summary.get("total_prompts", 0)
        st.metric(
            label="Total Prompts",
            value=total_prompts
        )
    
    # Global ASR
    with col2:
        global_asr = summary.get("global_asr")
        asr_display = f"{global_asr:.1%}" if global_asr is not None else "N/A"
        st.metric(
            label="Global ASR",
            value=asr_display
        )
        if global_asr is not None:
            color = get_asr_color(global_asr)
            st.markdown(f"<div style='width: 100%; height: 4px; background-color: {color}; border-radius: 2px;'></div>", 
                       unsafe_allow_html=True)
    
    # Over-Refusal Rate
    with col3:
        over_refusal_rate = summary.get("over_refusal_rate")
        refusal_display = f"{over_refusal_rate:.1%}" if over_refusal_rate is not None else "N/A"
        st.metric(
            label="Over-Refusal Rate",
            value=refusal_display
        )
        if over_refusal_rate is not None:
            color = get_over_refusal_color(over_refusal_rate)
            st.markdown(f"<div style='width: 100%; height: 4px; background-color: {color}; border-radius: 2px;'></div>", 
                       unsafe_allow_html=True)
    
    # Biased Categories Count
    with col4:
        biased_count = summary.get("biased_categories_count", 0)
        biased_sizes = summary.get("biased_sizes_count", 0)
        total_biased = biased_count + biased_sizes
        
        st.metric(
            label="Biased Slices",
            value=total_biased,
            delta=f"{biased_count} categories, {biased_sizes} sizes" if total_biased > 0 else None,
            delta_color="inverse" if total_biased > 0 else "normal"
        )
        
        if total_biased > 0:
            st.warning(f"⚠️ {biased_count} biased categories, {biased_sizes} biased size labels detected")

