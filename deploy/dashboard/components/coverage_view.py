"""Coverage metrics visualization component."""
import streamlit as st
import pandas as pd
from typing import Dict


def render_coverage_metrics(coverage_data: Dict) -> None:
    """Render coverage metrics with detailed breakdowns.
    
    Args:
        coverage_data: Coverage metrics data from API.
    """
    if not coverage_data:
        st.warning("No coverage data available.")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Prompts", coverage_data.get("total_prompts", 0))
    
    with col2:
        st.metric("Categories", coverage_data.get("num_categories", 0))
    
    with col3:
        st.metric("Unique Prompt IDs", coverage_data.get("unique_prompt_ids", 0))
    
    # Prompts per category
    prompts_per_category = coverage_data.get("prompts_per_category")
    if prompts_per_category:
        st.subheader("Prompts per Category")
        df_category = pd.DataFrame([
            {"Category": cat, "Count": count}
            for cat, count in prompts_per_category.items()
        ])
        df_category = df_category.sort_values("Count", ascending=False)
        st.dataframe(df_category, use_container_width=True, hide_index=True)
    
    # Prompts per size label
    prompts_per_size = coverage_data.get("prompts_per_size")
    if prompts_per_size:
        with st.expander("Prompts per Size Label"):
            df_size = pd.DataFrame([
                {"Size Label": size, "Count": count}
                for size, count in prompts_per_size.items()
            ])
            df_size = df_size.sort_values("Count", ascending=False)
            st.dataframe(df_size, use_container_width=True, hide_index=True)

