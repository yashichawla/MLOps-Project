"""Bias detection table component."""
import streamlit as st
import pandas as pd
from typing import Dict, List


def render_bias_table(bias_data: Dict) -> None:
    """Render bias detection results table.
    
    Args:
        bias_data: Bias report data from API.
    """
    if not bias_data or "by_category" not in bias_data:
        st.warning("No bias data available.")
        return
    
    categories = bias_data["by_category"]
    if not categories:
        st.info("No category data to display.")
        return
    
    # Create DataFrame
    df_data = []
    for cat in categories:
        df_data.append({
            "Category": cat["category"],
            "ASR": f"{cat['asr']:.3f}",
            "Deviation": f"{cat.get('asr_deviation', 0):.3f}",
            "Count": cat["count"],
            "Biased": "⚠️ Yes" if cat.get("biased_flag", False) else "✓ No"
        })
    
    df = pd.DataFrame(df_data)
    
    # Style the dataframe
    def highlight_biased(row):
        if row["Biased"] == "⚠️ Yes":
            return ['background-color: #fee2e2'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        df.style.apply(highlight_biased, axis=1),
        width='stretch',
        hide_index=True
    )
    
    # Show biased slices summary
    biased_slices = bias_data.get("biased_slices", {})
    biased_categories = biased_slices.get("category", [])
    
    if biased_categories:
        st.warning(f"⚠️ **{len(biased_categories)} biased categories detected:**")
        for biased_cat in biased_categories:
            st.write(f"- **{biased_cat['category']}**: ASR={biased_cat['asr']:.3f}, "
                    f"Deviation={biased_cat.get('asr_deviation', 0):.3f}, Count={biased_cat['count']}")
    else:
        st.success("✓ No biased categories detected.")

