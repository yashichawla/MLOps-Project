"""Model selector component for the dashboard."""
import streamlit as st
from typing import List, Dict, Optional
from datetime import datetime


def render_model_selector(api_client, auto_refresh: bool = False) -> Optional[str]:
    """Render model selector widget in sidebar.
    
    Args:
        api_client: MetricsAPIClient instance.
        auto_refresh: Whether to auto-refresh model list.
    
    Returns:
        Selected model name, or None if no selection.
    """
    st.sidebar.header("Model Selection")
    
    # Fetch available models
    models_response = api_client.list_models()
    
    if not models_response:
        st.sidebar.error("Failed to load models. Check API connection.")
        st.sidebar.caption(f"API URL: {api_client.base_url}")
        return None
    
    if "models" not in models_response:
        st.sidebar.error("Invalid response from API. Expected 'models' key.")
        st.sidebar.caption(f"Response keys: {list(models_response.keys()) if isinstance(models_response, dict) else 'Not a dict'}")
        return None
    
    models = models_response["models"]
    
    if not models:
        st.sidebar.warning("No models available.")
        return None
    
    # Create model options with metadata
    model_options = ["All Models"] + [model["name"] for model in models]
    model_display = {}
    
    for model in models:
        name = model["name"]
        has_metrics = "✓" if model.get("has_metrics") else "✗"
        has_bias = "✓" if model.get("has_bias_report") else "✗"
        last_updated = model.get("last_updated")
        
        if last_updated:
            try:
                # Parse ISO format timestamp
                dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                date_str = dt.strftime("%Y-%m-%d")
            except:
                date_str = "Unknown"
        else:
            date_str = "Unknown"
        
        model_display[name] = f"{name} ({has_metrics} metrics, {has_bias} bias, updated: {date_str})"
    
    # Display model selector
    selected = st.sidebar.selectbox(
        "Select Model",
        options=model_options,
        format_func=lambda x: model_display.get(x, x) if x != "All Models" else x
    )
    
    # Show model details if a specific model is selected
    if selected and selected != "All Models":
        selected_model = next((m for m in models if m["name"] == selected), None)
        if selected_model:
            with st.sidebar.expander("Model Details"):
                st.write(f"**Name:** {selected_model['name']}")
                st.write(f"**Has Metrics:** {selected_model.get('has_metrics', False)}")
                st.write(f"**Has Bias Report:** {selected_model.get('has_bias_report', False)}")
                if selected_model.get('last_updated'):
                    st.write(f"**Last Updated:** {selected_model['last_updated']}")
    
    return selected if selected != "All Models" else None

