"""Main Streamlit dashboard application."""
import streamlit as st
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add dashboard directory to path for imports
dashboard_dir = Path(__file__).parent
sys.path.insert(0, str(dashboard_dir))

from api_client import MetricsAPIClient
from config import config
from components.model_selector import render_model_selector
from components.summary_cards import render_summary_cards
from components.charts import (
    render_asr_by_category_chart,
    render_coverage_pie_chart,
    render_model_comparison_chart
)
from components.bias_table import render_bias_table
from components.coverage_view import render_coverage_metrics

# Page configuration
st.set_page_config(
    page_title="Break-The-Bot Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "api_client" not in st.session_state:
    st.session_state.api_client = MetricsAPIClient()
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


@st.cache_data(ttl=config.CACHE_TTL)
def fetch_all_models(_api_client: MetricsAPIClient):
    """Fetch all models summary with caching."""
    return _api_client.get_all_models()


@st.cache_data(ttl=config.CACHE_TTL)
def fetch_model_summary(_api_client: MetricsAPIClient, model_name: str):
    """Fetch model summary with caching."""
    return _api_client.get_model_summary(model_name)


@st.cache_data(ttl=config.CACHE_TTL)
def fetch_bias_report(_api_client: MetricsAPIClient, model_name: str):
    """Fetch bias report with caching."""
    return _api_client.get_bias_report(model_name)


@st.cache_data(ttl=config.CACHE_TTL)
def fetch_model_metrics(_api_client: MetricsAPIClient, model_name: str):
    """Fetch model metrics with caching."""
    return _api_client.get_model_metrics(model_name)


def render_sidebar():
    """Render sidebar with controls."""
    st.sidebar.title("📊 Break-The-Bot Dashboard")
    
    # API Configuration
    with st.sidebar.expander("⚙️ API Configuration"):
        api_url = st.text_input(
            "API URL",
            value=config.API_BASE_URL,
            help="Base URL of the Metrics API service"
        )
        if api_url != config.API_BASE_URL:
            st.session_state.api_client = MetricsAPIClient(base_url=api_url)
            st.cache_data.clear()
    
    # Health check
    health = st.session_state.api_client.get_health()
    if health:
        if health.get("status") == "healthy":
            st.sidebar.success("✅ API Connected")
            st.sidebar.caption(f"API: {st.session_state.api_client.base_url}")
        else:
            st.sidebar.warning("⚠️ API Degraded")
            st.sidebar.caption(f"API: {st.session_state.api_client.base_url}")
    else:
        st.sidebar.error("❌ API Not Connected")
        st.sidebar.caption(f"API URL: {st.session_state.api_client.base_url}")
        st.sidebar.caption("Check if API is running on port 8080")
    
    st.sidebar.divider()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh",
        value=st.session_state.auto_refresh,
        help=f"Automatically refresh every {config.AUTO_REFRESH_INTERVAL} seconds"
    )
    st.session_state.auto_refresh = auto_refresh
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # Last refresh time
    if st.session_state.last_refresh:
        st.sidebar.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    st.sidebar.divider()
    
    # Model selector
    selected_model = render_model_selector(st.session_state.api_client)
    
    return selected_model


def render_overview_page(api_client: MetricsAPIClient):
    """Render overview page with all models comparison."""
    st.header("📈 Model Overview")
    
    # Fetch all models data
    all_models_data = fetch_all_models(api_client)
    
    if not all_models_data:
        st.error("Failed to fetch models data. Please check API connection.")
        return
    
    models = all_models_data.get("models", [])
    if not models:
        st.info("No models available.")
        return
    
    # Model comparison chart
    st.subheader("Model Comparison")
    render_model_comparison_chart(all_models_data)
    
    # Models table
    st.subheader("Models Summary")
    
    # Create summary table
    import pandas as pd
    table_data = []
    for model in models:
        table_data.append({
            "Model": model["model_name"],
            "ASR": f"{model.get('global_asr', 0):.1%}" if model.get("global_asr") is not None else "N/A",
            "Over-Refusal": f"{model.get('over_refusal_rate', 0):.1%}" if model.get("over_refusal_rate") is not None else "N/A",
            "Total Prompts": model.get("coverage", {}).get("total_prompts", "N/A") if model.get("coverage") else "N/A"
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, width='stretch', hide_index=True)


def render_model_detail_page(api_client: MetricsAPIClient, model_name: str):
    """Render detailed view for selected model."""
    st.header(f"🔍 Model Details: {model_name}")
    
    # Fetch model data
    with st.spinner("Loading model data..."):
        summary_data = fetch_model_summary(api_client, model_name)
        bias_data = fetch_bias_report(api_client, model_name)
        metrics_data = fetch_model_metrics(api_client, model_name)
    
    if not summary_data:
        st.error(f"Failed to load data for model: {model_name}")
        return
    
    # Summary cards
    render_summary_cards(summary_data)
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🎯 Bias Detection", "📈 Coverage", "📋 Full Metrics"])
    
    with tab1:
        st.subheader("Summary Metrics")
        if metrics_data:
            coverage = metrics_data.get("coverage_metrics", {})
            over_refusal = metrics_data.get("over_refusal_metrics", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Prompts", coverage.get("total_prompts", 0))
                st.metric("Categories", coverage.get("num_categories", 0))
            
            with col2:
                st.metric("Over-Refusal Threshold", f"{over_refusal.get('threshold', 0):.2f}")
                st.metric("Safe Count", over_refusal.get("safe_count", 0))
                st.metric("Over-Refusal Count", over_refusal.get("over_refusal_count", 0))
    
    with tab2:
        st.subheader("Bias Detection Analysis")
        if bias_data:
            # Global metrics
            global_metrics = bias_data.get("global", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Global ASR", f"{global_metrics.get('asr', 0):.3f}")
            with col2:
                st.metric("Mean Refusal", f"{global_metrics.get('mean_refusal', 0):.3f}")
            with col3:
                st.metric("Sample Count", global_metrics.get("count", 0))
            
            st.divider()
            
            # ASR by category chart
            render_asr_by_category_chart(bias_data)
            
            st.divider()
            
            # Bias table
            render_bias_table(bias_data)
        else:
            st.warning("Bias report not available for this model.")
    
    with tab3:
        st.subheader("Coverage Metrics")
        if metrics_data:
            coverage = metrics_data.get("coverage_metrics", {})
            render_coverage_metrics(coverage)
            
            st.divider()
            
            # Coverage pie chart
            render_coverage_pie_chart(coverage)
        else:
            st.warning("Coverage metrics not available.")
    
    with tab4:
        st.subheader("Full Metrics Data")
        if metrics_data:
            st.json(metrics_data)
        if bias_data:
            st.subheader("Bias Report Data")
            st.json(bias_data)


def main():
    """Main application entry point."""
    # Title
    st.title("📊 Break-The-Bot Evaluation Metrics Dashboard")
    st.caption("Real-time metrics visualization for LLM safety evaluation")
    
    # Render sidebar and get selected model
    selected_model = render_sidebar()
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(config.AUTO_REFRESH_INTERVAL)
        st.cache_data.clear()
        st.rerun()
    
    # Main content area
    if selected_model:
        # Show model detail page
        render_model_detail_page(st.session_state.api_client, selected_model)
    else:
        # Show overview page
        render_overview_page(st.session_state.api_client)
    
    # Footer
    st.divider()
    st.caption(f"API: {st.session_state.api_client.base_url} | "
              f"Cache TTL: {config.CACHE_TTL}s | "
              f"Auto-refresh: {'ON' if st.session_state.auto_refresh else 'OFF'}")


if __name__ == "__main__":
    main()

