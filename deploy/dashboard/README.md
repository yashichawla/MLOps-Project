# Streamlit Dashboard Documentation

## Overview

The Streamlit Dashboard provides an interactive web interface to visualize evaluation metrics from the Break-The-Bot project. It connects to the Metrics API service (deployed on Cloud Run) to fetch and display real-time metrics, bias detection results, and model performance comparisons.

## Features

- **Model Overview**: Compare all evaluated models side-by-side
- **Model Details**: Deep dive into individual model metrics
- **Bias Detection**: Visualize bias detection results with highlighted biased categories
- **Coverage Metrics**: View prompt distribution across categories and size labels
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Auto-refresh**: Optional automatic data refresh
- **Caching**: Efficient data caching to reduce API calls

## Installation

### Prerequisites

- Python 3.11+
- Access to Metrics API service (local or Cloud Run)

### Setup

1. Install dependencies:
```bash
cd deploy/dashboard
pip install -r requirements-dashboard.txt
```

2. Set environment variables (optional):
```bash
export METRICS_API_URL=https://your-cloud-run-url
export AUTO_REFRESH_INTERVAL=30
export CACHE_TTL=300
```

## Running the Dashboard

### Local Execution

```bash
# From repository root
streamlit run deploy/dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

### With Custom API URL

```bash
METRICS_API_URL=https://your-api-url streamlit run deploy/dashboard/app.py
```

## Configuration

### Environment Variables

- `METRICS_API_URL`: Base URL of the Metrics API service (default: `http://localhost:8080`)
- `AUTO_REFRESH_INTERVAL`: Auto-refresh interval in seconds (default: 30)
- `CACHE_TTL`: Cache time-to-live in seconds (default: 300)
- `API_TIMEOUT`: API request timeout in seconds (default: 10)

### Streamlit Secrets (for Streamlit Cloud)

Create `.streamlit/secrets.toml`:
```toml
METRICS_API_URL = "https://your-cloud-run-url"
```

## Dashboard Pages

### Overview Page

Displays when "All Models" is selected:
- Model comparison chart (ASR vs Over-Refusal Rate)
- Models summary table
- Quick metrics overview

### Model Detail Page

Displays when a specific model is selected:
- **Summary Tab**: Key metrics and statistics
- **Bias Detection Tab**: 
  - Global bias metrics
  - ASR by category chart (with bias highlighting)
  - Bias detection table
- **Coverage Tab**:
  - Coverage statistics
  - Prompts per category breakdown
  - Coverage distribution pie chart
- **Full Metrics Tab**: Complete JSON data

## Components

### Model Selector

Located in the sidebar:
- Dropdown to select model
- Shows model metadata (last updated, has metrics, has bias report)
- Auto-refresh option

### Summary Cards

Four key metrics displayed at the top:
- Total Prompts
- Global ASR (with color coding)
- Over-Refusal Rate (with color coding)
- Biased Slices Count

### Charts

- **ASR by Category**: Bar chart showing ASR per category, with biased categories highlighted in red
- **Coverage Pie Chart**: Distribution of prompts across categories
- **Model Comparison Chart**: Side-by-side comparison of all models

### Bias Table

Interactive table showing:
- Category
- ASR
- Deviation from global ASR
- Sample count
- Biased flag

## Error Handling

The dashboard handles various error scenarios:

- **API Connection Errors**: Shows error message with retry option
- **Missing Data**: Displays informative messages
- **Invalid Responses**: Logs errors and shows user-friendly messages
- **Network Timeouts**: Shows timeout message with retry option

## Caching

The dashboard uses Streamlit's `@st.cache_data` decorator to cache API responses:
- Reduces API calls
- Improves performance
- Configurable TTL (default: 300 seconds)
- Manual cache clear via refresh button

## Auto-Refresh

Optional auto-refresh feature:
- Configurable interval (default: 30 seconds)
- Toggle in sidebar
- Visual indicator of last refresh time
- Automatically clears cache and reruns

## Troubleshooting

### Dashboard won't connect to API

1. Check API URL in sidebar configuration
2. Verify API is running and accessible
3. Check network connectivity
4. Verify API health endpoint: `curl {API_URL}/health`

### No models showing

1. Verify API has data (check `/dashboard/models` endpoint)
2. Ensure DAG has run and pushed metrics to GCS
3. Check API logs for errors

### Charts not rendering

1. Verify Plotly is installed: `pip install plotly`
2. Check browser console for JavaScript errors
3. Try clearing browser cache

### Slow performance

1. Reduce cache TTL if data changes frequently
2. Disable auto-refresh if not needed
3. Check API response times

## Deployment Options

### Option 1: Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set `METRICS_API_URL` in secrets
4. Deploy

### Option 2: Local Execution

Run locally for development or internal use:
```bash
streamlit run deploy/dashboard/app.py
```

### Option 3: Docker Container

Create Dockerfile and deploy to container platform:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY deploy/dashboard/requirements-dashboard.txt .
RUN pip install -r requirements-dashboard.txt
COPY deploy/dashboard/ .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## API Integration

The dashboard expects the following API endpoints:

- `GET /health` - Health check
- `GET /metrics/all` - All models summary
- `GET /metrics/{model_name}` - Model metrics
- `GET /metrics/{model_name}/bias` - Bias report
- `GET /metrics/{model_name}/summary` - Combined summary
- `GET /dashboard/models` - Models list

See [API Documentation](../api/README.md) for details.

## Development

### Project Structure

```
deploy/dashboard/
├── app.py                    # Main application
├── api_client.py             # API client
├── config.py                 # Configuration
├── components/               # Reusable components
│   ├── model_selector.py
│   ├── summary_cards.py
│   ├── charts.py
│   ├── bias_table.py
│   └── coverage_view.py
├── requirements-dashboard.txt
└── README.md
```

### Adding New Features

1. Create component in `components/` directory
2. Import and use in `app.py`
3. Add to appropriate tab or page
4. Update documentation

### Testing

Test with local API:
```bash
# Terminal 1: Run API
cd deploy
python -m uvicorn api.main:app --reload

# Terminal 2: Run Dashboard
streamlit run dashboard/app.py
```

## Support

For issues or questions:
1. Check API health: `{API_URL}/health`
2. Review API logs
3. Check dashboard logs in terminal
4. Verify data exists in GCS

