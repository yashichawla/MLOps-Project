FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy dashboard code
COPY . .

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Run Streamlit with proper PORT handling for Cloud Run
# --server.headless=true is required for Cloud Run (no browser available)
# --server.port uses PORT env var with fallback to 8501
# Use shell form to allow variable expansion
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]

