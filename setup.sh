#!/bin/bash
# Automated setup script for Break The Bot MLOps Project
# This script sets up the project on a clean system

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}=========================================="
echo "Break The Bot - Automated Setup"
echo "==========================================${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed"
        echo "Please install Python 3.11 or higher from https://www.python.org/"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
        print_error "Python 3.11 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_status "Python $PYTHON_VERSION found"
}

# Check prerequisites
echo -e "${BLUE}Step 1: Checking Prerequisites${NC}"
echo "----------------------------------------"

check_python_version

# Check Docker
if ! command_exists docker; then
    print_error "Docker is not installed"
    echo "Please install Docker from https://www.docker.com/get-started"
    exit 1
fi
print_status "Docker found: $(docker --version | awk '{print $3}' | cut -d, -f1)"

# Check Docker Compose
if ! command_exists docker && ! docker compose version >/dev/null 2>&1; then
    print_error "Docker Compose is not installed"
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi
print_status "Docker Compose found: $(docker compose version | awk '{print $4}')"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running"
    echo "Please start Docker Desktop or Docker daemon"
    exit 1
fi
print_status "Docker is running"

# Check Git
if ! command_exists git; then
    print_warning "Git is not installed (optional, but recommended)"
else
    print_status "Git found: $(git --version | awk '{print $3}')"
fi

# Check gcloud (optional)
if command_exists gcloud; then
    print_status "gcloud CLI found (optional for cloud deployment)"
else
    print_warning "gcloud CLI not found (optional, needed only for cloud deployment)"
fi

echo ""

# Step 2: Create virtual environment
echo -e "${BLUE}Step 2: Setting Up Virtual Environment${NC}"
echo "----------------------------------------"

if [ -d "$PROJECT_ROOT/venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$PROJECT_ROOT/venv"
    else
        print_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$PROJECT_ROOT/venv" ]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$PROJECT_ROOT/venv"
    print_status "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/Scripts/activate" ]; then
    source "$PROJECT_ROOT/venv/Scripts/activate"
else
    print_error "Could not find virtual environment activation script"
    exit 1
fi

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet
print_status "pip upgraded"

echo ""

# Step 3: Install dependencies
echo -e "${BLUE}Step 3: Installing Dependencies${NC}"
echo "----------------------------------------"

if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    print_info "Installing main dependencies..."
    pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
    print_status "Main dependencies installed"
else
    print_warning "requirements.txt not found, skipping main dependencies"
fi

if [ -f "$PROJECT_ROOT/deploy/requirements-api.txt" ]; then
    print_info "Installing API dependencies..."
    pip install -r "$PROJECT_ROOT/deploy/requirements-api.txt" --quiet
    print_status "API dependencies installed"
fi

if [ -f "$PROJECT_ROOT/deploy/dashboard/requirements-dashboard.txt" ]; then
    print_info "Installing dashboard dependencies..."
    pip install -r "$PROJECT_ROOT/deploy/dashboard/requirements-dashboard.txt" --quiet
    print_status "Dashboard dependencies installed"
fi

echo ""

# Step 4: Create required directories
echo -e "${BLUE}Step 4: Creating Required Directories${NC}"
echo "----------------------------------------"

directories=(
    ".secrets"
    "airflow_artifacts/logs"
    "data/processed"
    "data/metrics"
    "data/responses"
    "data/judge"
    "data/bias"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$PROJECT_ROOT/$dir" ]; then
        mkdir -p "$PROJECT_ROOT/$dir"
        print_status "Created directory: $dir"
    else
        print_info "Directory already exists: $dir"
    fi
done

echo ""

# Step 5: Set up environment variables
echo -e "${BLUE}Step 5: Setting Up Environment Variables${NC}"
echo "----------------------------------------"

ENV_FILE="$PROJECT_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    print_warning ".env file already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Keeping existing .env file"
        SKIP_ENV=true
    fi
fi

if [ "$SKIP_ENV" != "true" ]; then
    print_info "Creating .env file template..."
    
    cat > "$ENV_FILE" << 'EOF'
# Break The Bot Environment Variables
# Fill in the values below with your actual credentials

# SMTP Configuration (for Airflow email notifications)
# Generate Gmail App Password from: https://myaccount.google.com/apppasswords
AIRFLOW_SMTP_USER=your_email@gmail.com
AIRFLOW_SMTP_PASSWORD=your_16_digit_gmail_app_password

# HuggingFace Token (for victim LLM API calls)
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token

# Groq API Key (for LLM-as-a-Judge)
# Get from: https://console.groq.com/
GROQ_API_KEY=your_groq_api_key

# Airflow Web Port (optional, defaults to 8080)
AIRFLOW_WEB_PORT=8080
EOF
    
    print_status ".env file created"
    print_warning "Please edit .env file and fill in your actual credentials"
    echo ""
    print_info "Required values:"
    echo "  - AIRFLOW_SMTP_USER: Your Gmail address"
    echo "  - AIRFLOW_SMTP_PASSWORD: 16-digit Gmail App Password"
    echo "  - HF_TOKEN: HuggingFace API token"
    echo "  - GROQ_API_KEY: Groq API key"
    echo ""
    read -p "Press Enter to continue after editing .env file (or Ctrl+C to exit and edit manually)..."
fi

echo ""

# Step 6: Set up GCP credentials
echo -e "${BLUE}Step 6: Setting Up GCP Credentials${NC}"
echo "----------------------------------------"

GCP_KEY_FILE="$PROJECT_ROOT/.secrets/gcp-key.json"

if [ -f "$GCP_KEY_FILE" ]; then
    print_status "GCP credentials file already exists"
    print_info "Location: $GCP_KEY_FILE"
else
    print_warning "GCP credentials file not found"
    echo "Please place your GCP service account key JSON file at:"
    echo "  $GCP_KEY_FILE"
    echo ""
    print_info "To obtain a service account key:"
    echo "  1. Go to GCP Console → IAM & Admin → Service Accounts"
    echo "  2. Create or select a service account"
    echo "  3. Create a key (JSON format)"
    echo "  4. Save it as: .secrets/gcp-key.json"
    echo ""
    read -p "Press Enter after placing the GCP key file (or Ctrl+C to exit)..."
    
    if [ ! -f "$GCP_KEY_FILE" ]; then
        print_error "GCP credentials file still not found"
        print_warning "You can continue setup, but Airflow and API services will need this file to run"
    else
        print_status "GCP credentials file found"
    fi
fi

echo ""

# Step 7: Initialize Airflow
echo -e "${BLUE}Step 7: Initializing Airflow${NC}"
echo "----------------------------------------"

# Check if Airflow is already initialized
if docker compose ps postgres 2>/dev/null | grep -q "Up"; then
    print_warning "Airflow services appear to be running"
    read -p "Do you want to reinitialize Airflow? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping Airflow initialization"
        SKIP_AIRFLOW=true
    fi
fi

if [ "$SKIP_AIRFLOW" != "true" ]; then
    print_info "Initializing Airflow database..."
    print_info "This may take a few minutes..."
    
    if docker compose run --rm airflow-init; then
        print_status "Airflow initialized successfully"
    else
        print_error "Airflow initialization failed"
        print_warning "You can try running manually: docker compose run --rm airflow-init"
    fi
fi

echo ""

# Step 8: Summary and next steps
echo -e "${BLUE}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""

print_status "Setup completed successfully"
echo ""

echo -e "${BLUE}Next Steps:${NC}"
echo ""

echo "1. ${GREEN}Start Airflow Services:${NC}"
echo "   docker compose up -d webserver scheduler"
echo "   Access at: http://localhost:8080 (admin/admin)"
echo ""

echo "2. ${GREEN}Configure SMTP Connection in Airflow UI:${NC}"
echo "   - Go to Admin → Connections"
echo "   - Add connection: gmail_smtp"
echo "   - See setup.md for detailed instructions"
echo ""

echo "3. ${GREEN}Start API and Dashboard (optional):${NC}"
echo "   cd deploy"
echo "   ./start_all.sh"
echo "   API: http://localhost:8080"
echo "   Dashboard: http://localhost:8501"
echo ""

echo -e "${BLUE}Useful Commands:${NC}"
echo ""
echo "  # Start Airflow"
echo "  docker compose up -d webserver scheduler"
echo ""
echo "  # Stop Airflow"
echo "  docker compose down"
echo ""
echo "  # View Airflow logs"
echo "  docker compose logs -f webserver"
echo ""
echo "  # Start API only"
echo "  cd deploy && ./start_api_only.sh"
echo ""
echo "  # Start Dashboard only"
echo "  cd deploy && ./start_dashboard_only.sh"
echo ""

echo -e "${BLUE}Documentation:${NC}"
echo "  - Setup Guide: setup.md"
echo "  - Quick Start: quick_start.md"
echo "  - Deployment: deployment_guide.md"
echo ""

print_info "For detailed instructions, see setup.md"
echo ""

