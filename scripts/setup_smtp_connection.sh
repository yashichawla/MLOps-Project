#!/bin/bash
# Script to create Airflow SMTP connection for Gmail
# Usage: ./setup_smtp_connection.sh <gmail-address> <app-password>

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <gmail-address> <app-password>"
    echo ""
    echo "Example:"
    echo "  $0 your.email@gmail.com xxxx xxxx xxxx xxxx"
    echo ""
    echo "Note: Use Gmail App Password (16 characters), not your regular password"
    echo "To generate App Password:"
    echo "  1. Go to Google Account → Security"
    echo "  2. Enable 2-Step Verification"
    echo "  3. Go to App Passwords → Generate for 'Mail'"
    exit 1
fi

GMAIL_ADDRESS="$1"
APP_PASSWORD="$2"

echo "Setting up Airflow SMTP connection 'gmail_smtp'..."

# Check if running in Docker
if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER}" ]; then
    echo "Detected Docker environment"
    AIRFLOW_CMD="airflow"
else
    # Try to detect if we're in a Docker Compose setup
    if command -v docker-compose &> /dev/null || command -v docker &> /dev/null; then
        echo "Attempting to use Docker Compose..."
        # Try to find the webserver container
        CONTAINER=$(docker ps --filter "name=webserver" --format "{{.Names}}" | head -n 1)
        if [ -n "$CONTAINER" ]; then
            echo "Found container: $CONTAINER"
            AIRFLOW_CMD="docker exec -it $CONTAINER airflow"
        else
            echo "Warning: Could not find Airflow webserver container"
            echo "Please run this script from within the Airflow container or specify the container name"
            exit 1
        fi
    else
        # Assume Airflow is installed locally
        AIRFLOW_CMD="airflow"
    fi
fi

# Check if connection already exists
if $AIRFLOW_CMD connections get gmail_smtp &> /dev/null; then
    echo "Connection 'gmail_smtp' already exists. Deleting it first..."
    $AIRFLOW_CMD connections delete gmail_smtp
fi

# Create the connection
echo "Creating SMTP connection..."
$AIRFLOW_CMD connections add gmail_smtp \
    --conn-type email \
    --conn-host smtp.gmail.com \
    --conn-login "$GMAIL_ADDRESS" \
    --conn-password "$APP_PASSWORD" \
    --conn-port 587 \
    --conn-extra '{"starttls": true, "ssl": false}'

echo "✅ SMTP connection 'gmail_smtp' created successfully!"
echo ""
echo "To verify, run:"
echo "  $AIRFLOW_CMD connections get gmail_smtp"
echo ""
echo "Or check in Airflow UI: Admin → Connections → gmail_smtp"

