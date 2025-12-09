#!/bin/bash
# Script to create Airflow SMTP connection in Google Cloud Composer
# Usage: ./setup_smtp_connection_composer.sh <gmail-address> <app-password>

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

COMPOSER_ENV="mlops-airflow-composer"
LOCATION="us-central1"
PROJECT_ID="break-the-bot-480422"

echo "Setting up Airflow SMTP connection 'gmail_smtp' in Composer..."
echo "Environment: $COMPOSER_ENV"
echo "Location: $LOCATION"
echo ""

# Check if connection already exists
echo "Checking if connection 'gmail_smtp' already exists..."
EXISTING=$(gcloud composer environments run "$COMPOSER_ENV" \
  --location "$LOCATION" \
  --project "$PROJECT_ID" \
  connections -- get gmail_smtp 2>&1 || echo "NOT_FOUND")

if echo "$EXISTING" | grep -q "NOT_FOUND\|does not exist"; then
    echo "Connection does not exist. Creating it..."
else
    echo "Connection 'gmail_smtp' already exists. Deleting it first..."
    gcloud composer environments run "$COMPOSER_ENV" \
      --location "$LOCATION" \
      --project "$PROJECT_ID" \
      connections -- delete gmail_smtp || true
    echo "Deleted existing connection"
fi

# Create the connection
echo ""
echo "Creating SMTP connection..."
gcloud composer environments run "$COMPOSER_ENV" \
  --location "$LOCATION" \
  --project "$PROJECT_ID" \
  connections -- add gmail_smtp \
  --conn-type email \
  --conn-host smtp.gmail.com \
  --conn-login "$GMAIL_ADDRESS" \
  --conn-password "$APP_PASSWORD" \
  --conn-port 587 \
  --conn-extra '{"starttls": true, "ssl": false}'

echo ""
echo "✅ SMTP connection 'gmail_smtp' created successfully!"
echo ""
echo "To verify, run:"
echo "  gcloud composer environments run $COMPOSER_ENV --location $LOCATION connections -- get gmail_smtp"
echo ""
echo "Or check in Airflow UI: Admin → Connections → gmail_smtp"

