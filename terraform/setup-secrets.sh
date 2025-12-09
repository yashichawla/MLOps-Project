#!/bin/bash
# Script to set up Airflow Variables from Secret Manager after Composer deployment
# Run this after Terraform apply to configure Airflow with secrets

set -e

PROJECT_ID="${PROJECT_ID:-break-the-bot-480422}"
COMPOSER_ENV="${COMPOSER_ENV:-mlops-airflow-composer}"
LOCATION="${LOCATION:-us-central1}"

echo "Setting up Airflow Variables from Secret Manager..."

# Get Composer Airflow URI
AIRFLOW_URI=$(gcloud composer environments describe ${COMPOSER_ENV} \
  --location ${LOCATION} \
  --format="value(config.airflowUri)")

echo "Airflow URI: ${AIRFLOW_URI}"

# Get access token
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Function to get secret value
get_secret() {
  SECRET_NAME=$1
  gcloud secrets versions access latest --secret="${SECRET_NAME}" --project="${PROJECT_ID}"
}

# Set Airflow Variables from secrets
echo "Setting HF_TOKEN..."
HF_TOKEN=$(get_secret "composer-hf-token")
curl -X POST "${AIRFLOW_URI}/api/v1/variables" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"key\": \"HF_TOKEN\", \"value\": \"${HF_TOKEN}\", \"description\": \"HuggingFace API token\"}"

echo "Setting GROQ_API_KEY..."
GROQ_API_KEY=$(get_secret "composer-groq-api-key")
curl -X POST "${AIRFLOW_URI}/api/v1/variables" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"key\": \"GROQ_API_KEY\", \"value\": \"${GROQ_API_KEY}\", \"description\": \"Groq API key\"}"

echo "Setting SMTP credentials..."
SMTP_USER=$(get_secret "composer-smtp-user")
SMTP_PASSWORD=$(get_secret "composer-smtp-password")

curl -X POST "${AIRFLOW_URI}/api/v1/variables" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"key\": \"AIRFLOW__SMTP__SMTP_USER\", \"value\": \"${SMTP_USER}\", \"description\": \"SMTP username\"}"

curl -X POST "${AIRFLOW_URI}/api/v1/variables" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"key\": \"AIRFLOW__SMTP__SMTP_PASSWORD\", \"value\": \"${SMTP_PASSWORD}\", \"description\": \"SMTP password\"}"

curl -X POST "${AIRFLOW_URI}/api/v1/variables" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"key\": \"AIRFLOW__SMTP__SMTP_MAIL_FROM\", \"value\": \"${SMTP_USER}\", \"description\": \"SMTP from address\"}"

echo "✅ Airflow Variables configured successfully!"
echo ""
echo "You can also set these manually in Airflow UI:"
echo "  Admin → Variables → Add/Edit"

