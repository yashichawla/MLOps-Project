# Terraform Infrastructure for Cloud Composer

This directory contains Terraform configuration for deploying Cloud Composer (managed Airflow) with GitHub webhook integration.

## Structure

```
terraform/
├── main.tf                 # Provider and API enables
├── variables.tf            # Variable definitions
├── outputs.tf              # Output values
├── composer.tf             # Cloud Composer environment
├── webhook-trigger.tf      # Cloud Run webhook service
├── secrets.tf              # Secret Manager resources
├── iam.tf                  # IAM roles and bindings
├── versions.tf             # Provider versions
├── webhook-trigger/        # Webhook service code
│   ├── main.py            # FastAPI webhook handler
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # Container image
├── MIGRATION_GUIDE.md     # Step-by-step migration guide
└── README.md              # This file
```

## Quick Start

1. **Create `terraform.tfvars`**:
   ```hcl
   project_id = "break-the-bot-480422"
   region     = "us-central1"
   # ... other variables
   ```

2. **Initialize Terraform**:
   ```bash
   terraform init
   ```

3. **Plan and Apply**:
   ```bash
   terraform plan -out=tfplan
   terraform apply tfplan
   ```

4. **Follow Migration Guide**:
   See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) for detailed steps.

## Variables

See [variables.tf](./variables.tf) for all available variables. Key variables:

- `project_id`: GCP Project ID
- `region`: GCP Region
- `composer_environment_name`: Composer environment name
- `hf_token`: HuggingFace API token (sensitive)
- `groq_api_key`: Groq API key (sensitive)
- `smtp_user`: SMTP username (sensitive)
- `smtp_password`: SMTP password (sensitive)
- `github_webhook_secret`: GitHub webhook secret (sensitive)

## Outputs

After applying, Terraform outputs:

- `composer_airflow_uri`: Airflow web UI URL
- `composer_dags_bucket`: GCS bucket for DAGs
- `webhook_trigger_url`: Webhook service URL
- `composer_service_account`: Service account email

## Resources Created

- **Cloud Composer Environment**: Managed Airflow environment
- **GCS Buckets**: For DAGs and configs
- **Secret Manager Secrets**: For sensitive credentials
- **Service Accounts**: For Composer and webhook service
- **Cloud Run Service**: GitHub webhook trigger service
- **IAM Roles**: Permissions for service accounts

## Cost Estimate

- Cloud Composer: ~$300-500/month
- Cloud Run: ~$5-10/month
- GCS: ~$5-10/month
- Secret Manager: ~$0.30/month
- **Total**: ~$310-525/month

## Webhook Service

The webhook trigger service (`webhook-trigger/`) is a FastAPI application that:
- Receives GitHub webhook payloads
- Validates webhook signatures
- Checks if config files changed
- Triggers Composer DAG via Airflow REST API

### Deployment

The webhook service is deployed using Cloud Build, which automatically:
- Builds the Docker image
- Pushes to Container Registry
- Fetches the Composer Airflow URI
- Deploys to Cloud Run with the correct configuration

**Deploy using Cloud Build:**

```bash
cd terraform/webhook-trigger
gcloud builds submit --config cloudbuild.yaml
```

The Cloud Build configuration will:
1. Build the Docker image from `webhook-trigger/`
2. Push to Container Registry
3. Get the Composer Airflow URI automatically
4. Deploy to Cloud Run with environment variables:
   - `COMPOSER_ENVIRONMENT`: Composer environment name
   - `COMPOSER_LOCATION`: GCP region
   - `COMPOSER_DAG_ID`: DAG to trigger
   - `GCP_PROJECT_ID`: GCP project ID
   - `COMPOSER_AIRFLOW_URI`: Airflow URI (fetched automatically)
   - `GITHUB_WEBHOOK_SECRET`: From Secret Manager

**Get the webhook URL after deployment:**

```bash
gcloud run services describe composer-webhook-trigger \
  --region us-central1 \
  --format "value(status.url)"
```

**Configure GitHub Webhook:**

1. Go to your GitHub repository → Settings → Webhooks
2. Add webhook with:
   - **Payload URL**: `https://composer-webhook-trigger-XXXXX.us-central1.run.app/webhook`
   - **Content type**: `application/json`
   - **Secret**: Use the same secret from `GITHUB_WEBHOOK_SECRET`
   - **Events**: Select "Just the push event"
3. Save the webhook

**Testing the Webhook:**

After configuring the GitHub webhook, push a change to any of these config files:
- `config/data_sources.json`
- `config/attack_llm_config.json`
- `dags/config/data_sources.json`
- `dags/config/attack_llm_config.json`

The webhook will automatically trigger the DAG in Composer.

**Monitor Webhook Logs:**

```bash
gcloud run services logs read composer-webhook-trigger \
  --region us-central1 \
  --limit 20
```

## Documentation

- [Migration Guide](./MIGRATION_GUIDE.md): Step-by-step migration instructions
- [Terraform Documentation](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Cloud Composer Documentation](https://cloud.google.com/composer/docs)

