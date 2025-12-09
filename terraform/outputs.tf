output "composer_environment_name" {
  description = "Cloud Composer environment name"
  value       = google_composer_environment.mlops_airflow.name
}

output "composer_airflow_uri" {
  description = "Airflow web UI URL"
  value       = google_composer_environment.mlops_airflow.config[0].airflow_uri
}

output "composer_dags_bucket" {
  description = "GCS bucket for DAGs"
  value       = google_storage_bucket.composer_dags.name
}

output "composer_configs_bucket" {
  description = "GCS bucket for configs"
  value       = google_storage_bucket.composer_configs.name
}

# Webhook trigger URL is managed by Cloud Build
# Get it with: gcloud run services describe composer-webhook-trigger --region=us-central1 --format="value(status.url)"
output "webhook_trigger_url_note" {
  description = "Note: Webhook trigger URL is managed by Cloud Build. Get URL with: gcloud run services describe composer-webhook-trigger --region=us-central1 --format='value(status.url)'"
  value       = "Managed by Cloud Build - see terraform/webhook-trigger/cloudbuild.yaml"
}

output "composer_service_account" {
  description = "Composer worker service account email (using mlops-850)"
  value       = "mlops-850@${var.project_id}.iam.gserviceaccount.com"
}

output "webhook_service_account" {
  description = "Webhook trigger service account email (using mlops-850)"
  value       = "mlops-850@${var.project_id}.iam.gserviceaccount.com"
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository for webhook trigger service"
  value       = google_artifact_registry_repository.webhook_trigger.name
}

