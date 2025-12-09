# NOTE: Using mlops-850@break-the-bot-480422.iam.gserviceaccount.com for all services
# This service account has roles/owner which includes all needed permissions:
# - roles/composer.worker (for Composer)
# - roles/composer.user (for webhook trigger)
# - roles/storage.objectAdmin (for DVC/GCS)
# - roles/secretmanager.secretAccessor (for secrets)
# The service account is not managed by Terraform (exists externally)

# Service account for Cloud Composer workers
# DISABLED: Using mlops-850 instead
# resource "google_service_account" "composer_worker" {
#   account_id   = "composer-worker-sa"
#   display_name = "Cloud Composer Worker Service Account"
#   project      = var.project_id
# }

# Grant Composer worker role
# DISABLED: mlops-850 has roles/owner which includes this
# resource "google_project_iam_member" "composer_worker" {
#   project = var.project_id
#   role    = "roles/composer.worker"
#   member  = "serviceAccount:${google_service_account.composer_worker.email}"
# }

# Grant Storage Object Admin for DVC/GCS operations
# DISABLED: mlops-850 has roles/owner which includes this
# resource "google_project_iam_member" "composer_storage_admin" {
#   project = var.project_id
#   role    = "roles/storage.objectAdmin"
#   member  = "serviceAccount:${google_service_account.composer_worker.email}"
# }

# Grant Secret Manager accessor role
# DISABLED: mlops-850 has roles/owner which includes this
# resource "google_project_iam_member" "composer_secret_accessor" {
#   project = var.project_id
#   role    = "roles/secretmanager.secretAccessor"
#   member  = "serviceAccount:${google_service_account.composer_worker.email}"
# }

# Service account for webhook trigger service
# DISABLED: Using mlops-850 instead
# resource "google_service_account" "webhook_trigger" {
#   account_id   = "webhook-trigger-sa"
#   display_name = "Webhook Trigger Service Account"
#   project      = var.project_id
# }

# Grant Composer user role to trigger DAGs
# DISABLED: mlops-850 has roles/owner which includes this
# resource "google_project_iam_member" "webhook_composer_user" {
#   project = var.project_id
#   role    = "roles/composer.user"
#   member  = "serviceAccount:${google_service_account.webhook_trigger.email}"
# }

# Grant Secret Manager accessor for GitHub webhook secret
# DISABLED: mlops-850 has roles/owner which includes this
# resource "google_project_iam_member" "webhook_secret_accessor" {
#   project = var.project_id
#   role    = "roles/secretmanager.secretAccessor"
#   member  = "serviceAccount:${google_service_account.webhook_trigger.email}"
# }

# Service account for metrics API
# Note: mlops-850@break-the-bot-480422.iam.gserviceaccount.com exists but is not managed by Terraform
# If you want to manage it, uncomment below and update cloudbuild.yaml to use this service account
# resource "google_service_account" "metrics_api" {
#   account_id   = "metrics-api-sa"
#   display_name = "Metrics API Service Account"
#   project      = var.project_id
# }
#
# # Grant Storage Object Viewer for GCS access
# resource "google_project_iam_member" "metrics_api_storage_viewer" {
#   project = var.project_id
#   role    = "roles/storage.objectViewer"
#   member  = "serviceAccount:${google_service_account.metrics_api.email}"
# }

