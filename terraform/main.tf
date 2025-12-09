provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "composer_api" {
  service = "composer.googleapis.com"
  project = var.project_id

  disable_on_destroy = false
}

resource "google_project_service" "secretmanager_api" {
  service = "secretmanager.googleapis.com"
  project = var.project_id

  disable_on_destroy = false
}

# Wait for Secret Manager API to propagate
# Note: If you still see API errors, manually enable the API at:
# https://console.developers.google.com/apis/api/secretmanager.googleapis.com/overview?project=break-the-bot-480422
resource "time_sleep" "wait_for_secretmanager_api" {
  depends_on = [google_project_service.secretmanager_api]
  create_duration = "120s"
}

resource "google_project_service" "run_api" {
  service = "run.googleapis.com"
  project = var.project_id

  disable_on_destroy = false
}

resource "google_project_service" "storage_api" {
  service = "storage-api.googleapis.com"
  project = var.project_id

  disable_on_destroy = false
}

resource "google_project_service" "cloudbuild_api" {
  service = "cloudbuild.googleapis.com"
  project = var.project_id

  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry_api" {
  service = "artifactregistry.googleapis.com"
  project = var.project_id

  disable_on_destroy = false
}

