# Artifact Registry repository for webhook trigger service
resource "google_artifact_registry_repository" "webhook_trigger" {
  location      = var.region
  repository_id = "webhook-trigger"
  description   = "Docker repository for webhook trigger service"
  format        = "DOCKER"

  depends_on = [google_project_service.artifactregistry_api]
}

# Cloud Run service is now managed by Cloud Build
# See terraform/webhook-trigger/cloudbuild.yaml for deployment
# This keeps Terraform focused on infrastructure only (service accounts, IAM, secrets, etc.)

