# Secret for HuggingFace token
resource "google_secret_manager_secret" "hf_token" {
  secret_id = "composer-hf-token"
  project   = var.project_id

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  depends_on = [time_sleep.wait_for_secretmanager_api]
}

resource "google_secret_manager_secret_version" "hf_token" {
  secret      = google_secret_manager_secret.hf_token.id
  secret_data = var.hf_token
}

# Secret for Groq API key
resource "google_secret_manager_secret" "groq_api_key" {
  secret_id = "composer-groq-api-key"
  project   = var.project_id

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  depends_on = [time_sleep.wait_for_secretmanager_api]
}

resource "google_secret_manager_secret_version" "groq_api_key" {
  secret      = google_secret_manager_secret.groq_api_key.id
  secret_data = var.groq_api_key
}

# Secret for SMTP user
resource "google_secret_manager_secret" "smtp_user" {
  secret_id = "composer-smtp-user"
  project   = var.project_id

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  depends_on = [time_sleep.wait_for_secretmanager_api]
}

resource "google_secret_manager_secret_version" "smtp_user" {
  secret      = google_secret_manager_secret.smtp_user.id
  secret_data = var.smtp_user
}

# Secret for SMTP password
resource "google_secret_manager_secret" "smtp_password" {
  secret_id = "composer-smtp-password"
  project   = var.project_id

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  depends_on = [time_sleep.wait_for_secretmanager_api]
}

resource "google_secret_manager_secret_version" "smtp_password" {
  secret      = google_secret_manager_secret.smtp_password.id
  secret_data = var.smtp_password
}

# Secret for GitHub webhook secret
resource "google_secret_manager_secret" "github_webhook_secret" {
  secret_id = "github-webhook-secret"
  project   = var.project_id

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  depends_on = [time_sleep.wait_for_secretmanager_api]
}

resource "google_secret_manager_secret_version" "github_webhook_secret" {
  secret      = google_secret_manager_secret.github_webhook_secret.id
  secret_data = var.github_webhook_secret
}

