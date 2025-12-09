# GCS bucket for Composer DAGs
resource "google_storage_bucket" "composer_dags" {
  name          = "${var.project_id}-composer-dags"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 0
    }
    action {
      type = "Delete"
    }
  }
}

# GCS bucket for Composer configs (optional)
resource "google_storage_bucket" "composer_configs" {
  name          = "${var.project_id}-composer-configs"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true
}

# Cloud Composer Environment
resource "google_composer_environment" "mlops_airflow" {
  name   = var.composer_environment_name
  region = var.region

  lifecycle {
    # Prevent accidental destruction - uncomment if you want extra protection
    # prevent_destroy = true
    # Only recreate if image_version changes, not for package updates
    replace_triggered_by = []
  }

  config {
    # Composer 3 uses workloads_config instead of node_config for machine settings
    node_config {
      # Using mlops-850 service account (has roles/owner, includes all needed permissions)
      service_account = "mlops-850@${var.project_id}.iam.gserviceaccount.com"
    }

    # Workloads config for Composer 3
    # Note: Memory must be multiples of 0.25GB, and web_server must be at least 2GB
    workloads_config {
      scheduler {
        cpu        = 0.5
        memory_gb  = 2.0  # Must be multiple of 0.25GB
        storage_gb = 1
        count      = 1
      }
      web_server {
        cpu        = 0.5
        memory_gb  = 2.0  # Must be at least 2GB
        storage_gb = 1
      }
      worker {
        cpu        = 0.5
        memory_gb  = 2.0  # Must be multiple of 0.25GB
        storage_gb = 1
        min_count  = 1
        max_count  = 3
      }
    }

    software_config {
      image_version = var.airflow_version
      # python_version is not supported in Composer 3 - always uses Python 3

      # Environment variables
      # Note: Secret Manager secrets are accessed via Airflow Variables or Connections
      # For Composer, we'll use Airflow Variables to reference secrets
      env_variables = {
        PROJECT_ROOT        = "/home/airflow/gcs"
        SALAD_CONFIG_PATH   = "/home/airflow/gcs/dags/config/data_sources.json"
        SALAD_OUTPUT_PATH   = "/home/airflow/gcs/dags/dvc_project/data/processed/processed_data.csv"
        # Secrets will be accessed via Airflow Variables set after deployment
        # Or use Secret Manager integration in Airflow Connections
      }

      # Airflow configuration overrides
      # Note: AIRFLOW__* variables must be set via airflow_config_overrides, not env_variables
      # Format: "section-name" (use hyphens, not dots)
      airflow_config_overrides = {
        "email-email_backend" = "airflow.utils.email.send_email_smtp"
        "smtp-smtp_host"      = "smtp.gmail.com"
        "smtp-smtp_port"       = "587"
        "smtp-smtp_starttls"  = "True"
        "smtp-smtp_ssl"       = "False"
        # SMTP user and password will be set via Airflow Variables after deployment
      }

      # PyPI packages from composer_requirements.txt
      # Note: Using exact versions (==) to speed up installation and avoid timeouts
      # DVC GCS plugin is explicitly included as dvc-gs (same version as dvc)
      # uvicorn[standard] extras are not supported in Terraform, using uvicorn base package
      pypi_packages = {
        # Core Data Processing
        numpy              = "==1.26.4"
        pandas             = "==2.1.4"
        pyarrow            = "==16.0.0"
        datasets           = "==2.16.1"
        
        # DVC and Storage
        dvc                = "==3.0.0"
        dvc-gs             = "==3.0.2"
        gcsfs              = "==2023.10.0"
        
        # Data Validation
        great-expectations = "==0.18.21"
        
        # Environment and API Clients
        python-dotenv      = "==1.0.0"
        huggingface-hub    = "==0.30.2"
        groq               = "==0.34.1"
        
        # System Dependencies
        cffi               = "==1.16.0"
        setuptools         = "==66.1.1"
        tqdm               = "==4.66.0"
      }
    }
  }

  depends_on = [
    google_project_service.composer_api,
    google_storage_bucket.composer_dags
  ]
}

