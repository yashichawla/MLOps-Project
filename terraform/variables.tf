variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "break-the-bot-480422"
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "composer_environment_name" {
  description = "Cloud Composer environment name"
  type        = string
  default     = "mlops-airflow-composer"
}

variable "composer_node_count" {
  description = "Number of worker nodes in Composer environment"
  type        = number
  default     = 3
}

variable "composer_machine_type" {
  description = "Machine type for Composer nodes"
  type        = string
  default     = "n1-standard-1"
}

variable "composer_disk_size_gb" {
  description = "Disk size in GB for Composer nodes"
  type        = number
  default     = 30
}

variable "airflow_version" {
  description = "Airflow version for Composer"
  type        = string
  default     = "composer-3-airflow-2.9.3"
}

variable "python_version" {
  description = "Python version for Composer"
  type        = string
  default     = "3"
}

variable "github_webhook_secret" {
  description = "GitHub webhook secret for authentication"
  type        = string
  sensitive   = true
}

variable "hf_token" {
  description = "HuggingFace API token"
  type        = string
  sensitive   = true
}

variable "groq_api_key" {
  description = "Groq API key"
  type        = string
  sensitive   = true
}

variable "smtp_user" {
  description = "SMTP username for email notifications"
  type        = string
  sensitive   = true
}

variable "smtp_password" {
  description = "SMTP password for email notifications"
  type        = string
  sensitive   = true
}

variable "dvc_bucket_name" {
  description = "Existing DVC GCS bucket name"
  type        = string
  default     = "mlops-project-dvc-480422"
}

variable "composer_dag_id" {
  description = "DAG ID to trigger via webhook"
  type        = string
  default     = "salad_ml_evaluation_pipeline_v1"
}

