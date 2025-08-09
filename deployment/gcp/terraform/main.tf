# Terraform configuration for MeshAI SDK Development Environment on GCP
# Minimal cost setup with Cloud Run + Cloud SQL + Memorystore

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "sql-component.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "cloudbuild.googleapis.com",
    "container.googleapis.com",
    "compute.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "secretmanager.googleapis.com"
  ])
  
  service = each.key
  project = var.project_id
}

# Service Account for MeshAI services
resource "google_service_account" "meshai_service" {
  account_id   = "meshai-service"
  display_name = "MeshAI Service Account"
  description  = "Service account for MeshAI Registry and Runtime services"
}

# IAM roles for the service account
resource "google_project_iam_member" "meshai_roles" {
  for_each = toset([
    "roles/cloudsql.client",
    "roles/redis.editor",
    "roles/secretmanager.secretAccessor",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter"
  ])
  
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.meshai_service.email}"
}

# Random password for PostgreSQL
resource "random_password" "postgres_password" {
  length  = 16
  special = true
}

# Cloud SQL PostgreSQL Instance (Development - db-f1-micro)
resource "google_sql_database_instance" "postgres" {
  name                = "meshai-postgres-${var.environment}"
  database_version    = "POSTGRES_15"
  region              = var.region
  deletion_protection = false

  settings {
    tier              = "db-f1-micro"  # Minimal cost tier
    availability_type = "ZONAL"       # Single zone for cost savings
    disk_type         = "PD_SSD"
    disk_size         = 10            # Minimal 10GB
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      start_time                    = "03:00"
      point_in_time_recovery_enabled = false  # Disable for cost savings
      backup_retention_settings {
        retained_backups = 7
      }
    }

    ip_configuration {
      ipv4_enabled                                  = true
      private_network                              = null
      enable_private_path_for_google_cloud_services = false
      authorized_networks {
        name  = "allow-all"
        value = "0.0.0.0/0"  # Note: Restrict this in production
      }
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }
  }
}

# PostgreSQL Database
resource "google_sql_database" "meshai_db" {
  name     = "meshai"
  instance = google_sql_database_instance.postgres.name
}

# PostgreSQL User
resource "google_sql_user" "meshai_user" {
  name     = "meshai"
  instance = google_sql_database_instance.postgres.name
  password = random_password.postgres_password.result
}

# Memorystore Redis Instance (Basic tier, 1GB)
resource "google_redis_instance" "redis" {
  name               = "meshai-redis-${var.environment}"
  tier               = "BASIC"      # Basic tier for cost savings
  memory_size_gb     = 1            # Minimal 1GB
  region             = var.region
  redis_version      = "REDIS_7_0"
  display_name       = "MeshAI Redis Cache"
  
  # Basic tier doesn't support auth or transit encryption
  auth_enabled            = false
  transit_encryption_mode = "DISABLED"
  
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
  }
}

# Secret for database connection
resource "google_secret_manager_secret" "db_connection" {
  secret_id = "meshai-db-connection-${var.environment}"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "db_connection" {
  secret = google_secret_manager_secret.db_connection.name
  
  secret_data = jsonencode({
    host     = google_sql_database_instance.postgres.public_ip_address
    port     = "5432"
    database = google_sql_database.meshai_db.name
    username = google_sql_user.meshai_user.name
    password = random_password.postgres_password.result
    ssl_mode = "require"
  })
}

# Secret for Redis connection
resource "google_secret_manager_secret" "redis_connection" {
  secret_id = "meshai-redis-connection-${var.environment}"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "redis_connection" {
  secret = google_secret_manager_secret.redis_connection.name
  
  secret_data = jsonencode({
    host = google_redis_instance.redis.host
    port = google_redis_instance.redis.port
    url  = "redis://${google_redis_instance.redis.host}:${google_redis_instance.redis.port}"
  })
}

# Output values
output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP Region"
  value       = var.region
}

output "postgres_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.postgres.connection_name
}

output "postgres_public_ip" {
  description = "PostgreSQL public IP address"
  value       = google_sql_database_instance.postgres.public_ip_address
}

output "redis_host" {
  description = "Redis host address"
  value       = google_redis_instance.redis.host
}

output "redis_port" {
  description = "Redis port"
  value       = google_redis_instance.redis.port
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.meshai_service.email
}

output "db_secret_name" {
  description = "Database connection secret name"
  value       = google_secret_manager_secret.db_connection.name
}

output "redis_secret_name" {
  description = "Redis connection secret name"
  value       = google_secret_manager_secret.redis_connection.name
}