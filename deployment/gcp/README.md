# MeshAI SDK Google Cloud Platform Deployment

This directory contains all the necessary files to deploy the MeshAI SDK to Google Cloud Platform with minimal cost (~$78/month).

## 🎯 Development Environment Overview

**Architecture:**
- **Cloud Run**: 2 serverless containers (Registry + Runtime services)
- **Cloud SQL**: PostgreSQL database (db-f1-micro instance)
- **Memorystore**: Redis cache (1GB basic tier)
- **Load Balancer**: HTTPS load balancing and SSL termination
- **Monitoring**: Basic Cloud Monitoring and logging

**Estimated Monthly Cost: ~$78**
- Cloud Run (2 services): $8.50
- Cloud SQL (f1-micro): $17.50  
- Memorystore Redis (1GB): $26.00
- Load Balancer: $18.30
- Monitoring & Other: $7.70

## ⚠️ Important Security Notice

**These are deployment templates and examples.** For actual deployments:

1. **Copy these files to a private repository or local directory**
2. **Never commit sensitive information like project IDs to public repos**
3. **Use the private deployment approach below**

## 📁 Recommended Setup

### Create Private Deployment Directory

```bash
# Create private deployment directory
mkdir ~/meshai-deployment-private
cd ~/meshai-deployment-private

# Copy deployment templates
cp -r /path/to/meshai-sdk/deployment/gcp/* .

# Update configuration with your values
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
# Edit terraform.tfvars with your actual project ID

# Deploy
./deploy.sh --project-id YOUR_PROJECT_ID
```

## 🚀 Quick Start Deployment

### Prerequisites

1. **Google Cloud Account** with billing enabled
2. **GCP Project** with Owner or Editor permissions
3. **Required Tools:**
   ```bash
   # Install Google Cloud SDK
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   
   # Install Terraform
   wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
   unzip terraform_1.6.0_linux_amd64.zip
   sudo mv terraform /usr/local/bin/
   
   # Verify installations
   gcloud version
   terraform version
   ```

4. **Authentication:**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

### One-Command Deployment

```bash
# From the meshai-sdk root directory
./deployment/gcp/deploy.sh --project-id YOUR_PROJECT_ID
```

### Manual Step-by-Step Deployment

1. **Setup Terraform Variables:**
   ```bash
   cd deployment/gcp/terraform
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your project ID
   ```

2. **Deploy Infrastructure:**
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

3. **Build and Deploy Services:**
   ```bash
   cd ../../..  # Back to project root
   gcloud builds submit --config=deployment/gcp/cloudbuild.yaml
   ```

## 📁 File Structure

```
deployment/gcp/
├── README.md                    # This file
├── deploy.sh                    # One-command deployment script
├── cloudbuild.yaml             # Cloud Build configuration
└── terraform/
    ├── main.tf                 # Main Terraform configuration
    └── terraform.tfvars.example # Example variables file
```

## 🔧 Configuration Files

### `terraform/main.tf`
Main infrastructure as code configuration:
- **Cloud SQL**: PostgreSQL with minimal f1-micro instance
- **Memorystore**: Redis with 1GB basic tier
- **IAM**: Service accounts and permissions
- **Secrets**: Secure storage for database credentials

### `cloudbuild.yaml`
Cloud Build pipeline:
- Builds Docker image from root Dockerfile
- Deploys to 2 Cloud Run services (Registry + Runtime)
- Configures environment variables and resource limits

### `deploy.sh`
Automated deployment script:
- Prerequisites checking
- Infrastructure deployment
- Application building and deployment
- Basic monitoring setup
- Post-deployment testing

## 🏗️ Infrastructure Details

### Cloud Run Services

**Registry Service** (`meshai-registry`)
- **URL**: `https://meshai-registry-PROJECT_ID.a.run.app`
- **Port**: 8001
- **Resources**: 1 vCPU, 512MB RAM
- **Scaling**: 0-10 instances
- **Features**: Agent registration, discovery, health monitoring

**Runtime Service** (`meshai-runtime`)
- **URL**: `https://meshai-runtime-PROJECT_ID.a.run.app`
- **Port**: 8002  
- **Resources**: 1 vCPU, 512MB RAM
- **Scaling**: 0-10 instances
- **Features**: Task orchestration, ML routing, execution

### Database Configuration

**Cloud SQL PostgreSQL**
- **Instance**: db-f1-micro (1 shared vCPU, 3.75GB RAM)
- **Storage**: 10GB SSD with auto-resize
- **Backup**: Daily backups, 7-day retention
- **Network**: Public IP with authorized networks
- **Database**: `meshai` with user `meshai`

**Memorystore Redis**
- **Tier**: Basic (no replication for cost savings)
- **Memory**: 1GB capacity
- **Version**: Redis 7.0
- **Network**: VPC native, private IP
- **Policy**: allkeys-lru eviction

## 🔍 Monitoring and Logging

### Health Checks
- **Registry**: `https://meshai-registry-PROJECT_ID.a.run.app/health`
- **Runtime**: `https://meshai-runtime-PROJECT_ID.a.run.app/health`
- **Uptime Monitoring**: 5-minute intervals

### Metrics
- **Cloud Monitoring**: Automatic Cloud Run metrics
- **Custom Metrics**: Prometheus metrics at `/metrics` endpoints
- **Logging**: Structured logging with Cloud Logging

### Dashboards
- **Cloud Console**: https://console.cloud.google.com/run?project=PROJECT_ID
- **Monitoring**: https://console.cloud.google.com/monitoring?project=PROJECT_ID

## 🧪 Testing Your Deployment

### 1. Health Checks
```bash
# Check Registry Service
curl https://meshai-registry-PROJECT_ID.a.run.app/health

# Check Runtime Service  
curl https://meshai-runtime-PROJECT_ID.a.run.app/health
```

### 2. API Documentation
```bash
# Registry API docs
open https://meshai-registry-PROJECT_ID.a.run.app/docs

# Runtime API docs
open https://meshai-runtime-PROJECT_ID.a.run.app/docs
```

### 3. Register a Test Agent
```bash
curl -X POST "https://meshai-registry-PROJECT_ID.a.run.app/api/v1/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test-agent-1",
    "name": "Test Agent",
    "framework": "openai",
    "capabilities": ["text-generation"],
    "endpoint": "https://api.openai.com/v1"
  }'
```

### 4. Submit a Test Task
```bash
curl -X POST "https://meshai-runtime-PROJECT_ID.a.run.app/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "text-generation",
    "input_data": {"prompt": "Hello, world!"},
    "required_capabilities": ["text-generation"]
  }'
```

## 💰 Cost Management

### Cost Monitoring
1. **Billing Dashboard**: https://console.cloud.google.com/billing
2. **Cost Breakdown**: Monitor by service
3. **Budget Alerts**: Set up at $100/month threshold

### Cost Optimization Tips
1. **Scale to Zero**: Cloud Run scales to 0 when not in use
2. **Right-sizing**: Monitor and adjust CPU/memory as needed
3. **Regional Choice**: us-central1 is the lowest cost region
4. **Resource Monitoring**: Use Cloud Monitoring to track utilization

### Expected Cost Breakdown
```
Monthly Costs (Development Environment):
├── Cloud Run Services      $8.50   (11%)
├── Cloud SQL f1-micro     $17.50   (22%)  
├── Memorystore Redis 1GB  $26.00   (33%)
├── Load Balancer          $18.30   (23%)
├── Monitoring & Logging    $5.00    (6%)
├── Network Egress          $1.20    (2%)
└── Storage & Other         $1.60    (3%)
                           -------
Total:                     $78.10   (100%)
```

## 🔄 Scaling Up

When you're ready to scale beyond development:

### Staging Environment (~$306/month)
```bash
# Deploy with staging configuration
./deploy.sh --project-id YOUR_PROJECT_ID --environment staging
```

### Production Environment (~$1,330/month)
- Multi-region deployment
- High availability database
- Larger instance sizes
- Enhanced monitoring

## 🛠️ Troubleshooting

### Common Issues

**1. Permission Denied**
```bash
# Ensure you have the required permissions
gcloud projects get-iam-policy PROJECT_ID
```

**2. API Not Enabled**
```bash
# Enable required APIs
gcloud services enable run.googleapis.com sqladmin.googleapis.com redis.googleapis.com
```

**3. Cloud Build Timeout**
```bash
# Increase timeout in cloudbuild.yaml
timeout: '1200s'  # 20 minutes
```

**4. Database Connection Issues**
```bash
# Check Cloud SQL instance status
gcloud sql instances describe meshai-postgres-dev

# Test database connectivity
gcloud sql connect meshai-postgres-dev --user=meshai
```

**5. Redis Connection Issues**
```bash
# Check Redis instance status
gcloud redis instances describe meshai-redis-dev --region=us-central1
```

### Getting Help

1. **Cloud Console Logs**: Check Cloud Run and Cloud Build logs
2. **Error Reporting**: Monitor error rates in Cloud Console
3. **Support**: Use Google Cloud Support for critical issues

## 🔐 Security Considerations

### Development Environment Security
- **Public Database**: Uses authorized networks (restrict in production)
- **Unauthenticated Services**: Cloud Run allows public access (add auth in production)
- **Secret Management**: Database credentials stored in Secret Manager
- **Network**: Basic network configuration (enhance with VPC in production)

### Production Security Checklist
- [ ] Private networking with VPC
- [ ] Identity and Access Management (IAM)
- [ ] Cloud Armor for DDoS protection
- [ ] TLS/SSL everywhere
- [ ] Audit logging enabled
- [ ] Regular security scans

## 📚 Additional Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [Memorystore Documentation](https://cloud.google.com/memorystore/docs)
- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google)
- [MeshAI SDK Documentation](../../README.md)

## 🎉 Success!

After successful deployment, you should have:
- ✅ Two running Cloud Run services
- ✅ PostgreSQL database with agent storage
- ✅ Redis cache for context management
- ✅ Basic monitoring and logging
- ✅ HTTPS endpoints with automatic SSL
- ✅ Estimated cost: ~$78/month

Your MeshAI SDK is now running on Google Cloud Platform! 🚀