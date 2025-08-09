# MeshAI SDK Deployment

This directory contains deployment templates and examples for various cloud providers and environments.

## ⚠️ Security Notice

**These are templates and examples only.** For actual deployments:

1. **Copy deployment files to a private repository or local directory**
2. **Never commit sensitive information like:**
   - Project IDs
   - API keys
   - Database passwords
   - Service account keys

## 🌥️ Available Deployment Options

### Google Cloud Platform (GCP)
- **Location**: `gcp/`
- **Cost**: ~$78/month (development)
- **Services**: Cloud Run + Cloud SQL + Memorystore
- **Documentation**: [gcp/README.md](gcp/README.md)

### AWS (Coming Soon)
- **Cost**: ~$85/month (development) 
- **Services**: ECS Fargate + RDS + ElastiCache

### Azure (Coming Soon)
- **Cost**: ~$95/month (development)
- **Services**: Container Instances + PostgreSQL + Redis

## 🚀 Recommended Deployment Workflow

### Option 1: Private Deployment Repository (Recommended)

1. **Create a private repository for your deployment:**
   ```bash
   mkdir ~/meshai-deployment-private
   cd ~/meshai-deployment-private
   git init
   ```

2. **Copy deployment templates:**
   ```bash
   cp -r /path/to/meshai-sdk/deployment/* .
   ```

3. **Update configuration files with your values:**
   ```bash
   # Edit terraform.tfvars with your project ID
   cp gcp/terraform/terraform.tfvars.example gcp/terraform/terraform.tfvars
   vim gcp/terraform/terraform.tfvars
   ```

4. **Deploy:**
   ```bash
   ./gcp/deploy.sh --project-id YOUR_PROJECT_ID
   ```

### Option 2: Local Deployment Directory

1. **Create local deployment directory:**
   ```bash
   mkdir ~/meshai-deployment
   cp -r deployment/* ~/meshai-deployment/
   cd ~/meshai-deployment
   ```

2. **Configure and deploy:**
   ```bash
   # Update configuration
   cp gcp/terraform/terraform.tfvars.example gcp/terraform/terraform.tfvars
   # Edit with your values...
   
   # Deploy
   ./gcp/deploy.sh --project-id YOUR_PROJECT_ID
   ```

### Option 3: Environment Variables (For CI/CD)

Use environment variables instead of config files:
```bash
export TF_VAR_project_id="your-project-id"
export TF_VAR_region="us-central1"
export TF_VAR_environment="dev"

./gcp/deploy.sh
```

## 📁 What to Keep Private

### Always Private:
- `terraform.tfvars` - Contains project IDs and sensitive config
- `*.key` files - Service account keys
- `.env` files - Environment variables
- `terraform.tfstate*` - Terraform state files (contain secrets)

### Safe to Share (Templates):
- `*.tf` - Terraform infrastructure code (no secrets)
- `*.yaml` - Kubernetes/Docker configs (no secrets)
- `deploy.sh` - Deployment scripts (no hardcoded secrets)
- `README.md` - Documentation

## 🔒 Security Best Practices

1. **Use Secret Managers**: Google Secret Manager, AWS Secrets Manager, etc.
2. **Service Accounts**: Don't use personal credentials for deployments
3. **Private Networks**: Use VPCs and private networking in production
4. **Access Controls**: Implement proper IAM and RBAC
5. **Audit Logging**: Enable audit logs for all infrastructure changes

## 🏗️ Infrastructure as Code

All deployments use Infrastructure as Code (IaC) principles:
- **Terraform** for infrastructure provisioning
- **Cloud Build/GitHub Actions** for CI/CD
- **Docker** for containerization
- **Kubernetes/Cloud Run** for orchestration

## 📚 Next Steps

1. Choose your cloud provider
2. Copy deployment templates to a private location
3. Update configuration with your values
4. Follow the provider-specific README for deployment
5. Set up monitoring and alerting
6. Configure backup and disaster recovery

For detailed deployment instructions, see the provider-specific README files.