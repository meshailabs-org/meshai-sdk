# ⚠️ DEPLOYMENT TEMPLATES - DO NOT USE DIRECTLY

## Important Security Notice

**These are deployment templates and examples from the public MeshAI SDK repository.**

**🚫 DO NOT deploy directly from this public repository location.**

## Why Not Deploy Directly?

1. **Security Risk**: You'd be putting sensitive information in a public repo
2. **Project ID Exposure**: Your GCP project ID would be visible to everyone
3. **Credential Risk**: Terraform state files contain sensitive data
4. **Best Practices**: Infrastructure should be managed privately

## ✅ Correct Deployment Process

### Step 1: Create Private Deployment Directory
```bash
mkdir ~/meshai-deployment-private
cd ~/meshai-deployment-private
```

### Step 2: Copy Templates
```bash
# Copy all deployment files to your private directory
cp -r /path/to/meshai-sdk/deployment/gcp/* .
```

### Step 3: Configure Your Environment
```bash
# Update with your actual project ID
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
vim terraform/terraform.tfvars  # Edit with your project ID
```

### Step 4: Deploy from Private Location
```bash
# Deploy from your private directory (NOT the public repo)
./deploy.sh --project-id YOUR_PROJECT_ID
```

## 📚 What These Templates Provide

- **Infrastructure as Code**: Terraform configurations
- **Deployment Automation**: Bash scripts for easy deployment  
- **Cloud Build**: CI/CD pipeline configurations
- **Documentation**: Complete setup and usage guides
- **Cost Optimization**: Minimal cost development environment (~$78/month)

## 🔐 Security Best Practices

1. **Private Repository**: Keep deployment configs in private repos
2. **Environment Variables**: Use env vars instead of hardcoded values
3. **Secret Management**: Use Google Secret Manager for credentials
4. **Access Control**: Limit who can deploy infrastructure
5. **Audit Logging**: Enable audit logs for all deployments

## 🆘 Need Help?

1. Read the [deployment README](README.md) for detailed instructions
2. Check the [main deployment documentation](../README.md)
3. Review the [MeshAI SDK documentation](../../README.md)

Remember: These templates are here to help you deploy securely and efficiently, but always use them from a private location with your actual configuration.