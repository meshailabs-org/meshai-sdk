#!/bin/bash

# MeshAI SDK GCP Deployment Script - TEMPLATE
# 
# ⚠️  WARNING: This is a template script from the public repository.
# ⚠️  DO NOT deploy directly from here with your actual project ID.
# 
# Instructions:
# 1. Copy this entire gcp/ directory to a private location
# 2. Update terraform.tfvars with your actual project ID  
# 3. Run the deployment from your private copy
#
# Example:
#   mkdir ~/meshai-deployment-private
#   cp -r deployment/gcp/* ~/meshai-deployment-private/
#   cd ~/meshai-deployment-private
#   # Edit terraform/terraform.tfvars with your project ID
#   ./deploy.sh --project-id YOUR_PROJECT_ID

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=""
REGION="us-central1"
ENVIRONMENT="dev"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if required tools are installed
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command -v gcloud &> /dev/null; then
        missing_tools+=("gcloud")
    fi
    
    if ! command -v terraform &> /dev/null; then
        missing_tools+=("terraform")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        echo "Please install the missing tools and try again."
        echo ""
        echo "Installation guides:"
        echo "- gcloud: https://cloud.google.com/sdk/docs/install"
        echo "- terraform: https://terraform.io/downloads"
        echo "- docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    print_success "All prerequisites are installed"
}

# Function to get or validate project ID
setup_project() {
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${BLUE}Please enter your GCP Project ID:${NC}"
        read -r PROJECT_ID
    fi
    
    if [ -z "$PROJECT_ID" ]; then
        print_error "Project ID is required"
        exit 1
    fi
    
    print_status "Setting up project: $PROJECT_ID"
    
    # Set the project
    gcloud config set project "$PROJECT_ID"
    
    # Verify we can access the project
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        print_error "Cannot access project $PROJECT_ID. Please check:"
        echo "1. Project ID is correct"
        echo "2. You have access to the project"
        echo "3. You are authenticated with gcloud (run 'gcloud auth login')"
        exit 1
    fi
    
    print_success "Project setup complete"
}

# Function to deploy infrastructure with Terraform
deploy_infrastructure() {
    print_status "Deploying infrastructure with Terraform..."
    
    cd deployment/gcp/terraform
    
    # Check if terraform.tfvars exists
    if [ ! -f "terraform.tfvars" ]; then
        print_warning "terraform.tfvars not found. Creating from example..."
        cp terraform.tfvars.example terraform.tfvars
        sed -i "s/your-gcp-project-id/$PROJECT_ID/g" terraform.tfvars
        print_warning "Please review and update terraform.tfvars if needed"
    fi
    
    # Initialize Terraform
    print_status "Initializing Terraform..."
    terraform init
    
    # Plan deployment
    print_status "Planning infrastructure deployment..."
    terraform plan -var="project_id=$PROJECT_ID" -var="region=$REGION" -var="environment=$ENVIRONMENT"
    
    # Ask for confirmation
    echo -e "${YELLOW}Do you want to proceed with the deployment? (y/N):${NC}"
    read -r confirmation
    
    if [[ $confirmation =~ ^[Yy]$ ]]; then
        # Apply configuration
        print_status "Deploying infrastructure..."
        terraform apply -auto-approve -var="project_id=$PROJECT_ID" -var="region=$REGION" -var="environment=$ENVIRONMENT"
        
        print_success "Infrastructure deployment complete"
    else
        print_warning "Infrastructure deployment cancelled"
        exit 0
    fi
    
    cd ../../..
}

# Function to build and deploy application
deploy_application() {
    print_status "Building and deploying application..."
    
    # Enable Cloud Build API if not already enabled
    gcloud services enable cloudbuild.googleapis.com
    
    # Submit build to Cloud Build
    print_status "Submitting build to Cloud Build..."
    gcloud builds submit --config=deployment/gcp/cloudbuild.yaml \
        --substitutions=_DEPLOY_REGION="$REGION" .
    
    print_success "Application deployment complete"
}

# Function to setup monitoring and logging
setup_monitoring() {
    print_status "Setting up basic monitoring..."
    
    # Create uptime checks for both services
    gcloud alpha monitoring uptime create "meshai-registry-uptime" \
        --display-name="MeshAI Registry Service" \
        --http-check-path="/health" \
        --hostname="meshai-registry-$(gcloud config get-value project).a.run.app" \
        --port=443 \
        --use-ssl \
        --period=300 \
        --timeout=10 || print_warning "Failed to create uptime check for Registry"
    
    gcloud alpha monitoring uptime create "meshai-runtime-uptime" \
        --display-name="MeshAI Runtime Service" \
        --http-check-path="/health" \
        --hostname="meshai-runtime-$(gcloud config get-value project).a.run.app" \
        --port=443 \
        --use-ssl \
        --period=300 \
        --timeout=10 || print_warning "Failed to create uptime check for Runtime"
    
    print_success "Basic monitoring setup complete"
}

# Function to display deployment information
display_deployment_info() {
    print_success "🎉 MeshAI SDK Development Environment Deployment Complete!"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                    DEPLOYMENT SUMMARY                      ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Get service URLs
    REGISTRY_URL=$(gcloud run services describe meshai-registry --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "Not deployed yet")
    RUNTIME_URL=$(gcloud run services describe meshai-runtime --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "Not deployed yet")
    
    echo "🌐 Service URLs:"
    echo "   Registry Service: $REGISTRY_URL"
    echo "   Runtime Service:  $RUNTIME_URL"
    echo ""
    
    echo "🔍 Health Check URLs:"
    echo "   Registry Health:  $REGISTRY_URL/health"
    echo "   Runtime Health:   $RUNTIME_URL/health"
    echo ""
    
    echo "📊 Monitoring:"
    echo "   Cloud Console: https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"
    echo "   Cloud Run:     https://console.cloud.google.com/run?project=$PROJECT_ID"
    echo ""
    
    echo "💰 Estimated Monthly Cost: ~$78"
    echo "   - Cloud Run (2 services):     ~$8.50"
    echo "   - Cloud SQL (f1-micro):       ~$17.50"
    echo "   - Memorystore Redis (1GB):    ~$26.00"
    echo "   - Load Balancer:              ~$18.30"
    echo "   - Monitoring & Other:         ~$7.70"
    echo ""
    
    echo "🔧 Next Steps:"
    echo "   1. Test the services using the health check URLs"
    echo "   2. Register your first agent using the Registry API"
    echo "   3. Submit a test task using the Runtime API"
    echo "   4. Monitor costs in the Cloud Console billing section"
    echo ""
    
    echo "📚 API Documentation:"
    echo "   Registry API:  $REGISTRY_URL/docs"
    echo "   Runtime API:   $RUNTIME_URL/docs"
    echo ""
    
    print_success "Deployment information displayed above"
}

# Function to run post-deployment tests
run_tests() {
    print_status "Running basic deployment tests..."
    
    # Wait a moment for services to be ready
    sleep 30
    
    REGISTRY_URL=$(gcloud run services describe meshai-registry --region="$REGION" --format="value(status.url)" 2>/dev/null)
    RUNTIME_URL=$(gcloud run services describe meshai-runtime --region="$REGION" --format="value(status.url)" 2>/dev/null)
    
    if [ -n "$REGISTRY_URL" ]; then
        print_status "Testing Registry Service..."
        if curl -s -f "$REGISTRY_URL/health" > /dev/null; then
            print_success "Registry Service is healthy"
        else
            print_warning "Registry Service health check failed"
        fi
    fi
    
    if [ -n "$RUNTIME_URL" ]; then
        print_status "Testing Runtime Service..."
        if curl -s -f "$RUNTIME_URL/health" > /dev/null; then
            print_success "Runtime Service is healthy"
        else
            print_warning "Runtime Service health check failed"
        fi
    fi
}

# Main deployment function
main() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                 MeshAI SDK GCP Deployment                ║"
    echo "║              Development Environment Setup               ║"
    echo "║                                                         ║"
    echo "║   This will deploy a minimal cost (~$78/month) GCP      ║"
    echo "║   environment with Cloud Run + Cloud SQL + Redis       ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    # Check command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --project-id ID     GCP Project ID"
                echo "  --region REGION     GCP Region (default: us-central1)"
                echo "  --environment ENV   Environment name (default: dev)"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run deployment steps
    check_prerequisites
    setup_project
    deploy_infrastructure
    deploy_application
    setup_monitoring
    run_tests
    display_deployment_info
    
    print_success "🚀 MeshAI SDK deployment completed successfully!"
}

# Run main function
main "$@"