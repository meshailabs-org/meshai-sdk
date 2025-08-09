#!/bin/bash
# Create Cloud Build trigger for automated CI/CD

set -e

PROJECT_ID=${PROJECT_ID:-meshv1}
REPO_NAME=${REPO_NAME:-meshai-sdk}
BRANCH_NAME=${BRANCH_NAME:-main}

echo "Creating Cloud Build trigger for MeshAI SDK..."
echo "Project: $PROJECT_ID"
echo "Repository: $REPO_NAME"
echo "Branch: $BRANCH_NAME"

# Create the trigger
gcloud builds triggers create github \
  --project=$PROJECT_ID \
  --repo-name=$REPO_NAME \
  --repo-owner=meshailabs \
  --branch-pattern="^$BRANCH_NAME$" \
  --build-config=deployment/gcp/cloudbuild.yaml \
  --name=meshai-sdk-ci-cd \
  --description="Automated CI/CD pipeline for MeshAI SDK"

echo "✅ Cloud Build trigger created successfully!"
echo "🔗 View triggers at: https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT_ID"
echo ""
echo "Next steps:"
echo "1. Push code changes to the $BRANCH_NAME branch"
echo "2. Cloud Build will automatically trigger"
echo "3. Tests will run, Docker image will build"
echo "4. Services will deploy to Cloud Run"
echo "5. Integration tests will verify deployment"