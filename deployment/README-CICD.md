# CI/CD Pipeline Documentation

## Overview

MeshAI SDK uses a comprehensive CI/CD pipeline with multiple testing stages, automated deployments, and monitoring.

## Pipeline Components

### 1. GitHub Actions (`.github/workflows/ci-cd.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`

**Pipeline Stages:**

1. **Test & Code Quality**
   - Python 3.11 testing environment
   - PostgreSQL and Redis services for integration tests
   - Code formatting checks (Black, isort)
   - Linting (flake8, mypy) 
   - Unit tests with coverage reporting
   - Performance benchmarks

2. **Security Scanning**
   - Bandit security analysis
   - Safety dependency vulnerability check
   - Results uploaded as artifacts

3. **Docker Build**
   - Multi-platform Docker image build
   - Push to Google Container Registry
   - Image caching for faster builds

4. **Integration Testing**
   - Tests against live API endpoints
   - Only runs on main branch pushes

5. **Production Deployment**
   - Deploys to Google Cloud Run
   - Updates both Registry and Runtime services
   - Post-deployment health checks
   - Only on main branch with manual approval

### 2. Google Cloud Build (`deployment/gcp/cloudbuild.yaml`)

**Features:**
- Tests run before building Docker images
- Security scanning integrated
- Secrets management via Google Secret Manager
- Automated deployment to Cloud Run
- Post-deployment integration tests

**Secrets Configuration:**
```yaml
availableSecrets:
  secretManager:
    - database-url
    - redis-url
```

## Local Development Workflow

### Setup
```bash
# Install dependencies
make install

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Quick tests
make test

# All tests (unit, integration, performance)
make test-all

# Integration tests only
make integration

# Performance tests
make performance
```

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Security checks
make security

# Pre-commit checks
make pre-commit
```

### Local Services
```bash
# Start services locally
make local

# Stop services
make local-stop

# Check API health
make health
```

## Deployment

### Manual Deployment
```bash
# Deploy via Cloud Build
make deploy

# Or direct Cloud Build
gcloud builds submit --config deployment/gcp/cloudbuild.yaml .
```

### Automated Deployment
1. Push changes to `main` branch
2. GitHub Actions automatically triggers
3. All tests must pass
4. Manual approval required for production deployment
5. Automated health checks verify deployment

## Test Organization

### Unit Tests (`tests/`)
- Fast-running tests
- No external dependencies
- Code coverage tracking

### Integration Tests (`tests/integration/`)
- Test against live API endpoints
- Database and Redis integration
- Load balancer routing verification

### Performance Tests (`tests/benchmarks/`)
- Response time benchmarks
- Concurrent request handling
- Resource usage monitoring

## Monitoring & Alerts

### Built-in Monitoring
- Application metrics via Prometheus
- Health check endpoints
- Performance monitoring
- Error tracking

### Google Cloud Monitoring
- Uptime checks on critical endpoints
- Alert policies for service health
- Custom dashboards
- Log-based metrics

## Security

### Code Security
- Bandit static analysis
- Dependency vulnerability scanning
- Pre-commit security hooks
- Secret scanning

### Infrastructure Security
- Secrets stored in Google Secret Manager
- VPC networking for database access
- IAM roles and permissions
- HTTPS enforcement

## Configuration

### Environment Variables
- `MESHAI_DATABASE_URL`: PostgreSQL connection string
- `MESHAI_REDIS_URL`: Redis connection string
- `MESHAI_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENVIRONMENT`: Deployment environment (development, staging, production)

### GitHub Secrets Required
- `GCP_SERVICE_ACCOUNT_KEY`: Google Cloud service account key
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection string

### Google Cloud Secrets
```bash
# Create secrets
gcloud secrets create database-url --data-file=-
gcloud secrets create redis-url --data-file=-
```

## Troubleshooting

### Common Issues

1. **Tests failing locally but passing in CI**
   - Check Python version compatibility
   - Verify dependency versions match CI environment

2. **Deployment failures**
   - Check Google Cloud credentials
   - Verify secrets are properly configured
   - Review Cloud Build logs

3. **Integration test failures**
   - Ensure API endpoints are accessible
   - Check load balancer configuration
   - Verify SSL certificate status

### Debug Commands
```bash
# Check service health
curl https://api.meshai.dev/health

# View Cloud Build logs
gcloud builds log --region=us-central1

# Check Cloud Run service status
gcloud run services describe meshai-registry --region=us-central1

# Test integration endpoints locally
pytest tests/integration/ -v --tb=short
```

## Metrics & KPIs

### Development Metrics
- Test coverage percentage
- Build success rate
- Deployment frequency
- Mean time to recovery

### Application Metrics
- API response times
- Error rates
- Active agents count
- Task completion rates

### Infrastructure Metrics
- Service uptime
- Resource utilization
- Database performance
- Load balancer metrics

## Continuous Improvement

### Planned Enhancements
1. Automated performance regression detection
2. Blue-green deployment strategy
3. Automated rollback on failure
4. Enhanced security scanning
5. Multi-region deployment support

### Contributing

1. Create feature branch from `develop`
2. Run local tests: `make test-all`
3. Submit pull request to `main`
4. CI pipeline automatically runs
5. Manual code review required
6. Merge triggers automated deployment