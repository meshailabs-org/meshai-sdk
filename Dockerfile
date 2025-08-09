# MeshAI SDK Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for SQLite (if used)
RUN mkdir -p /app/data

# Expose ports for Registry (8001) and Runtime (8002)
EXPOSE 8001 8002

# Health check - check service based on SERVICE_TYPE
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD if [ "$SERVICE_TYPE" = "runtime" ]; then curl -f http://localhost:${PORT:-8002}/health; else curl -f http://localhost:${PORT:-8001}/health; fi

# Default command - start GCP service (controlled by SERVICE_TYPE env var)
CMD ["python", "scripts/start-gcp-service.py"]