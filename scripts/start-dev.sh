#!/bin/bash
"""
Development Environment Startup Script

Starts all MeshAI services in development mode with hot reload.
"""

# Set development environment variables
export MESHAI_DEBUG=true
export MESHAI_LOG_LEVEL=DEBUG
export MESHAI_REGISTRY_URL=http://localhost:8001
export MESHAI_RUNTIME_URL=http://localhost:8002
export MESHAI_ENABLE_METRICS=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 MeshAI Development Environment${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Registry Service:${NC} http://localhost:8001"
echo -e "${GREEN}Runtime Service:${NC}  http://localhost:8002"
echo -e "${GREEN}Health Checks:${NC}    /health endpoints available"
echo -e "${GREEN}Metrics:${NC}         /metrics endpoints available"
echo -e "${GREEN}API Docs:${NC}        /docs for interactive API documentation"
echo -e "${BLUE}========================================${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "scripts/start-all.py" ]; then
    echo -e "${RED}❌ Please run this script from the MeshAI SDK root directory${NC}"
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}📦 Installing dependencies...${NC}"
    python3 -m pip install -r requirements.txt
fi

# Start all services
echo -e "${GREEN}🚀 Starting all services...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

python3 scripts/start-all.py