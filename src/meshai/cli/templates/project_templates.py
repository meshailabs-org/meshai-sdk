"""
MeshAI Project Templates

Comprehensive project templates for different use cases.
"""

from typing import Dict, Any
from pathlib import Path


def get_project_template(template_type: str, project_name: str) -> Dict[str, str]:
    """
    Get project template files based on template type.
    
    Args:
        template_type: Type of template (simple, multi-agent, production, research)
        project_name: Name of the project
        
    Returns:
        Dictionary mapping file paths to file contents
    """
    
    templates = {
        "simple": get_simple_template,
        "multi-agent": get_multi_agent_template,
        "production": get_production_template,
        "research": get_research_template,
        "e-commerce": get_ecommerce_template,
        "support": get_support_template,
        "analytics": get_analytics_template
    }
    
    template_fn = templates.get(template_type, get_simple_template)
    return template_fn(project_name)


def get_simple_template(project_name: str) -> Dict[str, str]:
    """Simple single-agent application template"""
    
    files = {}
    
    # main.py
    files['main.py'] = f'''"""
{project_name} - Simple MeshAI Application

A single-agent application using MeshAI SDK.
"""

import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from meshai.core.registry import MeshRegistry
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.config import MeshConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main application entry point"""
    logger.info("Starting {project_name}...")
    
    # Load configuration
    config = MeshConfig()
    
    # Create registry
    registry = MeshRegistry(config=config)
    
    # Create and register agent
    from agents.assistant import AssistantAgent
    agent = AssistantAgent()
    await registry.register_agent(agent)
    
    logger.info(f"Agent '{{agent.name}}' registered successfully!")
    
    # Example interaction
    context = MeshContext()
    
    # Interactive loop
    print("\\n🤖 {project_name} is ready!")
    print("Type 'quit' to exit\\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            # Process user input
            task = TaskData(input=user_input)
            result = await agent.handle_task(task, context)
            
            print(f"Agent: {{result.get('result', 'No response')}}\\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error processing request: {{e}}")
    
    print("\\n👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
'''

    # agents/assistant.py
    files['agents/__init__.py'] = ''
    files['agents/assistant.py'] = f'''"""
Assistant Agent

Main agent for the {project_name} application.
"""

from typing import Dict, Any
from meshai.adapters.openai_adapter import OpenAIMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData


class AssistantAgent(OpenAIMeshAgent):
    """
    General-purpose assistant agent.
    """
    
    def __init__(self):
        super().__init__(
            model="gpt-3.5-turbo",
            agent_id="assistant-agent",
            name="Assistant",
            capabilities=["conversation", "help", "information"],
            system_prompt=(
                "You are a helpful AI assistant. "
                "Be concise, accurate, and professional in your responses."
            ),
            temperature=0.7
        )
    
    async def handle_task(
        self, 
        task_data: TaskData, 
        context: MeshContext
    ) -> Dict[str, Any]:
        """Handle incoming tasks"""
        # Custom pre-processing
        
        # Call parent handler
        result = await super().handle_task(task_data, context)
        
        # Custom post-processing
        
        return result
'''

    # config.yaml
    files['config.yaml'] = '''# MeshAI Configuration

meshai:
  # Registry settings
  registry:
    url: "http://localhost:8000"
    timeout: 30
  
  # Runtime settings  
  runtime:
    url: "http://localhost:8001"
    max_concurrent_tasks: 10
  
  # Logging
  logging:
    level: INFO
    format: json
  
  # Performance
  performance:
    enable_metrics: true
    enable_caching: true
    cache_ttl: 3600

# Agent configurations
agents:
  default:
    temperature: 0.7
    max_tokens: 2000
    timeout: 60
'''

    # requirements.txt
    files['requirements.txt'] = '''# MeshAI SDK
meshai-sdk

# Environment management
python-dotenv

# AI Framework
openai

# Additional utilities
pyyaml
'''

    # .env.example
    files['.env.example'] = '''# API Keys
OPENAI_API_KEY=your-openai-api-key-here

# MeshAI Settings
MESHAI_LOG_LEVEL=INFO
MESHAI_REGISTRY_URL=http://localhost:8000
MESHAI_RUNTIME_URL=http://localhost:8001

# Development
MESHAI_DEV_MODE=true
'''

    # README.md
    files['README.md'] = f'''# {project_name}

A simple MeshAI application with a single agent.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Project Structure

```
{project_name}/
├── main.py              # Entry point
├── agents/              # Agent definitions
│   └── assistant.py     # Main assistant agent
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
└── .env.example         # Environment template
```

## Development

Start development server:
```bash
meshai dev server --watch
```

Test the agent:
```bash
meshai agent test assistant-agent
```
'''

    return files


def get_multi_agent_template(project_name: str) -> Dict[str, str]:
    """Multi-agent workflow application template"""
    
    files = {}
    
    # main.py
    files['main.py'] = f'''"""
{project_name} - Multi-Agent MeshAI Application

A multi-agent workflow application using MeshAI SDK.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from meshai.core.registry import MeshRegistry
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.config import MeshConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrates multi-agent workflows"""
    
    def __init__(self, registry: MeshRegistry):
        self.registry = registry
        self.agents = {{}}
        
    async def register_agents(self):
        """Register all agents"""
        from agents.researcher import ResearcherAgent
        from agents.analyzer import AnalyzerAgent
        from agents.writer import WriterAgent
        
        # Create agents
        researcher = ResearcherAgent()
        analyzer = AnalyzerAgent()
        writer = WriterAgent()
        
        # Register with MeshAI
        await self.registry.register_agent(researcher)
        await self.registry.register_agent(analyzer)
        await self.registry.register_agent(writer)
        
        # Store references
        self.agents['researcher'] = researcher
        self.agents['analyzer'] = analyzer
        self.agents['writer'] = writer
        
        logger.info(f"Registered {{len(self.agents)}} agents")
        
    async def run_workflow(self, user_input: str) -> str:
        """Run the multi-agent workflow"""
        context = MeshContext()
        
        # Step 1: Research
        logger.info("Step 1: Researching...")
        research_task = TaskData(
            input=f"Research information about: {{user_input}}"
        )
        research_result = await self.agents['researcher'].handle_task(
            research_task, context
        )
        
        # Step 2: Analysis
        logger.info("Step 2: Analyzing...")
        analysis_task = TaskData(
            input=f"Analyze the following research: {{research_result['result']}}"
        )
        analysis_result = await self.agents['analyzer'].handle_task(
            analysis_task, context
        )
        
        # Step 3: Report Writing
        logger.info("Step 3: Writing report...")
        writing_task = TaskData(
            input=f"Write a comprehensive report based on: {{analysis_result['result']}}"
        )
        writing_result = await self.agents['writer'].handle_task(
            writing_task, context
        )
        
        return writing_result['result']


async def main():
    """Main application entry point"""
    logger.info("Starting {project_name}...")
    
    # Load configuration
    config = MeshConfig()
    
    # Create registry
    registry = MeshRegistry(config=config)
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(registry)
    await orchestrator.register_agents()
    
    # Interactive loop
    print("\\n🤖 {project_name} Multi-Agent System Ready!")
    print("Enter a topic to research and analyze")
    print("Type 'quit' to exit\\n")
    
    while True:
        try:
            user_input = input("Topic: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            print("\\n🔄 Processing with multi-agent workflow...")
            result = await orchestrator.run_workflow(user_input)
            
            print(f"\\n📋 Final Report:\\n{{result}}\\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Workflow error: {{e}}")
    
    print("\\n👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
'''

    # agents/__init__.py
    files['agents/__init__.py'] = '"""Agent definitions for multi-agent workflow"""'
    
    # agents/researcher.py
    files['agents/researcher.py'] = '''"""
Researcher Agent

Specializes in gathering and synthesizing information.
"""

from typing import Dict, Any
from meshai.adapters.google_adapter import GoogleMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData


class ResearcherAgent(GoogleMeshAgent):
    """Agent specialized in research and information gathering"""
    
    def __init__(self):
        super().__init__(
            model="gemini-pro",
            agent_id="researcher",
            name="Research Specialist",
            capabilities=["research", "information-gathering", "fact-checking"],
            system_prompt=(
                "You are a research specialist. Your role is to gather "
                "comprehensive, accurate information on topics. "
                "Focus on finding relevant facts, statistics, and insights."
            ),
            temperature=0.3
        )
'''

    # agents/analyzer.py
    files['agents/analyzer.py'] = '''"""
Analyzer Agent

Specializes in data analysis and insight extraction.
"""

from typing import Dict, Any
from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData


class AnalyzerAgent(AnthropicMeshAgent):
    """Agent specialized in analysis and insights"""
    
    def __init__(self):
        super().__init__(
            model="claude-3-sonnet-20240229",
            agent_id="analyzer",
            name="Data Analyst",
            capabilities=["analysis", "insights", "pattern-recognition"],
            system_prompt=(
                "You are a data analyst expert. Your role is to analyze "
                "information, identify patterns, extract insights, and "
                "provide strategic recommendations based on data."
            ),
            temperature=0.2
        )
'''

    # agents/writer.py
    files['agents/writer.py'] = '''"""
Writer Agent

Specializes in content creation and report writing.
"""

from typing import Dict, Any
from meshai.adapters.openai_adapter import OpenAIMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData


class WriterAgent(OpenAIMeshAgent):
    """Agent specialized in writing and content creation"""
    
    def __init__(self):
        super().__init__(
            model="gpt-4",
            agent_id="writer",
            name="Content Writer",
            capabilities=["writing", "content-creation", "reporting"],
            system_prompt=(
                "You are a professional content writer. Your role is to "
                "create clear, engaging, well-structured content based on "
                "research and analysis. Focus on clarity and readability."
            ),
            temperature=0.7
        )
'''

    # workflows/__init__.py
    files['workflows/__init__.py'] = ''
    
    # workflows/research_workflow.py
    files['workflows/research_workflow.py'] = '''"""
Research Workflow

Coordinates multi-agent research, analysis, and reporting.
"""

import asyncio
from typing import Dict, Any, List
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData


class ResearchWorkflow:
    """Manages research workflow across multiple agents"""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        
    async def execute(
        self, 
        topic: str, 
        context: MeshContext
    ) -> Dict[str, Any]:
        """Execute the complete research workflow"""
        
        results = {}
        
        # Phase 1: Parallel research from multiple perspectives
        research_tasks = [
            self._research_general(topic, context),
            self._research_technical(topic, context),
            self._research_business(topic, context)
        ]
        
        research_results = await asyncio.gather(*research_tasks)
        results['research'] = research_results
        
        # Phase 2: Comprehensive analysis
        combined_research = "\\n".join([r['content'] for r in research_results])
        analysis = await self._analyze_research(combined_research, context)
        results['analysis'] = analysis
        
        # Phase 3: Report generation
        report = await self._generate_report(analysis, context)
        results['report'] = report
        
        return results
    
    async def _research_general(self, topic: str, context: MeshContext):
        """General research perspective"""
        task = TaskData(
            input=f"Research general information about: {topic}"
        )
        result = await self.agents['researcher'].handle_task(task, context)
        return {'type': 'general', 'content': result['result']}
    
    async def _research_technical(self, topic: str, context: MeshContext):
        """Technical research perspective"""
        task = TaskData(
            input=f"Research technical aspects of: {topic}"
        )
        result = await self.agents['researcher'].handle_task(task, context)
        return {'type': 'technical', 'content': result['result']}
    
    async def _research_business(self, topic: str, context: MeshContext):
        """Business research perspective"""
        task = TaskData(
            input=f"Research business implications of: {topic}"
        )
        result = await self.agents['researcher'].handle_task(task, context)
        return {'type': 'business', 'content': result['result']}
    
    async def _analyze_research(self, research: str, context: MeshContext):
        """Analyze combined research"""
        task = TaskData(
            input=f"Analyze and extract key insights from: {research}"
        )
        result = await self.agents['analyzer'].handle_task(task, context)
        return result['result']
    
    async def _generate_report(self, analysis: str, context: MeshContext):
        """Generate final report"""
        task = TaskData(
            input=f"Generate a comprehensive report based on: {analysis}"
        )
        result = await self.agents['writer'].handle_task(task, context)
        return result['result']
'''

    # config.yaml
    files['config.yaml'] = '''# MeshAI Configuration

meshai:
  # Registry settings
  registry:
    url: "http://localhost:8000"
    timeout: 30
  
  # Runtime settings  
  runtime:
    url: "http://localhost:8001"
    max_concurrent_tasks: 10
  
  # Logging
  logging:
    level: INFO
    format: json
  
  # Performance
  performance:
    enable_metrics: true
    enable_caching: true
    cache_ttl: 3600

# Agent configurations
agents:
  researcher:
    temperature: 0.3
    max_tokens: 2000
  
  analyzer:
    temperature: 0.2
    max_tokens: 2500
  
  writer:
    temperature: 0.7
    max_tokens: 3000

# Workflow settings
workflows:
  research:
    max_parallel_tasks: 3
    timeout: 120
'''

    # requirements.txt
    files['requirements.txt'] = '''# MeshAI SDK
meshai-sdk

# Environment management
python-dotenv

# AI Frameworks
openai
anthropic
google-generativeai

# Additional utilities
pyyaml
asyncio
'''

    # .env.example
    files['.env.example'] = '''# API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# MeshAI Settings
MESHAI_LOG_LEVEL=INFO
MESHAI_REGISTRY_URL=http://localhost:8000
MESHAI_RUNTIME_URL=http://localhost:8001

# Development
MESHAI_DEV_MODE=true
'''

    # README.md
    files['README.md'] = f'''# {project_name}

A multi-agent MeshAI application with research, analysis, and writing capabilities.

## Architecture

This application uses three specialized agents:
- **Researcher** (Google Gemini): Information gathering
- **Analyzer** (Claude): Data analysis and insights
- **Writer** (GPT-4): Content creation and reporting

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Project Structure

```
{project_name}/
├── main.py              # Entry point and orchestrator
├── agents/              # Agent definitions
│   ├── researcher.py    # Research agent
│   ├── analyzer.py      # Analysis agent
│   └── writer.py        # Writing agent
├── workflows/           # Workflow definitions
│   └── research_workflow.py
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
└── .env.example         # Environment template
```

## Usage

The application runs a multi-agent workflow:
1. Research phase - Gathers information
2. Analysis phase - Extracts insights
3. Writing phase - Creates report

## Development

Test individual agents:
```bash
meshai agent test researcher
meshai agent test analyzer
meshai agent test writer
```

Run with debug logging:
```bash
MESHAI_LOG_LEVEL=DEBUG python main.py
```
'''

    return files


def get_production_template(project_name: str) -> Dict[str, str]:
    """Production-ready application template with Docker and monitoring"""
    
    files = get_multi_agent_template(project_name)  # Start with multi-agent base
    
    # Add production-specific files
    
    # Dockerfile
    files['Dockerfile'] = f'''FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 meshai && chown -R meshai:meshai /app
USER meshai

# Expose ports
EXPOSE 8080
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
'''

    # docker-compose.yml
    files['docker-compose.yml'] = f'''version: '3.8'

services:
  # Main application
  {project_name.lower().replace('-', '_')}:
    build: .
    container_name: {project_name.lower()}
    ports:
      - "8080:8080"
      - "9090:9090"  # Metrics port
    environment:
      - MESHAI_LOG_LEVEL=${{MESHAI_LOG_LEVEL:-INFO}}
      - MESHAI_REGISTRY_URL=${{MESHAI_REGISTRY_URL:-http://registry:8000}}
      - MESHAI_RUNTIME_URL=${{MESHAI_RUNTIME_URL:-http://runtime:8001}}
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - redis
      - postgres
    networks:
      - meshai-network
    restart: unless-stopped

  # Registry service
  registry:
    image: meshai/registry:latest
    container_name: meshai-registry
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://meshai:meshai@postgres:5432/meshai
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - meshai-network
    restart: unless-stopped

  # Runtime service
  runtime:
    image: meshai/runtime:latest
    container_name: meshai-runtime
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://meshai:meshai@postgres:5432/meshai
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - meshai-network
    restart: unless-stopped

  # Redis for caching and context
  redis:
    image: redis:7-alpine
    container_name: meshai-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - meshai-network
    restart: unless-stopped
    command: redis-server --appendonly yes

  # PostgreSQL for persistence
  postgres:
    image: postgres:15-alpine
    container_name: meshai-postgres
    environment:
      POSTGRES_DB: meshai
      POSTGRES_USER: meshai
      POSTGRES_PASSWORD: meshai
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - meshai-network
    restart: unless-stopped

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: meshai-prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - meshai-network
    restart: unless-stopped

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: meshai-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=redis-datasource
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - meshai-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  meshai-network:
    driver: bridge
'''

    # monitoring/prometheus.yml
    files['monitoring/prometheus.yml'] = '''global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'meshai-app'
    static_configs:
      - targets: ['app:9090']
        labels:
          service: 'application'
  
  - job_name: 'meshai-registry'
    static_configs:
      - targets: ['registry:9090']
        labels:
          service: 'registry'
  
  - job_name: 'meshai-runtime'
    static_configs:
      - targets: ['runtime:9090']
        labels:
          service: 'runtime'
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
        labels:
          service: 'redis'
'''

    # .github/workflows/deploy.yml
    files['.github/workflows/deploy.yml'] = f'''name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  PROJECT_ID: ${{{{ secrets.GCP_PROJECT_ID }}}}
  SERVICE_NAME: {project_name.lower()}
  REGION: us-central1

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=. --cov-report=xml
      env:
        MESHAI_TEST_MODE: true
    
    - name: Build Docker image
      run: |
        docker build -t $SERVICE_NAME:${{{{ github.sha }}}} .
    
    - name: Deploy to Cloud Run
      if: github.ref == 'refs/heads/main'
      run: |
        echo "Deploy to production"
        # Add deployment commands here
'''

    # Makefile
    files['Makefile'] = '''# MeshAI Production Makefile

.PHONY: help build run test deploy clean

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make run      - Run with docker-compose"
	@echo "  make test     - Run tests"
	@echo "  make deploy   - Deploy to production"
	@echo "  make clean    - Clean up containers and volumes"

build:
	docker-compose build

run:
	docker-compose up -d

logs:
	docker-compose logs -f

test:
	pytest tests/ --cov=. --cov-report=term-missing

deploy:
	./scripts/deploy.sh

stop:
	docker-compose down

clean:
	docker-compose down -v
	rm -rf logs/ data/

monitor:
	open http://localhost:3000  # Grafana
	open http://localhost:9091  # Prometheus
'''

    # scripts/deploy.sh
    files['scripts/deploy.sh'] = '''#!/bin/bash
set -e

echo "🚀 Deploying to production..."

# Build and push Docker image
docker build -t meshai-app:latest .
docker tag meshai-app:latest gcr.io/$PROJECT_ID/meshai-app:latest
docker push gcr.io/$PROJECT_ID/meshai-app:latest

# Deploy to Cloud Run
gcloud run deploy meshai-app \\
  --image gcr.io/$PROJECT_ID/meshai-app:latest \\
  --platform managed \\
  --region us-central1 \\
  --allow-unauthenticated

echo "✅ Deployment complete!"
'''

    return files


def get_research_template(project_name: str) -> Dict[str, str]:
    """Research and analysis workflow template"""
    
    files = {}
    
    # main.py focused on research
    files['main.py'] = f'''"""
{project_name} - Research & Analysis Platform

Advanced research and analysis using specialized AI agents.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from meshai.core.registry import MeshRegistry
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.config import MeshConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResearchPlatform:
    """Advanced research and analysis platform"""
    
    def __init__(self):
        self.config = MeshConfig()
        self.registry = MeshRegistry(self.config)
        self.agents = {{}}
        self.research_cache = {{}}
        
    async def initialize(self):
        """Initialize all research agents"""
        from researchers.academic_researcher import AcademicResearcher
        from researchers.market_researcher import MarketResearcher
        from researchers.technical_researcher import TechnicalResearcher
        from analyzers.data_analyst import DataAnalyst
        from analyzers.trend_analyst import TrendAnalyst
        
        # Create specialized researchers
        agents_to_register = [
            AcademicResearcher(),
            MarketResearcher(),
            TechnicalResearcher(),
            DataAnalyst(),
            TrendAnalyst()
        ]
        
        for agent in agents_to_register:
            await self.registry.register_agent(agent)
            self.agents[agent.agent_id] = agent
            
        logger.info(f"Initialized {{len(self.agents)}} research agents")
    
    async def conduct_research(
        self, 
        topic: str,
        research_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Conduct research on a topic"""
        
        context = MeshContext()
        results = {{
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "research_type": research_type,
            "findings": {{}}
        }}
        
        if research_type == "comprehensive":
            # Run all research types in parallel
            research_tasks = [
                self._academic_research(topic, context),
                self._market_research(topic, context),
                self._technical_research(topic, context)
            ]
            
            research_results = await asyncio.gather(*research_tasks)
            
            results["findings"]["academic"] = research_results[0]
            results["findings"]["market"] = research_results[1]
            results["findings"]["technical"] = research_results[2]
            
            # Analyze combined findings
            analysis = await self._analyze_findings(results["findings"], context)
            results["analysis"] = analysis
            
            # Generate trends
            trends = await self._identify_trends(analysis, context)
            results["trends"] = trends
            
        return results
    
    async def _academic_research(self, topic: str, context: MeshContext):
        """Conduct academic research"""
        task = TaskData(
            input=f"Conduct academic research on: {{topic}}. "
                  "Focus on peer-reviewed sources, academic papers, and scholarly insights."
        )
        result = await self.agents['academic-researcher'].handle_task(task, context)
        return result['result']
    
    async def _market_research(self, topic: str, context: MeshContext):
        """Conduct market research"""
        task = TaskData(
            input=f"Conduct market research on: {{topic}}. "
                  "Focus on market size, competitors, trends, and opportunities."
        )
        result = await self.agents['market-researcher'].handle_task(task, context)
        return result['result']
    
    async def _technical_research(self, topic: str, context: MeshContext):
        """Conduct technical research"""
        task = TaskData(
            input=f"Conduct technical research on: {{topic}}. "
                  "Focus on technical specifications, implementations, and best practices."
        )
        result = await self.agents['technical-researcher'].handle_task(task, context)
        return result['result']
    
    async def _analyze_findings(self, findings: Dict, context: MeshContext):
        """Analyze research findings"""
        task = TaskData(
            input=f"Analyze these research findings and extract key insights: {{findings}}"
        )
        result = await self.agents['data-analyst'].handle_task(task, context)
        return result['result']
    
    async def _identify_trends(self, analysis: str, context: MeshContext):
        """Identify trends from analysis"""
        task = TaskData(
            input=f"Identify key trends and future predictions from: {{analysis}}"
        )
        result = await self.agents['trend-analyst'].handle_task(task, context)
        return result['result']
    
    def save_research(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save research results to file"""
        if not filename:
            filename = f"research_{{results['topic'].replace(' ', '_')}}_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.json"
        
        output_dir = Path("research_output")
        output_dir.mkdir(exist_ok=True)
        
        import json
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Research saved to {{output_dir / filename}}")


async def main():
    """Main entry point"""
    platform = ResearchPlatform()
    await platform.initialize()
    
    print("\\n🔬 Research & Analysis Platform")
    print("=" * 50)
    print("Commands:")
    print("  research <topic> - Conduct comprehensive research")
    print("  analyze <topic>  - Quick analysis")
    print("  trends <topic>   - Trend identification")
    print("  quit            - Exit\\n")
    
    while True:
        try:
            user_input = input("Command: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            parts = user_input.split(' ', 1)
            if len(parts) < 2:
                print("Please provide a topic")
                continue
            
            command, topic = parts
            
            print(f"\\n🔄 Researching: {{topic}}...")
            
            if command == "research":
                results = await platform.conduct_research(topic, "comprehensive")
            elif command == "analyze":
                results = await platform.conduct_research(topic, "analysis")
            elif command == "trends":
                results = await platform.conduct_research(topic, "trends")
            else:
                print("Unknown command")
                continue
            
            # Display results
            print("\\n📊 Research Results:")
            print("-" * 50)
            
            if "findings" in results:
                for category, content in results["findings"].items():
                    print(f"\\n[{{category.upper()}}]")
                    print(content[:500] + "..." if len(content) > 500 else content)
            
            if "analysis" in results:
                print("\\n[ANALYSIS]")
                print(results["analysis"][:500] + "...")
            
            if "trends" in results:
                print("\\n[TRENDS]")
                print(results["trends"][:500] + "...")
            
            # Save results
            platform.save_research(results)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {{e}}")
    
    print("\\n👋 Research session ended")


if __name__ == "__main__":
    asyncio.run(main())
'''

    # Create researcher agents
    files['researchers/__init__.py'] = ''
    files['researchers/academic_researcher.py'] = '''"""Academic Researcher Agent"""

from meshai.adapters.google_adapter import GoogleMeshAgent

class AcademicResearcher(GoogleMeshAgent):
    def __init__(self):
        super().__init__(
            model="gemini-pro",
            agent_id="academic-researcher",
            name="Academic Researcher",
            capabilities=["academic-research", "literature-review", "citation"],
            system_prompt="You are an academic researcher specializing in finding and analyzing scholarly sources.",
            temperature=0.2
        )
'''

    files['researchers/market_researcher.py'] = '''"""Market Researcher Agent"""

from meshai.adapters.openai_adapter import OpenAIMeshAgent

class MarketResearcher(OpenAIMeshAgent):
    def __init__(self):
        super().__init__(
            model="gpt-4",
            agent_id="market-researcher",
            name="Market Researcher",
            capabilities=["market-research", "competitive-analysis", "trend-spotting"],
            system_prompt="You are a market research expert analyzing business opportunities and market dynamics.",
            temperature=0.4
        )
'''

    files['researchers/technical_researcher.py'] = '''"""Technical Researcher Agent"""

from meshai.adapters.anthropic_adapter import AnthropicMeshAgent

class TechnicalResearcher(AnthropicMeshAgent):
    def __init__(self):
        super().__init__(
            model="claude-3-sonnet-20240229",
            agent_id="technical-researcher", 
            name="Technical Researcher",
            capabilities=["technical-research", "implementation-details", "best-practices"],
            system_prompt="You are a technical researcher focusing on implementation details and technical specifications.",
            temperature=0.1
        )
'''

    # Create analyzer agents
    files['analyzers/__init__.py'] = ''
    files['analyzers/data_analyst.py'] = '''"""Data Analyst Agent"""

from meshai.adapters.openai_adapter import OpenAIMeshAgent

class DataAnalyst(OpenAIMeshAgent):
    def __init__(self):
        super().__init__(
            model="gpt-4",
            agent_id="data-analyst",
            name="Data Analyst",
            capabilities=["data-analysis", "statistical-analysis", "insight-extraction"],
            system_prompt="You are a data analyst expert in statistical analysis and extracting insights from research.",
            temperature=0.2
        )
'''

    files['analyzers/trend_analyst.py'] = '''"""Trend Analyst Agent"""

from meshai.adapters.anthropic_adapter import AnthropicMeshAgent

class TrendAnalyst(AnthropicMeshAgent):
    def __init__(self):
        super().__init__(
            model="claude-3-sonnet-20240229",
            agent_id="trend-analyst",
            name="Trend Analyst", 
            capabilities=["trend-analysis", "forecasting", "pattern-recognition"],
            system_prompt="You are a trend analyst identifying patterns and making future predictions.",
            temperature=0.3
        )
'''

    # Standard files
    files['requirements.txt'] = '''# MeshAI SDK
meshai-sdk

# AI Frameworks
openai
anthropic
google-generativeai

# Data handling
pandas
numpy
matplotlib

# Utilities
python-dotenv
pyyaml
'''

    files['.env.example'] = '''# API Keys
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
GOOGLE_API_KEY=your-key

# MeshAI
MESHAI_LOG_LEVEL=INFO
'''

    return files


def get_ecommerce_template(project_name: str) -> Dict[str, str]:
    """E-commerce assistant template"""
    
    files = {}
    
    files['main.py'] = f'''"""
{project_name} - E-commerce AI Assistant

Multi-agent system for e-commerce operations.
"""

import asyncio
from meshai.core.registry import MeshRegistry
from meshai.core.context import MeshContext

async def main():
    # E-commerce specific implementation
    pass

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    # Add other e-commerce specific files
    return files


def get_support_template(project_name: str) -> Dict[str, str]:
    """Customer support template"""
    
    files = {}
    
    files['main.py'] = f'''"""
{project_name} - AI Customer Support System

Intelligent customer support with multi-agent collaboration.
"""

import asyncio
from meshai.core.registry import MeshRegistry
from meshai.core.context import MeshContext

async def main():
    # Support system implementation
    pass

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return files


def get_analytics_template(project_name: str) -> Dict[str, str]:
    """Data analytics template"""
    
    files = {}
    
    files['main.py'] = f'''"""
{project_name} - AI Data Analytics Platform

Advanced analytics with specialized AI agents.
"""

import asyncio
from meshai.core.registry import MeshRegistry
from meshai.core.context import MeshContext

async def main():
    # Analytics platform implementation
    pass

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return files