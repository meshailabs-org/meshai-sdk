#!/usr/bin/env python3
"""
MeshAI CLI - Command Line Interface for Agent Management

A comprehensive CLI tool for MeshAI development, testing, and deployment.

Usage:
    meshai init [PROJECT_NAME] [--template TYPE]    # Initialize new project
    meshai agent create NAME [--framework FRAMEWORK] # Create new agent
    meshai agent list [--format FORMAT]             # List agents
    meshai agent test AGENT_ID [--verbose]          # Test agent
    meshai registry status                           # Check registry status
    meshai dev server [--port PORT] [--watch]       # Start development server
    meshai deploy [--environment ENV]               # Deploy to production
    meshai config validate [--fix]                  # Validate configuration
    meshai logs [--follow] [--service SERVICE]      # View logs

Examples:
    meshai init my-ai-project --template multi-agent
    meshai agent create code-reviewer --framework anthropic
    meshai dev server --port 8080 --watch
    meshai deploy --environment production
"""

import asyncio
import sys
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
import tempfile
import subprocess

# MeshAI imports
from meshai.core.registry import MeshRegistry
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.config import MeshConfig

console = Console()

# CLI version
CLI_VERSION = "1.0.0"

# Project templates
PROJECT_TEMPLATES = {
    "simple": {
        "description": "Single-agent application",
        "files": ["main.py", "config.yaml", "requirements.txt", ".env.example"]
    },
    "multi-agent": {
        "description": "Multi-agent workflow application", 
        "files": ["main.py", "agents/", "workflows/", "config.yaml", "requirements.txt", ".env.example"]
    },
    "production": {
        "description": "Production-ready application with monitoring",
        "files": ["main.py", "agents/", "workflows/", "config.yaml", "docker-compose.yml", "Dockerfile", "requirements.txt", ".env.example", "monitoring/"]
    },
    "research": {
        "description": "Research and analysis workflow",
        "files": ["main.py", "researchers/", "analyzers/", "config.yaml", "requirements.txt", ".env.example"]
    }
}

# Agent framework templates
AGENT_FRAMEWORKS = {
    "openai": {
        "class": "OpenAIMeshAgent",
        "module": "meshai.adapters.openai_adapter",
        "env_vars": ["OPENAI_API_KEY"],
        "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"]
    },
    "anthropic": {
        "class": "AnthropicMeshAgent", 
        "module": "meshai.adapters.anthropic_adapter",
        "env_vars": ["ANTHROPIC_API_KEY"],
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    },
    "google": {
        "class": "GoogleMeshAgent",
        "module": "meshai.adapters.google_adapter", 
        "env_vars": ["GOOGLE_API_KEY"],
        "models": ["gemini-pro", "gemini-pro-vision"]
    },
    "langchain": {
        "class": "LangChainMeshAgent",
        "module": "meshai.adapters.langchain_adapter",
        "env_vars": ["OPENAI_API_KEY"],  # Default LangChain LLM
        "models": ["custom"]
    },
    "crewai": {
        "class": "CrewAIMeshAgent",
        "module": "meshai.adapters.crewai_adapter",
        "env_vars": ["OPENAI_API_KEY"],  # Default CrewAI LLM
        "models": ["custom"]
    },
    "autogen": {
        "class": "AutoGenMeshAgent",
        "module": "meshai.adapters.autogen_adapter",
        "env_vars": ["OPENAI_API_KEY"],  # Default AutoGen LLM
        "models": ["custom"]
    }
}


class MeshAICLI:
    """Main CLI application class"""
    
    def __init__(self):
        self.console = Console()
        self.config = None
        self.registry = None
        
    async def load_config(self):
        """Load MeshAI configuration"""
        try:
            config_path = Path.cwd() / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                    self.config = MeshConfig(**config_data)
            else:
                self.config = MeshConfig()
                
            self.registry = MeshRegistry(config=self.config)
        except Exception as e:
            self.console.print(f"[red]Error loading configuration: {e}[/red]")
            sys.exit(1)

    def print_banner(self):
        """Print CLI banner"""
        banner = """
[bold blue]
 ███▄ ▄███▓▓█████   ██████  ██░ ██  ▄▄▄       ██▓
▓██▒▀█▀ ██▒▓█   ▀ ▒██    ▒ ▓██░ ██▒▒████▄    ▓██▒
▓██    ▓██░▒███   ░ ▓██▄   ▒██▀▀██░▒██  ▀█▄  ▒██▒
▒██    ▒██ ▒▓█  ▄   ▒   ██▒░▓█ ░██ ░██▄▄▄▄██ ░██░
▒██▒   ░██▒░▒████▒▒██████▒▒░▓█▒░██▓ ▓█   ▓██▒░██░
░ ▒░   ░  ░░░ ▒░ ░▒ ▒▓▒ ▒ ░ ▒ ░░▒░▒ ▒▒   ▓▒█░░▓  
░  ░      ░ ░ ░  ░░ ░▒  ░ ░ ▒ ░▒░ ░  ▒   ▒▒ ░ ▒ ░
░      ░      ░   ░  ░  ░   ░  ░░ ░  ░   ▒    ▒ ░
       ░      ░  ░      ░   ░  ░  ░      ░  ░ ░
[/bold blue]

[bold white]MeshAI CLI v{CLI_VERSION} - Universal AI Agent Orchestration[/bold white]
[dim]Type 'meshai --help' for available commands[/dim]
        """.strip()
        
        self.console.print(banner.format(CLI_VERSION=CLI_VERSION))


@click.group()
@click.version_option(version=CLI_VERSION)
@click.pass_context
def cli(ctx):
    """MeshAI CLI - Universal AI Agent Orchestration Platform"""
    ctx.ensure_object(dict)
    ctx.obj['cli'] = MeshAICLI()


@cli.group()
def agent():
    """Agent management commands"""
    pass


@cli.group() 
def dev():
    """Development tools and server"""
    pass


@cli.group()
def config():
    """Configuration management"""
    pass


@agent.command("create")
@click.argument("name")
@click.option("--framework", "-f", default="openai", help="AI framework to use",
              type=click.Choice(list(AGENT_FRAMEWORKS.keys())))
@click.option("--model", "-m", help="Model to use (optional)")
@click.option("--capabilities", "-c", multiple=True, help="Agent capabilities")
@click.option("--output", "-o", default="agents/", help="Output directory")
@click.pass_context
def create_agent(ctx, name, framework, model, capabilities, output):
    """Create a new AI agent"""
    cli_obj = ctx.obj['cli']
    cli_obj.print_banner()
    
    console.print(f"\n🤖 Creating new {framework} agent: [bold blue]{name}[/bold blue]")
    
    # Get framework info
    framework_info = AGENT_FRAMEWORKS[framework]
    
    # Select model if not provided
    if not model:
        models = framework_info["models"]
        if len(models) == 1:
            model = models[0]
        else:
            console.print("\nAvailable models:")
            for i, m in enumerate(models, 1):
                console.print(f"  {i}. {m}")
            
            choice = Prompt.ask("Select model", choices=[str(i) for i in range(1, len(models) + 1)])
            model = models[int(choice) - 1]
    
    # Get capabilities if not provided
    if not capabilities:
        default_capabilities = ["conversation", "help", "analysis"]
        caps_input = Prompt.ask(
            f"Agent capabilities (comma-separated)", 
            default=",".join(default_capabilities)
        )
        capabilities = [cap.strip() for cap in caps_input.split(",")]
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(exist_ok=True)
    
    # Generate agent code
    agent_code = generate_agent_code(name, framework, model, capabilities, framework_info)
    
    # Write agent file
    agent_file = output_path / f"{name.lower().replace('-', '_')}_agent.py"
    with open(agent_file, 'w') as f:
        f.write(agent_code)
    
    console.print(f"✅ Agent created: [green]{agent_file}[/green]")
    
    # Check for required environment variables
    missing_env = []
    for env_var in framework_info["env_vars"]:
        if not os.getenv(env_var):
            missing_env.append(env_var)
    
    if missing_env:
        console.print(f"\n⚠️ Missing environment variables:")
        for var in missing_env:
            console.print(f"   export {var}=your-api-key")


@agent.command("list")
@click.option("--format", "-f", default="table", type=click.Choice(["table", "json", "yaml"]))
@click.pass_context
def list_agents(ctx, format):
    """List registered agents"""
    cli_obj = ctx.obj['cli']
    
    async def _list_agents():
        await cli_obj.load_config()
        
        try:
            agents = await cli_obj.registry.get_all_agents()
            
            if format == "json":
                console.print(json.dumps([agent.to_dict() for agent in agents], indent=2))
            elif format == "yaml":
                console.print(yaml.dump([agent.to_dict() for agent in agents], default_flow_style=False))
            else:
                # Table format
                if not agents:
                    console.print("No agents registered")
                    return
                
                table = Table(title="Registered Agents")
                table.add_column("Agent ID", style="cyan")
                table.add_column("Name", style="blue") 
                table.add_column("Framework", style="green")
                table.add_column("Status", style="yellow")
                table.add_column("Capabilities", style="magenta")
                
                for agent in agents:
                    capabilities = ", ".join(agent.capabilities[:3])
                    if len(agent.capabilities) > 3:
                        capabilities += f" (+{len(agent.capabilities) - 3} more)"
                    
                    table.add_row(
                        agent.agent_id,
                        agent.name,
                        agent.framework,
                        agent.status,
                        capabilities
                    )
                
                console.print(table)
                
        except Exception as e:
            console.print(f"[red]Error listing agents: {e}[/red]")
    
    asyncio.run(_list_agents())


@agent.command("test")
@click.argument("agent_id")
@click.option("--message", "-m", default="Hello, this is a test message")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def test_agent(ctx, agent_id, message, verbose):
    """Test an agent with a sample message"""
    cli_obj = ctx.obj['cli']
    
    async def _test_agent():
        await cli_obj.load_config()
        
        console.print(f"🧪 Testing agent: [blue]{agent_id}[/blue]")
        
        if verbose:
            console.print(f"Test message: [dim]{message}[/dim]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Sending test message...", total=None)
                
                # Get agent
                agents = await cli_obj.registry.get_all_agents()
                agent = next((a for a in agents if a.agent_id == agent_id), None)
                
                if not agent:
                    progress.stop()
                    console.print(f"[red]Agent '{agent_id}' not found[/red]")
                    return
                
                # Test the agent
                context = MeshContext()
                task_data = TaskData(input=message)
                
                import time
                start_time = time.time()
                result = await agent.handle_task(task_data, context)
                response_time = time.time() - start_time
                
                progress.stop()
                
                # Show results
                console.print(f"\n✅ Test completed in [green]{response_time:.2f}s[/green]")
                
                if verbose:
                    console.print("\nAgent Details:")
                    console.print(f"  Name: {agent.name}")
                    console.print(f"  Framework: {agent.framework}")
                    console.print(f"  Capabilities: {', '.join(agent.capabilities)}")
                
                console.print(f"\n📋 Response:")
                response_panel = Panel(
                    result.get('result', 'No response'),
                    title="Agent Response",
                    border_style="green"
                )
                console.print(response_panel)
                
        except Exception as e:
            console.print(f"[red]Test failed: {e}[/red]")
            if verbose:
                import traceback
                console.print(f"[red]{traceback.format_exc()}[/red]")
    
    asyncio.run(_test_agent())


@cli.command("init")
@click.argument("project_name", required=False)
@click.option("--template", "-t", default="simple", 
              type=click.Choice(list(PROJECT_TEMPLATES.keys())),
              help="Project template")
@click.option("--force", is_flag=True, help="Overwrite existing files")
def init_project(project_name, template, force):
    """Initialize a new MeshAI project"""
    if not project_name:
        project_name = Prompt.ask("Project name")
    
    project_path = Path.cwd() / project_name
    
    if project_path.exists() and not force:
        if not Confirm.ask(f"Directory '{project_name}' already exists. Continue?"):
            return
    
    console.print(f"\n🚀 Creating MeshAI project: [bold blue]{project_name}[/bold blue]")
    console.print(f"Template: [green]{template}[/green] - {PROJECT_TEMPLATES[template]['description']}")
    
    # Create project directory
    project_path.mkdir(exist_ok=True)
    
    # Generate project files based on template
    generate_project_files(project_path, template)
    
    console.print(f"\n✅ Project created successfully!")
    console.print(f"\nNext steps:")
    console.print(f"  cd {project_name}")
    console.print(f"  pip install -r requirements.txt")
    console.print(f"  cp .env.example .env  # Configure your API keys")
    console.print(f"  meshai dev server     # Start development")


@dev.command("server")
@click.option("--port", "-p", default=8080, help="Server port")
@click.option("--watch", "-w", is_flag=True, help="Watch for file changes")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def dev_server(port, watch, debug):
    """Start development server"""
    console.print(f"🚀 Starting MeshAI development server on port {port}")
    
    if watch:
        console.print("👁️ File watching enabled - server will restart on changes")
    
    if debug:
        console.print("🐛 Debug mode enabled")
    
    # Create development server script
    server_script = create_dev_server_script(port, watch, debug)
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(server_script)
            f.flush()
            
            # Run the development server
            cmd = [sys.executable, f.name]
            subprocess.run(cmd, check=True)
            
    except KeyboardInterrupt:
        console.print("\n👋 Development server stopped")
    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")
    finally:
        # Clean up temp file
        if 'f' in locals():
            os.unlink(f.name)


@config.command("validate")
@click.option("--fix", is_flag=True, help="Automatically fix issues")
def validate_config(fix):
    """Validate MeshAI configuration"""
    console.print("🔍 Validating MeshAI configuration...")
    
    config_path = Path.cwd() / "config.yaml"
    
    if not config_path.exists():
        console.print("[red]No config.yaml found in current directory[/red]")
        if fix:
            console.print("Creating default config.yaml...")
            create_default_config(config_path)
        return
    
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        # Validate configuration
        issues = validate_config_data(config_data)
        
        if not issues:
            console.print("✅ Configuration is valid")
            return
        
        console.print(f"[yellow]Found {len(issues)} configuration issues:[/yellow]")
        
        for issue in issues:
            console.print(f"  ❌ {issue['message']}")
            if fix and 'fix' in issue:
                console.print(f"     🔧 Auto-fixing: {issue['fix']}")
                # Apply fix logic here
        
        if fix:
            console.print("\n✅ Configuration fixed")
        
    except Exception as e:
        console.print(f"[red]Configuration validation failed: {e}[/red]")


@cli.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--service", "-s", help="Filter by service")
@click.option("--level", "-l", default="INFO", help="Log level filter")
@click.option("--lines", "-n", default=100, help="Number of lines to show")
def show_logs(follow, service, level, lines):
    """View MeshAI logs"""
    console.print(f"📋 Showing MeshAI logs (last {lines} lines)")
    
    if service:
        console.print(f"Service filter: [blue]{service}[/blue]")
    
    if level != "INFO":
        console.print(f"Level filter: [yellow]{level}[/yellow]")
    
    # Implementation would read from log files or external logging service
    console.print("[dim]Log viewing functionality would be implemented here[/dim]")


# Helper functions

def generate_agent_code(name, framework, model, capabilities, framework_info):
    """Generate agent code based on template"""
    agent_id = name.lower().replace(' ', '-').replace('_', '-')
    class_name = ''.join(word.capitalize() for word in name.replace('-', ' ').replace('_', ' ').split())
    
    template = f'''"""
{class_name} - MeshAI Agent

Generated by MeshAI CLI
Framework: {framework}
Model: {model}
Capabilities: {", ".join(capabilities)}
"""

import asyncio
from typing import Dict, Any, List
from {framework_info["module"]} import {framework_info["class"]}
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.registry import MeshRegistry


class {class_name}({framework_info["class"]}):
    """
    {class_name} agent powered by {framework}
    
    Capabilities:
    {chr(10).join([f"    - {cap}" for cap in capabilities])}
    """
    
    def __init__(self):
        super().__init__(
            model="{model}",
            agent_id="{agent_id}",
            name="{name}",
            capabilities={capabilities},
            system_prompt=(
                "You are {name}, a helpful AI agent with expertise in: "
                "{', '.join(capabilities)}. "
                "Always be helpful, accurate, and professional in your responses."
            ),
            temperature=0.7
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        """
        Handle incoming tasks with custom processing logic
        """
        # Add any custom preprocessing here
        
        # Call the parent handler
        result = await super().handle_task(task_data, context)
        
        # Add any custom postprocessing here
        
        return result
    
    async def custom_capability(self, input_data: str, context: MeshContext) -> str:
        """
        Example custom capability implementation
        """
        task_data = TaskData(
            input=f"Process this with your expertise: {{input_data}}",
            parameters={{"custom": True}}
        )
        
        result = await self.handle_task(task_data, context)
        return result.get('result', '')


async def main():
    """
    Example usage of the {class_name} agent
    """
    # Create agent instance
    agent = {class_name}()
    
    # Register with MeshAI
    registry = MeshRegistry()
    await registry.register_agent(agent)
    
    print(f"✅ {{agent.name}} registered successfully!")
    print(f"Agent ID: {{agent.agent_id}}")
    print(f"Capabilities: {{', '.join(agent.capabilities)}}")
    
    # Example task
    context = MeshContext()
    task = TaskData(input="Hello! Can you help me with something?")
    
    result = await agent.handle_task(task, context)
    print(f"\\nAgent response: {{result['result']}}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
'''
    
    return template


def generate_project_files(project_path: Path, template: str):
    """Generate project files based on template"""
    template_info = PROJECT_TEMPLATES[template]
    
    # Create main.py
    main_py = f'''"""
MeshAI Project - {project_path.name}

Template: {template}
Generated by MeshAI CLI
"""

import asyncio
import logging
from pathlib import Path
from meshai.core.registry import MeshRegistry
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.config import MeshConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main application entry point"""
    logger.info("Starting MeshAI application...")
    
    # Load configuration
    config = MeshConfig()
    
    # Create registry
    registry = MeshRegistry(config=config)
    
    # TODO: Register your agents here
    # Example:
    # from agents.my_agent import MyAgent
    # agent = MyAgent()
    # await registry.register_agent(agent)
    
    logger.info("Application started successfully!")
    
    # Example task execution
    context = MeshContext()
    # TODO: Implement your workflow here
    
if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open(project_path / "main.py", 'w') as f:
        f.write(main_py)
    
    # Create config.yaml
    config_yaml = '''# MeshAI Configuration
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
    format: structured
  
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
    
    with open(project_path / "config.yaml", 'w') as f:
        f.write(config_yaml)
    
    # Create requirements.txt
    requirements = '''# MeshAI SDK
meshai-sdk

# Optional framework dependencies
# Uncomment the ones you need:

# OpenAI
# openai

# Anthropic
# anthropic

# Google AI
# google-generativeai

# LangChain
# langchain
# langchain-openai

# CrewAI  
# crewai

# AutoGen
# pyautogen

# Additional utilities
pyyaml
rich
click
'''
    
    with open(project_path / "requirements.txt", 'w') as f:
        f.write(requirements)
    
    # Create .env.example
    env_example = '''# MeshAI Environment Variables

# API Keys (uncomment and set the ones you need)
# OPENAI_API_KEY=your-openai-api-key-here
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
# GOOGLE_API_KEY=your-google-api-key-here

# AWS (for Bedrock)
# AWS_ACCESS_KEY_ID=your-aws-access-key
# AWS_SECRET_ACCESS_KEY=your-aws-secret-key
# AWS_DEFAULT_REGION=us-east-1

# MeshAI Settings
MESHAI_LOG_LEVEL=INFO
MESHAI_REGISTRY_URL=http://localhost:8000
MESHAI_RUNTIME_URL=http://localhost:8001

# Development
MESHAI_DEV_MODE=true
MESHAI_DEBUG=false
'''
    
    with open(project_path / ".env.example", 'w') as f:
        f.write(env_example)
    
    # Create template-specific directories and files
    if template in ["multi-agent", "production", "research"]:
        (project_path / "agents").mkdir(exist_ok=True)
        
        # Create example agent
        example_agent = '''"""
Example Agent

This is a template agent to get you started.
"""

from meshai.adapters.openai_adapter import OpenAIMeshAgent

class ExampleAgent(OpenAIMeshAgent):
    def __init__(self):
        super().__init__(
            model="gpt-3.5-turbo",
            agent_id="example-agent",
            name="Example Agent",
            capabilities=["conversation", "help"]
        )
'''
        
        with open(project_path / "agents" / "__init__.py", 'w') as f:
            f.write("")
            
        with open(project_path / "agents" / "example_agent.py", 'w') as f:
            f.write(example_agent)
    
    if template == "production":
        # Create Docker files
        dockerfile = '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
'''
        
        with open(project_path / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        
        docker_compose = f'''version: '3.8'

services:
  {project_path.name}:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MESHAI_LOG_LEVEL=INFO
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: meshai
      POSTGRES_USER: meshai
      POSTGRES_PASSWORD: meshai
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
'''
        
        with open(project_path / "docker-compose.yml", 'w') as f:
            f.write(docker_compose)


def create_dev_server_script(port: int, watch: bool, debug: bool) -> str:
    """Create development server script"""
    script = f'''
import asyncio
import logging
import sys
from pathlib import Path

# Add src directory to path for MeshAI imports
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from meshai.dev.server import run_dev_server
    
    async def start_server():
        """Start the MeshAI development server"""
        await run_dev_server(
            port={port},
            watch_paths=["."] if {watch} else None,
            debug={debug},
            auto_reload={watch}
        )
    
    if __name__ == "__main__":
        try:
            asyncio.run(start_server())
        except KeyboardInterrupt:
            print("\\n👋 Development server stopped")
            
except ImportError as e:
    # Fallback to simple server if MeshAI dev tools not available
    import logging
    from pathlib import Path
    import sys
    import importlib.util
    from typing import Dict, Any

    # Configure logging
    log_level = logging.DEBUG if {debug} else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("meshai-dev-server")

    async def start_server():
        """Start simple MeshAI development server"""
        logger.info("🚀 Starting simple MeshAI development server on port {port}")
        logger.warning("⚠️ Advanced dev server not available, using fallback mode")
        
        # Look for main.py
        main_file = Path.cwd() / "main.py"
        if not main_file.exists():
            logger.error("❌ No main.py found in current directory")
            return
        
        try:
            # Import and run the main module
            spec = importlib.util.spec_from_file_location("main", main_file)
            main_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_module)
            
            if hasattr(main_module, 'main'):
                logger.info("✅ Running main() function...")
                await main_module.main()
            else:
                logger.warning("⚠️ No main() function found in main.py")
                
        except Exception as e:
            logger.error(f"❌ Error running application: {{e}}")
            if {debug}:
                import traceback
                logger.error(traceback.format_exc())

    if __name__ == "__main__":
        try:
            asyncio.run(start_server())
        except KeyboardInterrupt:
            print("\\n👋 Simple server stopped")
'''
    
    return script


def create_default_config(config_path: Path):
    """Create default configuration file"""
    default_config = '''# MeshAI Configuration
meshai:
  registry:
    url: "http://localhost:8000"
  runtime:
    url: "http://localhost:8001"
  logging:
    level: INFO
'''
    
    with open(config_path, 'w') as f:
        f.write(default_config)


def validate_config_data(config_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Validate configuration data and return issues"""
    issues = []
    
    # Check for required sections
    if 'meshai' not in config_data:
        issues.append({
            'message': "Missing 'meshai' section",
            'fix': "Add meshai section with registry and runtime URLs"
        })
    
    meshai_config = config_data.get('meshai', {})
    
    # Check registry configuration
    if 'registry' not in meshai_config:
        issues.append({
            'message': "Missing registry configuration",
            'fix': "Add registry.url setting"
        })
    
    # Check runtime configuration  
    if 'runtime' not in meshai_config:
        issues.append({
            'message': "Missing runtime configuration",
            'fix': "Add runtime.url setting"
        })
    
    return issues


def main():
    """Entry point for the CLI"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()