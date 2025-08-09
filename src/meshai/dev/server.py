"""
MeshAI Development Server

Hot-reload development server with real-time agent testing and debugging.
"""

import asyncio
import logging
import json
import os
import sys
import signal
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import importlib.util
import importlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .watcher import FileWatcher
from .reloader import HotReloader
from ..core.registry import MeshRegistry
from ..core.context import MeshContext
from ..core.schemas import TaskData
from ..core.config import MeshConfig

logger = logging.getLogger(__name__)


class DevServer:
    """
    MeshAI Development Server with hot-reload capabilities
    
    Features:
    - Hot-reload on file changes
    - Real-time agent testing
    - WebSocket-based dashboard
    - Interactive debugging
    - Performance monitoring
    - Live configuration updates
    """
    
    def __init__(
        self,
        port: int = 8080,
        watch_paths: Optional[List[str]] = None,
        debug: bool = True,
        auto_reload: bool = True
    ):
        self.port = port
        self.debug = debug
        self.auto_reload = auto_reload
        self.watch_paths = watch_paths or ["."]
        
        # Server components
        self.app = FastAPI(title="MeshAI Dev Server", debug=debug)
        self.registry = None
        self.config = None
        self.hot_reloader = None
        self.file_watcher = None
        
        # State tracking
        self.agents = {}
        self.active_connections = []
        self.task_history = []
        self.performance_metrics = {
            "requests": 0,
            "errors": 0,
            "avg_response_time": 0,
            "uptime_start": datetime.now()
        }
        
        # Setup server
        self._setup_server()
        self._setup_routes()
        self._setup_websockets()
    
    def _setup_server(self):
        """Configure FastAPI server"""
        
        # CORS for development
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Static files for dashboard
        static_path = Path(__file__).parent / "static"
        if static_path.exists():
            self.app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def dashboard():
            """Development dashboard"""
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.get("/health")
        async def health():
            """Health check"""
            return {"status": "healthy", "uptime": self._get_uptime()}
        
        @self.app.get("/agents")
        async def list_agents():
            """List registered agents"""
            if not self.registry:
                return {"agents": []}
            
            try:
                agents = await self.registry.get_all_agents()
                return {
                    "agents": [
                        {
                            "id": agent.agent_id,
                            "name": agent.name,
                            "framework": agent.framework,
                            "capabilities": agent.capabilities,
                            "status": "active"
                        }
                        for agent in agents
                    ]
                }
            except Exception as e:
                logger.error(f"Error listing agents: {e}")
                return {"agents": [], "error": str(e)}
        
        @self.app.post("/agents/{agent_id}/test")
        async def test_agent(agent_id: str, request_data: dict, background_tasks: BackgroundTasks):
            """Test an agent with a message"""
            
            if not self.registry:
                raise HTTPException(status_code=500, detail="Registry not initialized")
            
            message = request_data.get("message", "Hello, this is a test")
            start_time = datetime.now()
            
            try:
                # Get agent
                agents = await self.registry.get_all_agents()
                agent = next((a for a in agents if a.agent_id == agent_id), None)
                
                if not agent:
                    raise HTTPException(status_code=404, detail="Agent not found")
                
                # Create task
                context = MeshContext()
                task_data = TaskData(input=message)
                
                # Execute task
                result = await agent.handle_task(task_data, context)
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Track performance
                self.performance_metrics["requests"] += 1
                self._update_avg_response_time(response_time)
                
                # Store in history
                task_record = {
                    "timestamp": start_time.isoformat(),
                    "agent_id": agent_id,
                    "message": message,
                    "response": result.get("result", ""),
                    "response_time": response_time,
                    "success": True
                }
                
                self.task_history.append(task_record)
                
                # Notify WebSocket clients
                background_tasks.add_task(
                    self._broadcast_update, 
                    {"type": "task_completed", "data": task_record}
                )
                
                return {
                    "success": True,
                    "result": result,
                    "response_time": response_time,
                    "timestamp": start_time.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Agent test error: {e}")
                self.performance_metrics["errors"] += 1
                
                error_record = {
                    "timestamp": start_time.isoformat(),
                    "agent_id": agent_id,
                    "message": message,
                    "error": str(e),
                    "response_time": (datetime.now() - start_time).total_seconds(),
                    "success": False
                }
                
                self.task_history.append(error_record)
                
                background_tasks.add_task(
                    self._broadcast_update,
                    {"type": "task_failed", "data": error_record}
                )
                
                return {
                    "success": False,
                    "error": str(e),
                    "timestamp": start_time.isoformat()
                }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get performance metrics"""
            return {
                **self.performance_metrics,
                "uptime": self._get_uptime(),
                "recent_tasks": self.task_history[-10:] if self.task_history else []
            }
        
        @self.app.post("/reload")
        async def manual_reload(background_tasks: BackgroundTasks):
            """Manually trigger reload"""
            background_tasks.add_task(self._reload_application)
            return {"message": "Reload triggered"}
        
        @self.app.get("/logs")
        async def get_logs():
            """Get recent logs"""
            # This would integrate with actual logging system
            return {"logs": ["Sample log entry"]}
    
    def _setup_websockets(self):
        """Setup WebSocket endpoints for real-time updates"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                # Send initial state
                await websocket.send_json({
                    "type": "connected",
                    "data": {
                        "agents": len(self.agents),
                        "uptime": self._get_uptime()
                    }
                })
                
                # Keep connection alive
                while True:
                    try:
                        # Ping every 30 seconds
                        await asyncio.sleep(30)
                        await websocket.send_json({"type": "ping"})
                    except:
                        break
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
    
    async def initialize(self):
        """Initialize the development server"""
        
        logger.info("🚀 Initializing MeshAI Development Server...")
        
        try:
            # Load configuration
            self.config = MeshConfig()
            self.registry = MeshRegistry(config=self.config)
            
            # Initialize hot reloader if enabled
            if self.auto_reload:
                self.hot_reloader = HotReloader(
                    watch_paths=self.watch_paths,
                    reload_callback=self._reload_application
                )
                
                self.file_watcher = FileWatcher(
                    paths=self.watch_paths,
                    callback=self._on_file_change
                )
            
            # Load existing agents
            await self._discover_and_load_agents()
            
            logger.info(f"✅ Dev server initialized with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize dev server: {e}")
            raise
    
    async def _discover_and_load_agents(self):
        """Discover and load agents from the project"""
        
        logger.info("🔍 Discovering agents...")
        
        # Look for agents in common locations, avoiding SDK source directories
        search_paths = [
            Path("agents"),
            Path("src/agents"), 
            Path("app/agents"),
        ]
        
        # Only include current directory if it doesn't look like an SDK source
        current_path = Path(".")
        if not (current_path / "src" / "meshai").exists():
            search_paths.append(current_path)
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            logger.debug(f"Searching for agents in: {search_path}")
            
            for py_file in search_path.rglob("*.py"):
                # Skip files in virtual environments and package directories
                if self._should_skip_file(py_file):
                    continue
                    
                try:
                    await self._load_agent_from_file(py_file)
                except ImportError as e:
                    logger.debug(f"Skipping {py_file}: missing dependency - {e}")
                except Exception as e:
                    logger.warning(f"Error loading file {py_file}: {e}")
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during agent discovery"""
        
        # Convert to string for easier path matching
        path_str = str(file_path)
        path_parts = file_path.parts
        
        # Skip private files
        if file_path.name.startswith("_"):
            return True
        
        # Skip common non-agent files
        skip_files = {
            "setup.py", "test_dev_server.py", "conftest.py", 
            "__init__.py", "__main__.py"
        }
        if file_path.name in skip_files:
            return True
        
        # Skip virtual environment directories (common patterns)
        venv_patterns = {
            "venv", "env", ".venv", ".env", "virtualenv", 
            "meshai-env", "python-env", ".virtualenv"
        }
        for part in path_parts:
            if part in venv_patterns:
                return True
        
        # Skip site-packages and other Python installation directories
        install_patterns = {
            "site-packages", "dist-packages", ".eggs", "build", 
            "lib64", "__pycache__", ".pytest_cache"
        }
        for part in path_parts:
            if part in install_patterns:
                return True
        
        # Skip test directories and files
        test_patterns = {"test", "tests", "testing"}
        for part in path_parts:
            if part in test_patterns:
                return True
        
        # Skip if file contains "test" in name  
        if "test" in file_path.name.lower():
            return True
        
        # Skip common SDK structure patterns
        sdk_patterns = {"adapters", "cli", "utils", "core", "dev"}
        for part in path_parts:
            if part in sdk_patterns:
                return True
        
        # Skip hidden directories (starting with .)
        for part in path_parts:
            if part.startswith(".") and len(part) > 1:
                return True
        
        return False
    
    async def _load_agent_from_file(self, file_path: Path):
        """Load an agent from a Python file"""
        
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(
                f"agent_{file_path.stem}", 
                file_path
            )
            if spec is None or spec.loader is None:
                logger.debug(f"Cannot create spec for {file_path}")
                return
                
            module = importlib.util.module_from_spec(spec)
            
            # Execute module in a try/catch to handle import errors
            spec.loader.exec_module(module)
            
            # Look for agent classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Check if it's an agent class
                if (isinstance(attr, type) and 
                    hasattr(attr, 'agent_id') and 
                    hasattr(attr, 'handle_task')):
                    
                    try:
                        # Instantiate and register
                        agent_instance = attr()
                        await self.registry.register_agent(agent_instance)
                        
                        self.agents[agent_instance.agent_id] = {
                            "instance": agent_instance,
                            "file": str(file_path),
                            "class_name": attr_name
                        }
                        
                        logger.info(f"📋 Loaded agent: {agent_instance.name}")
                        
                    except Exception as e:
                        logger.warning(f"Could not instantiate agent {attr_name}: {e}")
                        
        except Exception as e:
            logger.warning(f"Error loading file {file_path}: {e}")
    
    async def _on_file_change(self, file_path: str, event_type: str):
        """Handle file change events"""
        
        logger.info(f"📝 File changed: {file_path} ({event_type})")
        
        # Broadcast to WebSocket clients
        await self._broadcast_update({
            "type": "file_changed",
            "data": {
                "file": file_path,
                "event": event_type,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        # Trigger reload if it's a Python file
        if file_path.endswith(".py"):
            await self._reload_application()
    
    async def _reload_application(self):
        """Reload the application"""
        
        logger.info("🔄 Reloading application...")
        
        try:
            # Clear current agents
            old_agent_count = len(self.agents)
            self.agents.clear()
            
            # Reload modules
            if self.hot_reloader:
                await self.hot_reloader.reload()
            
            # Rediscover agents
            await self._discover_and_load_agents()
            
            new_agent_count = len(self.agents)
            
            logger.info(f"✅ Reload complete: {old_agent_count} → {new_agent_count} agents")
            
            # Notify clients
            await self._broadcast_update({
                "type": "reload_complete",
                "data": {
                    "agents": new_agent_count,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Reload failed: {e}")
            await self._broadcast_update({
                "type": "reload_failed", 
                "data": {"error": str(e)}
            })
    
    async def _broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all WebSocket connections"""
        
        if not self.active_connections:
            return
        
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time metric"""
        
        current_avg = self.performance_metrics["avg_response_time"]
        total_requests = self.performance_metrics["requests"]
        
        new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        self.performance_metrics["avg_response_time"] = new_avg
    
    def _get_uptime(self) -> str:
        """Get server uptime as formatted string"""
        
        uptime = datetime.now() - self.performance_metrics["uptime_start"]
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{hours}h {minutes}m {seconds}s"
    
    def _get_dashboard_html(self) -> str:
        """Get the development dashboard HTML"""
        
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>MeshAI Dev Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .agents {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .agent {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .agent-name {
            font-weight: bold;
        }
        .agent-capabilities {
            color: #666;
            font-size: 14px;
        }
        .test-form {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .status {
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }
        .status.connected {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.disconnected {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .log-entry {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 10px;
            margin-bottom: 5px;
            font-family: monospace;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 MeshAI Development Dashboard</h1>
            <p>Real-time agent development and testing</p>
        </div>
        
        <div class="status" id="connectionStatus">
            <span id="statusText">Connecting...</span>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="agentCount">-</div>
                <div class="stat-label">Active Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="requestCount">-</div>
                <div class="stat-label">Total Requests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="responseTime">-</div>
                <div class="stat-label">Avg Response Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="uptime">-</div>
                <div class="stat-label">Uptime</div>
            </div>
        </div>
        
        <div class="agents">
            <h3>Registered Agents</h3>
            <div id="agentList">Loading...</div>
        </div>
        
        <div class="test-form">
            <h3>Test Agent</h3>
            <div class="form-group">
                <label>Agent:</label>
                <select id="testAgent">
                    <option value="">Select an agent...</option>
                </select>
            </div>
            <div class="form-group">
                <label>Message:</label>
                <textarea id="testMessage" rows="3" placeholder="Enter test message...">Hello! Can you help me test your capabilities?</textarea>
            </div>
            <button class="btn" onclick="testAgent()">Test Agent</button>
            <div id="testResult"></div>
        </div>
        
        <div class="agents">
            <h3>Recent Activity</h3>
            <div id="activityLog">No recent activity</div>
        </div>
    </div>

    <script>
        let ws;
        let agents = [];
        
        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                document.getElementById('connectionStatus').className = 'status connected';
                document.getElementById('statusText').textContent = '🟢 Connected to dev server';
                loadData();
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
            
            ws.onclose = function() {
                document.getElementById('connectionStatus').className = 'status disconnected';
                document.getElementById('statusText').textContent = '🔴 Disconnected from dev server';
                setTimeout(connect, 2000);
            };
        }
        
        function handleMessage(message) {
            switch(message.type) {
                case 'task_completed':
                    addLogEntry(`✅ Task completed: ${message.data.agent_id}`, 'success');
                    loadMetrics();
                    break;
                case 'task_failed':
                    addLogEntry(`❌ Task failed: ${message.data.agent_id} - ${message.data.error}`, 'error');
                    loadMetrics();
                    break;
                case 'file_changed':
                    addLogEntry(`📝 File changed: ${message.data.file}`, 'info');
                    break;
                case 'reload_complete':
                    addLogEntry('🔄 Application reloaded successfully', 'success');
                    loadAgents();
                    break;
                case 'reload_failed':
                    addLogEntry(`❌ Reload failed: ${message.data.error}`, 'error');
                    break;
            }
        }
        
        function addLogEntry(text, type = 'info') {
            const log = document.getElementById('activityLog');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
            log.insertBefore(entry, log.firstChild);
            
            // Keep only last 20 entries
            while(log.children.length > 20) {
                log.removeChild(log.lastChild);
            }
        }
        
        async function loadData() {
            await Promise.all([loadAgents(), loadMetrics()]);
        }
        
        async function loadAgents() {
            try {
                const response = await fetch('/agents');
                const data = await response.json();
                agents = data.agents || [];
                
                document.getElementById('agentCount').textContent = agents.length;
                
                const agentList = document.getElementById('agentList');
                const testAgent = document.getElementById('testAgent');
                
                if (agents.length === 0) {
                    agentList.innerHTML = '<p>No agents registered</p>';
                    testAgent.innerHTML = '<option value="">No agents available</option>';
                    return;
                }
                
                agentList.innerHTML = agents.map(agent => `
                    <div class="agent">
                        <div class="agent-name">${agent.name} (${agent.framework})</div>
                        <div class="agent-capabilities">Capabilities: ${agent.capabilities.join(', ')}</div>
                    </div>
                `).join('');
                
                testAgent.innerHTML = '<option value="">Select an agent...</option>' +
                    agents.map(agent => `<option value="${agent.id}">${agent.name}</option>`).join('');
                    
            } catch (error) {
                console.error('Error loading agents:', error);
            }
        }
        
        async function loadMetrics() {
            try {
                const response = await fetch('/metrics');
                const data = await response.json();
                
                document.getElementById('requestCount').textContent = data.requests || 0;
                document.getElementById('responseTime').textContent = 
                    data.avg_response_time ? `${data.avg_response_time.toFixed(2)}s` : '-';
                document.getElementById('uptime').textContent = data.uptime || '-';
                
            } catch (error) {
                console.error('Error loading metrics:', error);
            }
        }
        
        async function testAgent() {
            const agentId = document.getElementById('testAgent').value;
            const message = document.getElementById('testMessage').value;
            
            if (!agentId || !message) {
                alert('Please select an agent and enter a message');
                return;
            }
            
            const resultDiv = document.getElementById('testResult');
            resultDiv.innerHTML = '<div style="color: #666;">Testing...</div>';
            
            try {
                const response = await fetch(`/agents/${agentId}/test`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message})
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultDiv.innerHTML = `
                        <div style="background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; margin-top: 10px; border-radius: 4px;">
                            <strong>✅ Success</strong> (${result.response_time.toFixed(2)}s)<br>
                            <div style="margin-top: 10px; white-space: pre-wrap;">${result.result.result || 'No response'}</div>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; margin-top: 10px; border-radius: 4px;">
                            <strong>❌ Error</strong><br>
                            <div style="margin-top: 10px;">${result.error}</div>
                        </div>
                    `;
                }
            } catch (error) {
                resultDiv.innerHTML = `
                    <div style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; margin-top: 10px; border-radius: 4px;">
                        <strong>❌ Network Error</strong><br>
                        <div style="margin-top: 10px;">${error.message}</div>
                    </div>
                `;
            }
        }
        
        // Initialize
        connect();
        setInterval(loadMetrics, 30000); // Update metrics every 30 seconds
    </script>
</body>
</html>
        '''
    
    async def start(self):
        """Start the development server"""
        
        # Initialize components
        await self.initialize()
        
        # Start file watcher
        if self.file_watcher:
            self.file_watcher.start()
        
        logger.info(f"🌟 Starting development server on port {self.port}")
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("🛑 Shutting down development server...")
            if self.file_watcher:
                self.file_watcher.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0", 
            port=self.port,
            log_level="info" if not self.debug else "debug",
            reload=False,  # We handle our own reloading
            access_log=self.debug
        )
        
        server = uvicorn.Server(config)
        await server.serve()


async def run_dev_server(
    port: int = 8080,
    watch_paths: Optional[List[str]] = None,
    debug: bool = True,
    auto_reload: bool = True
):
    """
    Run the MeshAI development server
    
    Args:
        port: Server port
        watch_paths: Paths to watch for changes
        debug: Enable debug mode
        auto_reload: Enable hot-reload
    """
    
    server = DevServer(
        port=port,
        watch_paths=watch_paths,
        debug=debug,
        auto_reload=auto_reload
    )
    
    await server.start()


if __name__ == "__main__":
    # Run the development server
    asyncio.run(run_dev_server())