"""
Google AI adapter for MeshAI SDK
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    import google.generativeai as genai
    from google.cloud import aiplatform
except ImportError:
    raise ImportError(
        "Google AI is not installed. Install with: pip install meshai-sdk[google]"
    )

import structlog

from ..core.agent import MeshAgent
from ..core.config import MeshConfig
from ..core.context import MeshContext
from ..core.schemas import TaskData
from ..exceptions.base import TaskExecutionError, ConfigurationError

logger = structlog.get_logger(__name__)


class GoogleMeshAgent(MeshAgent):
    """MeshAI adapter for Google Generative AI (Gemini)"""
    
    def __init__(
        self,
        model: str = "gemini-pro",
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        config: Optional[MeshConfig] = None,
        **kwargs
    ):
        if not agent_id:
            agent_id = f"gemini-{model.replace('-', '_')}"
        if not name:
            name = f"Google {model}"
        if not capabilities:
            capabilities = ["text-generation", "reasoning", "multimodal"]
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            framework="google",
            config=config,
            **kwargs
        )
        
        self.model_name = model
        
        # Configure API
        api_key = api_key or self._get_api_key()
        if not api_key:
            raise ConfigurationError("Google AI API key required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        
        logger.info(f"Google Gemini agent {agent_id} initialized")
    
    def _get_api_key(self) -> Optional[str]:
        import os
        return (
            getattr(self.config, 'google_api_key', None) or
            os.getenv('GOOGLE_API_KEY')
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        try:
            # Extract input
            if isinstance(task_data.input, dict):
                prompt = task_data.input.get("input", str(task_data.input))
            else:
                prompt = str(task_data.input)
            
            # Add context
            conversation_history = await context.get("conversation_history", [])
            if conversation_history:
                context_prompt = "\n".join([
                    f"{msg.get('type', 'user')}: {msg.get('content', '')}" 
                    for msg in conversation_history[-5:]
                ])
                prompt = f"Context:\n{context_prompt}\n\nCurrent task: {prompt}"
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Update context
            await self._update_context(context, prompt, response.text)
            
            return {
                "result": response.text,
                "type": "google_response",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Google AI task execution failed: {e}")
            raise TaskExecutionError(f"Google AI execution error: {e}")
    
    async def _update_context(self, context: MeshContext, prompt: str, response: str):
        conversation_history = await context.get("conversation_history", [])
        conversation_history.extend([
            {"type": "human", "content": prompt, "timestamp": datetime.utcnow().isoformat()},
            {"type": "ai", "content": response, "timestamp": datetime.utcnow().isoformat(), "source": "google"}
        ])
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        await context.set("conversation_history", conversation_history)


class VertexAIMeshAgent(MeshAgent):
    """MeshAI adapter for Google Vertex AI"""
    
    def __init__(
        self,
        model: str = "text-bison",
        project_id: str = None,
        location: str = "us-central1",
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        config: Optional[MeshConfig] = None,
        **kwargs
    ):
        if not agent_id:
            agent_id = f"vertex-{model.replace('-', '_')}"
        if not name:
            name = f"Vertex AI {model}"
        if not capabilities:
            capabilities = ["text-generation", "reasoning"]
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            framework="vertex-ai",
            config=config,
            **kwargs
        )
        
        self.model_name = model
        self.project_id = project_id or self._get_project_id()
        self.location = location
        
        if not self.project_id:
            raise ConfigurationError("Google Cloud project ID required")
        
        aiplatform.init(project=self.project_id, location=self.location)
        
        logger.info(f"Vertex AI agent {agent_id} initialized")
    
    def _get_project_id(self) -> Optional[str]:
        import os
        return (
            getattr(self.config, 'google_project_id', None) or
            os.getenv('GOOGLE_CLOUD_PROJECT') or
            os.getenv('GCLOUD_PROJECT')
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        try:
            # This is a simplified implementation
            # Real implementation would use specific Vertex AI model clients
            
            if isinstance(task_data.input, dict):
                prompt = task_data.input.get("input", str(task_data.input))
            else:
                prompt = str(task_data.input)
            
            # For this example, we'll use a simple text generation approach
            # In practice, you'd use the specific Vertex AI model
            result = f"Vertex AI processed: {prompt}"
            
            # Update context
            await self._update_context(context, prompt, result)
            
            return {
                "result": result,
                "type": "vertex_response",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Vertex AI task execution failed: {e}")
            raise TaskExecutionError(f"Vertex AI execution error: {e}")
    
    async def _update_context(self, context: MeshContext, prompt: str, response: str):
        conversation_history = await context.get("conversation_history", [])
        conversation_history.extend([
            {"type": "human", "content": prompt, "timestamp": datetime.utcnow().isoformat()},
            {"type": "ai", "content": response, "timestamp": datetime.utcnow().isoformat(), "source": "vertex"}
        ])
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        await context.set("conversation_history", conversation_history)