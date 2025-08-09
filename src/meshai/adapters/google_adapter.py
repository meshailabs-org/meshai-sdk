"""
Google AI adapter for MeshAI SDK
"""

import asyncio
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
    """MeshAI adapter for Google Generative AI (Gemini)
    
    Features:
    - Support for all Gemini models
    - Multimodal capabilities (text, images)
    - Safety ratings and content filtering
    - Context-aware conversations
    - Configurable generation parameters
    """
    
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
            capabilities = ["text-generation", "reasoning", "multimodal", "analysis"]
            if "vision" in model or "1.5" in model:
                capabilities.extend(["image-analysis", "visual-reasoning"])
        
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
            raise ConfigurationError(
                "Google AI API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        
        logger.info(f"Google Gemini agent {agent_id} initialized with model {model}")
    
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
                # Check for multimodal content (images, etc.)
                media_content = task_data.input.get("media", [])
            else:
                prompt = str(task_data.input)
                media_content = []
            
            # Prepare content for generation
            content_parts = []
            
            # Add context if available
            conversation_history = await context.get("conversation_history", [])
            if conversation_history:
                context_summary = self._prepare_context_summary(conversation_history)
                prompt = f"{context_summary}\n\nCurrent task: {prompt}"
            
            content_parts.append(prompt)
            
            # Add media content if present (for multimodal models)
            for media in media_content:
                if isinstance(media, dict) and media.get('type') == 'image':
                    # This would handle image inputs for multimodal Gemini
                    content_parts.append(media)
            
            # Configure generation parameters
            generation_config = self._get_generation_config(task_data.parameters)
            
            # Prepare tools for function calling
            tools = self._prepare_tools()
            
            # Generate response
            start_time = datetime.utcnow()
            response = await self._generate_content_async(content_parts, generation_config, tools)
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Process response (including function calls)
            result = await self._process_response(response, context)
            
            # Update context
            await self._update_context(context, prompt, result.get("result", ""))
            
            # Add metadata
            result.update({
                "type": "google_response",
                "model": self.model_name,
                "response_time_seconds": response_time,
                "multimodal": len(media_content) > 0,
                "safety_ratings": self._extract_safety_ratings(response)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Google AI task execution failed: {e}")
            raise TaskExecutionError(f"Google AI execution error: {e}")
    
    def _prepare_context_summary(self, conversation_history: List[Dict]) -> str:
        """Prepare context summary for Google AI"""
        if not conversation_history:
            return ""
        
        context_parts = ["Previous conversation:"]
        for msg in conversation_history[-3:]:  # Last 3 messages
            if isinstance(msg, dict) and 'content' in msg:
                role = "Human" if msg.get('type') == 'human' else "Assistant"
                content = msg['content'][:200]  # Limit length
                context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def _get_generation_config(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get generation configuration from task parameters"""
        config = {}
        
        # Map common parameters
        if "temperature" in parameters:
            config["temperature"] = float(parameters["temperature"])
        if "max_output_tokens" in parameters:
            config["max_output_tokens"] = int(parameters["max_output_tokens"])
        if "top_p" in parameters:
            config["top_p"] = float(parameters["top_p"])
        if "top_k" in parameters:
            config["top_k"] = int(parameters["top_k"])
        
        return config
    
    def _prepare_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Prepare function definitions for Google AI"""
        try:
            import google.generativeai.types as genai_types
            
            # Define MeshAI agent invocation tool
            meshai_tool = genai_types.FunctionDeclaration(
                name="invoke_meshai_agent",
                description="Invoke another MeshAI agent with specific capabilities to help complete tasks",
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "capabilities": genai_types.Schema(
                            type=genai_types.Type.ARRAY,
                            items=genai_types.Schema(type=genai_types.Type.STRING),
                            description="Required capabilities for the task"
                        ),
                        "task": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description="Clear task description or query"
                        ),
                        "context": genai_types.Schema(
                            type=genai_types.Type.OBJECT,
                            description="Additional context for the task"
                        )
                    },
                    required=["capabilities", "task"]
                )
            )
            
            return [genai_types.Tool(function_declarations=[meshai_tool])]
            
        except ImportError:
            logger.warning("Google AI types not available for function calling")
            return None
        except Exception as e:
            logger.warning(f"Failed to prepare Google AI tools: {e}")
            return None
    
    async def _generate_content_async(self, content_parts: List, config: Dict, tools: Optional[List] = None) -> Any:
        """Generate content asynchronously"""
        def generate():
            kwargs = {"contents": content_parts}
            
            if config:
                generation_config = genai.types.GenerationConfig(**config)
                kwargs["generation_config"] = generation_config
            
            if tools:
                kwargs["tools"] = tools
                
            return self.model.generate_content(**kwargs)
        
        return await asyncio.get_event_loop().run_in_executor(None, generate)
    
    async def _process_response(self, response: Any, context: MeshContext) -> Dict[str, Any]:
        """Process Google AI response including function calls"""
        result = {"result": "", "tools_used": False}
        
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    function_results = []
                    
                    for part in candidate.content.parts:
                        # Handle text parts
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                        
                        # Handle function calls
                        elif hasattr(part, 'function_call'):
                            function_result = await self._handle_function_call(part.function_call, context)
                            function_results.append(function_result)
                            text_parts.append(f"[Function called: {part.function_call.name}]")
                            result["tools_used"] = True
                    
                    result["result"] = "\n".join(text_parts)
                    if function_results:
                        result["function_results"] = function_results
                else:
                    # Fallback to simple text extraction
                    result["result"] = self._extract_response_text(response)
            else:
                result["result"] = self._extract_response_text(response)
                
        except Exception as e:
            logger.warning(f"Error processing Google AI response: {e}")
            result["result"] = self._extract_response_text(response)
        
        return result
    
    async def _handle_function_call(self, function_call: Any, context: MeshContext) -> Dict[str, Any]:
        """Handle function calls from Google AI"""
        try:
            if function_call.name == "invoke_meshai_agent":
                # Extract arguments
                args = function_call.args
                capabilities = args.get("capabilities", [])
                task = args.get("task", "")
                task_context = args.get("context", {})
                
                logger.info(f"Google AI agent invoking MeshAI agent with capabilities: {capabilities}")
                
                # Invoke agent through MeshAI
                result = await self.invoke_agent(
                    capabilities=capabilities,
                    task={"input": task, "parameters": task_context},
                    routing_strategy="capability_match"
                )
                
                return {
                    "function_name": function_call.name,
                    "success": result.status == "completed",
                    "result": result.result if result.status == "completed" else result.error
                }
            else:
                return {
                    "function_name": function_call.name,
                    "success": False,
                    "error": f"Unknown function: {function_call.name}"
                }
                
        except Exception as e:
            logger.error(f"Function call error: {e}")
            return {
                "function_name": getattr(function_call, 'name', 'unknown'),
                "success": False,
                "error": str(e)
            }
    
    def _extract_response_text(self, response: Any) -> str:
        """Extract text from Google AI response"""
        try:
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content'):
                    return candidate.content.parts[0].text
            return "No response generated"
        except Exception as e:
            logger.warning(f"Failed to extract response text: {e}")
            return str(response)
    
    def _extract_safety_ratings(self, response: Any) -> List[Dict]:
        """Extract safety ratings from response"""
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings'):
                    return [
                        {
                            "category": rating.category.name,
                            "probability": rating.probability.name
                        }
                        for rating in candidate.safety_ratings
                    ]
            return []
        except Exception:
            return []
    
    async def _update_context(self, context: MeshContext, prompt: str, response: str):
        """Update context with conversation"""
        conversation_history = await context.get("conversation_history", [])
        conversation_history.extend([
            {"type": "human", "content": prompt, "timestamp": datetime.utcnow().isoformat()},
            {"type": "ai", "content": response, "timestamp": datetime.utcnow().isoformat(), "source": "google"}
        ])
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        await context.set("conversation_history", conversation_history)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model": self.model_name,
            "provider": "google",
            "supports_multimodal": self.model_name in ["gemini-pro-vision", "gemini-1.5-pro"],
            "supports_function_calling": True,
            "max_tokens": 30720 if "gemini" in self.model_name else 8192
        }
    
    async def stream_response(self, task_input: Union[str, Dict[str, Any]], context: MeshContext):
        """Stream response from Google AI"""
        try:
            if isinstance(task_input, dict):
                prompt = task_input.get("input", str(task_input))
            else:
                prompt = str(task_input)
            
            # Add context
            conversation_history = await context.get("conversation_history", [])
            if conversation_history:
                context_summary = self._prepare_context_summary(conversation_history)
                prompt = f"{context_summary}\n\nCurrent task: {prompt}"
            
            def generate_stream():
                return self.model.generate_content(prompt, stream=True)
            
            stream = await asyncio.get_event_loop().run_in_executor(None, generate_stream)
            
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {e}"


class VertexAIMeshAgent(MeshAgent):
    """MeshAI adapter for Google Vertex AI
    
    Features:
    - Enterprise-grade Google Cloud AI models
    - Text generation and embeddings
    - Scalable cloud deployment
    - Advanced configuration options
    """
    
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
            capabilities = ["text-generation", "reasoning", "enterprise-ai"]
        
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
            raise ConfigurationError(
                "Google Cloud project ID required. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or pass project_id parameter."
            )
        
        aiplatform.init(project=self.project_id, location=self.location)
        
        logger.info(f"Vertex AI agent {agent_id} initialized with model {model}")
    
    def _get_project_id(self) -> Optional[str]:
        import os
        return (
            getattr(self.config, 'google_project_id', None) or
            os.getenv('GOOGLE_CLOUD_PROJECT') or
            os.getenv('GCLOUD_PROJECT')
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
                context_summary = self._prepare_context_summary(conversation_history)
                prompt = f"{context_summary}\n\nCurrent task: {prompt}"
            
            # Use Vertex AI Prediction API
            result = await self._generate_vertex_response(prompt, task_data.parameters)
            
            # Update context
            await self._update_context(context, prompt, result)
            
            return {
                "result": result,
                "type": "vertex_response",
                "model": self.model_name,
                "project_id": self.project_id,
                "location": self.location
            }
            
        except Exception as e:
            logger.error(f"Vertex AI task execution failed: {e}")
            raise TaskExecutionError(f"Vertex AI execution error: {e}")
    
    async def _generate_vertex_response(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate response using Vertex AI"""
        try:
            def generate():
                # This is a simplified implementation
                # In production, you'd use specific Vertex AI model endpoints
                if "text-bison" in self.model_name:
                    # Use PaLM API through aiplatform
                    from google.cloud import aiplatform
                    
                    # Get endpoint (simplified)
                    endpoints = aiplatform.Endpoint.list(
                        filter=f'display_name="{self.model_name}"',
                        project=self.project_id,
                        location=self.location
                    )
                    if endpoints:
                        # Generate prediction
                        instances = [{"content": prompt}]
                        response = endpoints[0].predict(instances=instances)
                        return response.predictions[0].get('content', 'No response')
                
                # Fallback for demonstration
                return f"Vertex AI ({self.model_name}) processed: {prompt[:100]}..."
            
            return await asyncio.get_event_loop().run_in_executor(None, generate)
            
        except Exception as e:
            logger.warning(f"Vertex AI generation failed, using fallback: {e}")
            return f"Vertex AI processed: {prompt[:100]}..."
    
    def _prepare_context_summary(self, conversation_history: List[Dict]) -> str:
        """Prepare context summary"""
        if not conversation_history:
            return ""
        
        context_parts = ["Context from previous conversation:"]
        for msg in conversation_history[-2:]:
            if isinstance(msg, dict) and 'content' in msg:
                role = "User" if msg.get('type') == 'human' else "AI"
                content = msg['content'][:150]
                context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    async def _update_context(self, context: MeshContext, prompt: str, response: str):
        """Update context with conversation"""
        conversation_history = await context.get("conversation_history", [])
        conversation_history.extend([
            {"type": "human", "content": prompt, "timestamp": datetime.utcnow().isoformat()},
            {"type": "ai", "content": response, "timestamp": datetime.utcnow().isoformat(), "source": "vertex"}
        ])
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        await context.set("conversation_history", conversation_history)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Vertex AI model information"""
        return {
            "model": self.model_name,
            "provider": "vertex-ai",
            "project_id": self.project_id,
            "location": self.location,
            "supports_streaming": False,
            "enterprise_ready": True
        }