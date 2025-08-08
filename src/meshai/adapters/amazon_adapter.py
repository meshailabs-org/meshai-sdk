"""
Amazon Bedrock adapter for MeshAI SDK
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    raise ImportError(
        "AWS SDK is not installed. Install with: pip install meshai-sdk[amazon]"
    )

import structlog

from ..core.agent import MeshAgent
from ..core.config import MeshConfig
from ..core.context import MeshContext
from ..core.schemas import TaskData
from ..exceptions.base import TaskExecutionError, ConfigurationError

logger = structlog.get_logger(__name__)


class BedrockMeshAgent(MeshAgent):
    """MeshAI adapter for Amazon Bedrock models"""
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        config: Optional[MeshConfig] = None,
        **kwargs
    ):
        if not agent_id:
            agent_id = f"bedrock-{model_id.replace('.', '_').replace(':', '_')}"
        if not name:
            name = f"Bedrock {model_id.split('.')[-1]}"
        if not capabilities:
            capabilities = ["text-generation", "reasoning", "analysis"]
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            framework="bedrock",
            config=config,
            **kwargs
        )
        
        self.model_id = model_id
        self.region = region
        
        # Initialize Bedrock client
        session_kwargs = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
        
        session = boto3.Session(**session_kwargs)
        self.bedrock_client = session.client("bedrock-runtime")
        
        logger.info(f"Bedrock agent {agent_id} initialized with model {model_id}")
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        try:
            # Extract input
            if isinstance(task_data.input, dict):
                prompt = task_data.input.get("input", str(task_data.input))
            else:
                prompt = str(task_data.input)
            
            # Add context
            conversation_history = await context.get("conversation_history", [])
            messages = self._prepare_messages(prompt, conversation_history)
            
            # Prepare request body based on model type
            body = self._prepare_request_body(messages, task_data.parameters)
            
            # Make Bedrock API call
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response["body"].read())
            result_text = self._extract_response_text(response_body)
            
            # Update context
            await self._update_context(context, prompt, result_text)
            
            return {
                "result": result_text,
                "type": "bedrock_response",
                "model": self.model_id,
                "usage": response_body.get("usage", {})
            }
            
        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            raise TaskExecutionError(f"Bedrock API error: {e}")
        except Exception as e:
            logger.error(f"Bedrock task execution failed: {e}")
            raise TaskExecutionError(f"Bedrock execution error: {e}")
    
    def _prepare_messages(self, prompt: str, conversation_history: List[Dict]) -> List[Dict]:
        """Prepare messages for the model"""
        messages = []
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Last 10 messages
            if isinstance(msg, dict) and 'content' in msg:
                role = "user" if msg.get('type') == 'human' else "assistant"
                messages.append({
                    "role": role,
                    "content": msg['content']
                })
        
        # Add current prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def _prepare_request_body(self, messages: List[Dict], parameters: Dict) -> Dict:
        """Prepare request body based on model type"""
        if "anthropic.claude" in self.model_id:
            # Claude format
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": parameters.get("max_tokens", 4096),
                "temperature": parameters.get("temperature", 0.7),
                "messages": messages
            }
        elif "amazon.titan" in self.model_id:
            # Titan format
            prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            return {
                "inputText": prompt_text,
                "textGenerationConfig": {
                    "maxTokenCount": parameters.get("max_tokens", 4096),
                    "temperature": parameters.get("temperature", 0.7),
                    "topP": parameters.get("top_p", 0.9)
                }
            }
        elif "ai21.j2" in self.model_id:
            # Jurassic format
            prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            return {
                "prompt": prompt_text,
                "maxTokens": parameters.get("max_tokens", 4096),
                "temperature": parameters.get("temperature", 0.7)
            }
        else:
            # Generic format
            prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            return {
                "prompt": prompt_text,
                "max_tokens": parameters.get("max_tokens", 4096),
                "temperature": parameters.get("temperature", 0.7)
            }
    
    def _extract_response_text(self, response_body: Dict) -> str:
        """Extract response text based on model type"""
        if "anthropic.claude" in self.model_id:
            # Claude format
            content = response_body.get("content", [])
            if content and isinstance(content[0], dict):
                return content[0].get("text", "")
            return ""
        elif "amazon.titan" in self.model_id:
            # Titan format
            results = response_body.get("results", [])
            if results:
                return results[0].get("outputText", "")
            return ""
        elif "ai21.j2" in self.model_id:
            # Jurassic format
            completions = response_body.get("completions", [])
            if completions:
                return completions[0].get("data", {}).get("text", "")
            return ""
        else:
            # Try common fields
            return (
                response_body.get("completion", "") or
                response_body.get("text", "") or
                response_body.get("generated_text", "") or
                str(response_body)
            )
    
    async def _update_context(self, context: MeshContext, prompt: str, response: str):
        """Update context with conversation"""
        conversation_history = await context.get("conversation_history", [])
        conversation_history.extend([
            {"type": "human", "content": prompt, "timestamp": datetime.utcnow().isoformat()},
            {"type": "ai", "content": response, "timestamp": datetime.utcnow().isoformat(), "source": "bedrock"}
        ])
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        await context.set("conversation_history", conversation_history)
    
    def get_available_models(self) -> List[str]:
        """Get list of available Bedrock models"""
        try:
            bedrock_client = boto3.client("bedrock", region_name=self.region)
            response = bedrock_client.list_foundation_models()
            return [model["modelId"] for model in response.get("modelSummaries", [])]
        except Exception as e:
            logger.error(f"Failed to list Bedrock models: {e}")
            return []