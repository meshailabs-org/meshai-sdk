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
            
            # Process response (including tool use)
            result = await self._process_bedrock_response(response_body, context)
            
            # Update context
            await self._update_context(context, prompt, result.get("result", ""))
            
            # Add metadata
            result.update({
                "type": "bedrock_response",
                "model": self.model_id,
                "usage": response_body.get("usage", {})
            })
            
            return result
            
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
    
    def _prepare_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Prepare tool definitions for Bedrock models (Claude format)"""
        if "anthropic.claude" not in self.model_id:
            return None  # Only Claude models support tools in Bedrock
            
        return [{
            "name": "invoke_meshai_agent",
            "description": "Invoke another MeshAI agent with specific capabilities to help complete tasks",
            "input_schema": {
                "type": "object",
                "properties": {
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required capabilities for the task"
                    },
                    "task": {
                        "type": "string",
                        "description": "Clear task description or query"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context for the task",
                        "additionalProperties": True
                    }
                },
                "required": ["capabilities", "task"]
            }
        }]
    
    def _prepare_request_body(self, messages: List[Dict], parameters: Dict) -> Dict:
        """Prepare request body based on model type"""
        if "anthropic.claude" in self.model_id:
            # Claude format with tool support
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": parameters.get("max_tokens", 4096),
                "temperature": parameters.get("temperature", 0.7),
                "messages": messages
            }
            
            # Add tools for Claude models
            tools = self._prepare_tools()
            if tools:
                body["tools"] = tools
            
            return body
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
    
    async def _process_bedrock_response(self, response_body: Dict, context: MeshContext) -> Dict[str, Any]:
        """Process Bedrock response including tool use"""
        result = {"result": "", "tools_used": False}
        
        try:
            if "anthropic.claude" in self.model_id:
                # Handle Claude format response
                content = response_body.get("content", [])
                text_parts = []
                tool_results = []
                
                for content_block in content:
                    if content_block.get("type") == "text":
                        text_parts.append(content_block.get("text", ""))
                    elif content_block.get("type") == "tool_use":
                        # Handle tool use
                        tool_result = await self._handle_bedrock_tool_use(content_block, context)
                        tool_results.append(tool_result)
                        text_parts.append(f"[Tool used: {content_block.get('name', 'unknown')}]")
                        result["tools_used"] = True
                
                result["result"] = "\n".join(text_parts)
                if tool_results:
                    result["tool_results"] = tool_results
            else:
                # Fallback to simple text extraction for other models
                result["result"] = self._extract_response_text(response_body)
                
        except Exception as e:
            logger.warning(f"Error processing Bedrock response: {e}")
            result["result"] = self._extract_response_text(response_body)
        
        return result
    
    async def _handle_bedrock_tool_use(self, tool_block: Dict, context: MeshContext) -> Dict[str, Any]:
        """Handle tool use in Bedrock response"""
        try:
            tool_name = tool_block.get("name")
            tool_input = tool_block.get("input", {})
            
            if tool_name == "invoke_meshai_agent":
                capabilities = tool_input.get("capabilities", [])
                task = tool_input.get("task", "")
                task_context = tool_input.get("context", {})
                
                logger.info(f"Bedrock agent invoking MeshAI agent with capabilities: {capabilities}")
                
                # Invoke agent through MeshAI
                result = await self.invoke_agent(
                    capabilities=capabilities,
                    task={"input": task, "parameters": task_context},
                    routing_strategy="capability_match"
                )
                
                return {
                    "tool_name": tool_name,
                    "tool_id": tool_block.get("id"),
                    "success": result.status == "completed",
                    "result": result.result if result.status == "completed" else result.error
                }
            else:
                return {
                    "tool_name": tool_name,
                    "tool_id": tool_block.get("id"),
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }
                
        except Exception as e:
            logger.error(f"Bedrock tool use error: {e}")
            return {
                "tool_name": tool_block.get("name", "unknown"),
                "tool_id": tool_block.get("id"),
                "success": False,
                "error": str(e)
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Bedrock model information"""
        return {
            "model": self.model_id,
            "provider": "amazon-bedrock",
            "region": self.region,
            "model_family": self._get_model_family(),
            "supports_streaming": self._supports_streaming(),
            "max_tokens": self._get_max_tokens()
        }
    
    def _get_model_family(self) -> str:
        """Get the model family"""
        if "anthropic.claude" in self.model_id:
            return "claude"
        elif "amazon.titan" in self.model_id:
            return "titan"
        elif "ai21.j2" in self.model_id:
            return "jurassic"
        elif "cohere" in self.model_id:
            return "cohere"
        elif "meta.llama" in self.model_id:
            return "llama"
        else:
            return "unknown"
    
    def _supports_streaming(self) -> bool:
        """Check if model supports streaming"""
        # Most Bedrock models support streaming
        return True
    
    def _get_max_tokens(self) -> int:
        """Get maximum tokens for the model"""
        if "claude-3" in self.model_id:
            return 200000  # Claude 3 has high context
        elif "titan" in self.model_id:
            return 4000
        elif "j2-ultra" in self.model_id:
            return 8192
        else:
            return 4096
    
    async def stream_response(self, task_input: Union[str, Dict[str, Any]], context: MeshContext):
        """Stream response from Bedrock"""
        try:
            if isinstance(task_input, dict):
                prompt = task_input.get("input", str(task_input))
            else:
                prompt = str(task_input)
            
            # Prepare messages
            conversation_history = await context.get("conversation_history", [])
            messages = self._prepare_messages(prompt, conversation_history)
            body = self._prepare_request_body(messages, {})
            
            # Use streaming API
            response = self.bedrock_client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json"
            )
            
            # Stream the response
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                
                if "anthropic.claude" in self.model_id:
                    # Claude streaming format
                    if chunk.get("type") == "content_block_delta":
                        delta = chunk.get("delta", {})
                        if "text" in delta:
                            yield delta["text"]
                elif "amazon.titan" in self.model_id:
                    # Titan streaming format
                    if "outputText" in chunk:
                        yield chunk["outputText"]
                
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {e}"


class SageMakerMeshAgent(MeshAgent):
    """MeshAI adapter for Amazon SageMaker endpoints
    
    Features:
    - Custom model deployments
    - Real-time inference endpoints
    - Auto-scaling and load balancing
    - Custom inference logic
    """
    
    def __init__(
        self,
        endpoint_name: str,
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
            agent_id = f"sagemaker-{endpoint_name.replace('-', '_')}"
        if not name:
            name = f"SageMaker {endpoint_name}"
        if not capabilities:
            capabilities = ["custom-inference", "ml-models"]
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            framework="sagemaker",
            config=config,
            **kwargs
        )
        
        self.endpoint_name = endpoint_name
        self.region = region
        
        # Initialize SageMaker client
        session_kwargs = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
        
        session = boto3.Session(**session_kwargs)
        self.sagemaker_client = session.client("sagemaker-runtime")
        
        logger.info(f"SageMaker agent {agent_id} initialized with endpoint {endpoint_name}")
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        try:
            # Extract input
            if isinstance(task_data.input, dict):
                payload = task_data.input
            else:
                payload = {"input": str(task_data.input)}
            
            # Add context if needed
            conversation_history = await context.get("conversation_history", [])
            if conversation_history:
                payload["context"] = conversation_history[-5:]  # Last 5 messages
            
            # Add parameters
            payload.update(task_data.parameters)
            
            # Invoke SageMaker endpoint
            start_time = datetime.utcnow()
            response = await self._invoke_endpoint_async(payload)
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Parse response
            result = self._parse_sagemaker_response(response)
            
            # Update context
            if isinstance(result, dict) and "text" in result:
                await self._update_context(context, str(payload), result["text"])
            
            return {
                "result": result,
                "type": "sagemaker_response",
                "endpoint": self.endpoint_name,
                "response_time_seconds": response_time
            }
            
        except ClientError as e:
            logger.error(f"SageMaker API error: {e}")
            raise TaskExecutionError(f"SageMaker API error: {e}")
        except Exception as e:
            logger.error(f"SageMaker task execution failed: {e}")
            raise TaskExecutionError(f"SageMaker execution error: {e}")
    
    async def _invoke_endpoint_async(self, payload: Dict) -> Dict:
        """Invoke SageMaker endpoint asynchronously"""
        import asyncio
        
        def invoke():
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=json.dumps(payload),
                ContentType="application/json"
            )
            return json.loads(response["Body"].read())
        
        return await asyncio.get_event_loop().run_in_executor(None, invoke)
    
    def _parse_sagemaker_response(self, response: Dict) -> Any:
        """Parse SageMaker response based on expected format"""
        # This depends on your model's output format
        if isinstance(response, dict):
            # Common patterns
            if "predictions" in response:
                return response["predictions"]
            elif "outputs" in response:
                return response["outputs"]
            elif "generated_text" in response:
                return response["generated_text"]
            else:
                return response
        else:
            return response
    
    async def _update_context(self, context: MeshContext, prompt: str, response: str):
        """Update context with conversation"""
        conversation_history = await context.get("conversation_history", [])
        conversation_history.extend([
            {"type": "human", "content": prompt, "timestamp": datetime.utcnow().isoformat()},
            {"type": "ai", "content": response, "timestamp": datetime.utcnow().isoformat(), "source": "sagemaker"}
        ])
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        await context.set("conversation_history", conversation_history)
    
    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get information about the SageMaker endpoint"""
        try:
            sagemaker_client = boto3.client("sagemaker", region_name=self.region)
            response = sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
            
            return {
                "endpoint_name": self.endpoint_name,
                "endpoint_arn": response.get("EndpointArn"),
                "endpoint_status": response.get("EndpointStatus"),
                "creation_time": response.get("CreationTime"),
                "instance_type": response.get("ProductionVariants", [{}])[0].get("InstanceType"),
                "model_name": response.get("ProductionVariants", [{}])[0].get("ModelName")
            }
        except Exception as e:
            logger.error(f"Failed to get endpoint info: {e}")
            return {"endpoint_name": self.endpoint_name, "error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get SageMaker model information"""
        endpoint_info = self.get_endpoint_info()
        return {
            "endpoint_name": self.endpoint_name,
            "provider": "amazon-sagemaker",
            "region": self.region,
            "status": endpoint_info.get("endpoint_status", "unknown"),
            "supports_streaming": False,  # Most SageMaker endpoints don't support streaming
            "custom_deployment": True
        }