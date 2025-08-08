"""
Framework adapters for MeshAI SDK
"""

# Import all adapters - some may fail if dependencies not installed
_available_adapters = {}

# LangChain adapter
try:
    from .langchain_adapter import LangChainMeshAgent, LangChainToolAdapter
    _available_adapters["langchain"] = {
        "LangChainMeshAgent": LangChainMeshAgent,
        "LangChainToolAdapter": LangChainToolAdapter,
    }
except ImportError:
    pass

# CrewAI adapter  
try:
    from .crewai_adapter import CrewAIMeshAgent
    _available_adapters["crewai"] = {
        "CrewAIMeshAgent": CrewAIMeshAgent,
    }
except ImportError:
    pass

# AutoGen adapter
try:
    from .autogen_adapter import AutoGenMeshAgent
    _available_adapters["autogen"] = {
        "AutoGenMeshAgent": AutoGenMeshAgent,
    }
except ImportError:
    pass

# Anthropic adapter
try:
    from .anthropic_adapter import AnthropicMeshAgent
    _available_adapters["anthropic"] = {
        "AnthropicMeshAgent": AnthropicMeshAgent,
    }
except ImportError:
    pass

# Google adapter
try:
    from .google_adapter import GoogleMeshAgent, VertexAIMeshAgent
    _available_adapters["google"] = {
        "GoogleMeshAgent": GoogleMeshAgent,
        "VertexAIMeshAgent": VertexAIMeshAgent,
    }
except ImportError:
    pass

# Amazon/AWS adapter
try:
    from .amazon_adapter import BedrockMeshAgent
    _available_adapters["amazon"] = {
        "BedrockMeshAgent": BedrockMeshAgent,
    }
except ImportError:
    pass

# OpenAI adapter
try:
    from .openai_adapter import OpenAIMeshAgent
    _available_adapters["openai"] = {
        "OpenAIMeshAgent": OpenAIMeshAgent,
    }
except ImportError:
    pass


def get_available_adapters() -> dict:
    """Get dictionary of available framework adapters"""
    return _available_adapters.copy()


def is_adapter_available(framework: str) -> bool:
    """Check if adapter is available for a framework"""
    return framework.lower() in _available_adapters


def get_adapter_classes(framework: str) -> dict:
    """Get adapter classes for a specific framework"""
    return _available_adapters.get(framework.lower(), {})


# Export what's available
__all__ = []

# Add available adapters to __all__
for framework, adapters in _available_adapters.items():
    for adapter_name in adapters.keys():
        globals()[adapter_name] = adapters[adapter_name]
        __all__.append(adapter_name)

# Always export utility functions
__all__.extend([
    "get_available_adapters",
    "is_adapter_available", 
    "get_adapter_classes",
])