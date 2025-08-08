"""
Serialization utilities for MeshAI SDK
"""

import json
import pickle
from typing import Any, Dict, Union, Optional
from datetime import datetime, date
import uuid

from pydantic import BaseModel


class MeshAIEncoder(json.JSONEncoder):
    """Custom JSON encoder for MeshAI data types"""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        
        return super().default(obj)


def serialize_data(
    data: Any, 
    format: str = "json",
    ensure_ascii: bool = False
) -> Union[str, bytes]:
    """
    Serialize data for transmission or storage.
    
    Args:
        data: Data to serialize
        format: Serialization format ("json", "pickle")
        ensure_ascii: Whether to escape non-ASCII characters in JSON
        
    Returns:
        Serialized data as string or bytes
        
    Raises:
        ValueError: If format is unsupported
        TypeError: If data is not serializable
    """
    if format.lower() == "json":
        try:
            return json.dumps(
                data,
                cls=MeshAIEncoder,
                ensure_ascii=ensure_ascii,
                separators=(',', ':')  # Compact format
            )
        except TypeError as e:
            raise TypeError(f"Data is not JSON serializable: {e}")
    
    elif format.lower() == "pickle":
        try:
            return pickle.dumps(data)
        except Exception as e:
            raise TypeError(f"Data is not pickle serializable: {e}")
    
    else:
        raise ValueError(f"Unsupported serialization format: {format}")


def deserialize_data(
    serialized_data: Union[str, bytes],
    format: str = "json",
    target_type: Optional[type] = None
) -> Any:
    """
    Deserialize data from transmission or storage.
    
    Args:
        serialized_data: Serialized data to deserialize
        format: Serialization format ("json", "pickle")
        target_type: Optional target type to validate/convert to
        
    Returns:
        Deserialized data
        
    Raises:
        ValueError: If format is unsupported or data is invalid
    """
    if format.lower() == "json":
        try:
            data = json.loads(serialized_data)
            
            # Convert to target type if specified
            if target_type and isinstance(target_type, type) and issubclass(target_type, BaseModel):
                return target_type(**data) if isinstance(data, dict) else target_type.model_validate(data)
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except Exception as e:
            raise ValueError(f"Failed to deserialize JSON: {e}")
    
    elif format.lower() == "pickle":
        try:
            return pickle.loads(serialized_data)
        except Exception as e:
            raise ValueError(f"Failed to deserialize pickle data: {e}")
    
    else:
        raise ValueError(f"Unsupported deserialization format: {format}")


def safe_serialize(data: Any, fallback: str = "null") -> str:
    """
    Safely serialize data with fallback for non-serializable objects.
    
    Args:
        data: Data to serialize
        fallback: Fallback value for non-serializable data
        
    Returns:
        JSON string
    """
    try:
        return serialize_data(data, "json")
    except (TypeError, ValueError):
        return fallback


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge into base
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def sanitize_data(data: Any, max_depth: int = 10) -> Any:
    """
    Sanitize data for serialization by removing circular references
    and limiting depth.
    
    Args:
        data: Data to sanitize
        max_depth: Maximum nesting depth
        
    Returns:
        Sanitized data
    """
    def _sanitize(obj: Any, depth: int, seen: set) -> Any:
        if depth >= max_depth:
            return "<max_depth_exceeded>"
        
        if id(obj) in seen:
            return "<circular_reference>"
        
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        if isinstance(obj, uuid.UUID):
            return str(obj)
        
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        
        seen.add(id(obj))
        
        try:
            if isinstance(obj, dict):
                return {
                    str(k): _sanitize(v, depth + 1, seen.copy())
                    for k, v in obj.items()
                }
            
            if isinstance(obj, (list, tuple)):
                return [
                    _sanitize(item, depth + 1, seen.copy())
                    for item in obj
                ]
            
            if hasattr(obj, '__dict__'):
                return {
                    str(k): _sanitize(v, depth + 1, seen.copy())
                    for k, v in obj.__dict__.items()
                    if not k.startswith('_')
                }
            
            return str(obj)
            
        except Exception:
            return "<serialization_error>"
    
    return _sanitize(data, 0, set())


def estimate_size(data: Any) -> int:
    """
    Estimate the size of data in bytes when serialized.
    
    Args:
        data: Data to estimate size for
        
    Returns:
        Estimated size in bytes
    """
    try:
        serialized = serialize_data(data, "json")
        return len(serialized.encode('utf-8'))
    except:
        # Fallback estimation
        return len(str(data).encode('utf-8'))