"""
MeshAI CLI Templates

This module contains project and agent templates for scaffolding.
"""

from .project_templates import get_project_template
from .agent_templates import get_agent_template

__all__ = ['get_project_template', 'get_agent_template']