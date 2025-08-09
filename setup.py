#!/usr/bin/env python3
"""
MeshAI SDK Setup
AI Agent Interoperability Platform
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="meshai-sdk",
    version="0.1.0",
    author="MeshAI Labs",
    author_email="dev@meshai.dev",
    description="SDK for MeshAI - AI Agent Interoperability Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meshailabs/meshai-sdk",
    project_urls={
        "Bug Tracker": "https://github.com/meshailabs/meshai-sdk/issues",
        "Documentation": "https://docs.meshai.dev",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "tenacity>=8.0.0",
        "python-jose[cryptography]>=3.3.0",
        "websockets>=11.0.0",
        "prometheus-client>=0.17.0",
        "structlog>=23.0.0",
        # CLI dependencies
        "click>=8.0.0",
        "rich>=13.0.0",
        "pyyaml>=6.0.0",
        # Development server dependencies
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
    ],
    extras_require={
        "langchain": [
            "langchain>=0.1.0",
            "langchain-core>=0.1.0",
            "langchain-community>=0.0.10",
        ],
        "crewai": [
            "crewai>=0.1.0",
        ],
        "autogen": [
            "pyautogen>=0.2.0",
        ],
        "anthropic": [
            "anthropic>=0.21.0",
        ],
        "google": [
            "google-generativeai>=0.4.0",
            "google-cloud-aiplatform>=1.42.0",
        ],
        "amazon": [
            "boto3>=1.34.0",
            "botocore>=1.34.0",
        ],
        "openai": [
            "openai>=1.12.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "watchdog>=3.0.0",  # File watching for dev server
        ],
        "all": [
            "langchain>=0.1.0",
            "langchain-core>=0.1.0",
            "langchain-community>=0.0.10",
            "crewai>=0.1.0",
            "pyautogen>=0.2.0",
            "anthropic>=0.21.0",
            "google-generativeai>=0.4.0",
            "google-cloud-aiplatform>=1.42.0",
            "boto3>=1.34.0",
            "botocore>=1.34.0",
            "openai>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "meshai=meshai.cli.main:main",
        ],
    },
)