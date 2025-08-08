# Changelog

All notable changes to the MeshAI SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-01-08

### Added
- **Advanced Context Management System** - Key differentiator feature
  - Persistent Redis-backed context storage
  - Agent-specific memory isolation with access control
  - Multiple conflict resolution strategies (Last Write Wins, Merge, Agent Priority, Version Branching)
  - Context versioning with rollback capabilities
  - Sophisticated sharing policies and permissions (Private, Shared, Public scopes)
  - Real-time context analytics and monitoring
  - CRDT-like conflict resolution for distributed agent coordination
- Enhanced `MeshContext` class with backward-compatible advanced features
- Comprehensive examples demonstrating context management patterns
- Complete test coverage for context management functionality

### Enhanced
- Context system now supports production-grade persistence and scaling
- Improved agent coordination through advanced memory management
- Better isolation and security for multi-tenant agent environments

## [0.1.0] - 2025-01-08

### Added
- Initial release of MeshAI SDK
- Core `MeshAgent` base class with lifecycle management
- Framework adapters for:
  - LangChain (agents, executors, runnables, tools)
  - CrewAI (crews and individual agents)
  - Anthropic Claude (with tool calling)
  - OpenAI GPT (with function calling)
  - Google Gemini (Generative AI)
  - Amazon Bedrock (multi-model support)
  - AutoGen (basic multi-agent support)
- Registry and Runtime client libraries
- Context management system for agent communication
- Comprehensive configuration management
- Metrics collection and monitoring
- CLI tools for agent and task management
- Examples and documentation
- Development tooling (black, isort, flake8, mypy)

### Features
- **Easy Integration**: Simple decorators and one-line framework wrapping
- **Cross-Framework Communication**: Agents can invoke other agents regardless of framework
- **Production Ready**: Health monitoring, metrics, error handling, retry logic
- **Context Sharing**: Persistent context and memory management between agents
- **Extensible**: Plugin architecture for custom framework adapters

### Developer Experience
- Simple `@register_agent` decorator
- Automatic agent registration and discovery
- Built-in health checks and monitoring endpoints
- CLI tools for testing and management
- Comprehensive examples and documentation