# Contributing to MeshAI SDK

Thank you for your interest in contributing to MeshAI SDK! This document provides guidelines and information for contributors.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/meshailabs-org/meshai-sdk.git
   cd meshai-sdk
   ```

2. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,all]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:
```bash
black src/ tests/ examples/
isort src/ tests/ examples/
flake8 src/ tests/ examples/
mypy src/
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
pytest tests/ --cov=meshai --cov-report=html
```

## Framework Adapters

When adding new framework adapters:

1. Create adapter in `src/meshai/adapters/`
2. Follow the existing adapter patterns
3. Add comprehensive tests
4. Update `__init__.py` to include new adapter
5. Add example usage in `examples/`
6. Update documentation

### Adapter Requirements

All adapters must:
- Inherit from `MeshAgent`
- Handle async operations properly
- Include error handling and logging
- Support context management
- Provide tool integration (if applicable)
- Include comprehensive docstrings

## Pull Request Process

1. **Fork the repository** and create your feature branch
2. **Write tests** for new functionality
3. **Update documentation** as needed
4. **Run all tests and linting** locally
5. **Submit pull request** with clear description

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Examples provided for new features
- [ ] Breaking changes are noted

## Issue Reporting

When reporting issues:

1. Use clear, descriptive titles
2. Provide minimal reproduction steps
3. Include environment details (Python version, OS, etc.)
4. Paste relevant error messages/logs
5. Mention which adapters/frameworks are involved

## Documentation

- Use clear, concise language
- Include code examples
- Document all public APIs
- Update README for major features
- Use proper docstring format

## Community

- Be respectful and inclusive
- Help others learn and contribute
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Join discussions in GitHub Issues

## Release Process

Releases are handled by maintainers:

1. Version bump in `setup.py` and `pyproject.toml`
2. Update CHANGELOG.md
3. Create GitHub release with notes
4. Publish to PyPI

## License

By contributing, you agree that your contributions will be licensed under the MIT License.