# MeshAI SDK Installation Guide

## Quick Install

### For Python 3.10+ Users

If you're encountering the "easy_install is deprecated" error, follow these steps:

```bash
# 1. Create a virtual environment
python3 -m venv meshai-env

# 2. Activate the virtual environment
# On Linux/Mac:
source meshai-env/bin/activate
# On Windows:
meshai-env\Scripts\activate.bat

# 3. Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel build

# 4. Install MeshAI SDK in development mode
pip install -e .

# 5. Verify installation
meshai --version
```

### Alternative: Using the Install Script

#### Linux/Mac:
```bash
chmod +x install.sh
./install.sh
```

#### Windows:
```cmd
install.bat
```

## Troubleshooting

### "easy_install is deprecated" Error

This occurs with newer Python versions (3.10+). The solution is to:
1. Always use a virtual environment
2. Upgrade pip, setuptools, and wheel before installing
3. Use `pip install -e .` instead of `python setup.py install`

### "externally-managed-environment" Error

Modern Linux distributions protect the system Python. Always use a virtual environment:

```bash
python3 -m venv meshai-env
meshai-env/bin/pip install -e .
```

### Missing Dependencies

If you encounter missing dependencies, install the full requirements:

```bash
pip install -r requirements.txt
```

### Python Version Issues

MeshAI SDK requires Python 3.8 or higher. Check your version:

```bash
python3 --version
```

If you need to install a specific Python version:
- **Ubuntu/Debian**: `sudo apt install python3.10 python3.10-venv`
- **macOS**: `brew install python@3.10`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

## Manual Installation

If the automated scripts don't work, follow these manual steps:

1. **Install Python 3.8+**
2. **Create virtual environment**:
   ```bash
   python3 -m venv meshai-env
   ```

3. **Activate virtual environment**:
   - Linux/Mac: `source meshai-env/bin/activate`
   - Windows: `meshai-env\Scripts\activate.bat`

4. **Install dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install httpx pydantic tenacity python-jose[cryptography]
   pip install websockets prometheus-client structlog
   pip install click rich pyyaml fastapi uvicorn[standard]
   ```

5. **Install MeshAI SDK**:
   ```bash
   pip install -e .
   ```

## Verify Installation

After installation, verify everything is working:

```bash
# Check CLI is available
meshai --version

# Show help
meshai --help

# List available commands
meshai agent --help
meshai init --help
meshai dev --help
```

## Next Steps

1. **Initialize a project**: `meshai init my-project`
2. **Start development server**: `meshai dev server`
3. **Create an agent**: `meshai agent create my-agent`

## Support

If you encounter issues:
1. Check the [GitHub Issues](https://github.com/meshailabs/meshai-sdk/issues)
2. Ensure you're using Python 3.8+
3. Try the manual installation steps
4. Create a new issue with your error message and Python version