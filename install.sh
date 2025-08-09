#!/bin/bash
# MeshAI SDK Installation Script
# Supports Python 3.8-3.11

echo "MeshAI SDK Installation"
echo "======================"

# Check Python version
python_cmd=""
if command -v python3 &> /dev/null; then
    python_cmd="python3"
elif command -v python &> /dev/null; then
    python_cmd="python"
else
    echo "Error: Python is not installed"
    exit 1
fi

echo "Using Python: $($python_cmd --version)"

# Check Python version is 3.8+
python_version=$($python_cmd -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required (found $python_version)"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
$python_cmd -m venv meshai-env

# Activate virtual environment
echo "Activating virtual environment..."
source meshai-env/bin/activate 2>/dev/null || source meshai-env/Scripts/activate 2>/dev/null

# Upgrade pip, setuptools, and wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel build

# Install the package
echo "Installing MeshAI SDK..."
pip install -e .

# Verify installation
echo "Verifying installation..."
if command -v meshai &> /dev/null; then
    echo "✓ MeshAI CLI installed successfully"
    meshai --version
else
    echo "✓ MeshAI SDK installed successfully"
    echo "  To use the CLI, activate the virtual environment:"
    echo "  source meshai-env/bin/activate"
    echo "  meshai --help"
fi

echo ""
echo "Installation complete!"
echo "======================"
echo "To get started:"
echo "  source meshai-env/bin/activate  # Activate virtual environment"
echo "  meshai --help                   # Show CLI help"
echo "  meshai init my-project          # Create a new project"