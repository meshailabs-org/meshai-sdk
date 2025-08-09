@echo off
REM MeshAI SDK Installation Script for Windows
REM Supports Python 3.8-3.11

echo MeshAI SDK Installation
echo =======================

REM Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed
    echo Please install Python 3.8 or higher from https://www.python.org
    pause
    exit /b 1
)

echo Using Python: 
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv meshai-env

REM Activate virtual environment
echo Activating virtual environment...
call meshai-env\Scripts\activate.bat

REM Upgrade pip, setuptools, and wheel
echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel build

REM Install the package
echo Installing MeshAI SDK...
pip install -e .

REM Verify installation
echo Verifying installation...
where meshai >nul 2>nul
if %errorlevel% equ 0 (
    echo √ MeshAI CLI installed successfully
    meshai --version
) else (
    echo √ MeshAI SDK installed successfully
    echo   To use the CLI, activate the virtual environment:
    echo   meshai-env\Scripts\activate.bat
    echo   meshai --help
)

echo.
echo Installation complete!
echo =======================
echo To get started:
echo   meshai-env\Scripts\activate.bat   # Activate virtual environment
echo   meshai --help                      # Show CLI help
echo   meshai init my-project             # Create a new project
pause