#!/usr/bin/env python3
"""
Test the MeshAI CLI tool

This script tests basic CLI functionality to ensure it's working correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def test_cli():
    """Test CLI functionality"""
    print("🧪 Testing MeshAI CLI...")
    
    # Add src to path to test locally
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Test 1: CLI help
    print("\n1. Testing CLI help...")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/meshai-cli.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ CLI help working")
            print(f"Output preview: {result.stdout[:200]}...")
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ CLI help timed out")
        return False
    except Exception as e:
        print(f"❌ CLI help error: {e}")
        return False
    
    # Test 2: CLI version
    print("\n2. Testing CLI version...")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/meshai-cli.py", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ CLI version working")
            print(f"Version: {result.stdout.strip()}")
        else:
            print(f"❌ CLI version failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ CLI version error: {e}")
    
    # Test 3: Agent subcommands
    print("\n3. Testing agent subcommands...")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/meshai-cli.py", "agent", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ Agent commands available")
        else:
            print(f"❌ Agent commands failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Agent commands error: {e}")
    
    # Test 4: Project init
    print("\n4. Testing project templates...")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/meshai-cli.py", "init", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and "template" in result.stdout:
            print("✅ Project templates available")
        else:
            print(f"❌ Project init failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Project init error: {e}")
    
    print("\n🎉 CLI basic tests completed!")
    return True


if __name__ == "__main__":
    success = test_cli()
    sys.exit(0 if success else 1)