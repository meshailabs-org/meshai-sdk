#!/usr/bin/env python3
"""
Comprehensive test runner for MeshAI SDK
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def run_command(cmd, description, exit_on_failure=True):
    """Run a command and handle the result"""
    print(f"\n🔍 {description}")
    print("=" * 50)
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
        return True
    else:
        print(f"❌ {description} - FAILED")
        if exit_on_failure:
            sys.exit(1)
        return False


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run MeshAI SDK tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--format", action="store_true", help="Check code formatting")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--all", action="store_true", help="Run all checks and tests")
    
    args = parser.parse_args()
    
    # If no specific flags, run basic tests
    if not any([args.quick, args.integration, args.performance, args.coverage, 
               args.format, args.lint, args.security, args.all]):
        args.quick = True
    
    if args.all:
        args.format = args.lint = args.security = args.coverage = True
        args.integration = args.performance = True
    
    print("🧪 MeshAI SDK Test Runner")
    print("=" * 50)
    
    success_count = 0
    total_count = 0
    
    # Code formatting
    if args.format:
        total_count += 1
        if run_command("black --check --diff src/ tests/ scripts/", 
                      "Code formatting check", exit_on_failure=False):
            success_count += 1
        
        total_count += 1
        if run_command("isort --check-only --diff src/ tests/ scripts/",
                      "Import sorting check", exit_on_failure=False):
            success_count += 1
    
    # Linting
    if args.lint:
        total_count += 1
        if run_command("flake8 src/ tests/ scripts/ --max-line-length=88 --extend-ignore=E203,W503",
                      "Code linting", exit_on_failure=False):
            success_count += 1
        
        total_count += 1
        if run_command("mypy src/meshai --ignore-missing-imports",
                      "Type checking", exit_on_failure=False):
            success_count += 1
    
    # Security checks
    if args.security:
        total_count += 1
        if run_command("bandit -r src/ -ll",
                      "Security scanning", exit_on_failure=False):
            success_count += 1
        
        total_count += 1
        if run_command("safety check",
                      "Dependency security check", exit_on_failure=False):
            success_count += 1
    
    # Unit tests
    if args.quick or args.coverage:
        total_count += 1
        test_cmd = "pytest tests/ -v --tb=short"
        if args.coverage:
            test_cmd += " --cov=src/meshai --cov-report=term-missing --cov-report=html"
        
        if run_command(test_cmd, "Unit tests", exit_on_failure=False):
            success_count += 1
    
    # Integration tests
    if args.integration:
        total_count += 1
        if run_command("pytest tests/integration/ -v --tb=short -m integration",
                      "Integration tests", exit_on_failure=False):
            success_count += 1
    
    # Performance tests
    if args.performance:
        total_count += 1
        if run_command("pytest tests/ -v --tb=short -m performance",
                      "Performance tests", exit_on_failure=False):
            success_count += 1
        
        total_count += 1
        if run_command("python tests/benchmarks/performance_benchmarks.py",
                      "Performance benchmarks", exit_on_failure=False):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    print(f"✅ Passed: {success_count}")
    print(f"❌ Failed: {total_count - success_count}")
    print(f"📈 Success Rate: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/")
    else:
        print("⚠️  Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()