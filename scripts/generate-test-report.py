#!/usr/bin/env python3
"""
Generate comprehensive test reports for MeshAI SDK
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os


def run_command(cmd):
    """Run command and capture output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def parse_coverage_xml():
    """Parse coverage XML report"""
    coverage_file = Path("coverage.xml")
    if not coverage_file.exists():
        return None
    
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        
        coverage_data = {
            "line_rate": float(root.get("line-rate", 0)) * 100,
            "branch_rate": float(root.get("branch-rate", 0)) * 100,
            "lines_covered": int(root.get("lines-covered", 0)),
            "lines_valid": int(root.get("lines-valid", 0)),
            "branches_covered": int(root.get("branches-covered", 0)),
            "branches_valid": int(root.get("branches-valid", 0)),
        }
        
        return coverage_data
    except Exception as e:
        print(f"Error parsing coverage XML: {e}")
        return None


def parse_pytest_json():
    """Parse pytest JSON report"""
    json_file = Path("test-results.json") 
    if not json_file.exists():
        return None
        
    try:
        with open(json_file) as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error parsing pytest JSON: {e}")
        return None


def generate_html_report():
    """Generate HTML test report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get coverage data
    coverage = parse_coverage_xml()
    
    # Get pytest results  
    pytest_results = parse_pytest_json()
    
    # Get git info
    success, git_commit, _ = run_command("git rev-parse HEAD")
    commit_hash = git_commit.strip()[:8] if success else "unknown"
    
    success, git_branch, _ = run_command("git branch --show-current")
    branch = git_branch.strip() if success else "unknown"
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MeshAI SDK Test Report</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .metric-card {{ display: inline-block; background: #f8f9fa; padding: 20px; margin: 10px; border-radius: 6px; min-width: 200px; text-align: center; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #28a745; }}
            .metric-label {{ color: #666; margin-top: 5px; }}
            .section {{ margin: 30px 0; }}
            .success {{ color: #28a745; }}
            .failure {{ color: #dc3545; }}
            .warning {{ color: #ffc107; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; }}
            .status-badge {{ padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }}
            .status-pass {{ background: #d4edda; color: #155724; }}
            .status-fail {{ background: #f8d7da; color: #721c24; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧪 MeshAI SDK Test Report</h1>
                <p>Generated on {timestamp}</p>
                <p>Branch: <code>{branch}</code> | Commit: <code>{commit_hash}</code></p>
            </div>
    """
    
    if coverage:
        html_content += f"""
            <div class="section">
                <h2>📊 Code Coverage</h2>
                <div class="metric-card">
                    <div class="metric-value">{coverage['line_rate']:.1f}%</div>
                    <div class="metric-label">Line Coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{coverage['branch_rate']:.1f}%</div>
                    <div class="metric-label">Branch Coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{coverage['lines_covered']}</div>
                    <div class="metric-label">Lines Covered</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{coverage['lines_valid']}</div>
                    <div class="metric-label">Total Lines</div>
                </div>
            </div>
        """
    
    if pytest_results:
        summary = pytest_results.get("summary", {})
        html_content += f"""
            <div class="section">
                <h2>🧪 Test Results</h2>
                <div class="metric-card">
                    <div class="metric-value success">{summary.get('passed', 0)}</div>
                    <div class="metric-label">Passed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value failure">{summary.get('failed', 0)}</div>
                    <div class="metric-label">Failed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value warning">{summary.get('skipped', 0)}</div>
                    <div class="metric-label">Skipped</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total', 0)}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
            </div>
        """
    
    # Add API health check
    html_content += """
        <div class="section">
            <h2>🔍 API Health Check</h2>
            <div id="api-health">Checking...</div>
        </div>
        
        <script>
            fetch('https://api.meshai.dev/health')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('api-health').innerHTML = `
                        <div class="metric-card">
                            <div class="metric-value success">✅</div>
                            <div class="metric-label">API Status: ${data.status}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.total_agents}</div>
                            <div class="metric-label">Total Agents</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${Math.floor(data.uptime_seconds / 3600)}h</div>
                            <div class="metric-label">Uptime</div>
                        </div>
                    `;
                })
                .catch(error => {
                    document.getElementById('api-health').innerHTML = `
                        <div class="metric-card">
                            <div class="metric-value failure">❌</div>
                            <div class="metric-label">API Unavailable</div>
                        </div>
                    `;
                });
        </script>
    </body>
    </html>
    """
    
    # Write report
    report_path = Path("test-report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    print(f"✅ Test report generated: {report_path.absolute()}")
    return report_path


def main():
    """Main function"""
    print("📊 Generating MeshAI SDK Test Report")
    print("=" * 50)
    
    # Run tests with coverage and JSON output
    print("Running tests with coverage...")
    success, stdout, stderr = run_command(
        "pytest tests/ --cov=src/meshai --cov-report=xml --json-report --json-report-file=test-results.json -v"
    )
    
    if not success:
        print("⚠️  Tests failed, generating report anyway...")
    
    # Generate HTML report
    report_path = generate_html_report()
    
    print("=" * 50)
    print(f"🎉 Test report complete: {report_path}")
    print("📊 View coverage details in htmlcov/index.html")
    print("🔍 Open test-report.html in your browser")


if __name__ == "__main__":
    main()