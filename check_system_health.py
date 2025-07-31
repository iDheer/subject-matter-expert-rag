#!/usr/bin/env python3
"""
🔍 SME System Health Check and Diagnostic Tool
=============================================

This script performs comprehensive health checks on the SME system
and provides diagnostic information and recommendations.

Run this if you're experiencing issues or want to verify system status.
"""

import os
import sys
import json
import subprocess
import time
import requests
from pathlib import Path
import platform

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def print_status(message, status="INFO"):
    """Print status message with icon"""
    icons = {
        "OK": "✅",
        "WARNING": "⚠️", 
        "ERROR": "❌",
        "INFO": "ℹ️"
    }
    print(f"{icons.get(status, 'ℹ️')} {message}")

def check_python_environment():
    """Check Python version and required packages"""
    print_header("Python Environment Check")
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 10):
        print_status(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}", "OK")
    else:
        print_status(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} - Upgrade recommended", "WARNING")
    
    # Check required packages
    required_packages = [
        "llama_index",
        "elasticsearch", 
        "sentence_transformers",
        "torch",
        "transformers",
        "requests"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"{package} installed", "OK")
        except ImportError:
            print_status(f"{package} missing", "ERROR")

def check_system_resources():
    """Check system resources"""
    print_header("System Resources Check")
    
    # Memory check
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_used_percent = memory.percent
        
        if memory_gb >= 16:
            print_status(f"RAM: {memory_gb:.1f}GB total ({memory_used_percent:.1f}% used)", "OK")
        elif memory_gb >= 8:
            print_status(f"RAM: {memory_gb:.1f}GB total ({memory_used_percent:.1f}% used) - More recommended", "WARNING")
        else:
            print_status(f"RAM: {memory_gb:.1f}GB total ({memory_used_percent:.1f}% used) - Insufficient", "ERROR")
            
        # Disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        disk_used_percent = (disk.used / disk.total) * 100
        
        if disk_free_gb >= 10:
            print_status(f"Disk: {disk_free_gb:.1f}GB free ({disk_used_percent:.1f}% used)", "OK")
        else:
            print_status(f"Disk: {disk_free_gb:.1f}GB free ({disk_used_percent:.1f}% used) - Low space", "WARNING")
            
    except ImportError:
        print_status("psutil not installed - cannot check system resources", "WARNING")

def check_gpu():
    """Check GPU availability"""
    print_header("GPU Check")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print_status(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB VRAM)", "OK")
        else:
            print_status("CUDA not available - will use CPU", "WARNING")
    except ImportError:
        print_status("PyTorch not installed - cannot check GPU", "ERROR")

def check_docker():
    """Check Docker status"""
    print_header("Docker Service Check")
    
    try:
        # Check if Docker is running
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print_status(f"Docker installed: {result.stdout.strip()}", "OK")
            
            # Check running containers
            result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print_status("Docker containers:", "INFO")
                print(result.stdout)
            else:
                print_status("Could not list Docker containers", "WARNING")
        else:
            print_status("Docker not found or not running", "ERROR")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("Docker not found or not responding", "ERROR")

def check_elasticsearch():
    """Check Elasticsearch connection"""
    print_header("Elasticsearch Check")
    
    try:
        response = requests.get('http://localhost:9200', timeout=5)
        if response.status_code == 200:
            es_info = response.json()
            print_status(f"Elasticsearch running: {es_info.get('version', {}).get('number', 'unknown')}", "OK")
            
            # Check cluster health
            health_response = requests.get('http://localhost:9200/_cluster/health', timeout=5)
            if health_response.status_code == 200:
                health = health_response.json()
                status = health.get('status', 'unknown')
                if status == 'green':
                    print_status(f"Cluster health: {status}", "OK")
                elif status == 'yellow':
                    print_status(f"Cluster health: {status}", "WARNING")
                else:
                    print_status(f"Cluster health: {status}", "ERROR")
        else:
            print_status(f"Elasticsearch responded with status {response.status_code}", "ERROR")
    except requests.exceptions.RequestException:
        print_status("Elasticsearch not accessible at localhost:9200", "ERROR")

def check_ollama():
    """Check Ollama service"""
    print_header("Ollama Service Check")
    
    try:
        response = requests.get('http://localhost:11434/api/version', timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            print_status(f"Ollama running: {version_info.get('version', 'unknown')}", "OK")
            
            # Check available models
            models_response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if models_response.status_code == 200:
                models = models_response.json()
                model_list = [model['name'] for model in models.get('models', [])]
                if model_list:
                    print_status(f"Available models: {', '.join(model_list)}", "OK")
                else:
                    print_status("No models installed", "WARNING")
        else:
            print_status(f"Ollama responded with status {response.status_code}", "ERROR")
    except requests.exceptions.RequestException:
        print_status("Ollama not accessible at localhost:11434", "ERROR")

def check_data_directory():
    """Check data directory and files"""
    print_header("Data Directory Check")
    
    data_dirs = ['data_large', 'gpu_chapter_data', 'gpu_chapter_elasticsearch_storage']
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = list(Path(data_dir).rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            dir_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024**2)
            print_status(f"{data_dir}: {file_count} files ({dir_size:.1f}MB)", "OK")
        else:
            print_status(f"{data_dir}: not found", "WARNING")

def check_configuration_files():
    """Check configuration files"""
    print_header("Configuration Files Check")
    
    config_files = [
        'requirements.txt',
        'enhanced_requirements.txt', 
        'docker-compose-elasticsearch.yml',
        'docker-compose-neo4j.yml',
        '.env.example'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print_status(f"{config_file}: found", "OK")
        else:
            print_status(f"{config_file}: missing", "WARNING")

def run_quick_tests():
    """Run quick functionality tests"""
    print_header("Quick Functionality Tests")
    
    # Test Python imports
    test_imports = [
        'llama_index.core',
        'elasticsearch',
        'sentence_transformers',
        'torch'
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print_status(f"Import {module}: OK", "OK")
        except ImportError as e:
            print_status(f"Import {module}: FAILED - {e}", "ERROR")

def generate_recommendations():
    """Generate recommendations based on checks"""
    print_header("Recommendations")
    
    # This would be expanded based on the actual check results
    recommendations = [
        "✅ Run 'pip install -r requirements.txt' to install missing packages",
        "✅ Start Elasticsearch with 'docker-compose -f docker-compose-elasticsearch.yml up -d'",
        "✅ Start Ollama with 'ollama serve' and install models with 'ollama pull qwen3:4b'",
        "✅ Add sample documents to the data_large/ directory",
        "✅ Run 'python SME_1_build_elasticsearch_database.py' to build the database",
        "✅ Test the system with 'python SME_2_query_elasticsearch_system.py'",
        "⚠️  Consider upgrading RAM to 16GB+ for better performance",
        "⚠️  Monitor GPU temperature during intensive operations",
        "ℹ️  Check the SME_SETUP_MANUAL.md for detailed troubleshooting"
    ]
    
    for rec in recommendations:
        print(rec)

def main():
    """Main diagnostic function"""
    print("🔍 SME System Health Check")
    print("=" * 60)
    print("This diagnostic tool will check your SME system configuration")
    print("and identify any issues that need attention.\n")
    
    start_time = time.time()
    
    # Run all checks
    check_python_environment()
    check_system_resources()
    check_gpu()
    check_docker()
    check_elasticsearch()
    check_ollama()
    check_data_directory()
    check_configuration_files()
    run_quick_tests()
    generate_recommendations()
    
    # Summary
    end_time = time.time()
    print_header("Summary")
    print_status(f"Health check completed in {end_time - start_time:.1f} seconds", "INFO")
    print_status("Review any ERROR or WARNING items above", "INFO")
    print_status("Refer to SME_SETUP_MANUAL.md for detailed solutions", "INFO")
    
    print("\n🎯 Next Steps:")
    print("1. Fix any ERROR items before proceeding")
    print("2. Address WARNING items for optimal performance")
    print("3. Run 'python SME_1_build_elasticsearch_database.py' if system is healthy")
    print("4. Start querying with 'python SME_2_query_elasticsearch_system.py'")

if __name__ == "__main__":
    main()
