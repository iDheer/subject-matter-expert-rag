#!/usr/bin/env python3
"""
🚀 QUICK SETUP AND INSTALLATION SCRIPT
====================================

This script helps you quickly set up the Enhanced Knowledge Graph System
with all dependencies and services.

Run this first before testing the system!
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("🧠 ENHANCED KNOWLEDGE GRAPH SYSTEM - QUICK SETUP")
    print("=" * 60)
    print("This script will help you set up all dependencies and services")
    print("for the Enhanced Chapter-Based Knowledge Graph System.")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        # Install main requirements
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'enhanced_requirements.txt'
        ], check=True)
        
        print("✅ Python dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def install_spacy_model():
    """Install spaCy English model"""
    print("\n🧠 Installing spaCy English model...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'
        ], check=True)
        
        print("✅ spaCy model installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install spaCy model: {e}")
        print("💡 Try manually: python -m spacy download en_core_web_sm")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        
        # Download required datasets
        datasets = [
            'punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker',
            'words', 'stopwords', 'wordnet', 'omw-1.4'
        ]
        
        for dataset in datasets:
            try:
                nltk.download(dataset, quiet=True)
            except:
                pass  # Some datasets might already exist
        
        print("✅ NLTK data downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def check_service(name, url, timeout=5):
    """Check if a service is running"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {name} is running")
            return True
        else:
            print(f"⚠️ {name} responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print(f"❌ {name} is not running")
        return False

def check_services():
    """Check if required services are running"""
    print("\n🔍 Checking required services...")
    
    services = {
        'Elasticsearch': 'http://localhost:9200',
        'Ollama': 'http://localhost:11434/api/tags'
    }
    
    all_running = True
    
    for name, url in services.items():
        if not check_service(name, url):
            all_running = False
    
    # Check Neo4j separately (different protocol)
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "knowledge123"))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        print("✅ Neo4j is running")
    except Exception:
        print("❌ Neo4j is not running")
        all_running = False
    
    return all_running

def setup_ollama_model():
    """Setup Ollama model"""
    print("\n🤖 Checking Ollama model...")
    
    try:
        # Check if qwen3:4b is available
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            if any('qwen3:4b' in name for name in model_names):
                print("✅ qwen3:4b model is available")
                return True
            else:
                print("⚠️ qwen3:4b model not found")
                print("💡 Install with: ollama pull qwen3:4b")
                return False
        else:
            print("❌ Could not check Ollama models")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check Ollama: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'gpu_knowledge_graph_data',
        'enhanced_visualizations',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def show_service_setup_instructions():
    """Show instructions for setting up services"""
    print("\n🛠️ SERVICE SETUP INSTRUCTIONS")
    print("=" * 40)
    
    print("\n1. Elasticsearch:")
    print("   docker-compose -f docker-compose-elasticsearch.yml up -d")
    
    print("\n2. Neo4j:")
    print("   - Install Neo4j Desktop from https://neo4j.com/download/")
    print("   - Create a database with password: knowledge123")
    print("   - Or use Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/knowledge123 neo4j")
    
    print("\n3. Ollama:")
    print("   - Install Ollama from https://ollama.ai/")
    print("   - Start: ollama serve")
    print("   - Install model: ollama pull qwen3:4b")

def check_gpu():
    """Check GPU availability"""
    print("\n🚀 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️ No GPU available, will use CPU")
            return False
    except Exception as e:
        print(f"⚠️ Could not check GPU: {e}")
        return False

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    print("\n" + "="*60)
    print("INSTALLING DEPENDENCIES")
    print("="*60)
    
    success = True
    
    # Install Python packages
    if not install_python_dependencies():
        success = False
    
    # Install spaCy model
    if not install_spacy_model():
        success = False
    
    # Download NLTK data
    if not download_nltk_data():
        success = False
    
    # Create directories
    create_directories()
    
    # Check GPU
    check_gpu()
    
    # Check services
    print("\n" + "="*60)
    print("CHECKING SERVICES")
    print("="*60)
    
    services_ok = check_services()
    
    if not services_ok:
        print("\n⚠️ Some services are not running!")
        show_service_setup_instructions()
        print("\n💡 Start the services and run this script again to verify.")
    else:
        print("\n🎉 All services are running!")
    
    # Check Ollama model
    setup_ollama_model()
    
    # Final summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    if success and services_ok:
        print("🎉 SETUP COMPLETE! Ready to test the system.")
        print("\nNext steps:")
        print("  1. Run: python KG_SYSTEM_TESTER.py")
        print("  2. Or run individual components manually")
        print("  3. Check KNOWLEDGE_GRAPH_QUICKSTART_TEST.md for details")
    else:
        print("⚠️ Setup completed with some issues.")
        print("Please fix the issues above before testing.")
    
    return success and services_ok

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    exit(0 if success else 1)
