#!/usr/bin/env python3
"""
Check and configure Ollama for GPU usage
"""
import os
import subprocess
import requests
import json
import torch

def check_gpu_availability():
    """Check if CUDA is available"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"🟢 CUDA Available: {gpu_count} GPU(s)")
        print(f"   GPU 0: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        print("🔴 CUDA not available")
        return False

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version = response.json().get("version", "unknown")
            print(f"🟢 Ollama service is running (version: {version})")
            return True
    except:
        pass
    
    print("🔴 Ollama service is not running")
    return False

def check_ollama_models():
    """Check available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"🟢 Available Ollama models: {len(models)}")
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0) / (1024**3)  # GB
                print(f"   - {name} ({size:.1f} GB)")
            return models
    except Exception as e:
        print(f"🔴 Error checking models: {e}")
    return []

def pull_qwen_model():
    """Pull Qwen 4B model if not available"""
    try:
        print("📥 Pulling qwen3:4b model...")
        result = subprocess.run(
            ["ollama", "pull", "qwen3:4b"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        if result.returncode == 0:
            print("✅ qwen3:4b model pulled successfully")
            return True
        else:
            print(f"❌ Error pulling model: {result.stderr}")
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
    return False

def configure_ollama_gpu():
    """Configure Ollama to use GPU"""
    print("🔧 Configuring Ollama for GPU usage...")
    
    # Set environment variables
    env_vars = {
        "CUDA_VISIBLE_DEVICES": "0",
        "OLLAMA_GPU_LAYERS": "35",  # Use GPU layers
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   Set {key}={value}")
    
    print("✅ Environment configured for GPU usage")

def test_model_inference():
    """Test model with a simple query"""
    try:
        print("🧪 Testing model inference...")
        
        data = {
            "model": "qwen3:4b",
            "prompt": "What is 2+2?",
            "stream": False,
            "options": {
                "num_gpu": 1,
                "gpu_memory_utilization": 0.8
            }
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            print(f"✅ Model response: {answer}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"❌ Error testing model: {e}")
    
    return False

def main():
    print("🔍 Checking Ollama GPU Configuration\n")
    
    # Check GPU
    gpu_available = check_gpu_availability()
    print()
    
    # Check Ollama service
    ollama_running = check_ollama_service()
    print()
    
    if not ollama_running:
        print("❌ Ollama is not running. Please start it with:")
        print("   ollama serve")
        return
    
    # Check models
    models = check_ollama_models()
    qwen_available = any("qwen3:4b" in model.get("name", "") for model in models)
    print()
    
    if not qwen_available:
        print("📥 qwen3:4b model not found. Pulling it...")
        if not pull_qwen_model():
            return
        print()
    else:
        print("✅ qwen3:4b model is available")
    
    # Configure for GPU
    if gpu_available:
        configure_ollama_gpu()
        print()
    
    # Test inference
    test_model_inference()
    
    print("\n🎉 Ollama GPU configuration complete!")
    if gpu_available:
        print("💡 Tips:")
        print("   - Restart Ollama service if needed: ollama serve")
        print("   - Monitor GPU usage: nvidia-smi")
        print("   - The model should now load in GPU memory")

if __name__ == "__main__":
    main()
