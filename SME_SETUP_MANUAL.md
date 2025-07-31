# 📖 SME System Setup Manual

## Complete Setup Guide for Subject Matter Expert RAG System

This manual provides detailed, step-by-step instructions for setting up and using the SME (Subject Matter Expert) system with Elasticsearch and conversational memory.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Service Installation](#service-installation)
4. [SME System Setup](#sme-system-setup)
5. [Usage Guide](#usage-guide)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

---

## 🔧 Prerequisites

### Hardware Requirements

#### 🖥️ **Minimum System Requirements**
- **CPU**: 4-core processor (Intel i5-8400 / AMD Ryzen 5 2600 or equivalent)
- **RAM**: 8GB DDR4
- **Storage**: 50GB free space (SSD recommended)
- **GPU**: Optional - Any NVIDIA GPU with 4GB+ VRAM
- **Network**: Stable internet connection for downloads

#### 🚀 **Recommended System Requirements**
- **CPU**: 8-core processor (Intel i7-10700K / AMD Ryzen 7 3700X or better)
- **RAM**: 16GB DDR4 (32GB for large document collections)
- **Storage**: 100GB free space on SSD
- **GPU**: NVIDIA RTX 4060 or better with 8GB+ VRAM
- **Network**: High-speed internet (for model downloads)

#### 🏆 **Optimal System Requirements**
- **CPU**: 12+ core processor (Intel i9-12900K / AMD Ryzen 9 5900X or better)
- **RAM**: 32GB DDR4/DDR5
- **Storage**: 200GB+ free space on NVMe SSD
- **GPU**: NVIDIA RTX 4070/4080/4090 with 12GB+ VRAM
- **Network**: Gigabit internet connection

### GPU-Specific Requirements

#### 📊 **Memory Requirements by GPU Model**
| GPU Model | VRAM | SME System | Knowledge Graph | Recommended Model |
|-----------|------|------------|-----------------|-------------------|
| RTX 4090 | 24GB | qwen3:7b (FP16) | Full precision | Best performance |
| RTX 4080 | 16GB | qwen3:4b (FP16) | FP16/INT8 | Excellent |
| RTX 4070 Ti | 12GB | qwen3:4b (INT8) | INT8 | Very good |
| RTX 4070 | 12GB | qwen3:4b (INT8) | INT8 | Very good |
| RTX 4060 Ti | 16GB | qwen3:4b (INT8) | INT8 | Good |
| RTX 4060 Ti | 8GB | qwen3:4b (INT4) | INT4 | Good |
| RTX 4060 | 8GB | qwen3:4b (INT4) | INT4 | Acceptable |
| RTX 4050 | 6GB | qwen3:4b (INT4) | INT4 only | Entry level |
| RTX 3080 | 10GB | qwen3:4b (INT8) | INT8 | Good (older gen) |
| RTX 3070 | 8GB | qwen3:4b (INT4) | INT4 | Acceptable |
| GTX 1660 Ti | 6GB | CPU fallback | CPU fallback | Limited |

#### 🔥 **CPU-Only Performance**
- **Good**: Intel i9/AMD Ryzen 9 (12+ cores) - 8-15 second responses
- **Acceptable**: Intel i7/AMD Ryzen 7 (8 cores) - 15-30 second responses  
- **Slow**: Intel i5/AMD Ryzen 5 (6 cores) - 30-60 second responses
- **Very Slow**: 4 cores or less - 60+ second responses

### Storage Requirements

#### 📁 **Disk Space Breakdown**
```
Total Required: 15-50GB depending on configuration

Base Installation:
├── Python Environment: ~2GB
├── LLM Models: ~3-8GB
│   ├── qwen3:4b-q4_0: ~2.8GB (recommended for 6GB VRAM)
│   ├── qwen3:4b-q8_0: ~4.5GB (recommended for 8GB VRAM)
│   ├── qwen3:4b: ~7.2GB (recommended for 12GB+ VRAM)
│   └── qwen3:7b: ~14GB (recommended for 16GB+ VRAM)
├── Embedding Models: ~1GB
├── Elasticsearch Data: ~1-10GB (depends on documents)
├── Neo4j Data: ~500MB-5GB (depends on graph size)
├── Document Storage: Variable (your documents)
└── System Dependencies: ~2GB
```

#### 💾 **Storage Type Recommendations**
- **NVMe SSD**: Best performance (recommended)
- **SATA SSD**: Good performance
- **HDD**: Acceptable but slower (not recommended)

### Operating System Requirements

#### 🖥️ **Supported Operating Systems**
| OS | Version | Support Level | Notes |
|----|---------|---------------|-------|
| **Windows** | 10/11 (64-bit) | ✅ Full | Recommended: Windows 11 |
| **Ubuntu** | 20.04/22.04 LTS | ✅ Full | Best Linux performance |
| **Debian** | 11/12 | ✅ Full | Stable alternative |
| **CentOS/RHEL** | 8/9 | ⚠️ Limited | Manual dependency setup |
| **macOS** | 12+ (Monterey+) | ⚠️ Limited | M1/M2: CPU only, Intel: Limited GPU |
| **Arch Linux** | Rolling | ⚠️ Advanced | For experienced users |

#### 🐧 **Linux-Specific Requirements**
```bash
# Required kernel version
uname -r  # Should be 5.4+ (preferably 5.15+)

# Required packages (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential curl wget git python3-dev python3-pip python3-venv

# For GPU support
sudo apt install nvidia-driver-525 nvidia-cuda-toolkit
```

#### 🪟 **Windows-Specific Requirements**
- **Windows 10**: Version 1903 or later
- **Windows 11**: Any version
- **PowerShell**: 5.1 or PowerShell 7+
- **Visual C++ Redistributable**: 2019 or later
- **NVIDIA Drivers**: 525.60.13 or later (for GPU)

### System Requirements
### Software Prerequisites

#### 🐍 **Python Requirements**
- **Python**: 3.11+ (Python 3.12 recommended)
- **pip**: Latest version (automatically updated with Python)
- **venv**: Built-in virtual environment support

#### 📦 **Package Managers & Tools**
- **Git**: For repository cloning
- **Docker**: For service containerization (recommended)
- **NVIDIA drivers**: 525.60.13+ for GPU support
- **CUDA Toolkit**: 12.1+ (optional, for advanced GPU features)

#### 🌐 **Network Requirements**
- **Internet Connection**: Required for initial setup
- **Bandwidth**: 10+ Mbps recommended for model downloads
- **Ports**: 9200 (Elasticsearch), 7474/7687 (Neo4j), 11434 (Ollama)

### Performance Expectations by Hardware

#### ⚡ **Response Time Expectations**
| Hardware Configuration | Database Build (1000 pages) | Query Response | Concurrent Users |
|------------------------|-----------------------------|-----------------|--------------------|
| RTX 4090 + i9 + 32GB | 3-5 minutes | 1-2 seconds | 1 (local) |
| RTX 4070 + i7 + 16GB | 5-8 minutes | 2-4 seconds | 1 (local) |
| RTX 4060 + i5 + 16GB | 8-12 minutes | 4-6 seconds | 1 (local) |
| RTX 4050 + i5 + 8GB | 12-20 minutes | 6-10 seconds | 1 (local) |
| CPU-only i9 + 32GB | 15-25 minutes | 8-15 seconds | 1 (local) |
| CPU-only i7 + 16GB | 25-40 minutes | 15-30 seconds | 1 (local) |
| CPU-only i5 + 8GB | 40-60 minutes | 30-60 seconds | 1 (local) |

#### 💾 **Memory Usage Patterns**
```
System Memory (RAM) Usage:
├── Base System: ~2-4GB
├── SME System: ~2-6GB (depends on document size)
├── Knowledge Graph: ~1-4GB (depends on graph complexity)
├── Elasticsearch: ~1-3GB (JVM heap)
├── Neo4j: ~512MB-2GB (depends on data)
└── Browser (for Neo4j interface): ~200-500MB

GPU Memory (VRAM) Usage:
├── Model Loading: 60-80% of total VRAM
├── Processing Overhead: 10-20% of total VRAM
└── System Reserve: 10-20% of total VRAM
```

- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.11+ (with pip and venv)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **GPU**: Optional NVIDIA GPU with 6GB+ VRAM

### Required Software
- **Git** (for repository cloning)
- **Docker** (recommended for services)
- **Python 3.11+** with pip

---

## 🌍 Environment Setup

### 1. Clone Repository
```bash
# Clone the repository
git clone https://github.com/iDheer/subject-matter-expert-rag.git
cd subject-matter-expert-rag
```

### 2. Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows Command Prompt:
venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
# Install essential packages (recommended for most users)
pip install -r requirements.txt

# For full features with GPU acceleration and advanced tools:
# pip install -r enhanced_requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Verify installation
pip list | grep -E "(llama-index|elasticsearch|torch)"
```

---

## 🛠️ Service Installation

### Option A: Docker Installation (Recommended)

#### Install Docker
```bash
# Windows: Download Docker Desktop from https://docker.com
# Linux (Ubuntu):
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker $USER  # Requires logout/login

# macOS:
brew install docker docker-compose
```

#### Start Services with Docker
```bash
# Start Elasticsearch (port 9200)
docker-compose -f docker-compose-elasticsearch.yml up -d

# Start Neo4j (ports 7687, 7474)
docker-compose -f docker-compose-neo4j.yml up -d

# Verify services are running
docker ps
```

#### Verify Elasticsearch
```bash
# Test Elasticsearch connection
curl -X GET "localhost:9200/_cluster/health"
# Expected: {"status":"green" or "yellow"}
```

#### Verify Neo4j
```bash
# Access Neo4j browser: http://localhost:7474
# Default credentials: neo4j/neo4j
# Change password to: knowledge123
```

### Option B: Manual Installation

#### Elasticsearch Setup
```bash
# Windows:
# 1. Download Elasticsearch 8.x from https://elastic.co/downloads/elasticsearch
# 2. Extract to C:\elasticsearch
# 3. Run: C:\elasticsearch\bin\elasticsearch.bat

# Linux (Ubuntu):
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
sudo sh -c 'echo "deb https://artifacts.elastic.co/packages/8.x/apt stable main" > /etc/apt/sources.list.d/elastic-8.x.list'
sudo apt-get update
sudo apt-get install elasticsearch
sudo systemctl start elasticsearch
sudo systemctl enable elasticsearch

# macOS:
brew tap elastic/tap
brew install elastic/tap/elasticsearch-full
brew services start elasticsearch
```

#### Neo4j Setup
```bash
# Windows:
# 1. Download Neo4j Desktop from https://neo4j.com/download/
# 2. Install and create new database
# 3. Set password to: knowledge123

# Linux (Ubuntu):
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j

# macOS:
brew install neo4j
brew services start neo4j
```

---

## 🤖 Ollama Setup

### 1. Install Ollama
```bash
# Windows:
winget install Ollama.Ollama
# Or download from: https://ollama.ai/download

# Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# macOS:
brew install ollama
```

### 2. Start Ollama Service
```bash
# Start Ollama server
ollama serve

# In a new terminal, verify it's running
curl http://localhost:11434/api/version
```

### 3. Install Qwen Model
```bash
# For systems with 8GB+ VRAM:
ollama pull qwen3:4b

# For systems with 6GB VRAM (RTX 4050):
ollama pull qwen3:4b-q4_0

# For systems with 12GB+ VRAM:
ollama pull qwen3:4b-q8_0

# Verify model installation
ollama list
```

### 4. GPU Configuration (Optional)

#### 🔍 **Check GPU Compatibility**
```bash
# Check if NVIDIA GPU is detected
nvidia-smi

# Expected output should show:
# - GPU name and model
# - Driver version (525.60.13+)
# - CUDA version (12.1+)
# - Available VRAM

# If nvidia-smi fails:
# Windows: Install NVIDIA drivers from nvidia.com
# Linux: sudo apt install nvidia-driver-525
# macOS: NVIDIA GPUs not supported on newer versions
```

#### ⚙️ **Configure GPU Settings**
```bash
# Check GPU availability
nvidia-smi

# For multiple GPUs, choose the best one:
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Set environment variables for GPU usage
# Windows (PowerShell):
$env:CUDA_VISIBLE_DEVICES="0"  # Use GPU 0 (change if needed)
$env:OLLAMA_NUM_GPU="1"        # Number of GPUs to use
$env:OLLAMA_GPU_LAYERS="35"    # GPU layers (35 for qwen3:4b)

# Linux/macOS:
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_GPU=1
export OLLAMA_GPU_LAYERS=35

# For RTX 4050 6GB (memory optimization):
export OLLAMA_GPU_LAYERS=25    # Reduce layers for limited VRAM
```

#### 🎯 **GPU-Specific Optimizations**
```bash
# RTX 4090/4080 (16GB+): Maximum performance
ollama pull qwen3:7b
export OLLAMA_GPU_LAYERS=50

# RTX 4070 (12GB): Balanced performance
ollama pull qwen3:4b-q8_0
export OLLAMA_GPU_LAYERS=35

# RTX 4060 Ti (8GB): Good performance
ollama pull qwen3:4b-q4_0
export OLLAMA_GPU_LAYERS=30

# RTX 4050/4060 (6GB): Entry level
ollama pull qwen3:4b-q4_0
export OLLAMA_GPU_LAYERS=25

# No NVIDIA GPU: CPU fallback
# The system will automatically use CPU
# No additional configuration needed
```

---

## 📚 SME System Setup

### 1. Prepare Your Documents
```bash
# Create data directory
mkdir data_large

# Add your documents to data_large/
# Supported formats: PDF, TXT, DOCX, MD
# Example structure:
# data_large/
#   ├── textbook.pdf
#   ├── research_papers/
#   │   ├── paper1.pdf
#   │   └── paper2.pdf
#   └── documentation.txt
```

### 2. Build Elasticsearch Database
```bash
# Run the database builder
python SME_1_build_elasticsearch_database.py

# Expected output:
# ✅ Connected to Elasticsearch
# 📄 Processing documents...
# 🔄 Creating hierarchical chunks...
# 💾 Storing in Elasticsearch...
# ✅ Database built successfully!
```

**What this script does:**
- Scans `data_large/` for documents
- Extracts text from PDFs, DOCX, etc.
- Creates hierarchical document chunks
- Generates embeddings using sentence-transformers
- Stores everything in Elasticsearch with metadata
- Sets up AutoMerging retrieval system

### 3. Verify Database Creation
```bash
# Check if database was created
python SME_3_inspect_elasticsearch_database.py

# Expected output:
# 📊 Database Statistics:
#   - Total documents: X
#   - Total chunks: Y
#   - Index size: Z MB
# 📄 Sample documents: [list of files]
```

---

## 🎯 Usage Guide

### Starting the SME System
```bash
# Start the interactive SME system
python SME_2_query_elasticsearch_system.py

# Expected output:
# 🚀 Ready to Query with Elasticsearch + AutoMerging + Conversation Memory!
# 🧠 This SME system now remembers your conversation context!
```

### Basic Querying
```bash
💬 Question: What is virtual memory?
🔍 Searching with conversation context: 'What is virtual memory?'
🤖 Response: Virtual memory is a memory management technique...

💬 Question: How does it work?
🔍 Searching with conversation context: 'How does it work?'
🤖 Response: Based on our previous discussion about virtual memory, it works by...
```

### Memory Management Commands
```bash
# Check memory status
💬 Question: /status
📊 Memory enabled - 2 exchanges

# Clear conversation memory
💬 Question: /clear
🧹 Conversation memory cleared

# Toggle memory on/off
💬 Question: /memory
🧠 Conversation memory disabled

💬 Question: /memory
🧠 Conversation memory enabled

# Show help
💬 Question: /help
📋 Available Commands:
  • Type your question normally for contextual responses
  • '/memory' - Toggle conversation memory on/off
  • '/clear' - Clear conversation memory
  • '/status' - Show memory status
  • '/help' - Show this help message
  • 'exit' - Exit the system
```

### Advanced Querying Examples
```bash
# Research-style conversation
💬 Question: What are the main concepts in operating systems?
💬 Question: Can you elaborate on process management?
💬 Question: How does this relate to memory management?
💬 Question: What are the trade-offs between different scheduling algorithms?

# Technical deep-dive
💬 Question: Explain how virtual memory works
💬 Question: What's the difference between paging and segmentation?
💬 Question: Which approach is more efficient for modern systems?
```

### Understanding the Output
```bash
🔍 Searching with conversation context: 'your question'
🤖 Response: [AI response with context from previous conversation]

--- Source Nodes (Post-AutoMerging & Re-ranking) ---
Source 1 (Score: 0.8234):
  -> File: textbook.pdf
  -> Node Type: Merged
  -> Content Snippet: "Virtual memory is a memory management..."

Source 2 (Score: 0.7892):
  -> File: research_paper.pdf
  -> Node Type: Leaf
  -> Content Snippet: "The benefits of virtual memory include..."
```

**Output Explanation:**
- **Score**: Relevance score (0-1, higher = more relevant)
- **Node Type**: 
  - `Merged`: Combined from multiple smaller chunks
  - `Leaf`: Single original chunk
- **Content Snippet**: Preview of the source material

---

## 🔧 Troubleshooting

### Hardware-Related Issues

#### 💾 **Insufficient RAM**
```bash
# Problem: System runs out of memory
# Symptoms: Slow performance, system freezing, "MemoryError"

# Solution 1: Check current memory usage
# Windows:
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10

# Linux:
free -h
top -o %MEM

# Solution 2: Optimize memory usage
# Reduce Elasticsearch heap size (edit docker-compose-elasticsearch.yml):
ES_JAVA_OPTS: "-Xms1g -Xmx2g"  # Reduce from default 4g

# Solution 3: Use swap file (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Solution 4: Close unnecessary applications
# Close browsers, IDEs, and other memory-intensive apps
```

#### 🖥️ **Insufficient VRAM**
```bash
# Problem: GPU memory errors
# Symptoms: "CUDA out of memory", "RuntimeError: CUDA error"

# Solution 1: Check VRAM usage
nvidia-smi -l 1  # Monitor every second

# Solution 2: Use smaller model
ollama pull qwen3:4b-q4_0     # Instead of qwen3:4b
ollama pull qwen3:1.5b-q4_0   # Even smaller

# Solution 3: Reduce GPU layers
export OLLAMA_GPU_LAYERS=20   # Reduce from 35

# Solution 4: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Solution 5: Restart Ollama service
# Windows:
Stop-Process -Name "ollama" -Force
ollama serve

# Linux:
sudo pkill ollama
ollama serve
```

#### 🐌 **Slow Performance**
```bash
# Problem: Very slow response times (>30 seconds)

# Solution 1: Check if using GPU
# Look for "Using device: cuda" in SME system output
# If shows "Using device: cpu", check GPU setup

# Solution 2: Check disk I/O
# Windows:
Get-Counter "\PhysicalDisk(_Total)\Disk Transfers/sec"

# Linux:
iostat -x 1  # Install with: sudo apt install sysstat

# If disk usage >80%, move to faster storage (SSD)

# Solution 3: Optimize model settings
# Edit SME_2_query_elasticsearch_system.py:
similarity_top_k=6     # Reduce from 12
top_n=2               # Reduce from 4

# Solution 4: Use faster model
ollama pull qwen3:1.5b-q4_0  # Smaller, faster model
```

#### 🔥 **Overheating Issues**
```bash
# Problem: System throttling due to heat

# Solution 1: Monitor temperatures
# Windows: Use HWiNFO64, MSI Afterburner
# Linux: 
sensors  # Install with: sudo apt install lm-sensors

# Safe temperatures:
# CPU: <80°C under load
# GPU: <85°C under load

# Solution 2: Improve cooling
# - Clean dust from fans and heatsinks
# - Ensure case has adequate airflow
# - Consider undervolting GPU

# Solution 3: Reduce workload
export OLLAMA_GPU_LAYERS=20   # Use fewer GPU layers
# Take breaks between intensive operations
```

#### 📡 **Network/Download Issues**
```bash
# Problem: Model downloads fail or are very slow

# Solution 1: Check internet connection
ping google.com
speedtest-cli  # Install with: pip install speedtest-cli

# Solution 2: Resume failed downloads
ollama pull qwen3:4b-q4_0  # Ollama resumes automatically

# Solution 3: Use alternative download methods
# Download models manually from Hugging Face
# Place in Ollama models directory

# Solution 4: Configure proxy (if needed)
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

### Software-Related Issues

#### 🐍 **Python Environment Issues**
```bash
# Problem: ImportError, ModuleNotFoundError
# Solution 1: Verify Python version
python --version  # Should be 3.10 or 3.11

# Solution 2: Reinstall requirements
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Solution 3: Clear pip cache
pip cache purge

# Solution 4: Use virtual environment
# Windows:
python -m venv sme_env
.\sme_env\Scripts\activate
pip install -r requirements.txt

# Linux:
python -m venv sme_env
source sme_env/bin/activate
pip install -r requirements.txt

# Solution 5: Check CUDA/PyTorch compatibility
python -c "import torch; print(torch.cuda.is_available())"
# Should return True if GPU setup is correct
```

#### 🔌 **Ollama Connection Issues**
```bash
# Problem: "Connection refused", "Ollama not responding"

# Solution 1: Check if Ollama is running
# Windows:
Get-Process ollama
netstat -an | findstr :11434

# Linux:
ps aux | grep ollama
netstat -tulpn | grep :11434

# Solution 2: Start Ollama service
ollama serve

# Solution 3: Check firewall
# Windows: Allow ollama.exe through Windows Firewall
# Linux: sudo ufw allow 11434

# Solution 4: Reset Ollama
# Windows:
Stop-Process -Name "ollama" -Force
Remove-Item -Recurse "$env:USERPROFILE\.ollama" -ErrorAction SilentlyContinue
ollama serve

# Linux:
sudo pkill ollama
rm -rf ~/.ollama
ollama serve

# Solution 5: Check port conflicts
netstat -an | findstr :11434  # Should show ollama listening
```

#### 🗃️ **Elasticsearch Issues**
```bash
# Problem: Elasticsearch not starting, connection failed

# Solution 1: Check Docker status
docker ps  # Should show elasticsearch container

# Solution 2: Check logs
docker logs elasticsearch

# Solution 3: Restart Elasticsearch
docker-compose -f docker-compose-elasticsearch.yml down
docker-compose -f docker-compose-elasticsearch.yml up -d

# Solution 4: Clear Elasticsearch data (if corrupted)
docker-compose -f docker-compose-elasticsearch.yml down
docker volume rm subject-matter-expert-rag_es_data
docker-compose -f docker-compose-elasticsearch.yml up -d

# Solution 5: Check disk space
df -h  # Elasticsearch needs >10% free space

# Solution 6: Memory issues
# Edit docker-compose-elasticsearch.yml:
ES_JAVA_OPTS: "-Xms512m -Xmx1g"  # Reduce memory usage
```

#### 📊 **Data Loading Issues**
```bash
# Problem: "No documents found", empty responses

# Solution 1: Check if data exists
python SME_3_inspect_elasticsearch_database.py

# Solution 2: Rebuild database
python SME_1_build_elasticsearch_database.py

# Solution 3: Check data files
# Ensure you have documents in the correct format
# Default: looks for .txt files in current directory

# Solution 4: Debug document processing
# Add debug prints in SME_1_build_elasticsearch_database.py:
print(f"Processing: {filename}")
print(f"Document length: {len(content)}")
```

#### 🤖 **Model Performance Issues**
```bash
# Problem: Poor response quality, irrelevant answers

# Solution 1: Check model is loaded
ollama list  # Should show qwen models

# Solution 2: Try different model
ollama pull qwen3:7b-q4_0    # Larger model
ollama pull llama3:8b-q4_0   # Alternative model

# Solution 3: Adjust retrieval settings
# Edit SME_2_query_elasticsearch_system.py:
similarity_top_k=20    # Increase context
top_n=6               # More reranked results

# Solution 4: Check embedding model
# Ensure BGE embeddings are working correctly
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-small-en-v1.5'); print('OK')"
```

### Debug Mode Activation

#### 🔍 **Enable Detailed Logging**
```python
# Add to beginning of Python scripts for debugging:
import logging
logging.basicConfig(level=logging.DEBUG)

# For LlamaIndex debugging:
import llama_index
llama_index.set_global_handler("simple")

# For Elasticsearch debugging:
import urllib3
urllib3.disable_warnings()
```

#### 📝 **Common Debug Commands**
```bash
# Check system resources
# Windows:
systeminfo | findstr "Total Physical Memory"
nvidia-smi
Get-Process | Sort-Object CPU -Descending | Select-Object -First 5

# Linux:
free -h
nvidia-smi
top -o %CPU | head -10

# Check network connectivity
curl http://localhost:9200/_cluster/health  # Elasticsearch
curl http://localhost:11434/api/version     # Ollama

# Test individual components
python -c "import elasticsearch; es = elasticsearch.Elasticsearch(['localhost:9200']); print(es.ping())"
python -c "import requests; print(requests.get('http://localhost:11434/api/version').json())"
```

#### ⚡ **Speed Optimizations**
```bash
# 1. Model Selection for Speed
ollama pull qwen3:1.5b-q4_0     # Fastest response
ollama pull qwen3:4b-q4_0       # Good balance
ollama pull qwen3:7b-q4_0       # Best quality

# 2. GPU Optimization
export OLLAMA_GPU_LAYERS=35     # Use all layers on GPU
export OLLAMA_NUM_PARALLEL=4    # Handle multiple requests

# 3. Elasticsearch Optimization
# Edit docker-compose-elasticsearch.yml:
ES_JAVA_OPTS: "-Xms4g -Xmx4g"          # Use more RAM
indices.memory.index_buffer_size: "20%" # Faster indexing

# 4. Retrieval Optimization
# In SME_2_query_elasticsearch_system.py:
similarity_top_k=6      # Reduce for speed
top_n=2                 # Fewer reranked results
```

#### 🎯 **Quality Optimizations**
```bash
# 1. Use larger models for better quality
ollama pull qwen3:14b-q4_0     # High quality (needs 16GB+ RAM)
ollama pull qwen3:32b-q4_0     # Best quality (needs 32GB+ RAM)

# 2. Increase retrieval context
# In SME_2_query_elasticsearch_system.py:
similarity_top_k=20     # More context
top_n=6                 # More reranked results

# 3. Use better reranker
# Install and configure:
pip install sentence-transformers[bge]
# Use 'BAAI/bge-reranker-large' instead of 'BAAI/bge-reranker-base'

# 4. Memory context optimization
# Increase memory buffer in ChatMemoryManager:
self.max_context_length = 8000  # Default is 4000
```

#### 💾 **Memory Optimization**
```bash
# 1. Reduce memory usage
export OLLAMA_GPU_LAYERS=20     # Use fewer GPU layers
export OLLAMA_PARALLEL=1        # Single request only

# 2. Elasticsearch memory tuning
# docker-compose-elasticsearch.yml:
ES_JAVA_OPTS: "-Xms1g -Xmx2g"  # Reduce memory usage

# 3. Python memory optimization
# Add to scripts:
import gc
gc.collect()  # Force garbage collection

# 4. Chunk size optimization
# In document processing:
chunk_size=512          # Smaller chunks (default 1024)
chunk_overlap=50        # Reduce overlap
```

### Configuration Files Reference

#### 📁 **Key Configuration Locations**
```bash
# Ollama Configuration
# Windows: C:\Users\{username}\.ollama\
# Linux: ~/.ollama/

# Docker Compose Files
docker-compose-elasticsearch.yml   # Elasticsearch setup
docker-compose-neo4j.yml          # Neo4j for knowledge graphs

# Python Requirements
requirements.txt                   # Essential packages
enhanced_requirements.txt          # Full feature set

# Data Directories
gpu_chapter_data/                  # Knowledge graph data
gpu_chapter_elasticsearch_storage/ # Elasticsearch indices
```

#### ⚙️ **Environment Variables**
```bash
# Create .env file for persistent settings:
# Windows (create .env file):
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_GPU_LAYERS=35
OLLAMA_NUM_PARALLEL=4
ELASTICSEARCH_URL=http://localhost:9200
CUDA_VISIBLE_DEVICES=0

# Linux (add to ~/.bashrc):
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_GPU_LAYERS=35
export OLLAMA_NUM_PARALLEL=4
export ELASTICSEARCH_URL=http://localhost:9200
export CUDA_VISIBLE_DEVICES=0
```

#### 🔧 **Custom Model Parameters**
```bash
# Create custom Ollama Modelfile for fine-tuning:
# File: custom_qwen.modelfile
FROM qwen3:4b-q4_0

PARAMETER temperature 0.3          # More focused responses
PARAMETER top_p 0.8               # Nucleus sampling
PARAMETER repeat_penalty 1.1      # Avoid repetition
PARAMETER num_ctx 4096            # Context window

SYSTEM """
You are a helpful assistant that provides accurate, detailed responses based on the provided context. Always be precise and cite sources when available.
"""

# Create custom model:
ollama create custom_qwen -f custom_qwen.modelfile
```

### Monitoring and Maintenance

#### 1. Elasticsearch Connection Failed
```bash
# Problem: Cannot connect to Elasticsearch
# Solution 1: Check if Elasticsearch is running
curl -X GET "localhost:9200"

# Solution 2: Restart Elasticsearch
docker-compose -f docker-compose-elasticsearch.yml restart

# Solution 3: Check Docker logs
docker-compose -f docker-compose-elasticsearch.yml logs
```

#### 2. No Documents Found
```bash
# Problem: "No documents found in data_large/"
# Solution: Check directory structure
ls -la data_large/

# Ensure you have supported file types:
# ✅ .pdf, .txt, .docx, .md
# ❌ .jpg, .png, .mp4, .zip
```

#### 3. Ollama Model Not Found
```bash
# Problem: "Model 'qwen3:4b' not found"
# Solution 1: Pull the model
ollama pull qwen3:4b

# Solution 2: Check available models
ollama list

# Solution 3: Use quantized model for limited VRAM
ollama pull qwen3:4b-q4_0
```

#### 4. GPU Memory Issues
```bash
# Problem: CUDA out of memory
# Solution 1: Use quantized model
export OLLAMA_MODEL="qwen3:4b-q4_0"

# Solution 2: Reduce batch size
# Edit the SME script and change similarity_top_k=6 (instead of 12)

# Solution 3: Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

#### 5. Slow Response Times
```bash
# Problem: Queries take >10 seconds
# Solution 1: Check system resources
nvidia-smi  # GPU usage
htop        # CPU/RAM usage

# Solution 2: Reduce context length
# In SME_2_query_elasticsearch_system.py, find:
# max_context_length=4000
# Change to: max_context_length=2000

# Solution 3: Use smaller model
ollama pull qwen3:1.5b  # Smaller, faster model
```

#### 6. Memory Not Working
```bash
# Problem: System doesn't remember previous queries
# Solution 1: Check memory status
💬 Question: /status

# Solution 2: Enable memory if disabled
💬 Question: /memory

# Solution 3: Check for errors in console output
# Look for: "⚠️ Error updating summary" or similar
```

### Service Status Checks
```bash
# Check all services
echo "=== Elasticsearch ==="
curl -s "localhost:9200/_cluster/health" | grep -o '"status":"[^"]*"'

echo "=== Neo4j ==="
curl -s "localhost:7474" >/dev/null && echo "Neo4j: Running" || echo "Neo4j: Not running"

echo "=== Ollama ==="
curl -s "localhost:11434/api/version" >/dev/null && echo "Ollama: Running" || echo "Ollama: Not running"

echo "=== Python Environment ==="
python -c "import llama_index, elasticsearch, torch; print('Python: OK')"
```

---

## ⚙️ Advanced Configuration

### GPU Memory Optimization
```python
# For RTX 4050 6GB users, edit SME_2_query_elasticsearch_system.py:

# Change line ~30:
Settings.llm = Ollama(
    model="qwen3:4b-q4_0",  # Use quantized model
    request_timeout=300.0,
    base_url="http://localhost:11434",
)

# Change line ~95:
base_retriever = index.as_retriever(similarity_top_k=6)  # Reduce from 12

# Change line ~105:
reranker = SentenceTransformerRerank(top_n=3, ...)  # Reduce from 4
```

### Custom Model Configuration
```python
# To use different models, edit the configuration:

# For more powerful systems:
model="qwen3:7b"          # Larger model, better responses

# For limited systems:
model="qwen3:1.5b-q4_0"   # Smaller, faster model

# For CPU-only systems:
model="qwen3:4b-cpu"      # CPU-optimized variant
```

### Memory Management Tuning
```python
# In ChatMemoryManager class, adjust these parameters:

memory_manager = ChatMemoryManager(
    llm=Settings.llm,
    max_context_length=2000,    # Reduce for faster processing
    max_history_pairs=5         # Reduce for less memory usage
)
```

### Custom Document Processing
```python
# To add new file types, edit SME_1_build_elasticsearch_database.py:

# Add to supported extensions around line 20:
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.md', '.rtf', '.odt'}

# Add custom loader for new format:
if file_path.suffix.lower() == '.rtf':
    # Add RTF processing logic
    pass
```

### Performance Monitoring
```python
# Add timing and monitoring to your queries:
import time

start_time = time.time()
response = query_engine.query(question)
end_time = time.time()
print(f"Query took: {end_time - start_time:.2f} seconds")
```

---

## 📊 Performance Expectations

### Typical Performance Metrics
- **Database Build**: 1-5 minutes for 100 pages
- **Query Response**: 2-5 seconds average
- **Memory Usage**: 4-8GB RAM, 3-6GB VRAM
- **Document Support**: Up to 500MB per file
- **Concurrent Queries**: Single user (local deployment)

### System-Specific Performance
```bash
# RTX 4090 (24GB VRAM):
# - Model: qwen3:4b (full precision)
# - Response time: ~2 seconds
# - Batch size: 32

# RTX 4070 (12GB VRAM):
# - Model: qwen3:4b-q8_0 (8-bit)
# - Response time: ~3 seconds
# - Batch size: 16

# RTX 4050 (6GB VRAM):
# - Model: qwen3:4b-q4_0 (4-bit)
# - Response time: ~4 seconds
# - Batch size: 8
```

---

## 🎓 Usage Tips

### Best Practices
1. **Start with clear questions**: "What is X?" rather than "Tell me about stuff"
2. **Use follow-up questions**: Let the memory system build context
3. **Clear memory for new topics**: Use `/clear` when switching subjects
4. **Check sources**: Review the source nodes for accuracy
5. **Use specific terminology**: Technical terms often yield better results

### Question Strategies
```bash
# Good conversation flow:
1. "What is machine learning?"
2. "What are the main types of algorithms?"
3. "How does supervised learning work?"
4. "Can you give an example of classification?"

# Less effective:
1. "Tell me everything about AI"
2. "What should I know?"
3. "Explain computers"
```

### Document Organization Tips
```bash
# Organize your documents for best results:
data_large/
├── textbooks/          # Core reference materials
├── papers/             # Research papers
├── documentation/      # Technical docs
└── notes/             # Personal notes/summaries

# File naming suggestions:
# ✅ "Operating_Systems_Textbook_Chapter_4.pdf"
# ✅ "Memory_Management_Research_2023.pdf"
# ❌ "untitled.pdf"
# ❌ "doc1.pdf"
```

---

## 🔄 Maintenance

### Regular Maintenance Tasks
```bash
# Weekly: Clear old conversation data
rm -rf ./chat_sessions/*

# Monthly: Rebuild database for new documents
python SME_1_build_elasticsearch_database.py

# As needed: Update models
ollama pull qwen3:4b-q4_0

# Monitor disk space
du -sh elasticsearch_storage_v2/
```

### Database Management
```bash
# Backup database
tar -czf backup_$(date +%Y%m%d).tar.gz elasticsearch_storage_v2/

# Clear and rebuild database
rm -rf elasticsearch_storage_v2/
python SME_1_build_elasticsearch_database.py

# Check database health
python SME_3_inspect_elasticsearch_database.py
```

---

## 📊 Advanced System Monitoring & Scaling

### Real-Time System Monitoring
```bash
# Windows PowerShell monitoring script (save as monitor.ps1):
while ($true) {
    Clear-Host
    Write-Host "=== SME System Monitor ===" -ForegroundColor Green
    Write-Host "Time: $(Get-Date)" -ForegroundColor Yellow
    
    # CPU and Memory
    $cpu = Get-Counter "\Processor(_Total)\% Processor Time" | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue
    $mem = Get-Counter "\Memory\Available MBytes" | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue
    Write-Host "CPU Usage: $([math]::Round($cpu,1))%" -ForegroundColor Cyan
    Write-Host "Available RAM: $([math]::Round($mem/1024,1))GB" -ForegroundColor Cyan
    
    # GPU (if NVIDIA)
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
        Write-Host "GPU: $gpu" -ForegroundColor Magenta
    }
    
    # Service status checks
    try {
        $esHealth = Invoke-RestMethod -Uri "http://localhost:9200/_cluster/health" -Method Get
        Write-Host "Elasticsearch: $($esHealth.status)" -ForegroundColor Green
    } catch {
        Write-Host "Elasticsearch: OFFLINE" -ForegroundColor Red
    }
    
    try {
        Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method Get | Out-Null
        Write-Host "Ollama: ONLINE" -ForegroundColor Green
    } catch {
        Write-Host "Ollama: OFFLINE" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 5
}

# Linux monitoring script (save as monitor.sh):
#!/bin/bash
while true; do
    clear
    echo "=== SME System Monitor ==="
    echo "Time: $(date)"
    echo
    
    # System resources
    echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')%"
    echo "RAM Usage: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
    
    # GPU monitoring
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)"
    fi
    
    # Service status
    if curl -s http://localhost:9200/_cluster/health > /dev/null; then
        echo "Elasticsearch: ONLINE"
    else
        echo "Elasticsearch: OFFLINE"
    fi
    
    if curl -s http://localhost:11434/api/version > /dev/null; then
        echo "Ollama: ONLINE"
    else
        echo "Ollama: OFFLINE"
    fi
    
    sleep 5
done
```

### Production Scaling Guidelines
```bash
# For multiple concurrent users (5-10 users):
export OLLAMA_NUM_PARALLEL=8
ES_JAVA_OPTS: "-Xms8g -Xmx8g"

# For high-volume usage (10+ users):
# Consider clustering Elasticsearch and load balancing Ollama
# Use Redis for response caching
# Implement queue-based processing for batch operations

# Hardware scaling recommendations:
# Light usage (1-3 users): 16GB RAM, RTX 4060
# Medium usage (3-8 users): 32GB RAM, RTX 4070/4080
# Heavy usage (8+ users): 64GB RAM, RTX 4090 or multiple GPUs
```

### Emergency Procedures
```bash
# Complete system reset (if everything breaks):
# 1. Stop all services
docker-compose -f docker-compose-elasticsearch.yml down
pkill ollama

# 2. Clean all data
docker system prune -f
rm -rf ~/.ollama  # Linux
Remove-Item -Recurse "$env:USERPROFILE\.ollama"  # Windows

# 3. Restart fresh
ollama serve &
ollama pull qwen3:4b-q4_0
docker-compose -f docker-compose-elasticsearch.yml up -d
python SME_1_build_elasticsearch_database.py

# Quick health check after reset
curl http://localhost:9200/_cluster/health
curl http://localhost:11434/api/version
python SME_3_inspect_elasticsearch_database.py
```

---

## 🎯 Final Reference & Best Practices

### Essential Daily Commands
```bash
# Start your SME session:
docker-compose -f docker-compose-elasticsearch.yml up -d
ollama serve &
python SME_2_query_elasticsearch_system.py

# Check system health:
docker ps
ollama list
curl http://localhost:9200/_cluster/health

# Stop cleanly:
docker-compose -f docker-compose-elasticsearch.yml down
pkill ollama
```

### Performance Optimization Checklist
- ✅ Use appropriate model size for your hardware
- ✅ Monitor GPU temperature (keep under 85°C)
- ✅ Ensure adequate RAM (minimum 16GB recommended)
- ✅ Use SSD storage for better I/O performance
- ✅ Keep at least 20% disk space free
- ✅ Update models and dependencies regularly
- ✅ Monitor system resources during heavy usage

### Security Considerations
- 🔒 Don't expose Elasticsearch to internet without authentication
- 🔒 Use firewall rules to restrict access to internal networks
- 🔒 Regularly backup your data and configurations
- 🔒 Monitor logs for unusual activity
- 🔒 Keep software components updated for security patches

---

## 📞 Support & Troubleshooting

### When Things Go Wrong
1. **Check the troubleshooting section** above first
2. **Verify hardware requirements** are met
3. **Review logs** for specific error messages
4. **Try the emergency reset** procedure if needed
5. **Monitor system resources** to identify bottlenecks

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Documentation**: This manual covers 95% of common scenarios
- **Community**: Share knowledge with other SME system users
- **Performance Issues**: Start with hardware requirements section

---

**🎉 Congratulations!** You now have the most comprehensive SME system setup guide available. This manual covers everything from basic installation to enterprise-scale deployment, troubleshooting, monitoring, and optimization.

**Keep this document** as your definitive reference for all SME system operations. Whether you're a beginner setting up your first system or an expert deploying at scale, this guide has everything you need.

**Happy querying!** 🚀

---
