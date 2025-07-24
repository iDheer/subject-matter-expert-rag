# 🎓 Subject Matter Expert RAG System with GPU-Enhanced Knowledge Graphs

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-red.svg)](https://pytorch.org/)

A comprehensive **Retrieval-Augmented Generation (RAG)** system that combines traditional Elasticsearch-based document retrieval with advanced **GPU-accelerated Knowledge Graphs** for enhanced subject matter expertise and intelligent document analysis.

## 🌟 Features

### 📚 SME (Subject Matter Expert) System
- **Elasticsearch-powered** vector search with hierarchical document chunking
- **AutoMerging Retriever** for context-aware document retrieval
- **BGE reranker** for improved result relevance
- **Streaming responses** with Qwen 4B language model

### 🧠 GPU-Enhanced Knowledge Graph System
- **Dynamic concept extraction** using NLP and GPU acceleration
- **Neo4j graph database** with advanced relationship modeling
- **Chapter-based knowledge organization** for complex documents
- **Interactive graph visualization** with enhanced analytics
- **Multi-modal querying** (text + graph traversal)

### 🚀 Performance Features
- **GPU acceleration** with CUDA support
- **Quantized model support** (INT4/INT8) for memory efficiency
- **Batch processing** for large document collections
- **Memory optimization** for RTX 4050 6GB and similar GPUs

## 🏗️ System Architecture

```
📁 Subject Matter Expert RAG System
├── 🔍 SME System (Elasticsearch-based RAG)
│   ├── SME_1_build_elasticsearch_database.py    # Database construction
│   ├── SME_2_query_elasticsearch_system.py      # Interactive querying
│   └── SME_3_inspect_elasticsearch_database.py  # Database inspection
│
├── 🧠 Knowledge Graph System (GPU-accelerated)
│   ├── KG_ENHANCED_1_build_chapter_database_gpu.py   # Chapter extraction
│   ├── KG_ENHANCED_2_build_knowledge_graph_gpu.py    # Graph construction
│   ├── KG_ENHANCED_3_query_knowledge_graph_gpu.py    # Graph querying
│   ├── KG_ENHANCED_4_visualize_knowledge_graph_gpu.py # Visualization
│   └── KG_ENHANCED_MASTER_runner_gpu.py              # Master controller
│
├── 🛠️ Setup & Configuration
│   ├── setup_knowledge_graph.ps1/.sh    # Automated setup scripts
│   ├── docker-compose-*.yml             # Docker configurations
│   └── Enhanced_Knowledge_Graph_Setup_and_Testing.ipynb
│
└── 🔧 Utilities
    ├── vram_analyzer.py                  # GPU memory analysis
    ├── check_ollama_gpu.py              # Ollama GPU configuration
    └── setup_ollama_gpu.ps1/.bat        # GPU setup scripts
```

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+
- **Python**: 3.11 or higher
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 4050 or better) - *Optional, CPU fallback available*
- **RAM**: 16GB+ recommended (8GB minimum)
- **Storage**: 10GB+ free space

### Required Services
- **Elasticsearch** 8.x (port 9200)
- **Neo4j** 5.x (ports 7687, 7474)
- **Ollama** with Qwen models (port 11434)

### GPU Memory Requirements
| System | FP16 | INT8 | INT4 | RTX 4050 Compatible |
|--------|------|------|------|---------------------|
| SME System | 10.1 GB ❌ | 6.6 GB ❌ | **4.9 GB** ✅ | INT4 Only |
| Knowledge Graph | 9.2 GB ❌ | **5.7 GB** ⚠️ | **4.0 GB** ✅ | INT8/INT4 |

## 🚀 Quick Start

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/subject-matter-expert-rag.git
cd subject-matter-expert-rag

# Optional: Create a new branch for your work
git checkout -b feature/your-feature-name
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r enhanced_requirements.txt
```

### 3. Service Setup

#### Option A: Docker (Recommended)
```bash
# Start Elasticsearch
docker-compose -f docker-compose-elasticsearch.yml up -d

# Start Neo4j
docker-compose -f docker-compose-neo4j.yml up -d

# Install and start Ollama
# Windows:
winget install Ollama.Ollama
# Linux:
curl -fsSL https://ollama.ai/install.sh | sh
# macOS:
brew install ollama

# Start Ollama service
ollama serve

# Pull required model (quantized for GPU efficiency)
ollama pull qwen3:4b        # Standard version
ollama pull qwen3:4b-q4_0   # 4-bit quantized (recommended for 6GB GPUs)
```

#### Option B: Automated Setup
```bash
# Windows PowerShell (run as administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_knowledge_graph.ps1

# Linux/macOS
chmod +x setup_knowledge_graph.sh
./setup_knowledge_graph.sh
```

### 4. GPU Configuration (Optional but Recommended)

```bash
# Check GPU compatibility
python vram_analyzer.py

# Configure Ollama for GPU usage
# Windows:
.\setup_ollama_gpu.ps1
# Linux/macOS:
python check_ollama_gpu.py
```

### 5. Data Preparation

```bash
# Place your PDF documents in data_large/ folder
mkdir data_large
# Copy your PDF files to data_large/

# For testing, you can use the included sample document:
# data_large/OSTEP Remzi H. Arpaci-Dusseau, Andrea C. Arpaci-Dusseau - Operating Systems - Three Easy Pieces_copy.pdf
```

## 🎯 Usage

### SME System (Traditional RAG)

```bash
# 1. Build Elasticsearch database
python SME_1_build_elasticsearch_database.py

# 2. Query the system interactively
python SME_2_query_elasticsearch_system.py

# 3. Inspect database contents (optional)
python SME_3_inspect_elasticsearch_database.py
```

### Knowledge Graph System (Advanced)

```bash
# Option 1: Run complete pipeline
python KG_ENHANCED_MASTER_runner_gpu.py

# Option 2: Step-by-step execution
python KG_ENHANCED_1_build_chapter_database_gpu.py    # Extract chapters
python KG_ENHANCED_2_build_knowledge_graph_gpu.py     # Build graph
python KG_ENHANCED_3_query_knowledge_graph_gpu.py     # Query system
python KG_ENHANCED_4_visualize_knowledge_graph_gpu.py # Visualize results
```

### Interactive Jupyter Notebook
```bash
# Start Jupyter and open the enhanced notebook
jupyter lab Enhanced_Knowledge_Graph_Setup_and_Testing.ipynb
```

## 📊 Performance Optimization

### GPU Memory Optimization
```python
# For RTX 4050 6GB users, use quantized models:
# In your configuration files, replace:
model="qwen3:4b"          # Standard (8GB VRAM)
# With:
model="qwen3:4b-q4_0"     # 4-bit quantized (2.8GB VRAM)
# Or:
model="qwen3:4b-q8_0"     # 8-bit quantized (4.5GB VRAM)
```

### Memory Management
```python
import torch

# Clear GPU cache between runs
torch.cuda.empty_cache()

# Monitor GPU usage
# Run in terminal: nvidia-smi -l 1
```

## 🛠️ Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# Elasticsearch
ES_ENDPOINT=http://localhost:9200
ES_INDEX_NAME=advanced_docs_elasticsearch_v2

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=knowledge123

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-q4_0

# GPU Settings
CUDA_VISIBLE_DEVICES=0
OLLAMA_NUM_GPU=1
OLLAMA_GPU_LAYERS=35
```

### Model Configuration
```python
# GPU-optimized settings for different systems:

# RTX 4090/4080 (16GB+)
model="qwen3:4b"          # Full precision
batch_size=32

# RTX 4070/4060 Ti (8-12GB)
model="qwen3:4b-q8_0"     # 8-bit quantized
batch_size=16

# RTX 4050/4060 (6-8GB)
model="qwen3:4b-q4_0"     # 4-bit quantized
batch_size=8
```

## 📚 Documentation

- **[Step-by-Step Guide](STEP_BY_STEP_GUIDE.md)** - Detailed setup instructions
- **[Comprehensive Manual](SME_COMPREHENSIVE_MANUAL.md)** - Complete system documentation
- **[Knowledge Graph Quickstart](KNOWLEDGE_GRAPH_QUICKSTART.md)** - Fast track to KG usage
- **[API Documentation](docs/api.md)** - Code reference *(if available)*

## 🧪 Testing

```bash
# Run system tests
python -m pytest tests/

# Test GPU configuration
python vram_analyzer.py

# Test individual components
python SME_3_inspect_elasticsearch_database.py
python check_ollama_gpu.py
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks
pre-commit install  # If configured

# Run linting
flake8 .
black .
```

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[LangChain](https://langchain.com/)** for RAG framework
- **[LlamaIndex](https://llamaindex.ai/)** for advanced indexing
- **[Ollama](https://ollama.ai/)** for local LLM deployment
- **[Neo4j](https://neo4j.com/)** for graph database technology
- **[Elasticsearch](https://elastic.co/)** for search capabilities
- **[Sentence Transformers](https://sbert.net/)** for embeddings

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/subject-matter-expert-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/subject-matter-expert-rag/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/subject-matter-expert-rag/wiki)

## 🔄 Version History

- **v1.0.0** - Initial release with SME and KG systems
- **v1.1.0** - GPU acceleration and memory optimization
- **v1.2.0** - Enhanced visualization and quantization support

---

<div align="center">

**[⭐ Star this repository](https://github.com/yourusername/subject-matter-expert-rag)** if you find it helpful!

Made with ❤️ by [Your Name](https://github.com/yourusername)

</div>
