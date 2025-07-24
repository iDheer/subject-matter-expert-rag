# 🎓 Subject Matter Expert RAG System with Knowledge Graphs

A comprehensive **Retrieval-Augmented Generation (RAG)** system combining traditional Elasticsearch-based document retrieval with advanced **GPU-accelerated Knowledge Graphs** for enhanced subject matter expertise.

## 🎯 System Overview

This system provides two complementary approaches:

1. **📚 SME (Subject Matter Expert) System** - Traditional RAG with Elasticsearch
2. **🧠 Knowledge Graph System** - Advanced GPU-accelerated knowledge graphs with dynamic concept extraction

## 🏗️ Architecture

```
📁 Subject Matter Expert RAG System
├── 🔍 SME System (Elasticsearch-based RAG)
│   ├── SME_1_build_elasticsearch_database.py
│   ├── SME_2_query_elasticsearch_system.py
│   └── SME_3_inspect_elasticsearch_database.py
│
├── 🧠 Knowledge Graph System (GPU-accelerated)
│   ├── KG_ENHANCED_1_build_chapter_database_gpu.py
│   ├── KG_ENHANCED_2_build_knowledge_graph_gpu.py
│   ├── KG_ENHANCED_3_query_knowledge_graph_gpu.py
│   ├── KG_ENHANCED_4_visualize_knowledge_graph_gpu.py
│   └── KG_ENHANCED_MASTER_runner_gpu.py (Master Runner)
│
└── 🛠️ Setup & Configuration
    ├── Enhanced_Knowledge_Graph_Setup_and_Testing.ipynb
    ├── KG_SETUP_ENHANCED_SYSTEM.py
    ├── setup_knowledge_graph.ps1 (Windows)
    └── setup_knowledge_graph.sh (Linux/macOS)
```

## 🚀 Quick Start

### 1. Prerequisites

**Required Services:**
- **Elasticsearch** (port 9200)
- **Neo4j** (port 7687, 7474)
- **Ollama** with `qwen3:4b` model (port 11434)

**System Requirements:**
- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ (optional, CPU fallback available)
- Docker Desktop
- 8GB+ RAM (16GB+ recommended for GPU)

### 2. Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd subject-matter-expert-rag

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
# For basic SME system:
pip install -r requirements.txt

# For enhanced system with Knowledge Graphs:
pip install -r enhanced_requirements.txt

# 4. Setup services
# Windows PowerShell:
.\setup_knowledge_graph.ps1
# Linux/macOS:
./setup_knowledge_graph.sh

# Or manually:
docker-compose -f docker-compose-elasticsearch.yml up -d
docker-compose -f docker-compose-neo4j.yml up -d
ollama serve
ollama pull qwen3:4b
```

### 3. Setup Data

```bash
# Add your documents to data_large/ directory
mkdir data_large
# Copy your PDF, DOCX, TXT files here
```

## 📊 Usage Guide

### 🔍 SME System (Traditional RAG)

**Best for:** Document retrieval, Q&A, basic semantic search

```bash
# 1. Build Elasticsearch database
python SME_1_build_elasticsearch_database.py

# 2. Query the system
python SME_2_query_elasticsearch_system.py

# 3. Inspect database
python SME_3_inspect_elasticsearch_database.py
```

**Features:**
- Hierarchical document parsing
- Elasticsearch vector storage
- AutoMerging retrieval
- Re-ranking with BGE models

### 🧠 Knowledge Graph System (Advanced)

**Best for:** Concept relationships, learning paths, intelligent tutoring

#### Option A: Master Runner (Recommended)
```bash
# Run everything with GPU optimization
python KG_ENHANCED_MASTER_runner_gpu.py

# Quick test mode
python KG_ENHANCED_MASTER_runner_gpu.py --quick

# Step-by-step interactive mode
python KG_ENHANCED_MASTER_runner_gpu.py --interactive
```

#### Option B: Step-by-Step Execution
```bash
# 1. Build chapter database with dynamic concepts
python KG_ENHANCED_1_build_chapter_database_gpu.py

# 2. Create knowledge graph with clustering
python KG_ENHANCED_2_build_knowledge_graph_gpu.py

# 3. Interactive query system
python KG_ENHANCED_3_query_knowledge_graph_gpu.py

# 4. Generate 3D visualizations
python KG_ENHANCED_4_visualize_knowledge_graph_gpu.py
```

#### Option C: Quick Start Guide
```bash
# Follow the detailed quickstart guide
# See: KNOWLEDGE_GRAPH_QUICKSTART.md
```

**Features:**
- Dynamic concept extraction (10-30 concepts per chapter)
- GPU-accelerated processing throughout
- Neo4j knowledge graph with concept clustering
- 3D interactive visualizations
- Learning path recommendations

## 🔥 Knowledge Graph Features

### 🎯 Dynamic Concept Extraction
- **5 Intelligent Methods:** Explicit objectives, key terms, technical concepts, section concepts, question concepts
- **Adaptive Count:** 10-30 concepts per chapter (not hardcoded)
- **GPU Acceleration:** CUDA-optimized processing

### 🕸️ Advanced Knowledge Graph
- **Neo4j Storage:** Complex relationships and prerequisites
- **Concept Clustering:** K-means with GPU acceleration
- **Relation Extraction:** Hybrid NLTK + Elasticsearch approach
- **Prerequisite Detection:** Automatic dependency mapping

### 📊 3D Visualizations
- **Interactive Dashboards:** Plotly-based exploration
- **Concept Clusters:** 3D t-SNE with GPU acceleration
- **Network Analysis:** NetworkX integration
- **Performance Analytics:** Real-time metrics

### 🔍 Intelligent Query System
- **Semantic Search:** GPU-accelerated embeddings
- **Context Awareness:** Multi-hop reasoning
- **Concept Expansion:** Related concept suggestions
- **Source Attribution:** Detailed provenance tracking

## 🎮 GPU Optimization

### Automatic GPU Detection
```python
# System automatically detects and uses GPU when available
Device: NVIDIA GeForce RTX 4090  # Example
GPU Acceleration: ✅ ENABLED
Memory: 24.0 GB VRAM
Expected speedup: 10-20x for embeddings
```

### Performance Modes
- **GPU Mode:** 10-20x faster processing, batch_size=64-128
- **CPU Mode:** Multi-threaded optimization, batch_size=8-16
- **Hybrid Mode:** GPU for ML, CPU for graph operations

### GPU-Optimized Libraries
- **PyTorch with CUDA 12.1**
- **CuPy** for GPU-accelerated NumPy
- **GPU-accelerated sentence transformers**
- **spaCy with CUDA support**
- **NVIDIA ML monitoring**

## 📋 Configuration

### Model Configuration
```python
# LLM: Ollama with qwen3:4b (2.5GB)
# Embeddings: sentence-transformers/all-mpnet-base-v2
# Device: Auto-detect CUDA/CPU
# Batch sizes: GPU=64, CPU=16
```

### Database Configuration
```python
# Elasticsearch: advanced_docs_elasticsearch_v2
# Neo4j: gpu_chapter_knowledge_v1 
# Storage: Separate from original system
```

## 🧪 Testing & Validation

### Quick Testing
```bash
# Test SME system
python SME_1_build_elasticsearch_database.py --test
python SME_2_query_elasticsearch_system.py

# Test Knowledge Graph system  
python KG_ENHANCED_MASTER_runner_gpu.py --quick
```

### Manual Testing
```bash
# Check services
curl http://localhost:9200                    # Elasticsearch
curl http://localhost:11434/api/tags          # Ollama
# Neo4j Browser: http://localhost:7474
```

## 📊 Performance Expectations

### SME System
- **Index Speed:** ~100-500 docs/minute
- **Query Speed:** <2 seconds
- **Memory Usage:** 2-4GB RAM

### Knowledge Graph System
- **GPU Mode:** 5-20x faster than CPU
- **Concept Extraction:** 20-30 concepts/chapter
- **Graph Building:** ~50-200 nodes/minute
- **Memory Usage:** 4-8GB (CPU), 2-6GB VRAM (GPU)

## 🛠️ Troubleshooting

### Common Issues

**1. GPU Not Available**
```bash
# Check CUDA installation
nvidia-smi
# Install CUDA 12.1+ drivers
# System will automatically fallback to CPU
```

**2. Services Not Running**
```bash
# Elasticsearch
docker-compose -f docker-compose-elasticsearch.yml up -d

# Neo4j
docker-compose -f docker-compose-neo4j.yml up -d

# Ollama
ollama serve
ollama pull qwen3:4b
```

**3. Memory Issues**
```bash
# Use quick mode for testing
python KG_ENHANCED_COMPLETE_RUNNER.py --quick

# Reduce batch sizes in config
# Monitor memory usage during processing
```

**4. Import Errors**
```bash
# Ensure virtual environment is activated
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Reinstall dependencies
pip install -r enhanced_requirements.txt
```

### Interactive Troubleshooting
```bash
# Launch comprehensive troubleshooting notebook
jupyter notebook Enhanced_Knowledge_Graph_Setup_and_Testing.ipynb

# Run system diagnostics
python KG_SETUP_ENHANCED_SYSTEM.py
```

## 📚 Documentation

- **`SME_COMPREHENSIVE_MANUAL.md`** - Complete setup guide for normal users
- **`KNOWLEDGE_GRAPH_QUICKSTART.md`** - Enhanced knowledge graph quick start
- **`Enhanced_Knowledge_Graph_Setup_and_Testing.ipynb`** - Interactive setup
- **`enhanced_visualizations/`** - Generated visualizations and reports

## 🔧 Advanced Configuration

### Environment Variables
```bash
# GPU Configuration
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Performance Tuning
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

### Custom Configuration
```python
# Modify config in individual scripts
BATCH_SIZE = 64  # GPU mode
BATCH_SIZE = 16  # CPU mode
MAX_CONCEPTS = 30
CLUSTERING_ALGORITHM = "kmeans"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -am 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LlamaIndex** for RAG framework
- **Elasticsearch** for vector storage
- **Neo4j** for knowledge graphs
- **Ollama** for local LLM inference
- **NVIDIA CUDA** for GPU acceleration
- **HuggingFace** for transformer models

---

## 🚀 Ready to Get Started?

1. **For basic SME system:** Start with `SME_1_build_elasticsearch_database.py`
2. **For knowledge graphs:** Run `KG_ENHANCED_MASTER_runner_gpu.py --quick` 
3. **For step-by-step setup:** Follow `SME_COMPREHENSIVE_MANUAL.md`
4. **For interactive setup:** Launch `Enhanced_Knowledge_Graph_Setup_and_Testing.ipynb`

**Need help?** Check the troubleshooting section or open an issue!

🎉 **Happy knowledge graphing!** 🎉
