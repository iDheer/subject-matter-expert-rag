# 🎓 Subject Matter Expert RAG System with GPU-Enhanced Knowledge Graphs

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-red.svg)](https://pytorch.org/)

A comprehensive **Retrieval-Augmented Generation (RAG)** system that combines traditional Elasticsearch-based document retrieval with advanced **GPU-accelerated Knowledge Graphs** for enhanced subject matter expertise and intelligent document analysis.

## 🌟 Key Features

### 📚 SME (Subject Matter Expert) System
- **Elasticsearch-powered** vector search with hierarchical document chunking
- **AutoMerging Retriever** for context-aware document retrieval
- **BGE reranker** for improved result relevance
- **Streaming responses** with Qwen 4B language model
- **🧠 Conversational Memory** - Remembers context across queries for natural dialogue
- **Intelligent Summarization** - Uses Qwen to maintain conversation history within context limits

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

## 🏗️ Project Structure

```
📁 Subject Matter Expert RAG System
├── 🔍 SME System (Elasticsearch + Conversational Memory)
│   ├── SME_1_build_elasticsearch_database.py    # Build document database
│   ├── SME_2_query_elasticsearch_system.py      # Interactive querying with memory
│   └── SME_3_inspect_elasticsearch_database.py  # Database inspection & analysis
│
├── 🧠 Knowledge Graph System (GPU-accelerated)
│   ├── KG_ENHANCED_1_build_chapter_database_gpu.py   # Chapter extraction & processing
│   ├── KG_ENHANCED_2_build_knowledge_graph_gpu.py    # Graph construction & relationships
│   ├── KG_ENHANCED_3_query_knowledge_graph_gpu.py    # Intelligent graph querying
│   ├── KG_ENHANCED_4_visualize_knowledge_graph_gpu.py # Interactive visualization
│   └── KG_ENHANCED_MASTER_runner_gpu.py              # Complete pipeline runner
│
├── 🛠️ Configuration & Services
│   ├── docker-compose-elasticsearch.yml         # Elasticsearch container setup
│   ├── docker-compose-neo4j.yml                # Neo4j graph database setup
│   ├── requirements.txt                        # Essential Python dependencies
│   └── enhanced_requirements.txt               # Full-featured dependencies
│
├── 🧪 Testing & Analysis
│   ├── questions.json                          # Test questions for evaluation
│   └── rag_test_results.txt                   # System performance results
│
└── 📖 Documentation
    ├── README.md                               # This comprehensive guide
    └── SME_SETUP_MANUAL.md                    # Detailed setup instructions
```

## 📋 System Requirements

### Hardware Requirements
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

## 🚀 Quick Start Guide

### 1. Repository Setup
```bash
# Clone the repository
git clone https://github.com/iDheer/subject-matter-expert-rag.git
cd subject-matter-expert-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For full features with GPU acceleration:
# pip install -r enhanced_requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

**Requirements Files:**
- `requirements.txt` - Essential dependencies for core functionality
- `enhanced_requirements.txt` - Full-featured with GPU acceleration, advanced tools, and development utilities

### 2. Service Setup

#### Option A: Docker (Recommended)
```bash
# Start Elasticsearch
docker-compose -f docker-compose-elasticsearch.yml up -d

# Start Neo4j
docker-compose -f docker-compose-neo4j.yml up -d
```

#### Option B: Manual Installation
```bash
# Install Elasticsearch 8.x
# Windows: Download from https://www.elastic.co/downloads/elasticsearch
# Linux: 
sudo apt-get install elasticsearch
# macOS: 
brew install elasticsearch

# Install Neo4j 5.x
# Windows: Download from https://neo4j.com/download/
# Linux: 
sudo apt-get install neo4j
# macOS: 
brew install neo4j

# Start services
sudo systemctl start elasticsearch
sudo systemctl start neo4j
```

### 3. Ollama Setup
```bash
# Install Ollama
# Windows:
winget install Ollama.Ollama
# Linux:
curl -fsSL https://ollama.ai/install.sh | sh
# macOS:
brew install ollama

# Start Ollama service
ollama serve

# Pull required model (quantized for GPU efficiency)
ollama pull qwen3:4b        # Standard version (8GB VRAM)
ollama pull qwen3:4b-q4_0   # 4-bit quantized (recommended for 6GB GPUs)
ollama pull qwen3:4b-q8_0   # 8-bit quantized (balanced option)
```

### 4. Data Preparation
```bash
# Create data directory
mkdir data_large

# Place your PDF documents in data_large/ folder
# Supported formats: PDF, TXT, DOCX
# Example structure:
# data_large/
#   ├── textbook.pdf
#   ├── research_paper.pdf
#   └── documentation.pdf
```

## 🎯 Usage Instructions

### SME System (Elasticsearch + Conversational Memory)

#### Step 1: Build Database
```bash
python SME_1_build_elasticsearch_database.py
```
**What it does:**
- Processes documents in `data_large/` folder
- Creates hierarchical document chunks
- Builds Elasticsearch vector index
- Sets up AutoMerging retrieval system

#### Step 2: Interactive Querying
```bash
python SME_2_query_elasticsearch_system.py
```
**Features:**
- **Natural conversations** with memory
- **Context-aware responses** that reference previous queries
- **Intelligent summarization** when conversation gets long
- **Memory management** commands

**Example Conversation:**
```
💬 Question: What is virtual memory?
🤖 Response: Virtual memory is a memory management technique that provides an idealized abstraction of storage resources...

💬 Question: How does paging work with it?
🤖 Response: Building on our discussion of virtual memory, paging works by dividing memory into fixed-size blocks...

💬 Question: /status
📊 Memory enabled - 2 exchanges

💬 Question: /clear
🧹 Conversation memory cleared
```

**Available Commands:**
- `/memory` - Toggle conversation memory on/off
- `/clear` - Clear conversation history
- `/status` - Show memory status
- `/help` - Display all commands
- `exit` - Exit the system

#### Step 3: Database Inspection (Optional)
```bash
python SME_3_inspect_elasticsearch_database.py
```
**Features:**
- View database statistics
- Browse document chunks
- Analyze retrieval performance
- Export database contents

### Knowledge Graph System (GPU-Accelerated)

#### Option 1: Complete Pipeline
```bash
python KG_ENHANCED_MASTER_runner_gpu.py
```
**What it does:**
- Runs the complete KG pipeline
- Processes all documents
- Builds comprehensive knowledge graph
- Enables interactive querying and visualization

#### Option 2: Step-by-Step Execution

**Step 1: Extract Chapters**
```bash
python KG_ENHANCED_1_build_chapter_database_gpu.py
```
- Extracts chapters from documents
- Creates structured chapter database
- Prepares data for graph construction

**Step 2: Build Knowledge Graph**
```bash
python KG_ENHANCED_2_build_knowledge_graph_gpu.py
```
- Extracts concepts and relationships
- Builds Neo4j knowledge graph
- Creates entity connections

**Step 3: Query System**
```bash
python KG_ENHANCED_3_query_knowledge_graph_gpu.py
```
- Interactive graph querying
- Multi-modal search (text + graph)
- Relationship exploration

**Step 4: Visualization**
```bash
python KG_ENHANCED_4_visualize_knowledge_graph_gpu.py
```
- Interactive graph visualization
- Network analysis
- Concept relationship mapping

## ⚙️ Configuration

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

### Model Configuration for Different GPUs
```python
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

## 🧪 Testing & Evaluation

### Test Questions
The system includes pre-defined test questions in `questions.json`:
```json
{
  "questions": [
    "What is virtual memory and how does it work?",
    "Explain the concept of process scheduling",
    "How do deadlocks occur and how can they be prevented?"
  ]
}
```

### Performance Results
Check `rag_test_results.txt` for system performance metrics:
- Response time analysis
- Retrieval accuracy scores
- Memory usage statistics
- GPU utilization data

### Custom Testing
```bash
# Run your own test questions
python SME_2_query_elasticsearch_system.py
# Then use questions from questions.json or create your own
```

## 🛠️ Troubleshooting

### Common Issues

**1. Elasticsearch Connection Failed**
```bash
# Check if Elasticsearch is running
curl -X GET "localhost:9200/_cluster/health"

# Restart Elasticsearch
docker-compose -f docker-compose-elasticsearch.yml restart
```

**2. Ollama Model Not Found**
```bash
# List available models
ollama list

# Pull required model
ollama pull qwen3:4b-q4_0
```

**3. GPU Memory Issues**
```bash
# Check GPU memory usage
nvidia-smi

# Use quantized model
export OLLAMA_MODEL="qwen3:4b-q4_0"
```

**4. Neo4j Connection Issues**
```bash
# Check Neo4j status
docker-compose -f docker-compose-neo4j.yml logs

# Reset Neo4j password
docker exec -it neo4j cypher-shell -u neo4j -p neo4j
# Then: ALTER USER neo4j SET PASSWORD 'knowledge123'
```

### Performance Optimization

**For RTX 4050 6GB Users:**
```python
# Use these settings in your configuration:
model="qwen3:4b-q4_0"     # 4-bit quantized (2.8GB VRAM)
batch_size=8              # Smaller batch size
max_context_length=2000   # Reduced context for memory
```

**Memory Management:**
```python
import torch
# Clear GPU cache between operations
torch.cuda.empty_cache()
```

## 📊 System Capabilities

### SME System Features
- **Document Types**: PDF, TXT, DOCX, MD
- **Max Document Size**: 500MB per file
- **Concurrent Users**: Single user (local deployment)
- **Response Time**: 2-5 seconds average
- **Memory Context**: Up to 4,000 characters with auto-summarization

### Knowledge Graph Features
- **Node Types**: Concepts, Entities, Chapters, Documents
- **Relationship Types**: 15+ semantic relationship types
- **Graph Size**: Handles 10,000+ nodes efficiently
- **Query Types**: Cypher, natural language, hybrid
- **Visualization**: Interactive web-based interface

## 🚀 Advanced Usage

### Batch Processing
```bash
# Process multiple document sets
for folder in data_batch_*/; do
    cp -r "$folder"/* data_large/
    python SME_1_build_elasticsearch_database.py
    python KG_ENHANCED_MASTER_runner_gpu.py
done
```

### Custom Model Integration
```python
# Replace Qwen with custom model
Settings.llm = Ollama(
    model="your-custom-model:latest",
    request_timeout=300.0,
    base_url="http://localhost:11434",
)
```

### API Integration
```python
# Basic API wrapper (extend as needed)
from SME_2_query_elasticsearch_system import query_engine

def api_query(question: str) -> str:
    response = query_engine.query(question)
    return response.response
```

## 📈 Performance Benchmarks

### SME System Performance
- **Retrieval Speed**: ~200ms for similarity search
- **Response Generation**: ~2-4 seconds
- **Memory Usage**: 4-8GB RAM + 3-6GB VRAM
- **Accuracy**: 85-92% on domain-specific questions

### Knowledge Graph Performance
- **Graph Construction**: ~5-10 minutes for 1000 pages
- **Query Response**: ~500ms for complex queries
- **Visualization Load**: ~2-3 seconds for 1000 nodes
- **Memory Usage**: 6-12GB RAM + 4-8GB VRAM

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[LangChain](https://langchain.com/)** for RAG framework
- **[LlamaIndex](https://llamaindex.ai/)** for advanced indexing
- **[Ollama](https://ollama.ai/)** for local LLM deployment
- **[Neo4j](https://neo4j.com/)** for graph database technology
- **[Elasticsearch](https://elastic.co/)** for search capabilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/iDheer/subject-matter-expert-rag/issues)
- **Documentation**: This README + `SME_SETUP_MANUAL.md`
- **Performance**: Check `rag_test_results.txt` for benchmarks

---

<div align="center">

**Ready to enhance your document analysis with AI?** 🚀

Start with: `python SME_1_build_elasticsearch_database.py`

</div>
