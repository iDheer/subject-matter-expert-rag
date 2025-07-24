# SME System Setup Guide - For Normal Users

## 🎯 What This Guide Covers

This guide helps you set up the **Subject Matter Expert (SME) RAG System** - a document-based AI assistant that can answer questions about your documents. Perfect for:

- Creating AI assistants for your company documents
- Building learning systems for educational content  
- Setting up Q&A systems for technical documentation
- Creating subject matter expert chatbots

## 🚀 Quick Setup (15 minutes)

### Step 1: Check Requirements
- **Computer**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 5GB free space
- **Internet**: Required for downloading models

### Step 2: Install Prerequisites

#### Install Python 3.12
1. Download from [python.org](https://python.org)
2. During installation, check "Add Python to PATH"
3. Verify: Open command prompt and type `python --version`

#### Install Docker Desktop
1. Download from [docker.com](https://docker.com)
2. Install and start Docker Desktop
3. Verify: Type `docker --version` in command prompt

### Step 3: Download the System
```bash
# Download and extract the system files
# Place your documents in the data_large/ folder
```

### Step 4: Setup Python Environment
```bash
# Open command prompt in the system folder
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (macOS/Linux)
source venv/bin/activate

# Install basic dependencies
pip install -r requirements.txt
```
- **CPU**: 4 cores (8 cores recommended)
- **Storage**: 20GB free space
- **Network**: Internet for model downloads

### Recommended Requirements
- **RAM**: 16-32GB
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- **GPU**: NVIDIA RTX 3060/4050 or better (6GB/6GB+ VRAM)
- **Storage**: SSD with 50GB+ free space
- **Network**: High-speed internet for initial setup

### Software Dependencies
- **Python**: 3.8 - 3.11
- **Docker**: 20.10+ with Docker Compose
- **Git**: For version control
- **CUDA**: 11.8+ (for GPU acceleration)

---

## Installation and Setup

### Step 1: Environment Preparation

#### Windows Setup
```powershell
# Install Python (if not installed)
# Download from python.org or use Microsoft Store

# Install Docker Desktop
# Download from docker.com

# Verify installations
python --version
docker --version
docker-compose --version
git --version
```

#### Linux Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
### Step 5: Start the Database
```bash
# Start Elasticsearch (this stores your documents)
docker-compose -f docker-compose-elasticsearch.yml up -d

# Wait 30 seconds for startup
# Check if it's working (should show JSON response)
curl http://localhost:9200
```

### Step 6: Install the AI Model
```bash
# Install Ollama (the AI engine)
# Windows: Download from ollama.ai and run installer
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Download the AI model (this will take a few minutes)
ollama pull qwen3:4b

# Verify it's installed
ollama list
```

### Step 7: Add Your Documents
```bash
# Put your files in the data_large folder
# Supported: PDF, Word docs (.docx), text files (.txt)
# Example:
#   data_large/manual.pdf
#   data_large/policies.docx  
#   data_large/procedures.txt
```

### Step 8: Build Your AI Assistant
```bash
# This processes your documents (takes 5-15 minutes depending on size)
python SME_1_build_elasticsearch_database.py

# You'll see progress like:
# ✅ Loading documents from data_large/
# ✅ Processing 50 documents...
# ✅ Creating searchable database...
# ✅ System ready!
```

### Step 9: Test Your AI Assistant
```bash
# Start the Q&A system
python SME_2_query_elasticsearch_system.py

# Try asking questions like:
# "What is our vacation policy?"
# "How do I reset a password?"
# "Explain the safety procedures"
```

## 🎯 Basic Usage

### Asking Questions
Once your system is running, you can:

1. **Ask natural questions** about your documents
2. **Get detailed answers** with source references
3. **Follow up** with related questions
4. **Search specific topics** or keywords

### Example Session
```
🤖 SME Assistant: How can I help you today?
👤 You: What are the company's remote work policies?

🤖 SME Assistant: Based on your HR Policy Manual, here are the key remote work policies:

1. **Eligibility**: Full-time employees after 90-day probation
2. **Equipment**: Company provides laptop and stipend for home office
3. **Schedule**: Core hours 10 AM - 3 PM in company timezone
4. **Communication**: Daily check-ins with manager required

📄 Sources: HR_Policy_Manual.pdf (pages 23-26)

👤 You: What's the equipment stipend amount?

🤖 SME Assistant: The home office equipment stipend is $500 annually for remote employees, covering ergonomic chair, desk, monitor, and other office supplies.

📄 Sources: HR_Policy_Manual.pdf (page 24)
```

## 🔧 Customization Options

### Adding More Documents
```bash
# Add new files to data_large/
# Re-run the indexing
python SME_1_build_elasticsearch_database.py --update
```

### Changing AI Model
```bash
# For better quality (but slower responses)
ollama pull qwen3:8b

# Update the model in SME_2_query_elasticsearch_system.py
# Change: MODEL_NAME = "qwen3:4b" 
# To: MODEL_NAME = "qwen3:8b"
```

### Adjusting Response Style
Edit `SME_2_query_elasticsearch_system.py` and modify the prompt:
```python
# Find this section and customize:
SYSTEM_PROMPT = """
You are a helpful AI assistant. Answer questions based on the provided documents.
Be concise and professional. Always cite your sources.
"""
```
# --- Index build complete ---
"""
```

## 🚨 Common Issues & Solutions

### "Docker is not running"
**Problem**: Cannot start Elasticsearch
**Solution**: 
1. Open Docker Desktop
2. Wait for the whale icon to be solid (not flashing)
3. Try the command again

### "Ollama connection failed" 
**Problem**: AI model not responding
**Solution**:
```bash
# Restart Ollama
ollama serve

# Check if model is installed
ollama list

# Re-download if needed
ollama pull qwen3:4b
```

### "No documents found"
**Problem**: System can't find your files
**Solution**:
1. Check files are in `data_large/` folder
2. Ensure files are PDF, DOCX, or TXT format
3. Check file permissions (not locked/protected)

### "Out of memory" errors
**Problem**: System runs out of RAM
**Solution**:
1. Close other applications
2. Use smaller model: `ollama pull qwen3:4b` instead of 8b
3. Process fewer documents at once

### Poor Answer Quality
**Problem**: AI gives wrong or vague answers
**Solution**:
1. Use better model: `ollama pull qwen3:8b`
2. Add more relevant documents
3. Be more specific in questions
4. Check document quality (readable text, not scanned images)

## 🎓 Advanced Features

### Knowledge Graph System
For more intelligent tutoring and learning paths:

```bash
# Install enhanced requirements
pip install -r enhanced_requirements.txt

# Setup Neo4j database
docker-compose -f docker-compose-neo4j.yml up -d

# Run the enhanced system
python KG_ENHANCED_MASTER_runner_gpu.py --quick
```

This adds:
- **Learning paths**: Automatically finds prerequisites
- **Concept relationships**: Shows how topics connect
- **Visual knowledge maps**: Interactive diagrams
- **Personalized recommendations**: Suggests next topics

See `KNOWLEDGE_GRAPH_QUICKSTART.md` for detailed instructions.

## 🔍 Checking System Status

### Quick Health Check
```bash
# Check all services
python SME_3_inspect_elasticsearch_database.py
```

### Manual Service Check
```bash
# Elasticsearch (should return JSON)
curl http://localhost:9200

# Ollama (should list models)
ollama list

# Docker containers (should show running)
docker ps
```

## 📊 Performance Tips

### For Better Speed
1. **Use SSD storage** for documents
2. **Close unnecessary apps** during indexing
3. **Use qwen3:4b model** for faster responses
4. **Enable GPU** if you have NVIDIA card

### For Better Quality
1. **Use qwen3:8b model** for more accurate answers
2. **Add more relevant documents** to the knowledge base
3. **Organize documents well** with clear structure
4. **Use descriptive filenames** for better context

## 🤝 Getting Help

### Documentation
- **README.md** - Quick overview
- **KNOWLEDGE_GRAPH_QUICKSTART.md** - Advanced features guide

### Common Questions
**Q: Can I use this offline?**
A: Yes! Once setup, everything runs locally.

**Q: What file formats are supported?**
A: PDF, Word (.docx), and text (.txt) files work best.

**Q: How many documents can I process?**
A: Hundreds to thousands, depending on your computer's memory.

**Q: Is my data private?**
A: Yes! Everything runs on your computer, nothing is sent to external servers.

---

## 🎉 You're Ready!

Your SME system is now set up! Here's what you can do next:

1. **Start asking questions** about your documents
2. **Experiment with different queries** to see what works best
3. **Add more documents** to expand the knowledge base
4. **Try the advanced knowledge graph features** for learning applications

**Remember**: The system learns from your documents, so the quality of answers depends on the quality and completeness of your source materials.

Happy querying! 🚀

---

## Knowledge Graph Integration

### Overview
The knowledge graph transforms your RAG system from simple document search into an intelligent learning companion that understands:
- **Hierarchical Structure**: Courses → Chapters → Objectives → Concepts
- **Prerequisites**: What you need to learn before tackling new topics
- **Learning Paths**: Personalized sequences of topics based on your progress
- **Skill Progression**: Adaptive difficulty based on your demonstrated knowledge

### Knowledge Graph Architecture

```
Documents Analysis → Knowledge Extraction → Neo4j Graph → Learning Recommendations
                                                     ↓
                            Visualizations ← Query Interface ← Progress Tracking
```

### Setting Up the Knowledge Graph

#### Step 1: Install Dependencies
```bash
# Install additional Python packages
pip install neo4j networkx matplotlib seaborn plotly

# Or update requirements.txt and reinstall
pip install -r requirements.txt
```

#### Step 2: Start Neo4j
```bash
# Start Neo4j container
docker-compose -f docker-compose-neo4j.yml up -d

# Wait for startup (about 30 seconds)
# Access Neo4j Browser at http://localhost:7474
# Username: neo4j, Password: knowledge123
```

#### Step 3: Automated Setup (Recommended)
```bash
# Windows
.\setup_knowledge_graph.ps1

# Linux/macOS
./setup_knowledge_graph.sh
```

#### Step 4: Build Knowledge Graph
```bash
python 5_build_knowledge_graph.py
```

### Knowledge Graph Features

#### Interactive Explorer
```bash
python 6_query_knowledge_graph.py
```

**Capabilities:**
- Search for concepts across the knowledge graph
- Get personalized learning path recommendations
- Track your progress through topics
- Find prerequisites for any concept
- View learning statistics and analytics

#### Visualizations
```bash
python 7_visualize_knowledge_graph.py
```

**Generated Visualizations:**
- Interactive network diagrams showing concept relationships
- Hierarchical sunburst charts for course structure
- Analytics dashboards with learning metrics
- Personal learning path visualizations

#### Advanced Querying
Access Neo4j Browser at `http://localhost:7474` for custom Cypher queries:

```cypher
// Find learning paths from beginner to advanced topics
MATCH path = (start:KnowledgeNode {difficulty: 'beginner'})-[*1..4]->(end:KnowledgeNode {difficulty: 'advanced'})
RETURN path LIMIT 5

// Analyze prerequisites for any concept
MATCH (concept:KnowledgeNode)<-[:PREREQUISITE_FOR*]-(prereq)
RETURN concept.title, count(prereq) as prerequisite_count
ORDER BY prerequisite_count DESC LIMIT 10
```

### Knowledge Graph Structure

The system automatically creates a hierarchical structure:

1. **Course Level** - Complete subjects or domains
2. **Chapter Level** - Major sections within courses  
3. **Learning Objective Level** - Specific learning goals
4. **Concept Level** - Individual definitions and principles

Relationships include prerequisites, containment, and teaching relationships.

---

## Domain Adaptation Guide

### Adapting to Your Domain

#### 1. Document Preparation
```bash
# Organize your domain documents
data_large/
├── textbooks/           # Core academic content
├── papers/             # Research papers
├── manuals/            # Technical documentation
├── faqs/               # Frequently asked questions
└── glossary/           # Domain-specific terminology
```

#### 2. Domain-Specific Configuration

Create a domain configuration file:
```python
# domain_config.py
DOMAIN_SETTINGS = {
    "medical": {
        "chunk_size": 256,  # Smaller chunks for precise medical info
        "chunk_overlap": 50,
        "similarity_top_k": 15,
        "rerank_top_n": 5,
        "embedding_model": "sentence-transformers/all-mpnet-base-v2"
    },
    "legal": {
        "chunk_size": 512,  # Larger chunks for legal context
        "chunk_overlap": 100,
        "similarity_top_k": 12,
        "rerank_top_n": 4,
        "embedding_model": "sentence-transformers/all-mpnet-base-v2"
    },
    "technical": {
        "chunk_size": 384,  # Medium chunks for technical docs
        "chunk_overlap": 75,
        "similarity_top_k": 10,
        "rerank_top_n": 3,
        "embedding_model": "sentence-transformers/all-mpnet-base-v2"
    }
}
```

#### 3. Custom Prompts for Your Domain

Edit the system prompts in the query script:
```python
# Example: Medical domain prompt
MEDICAL_SYSTEM_PROMPT = """
You are a medical AI assistant. Provide accurate, evidence-based answers.
Always include disclaimers about consulting healthcare professionals.
Cite specific sources when providing medical information.
If uncertain, clearly state limitations and recommend professional consultation.
"""

# Example: Legal domain prompt  
LEGAL_SYSTEM_PROMPT = """
You are a legal research assistant. Provide information based on the documents.
Always include disclaimers that this is not legal advice.
Cite specific cases, statutes, or regulations when relevant.
Recommend consulting qualified legal professionals for specific cases.
"""
```

#### 4. Domain-Specific Metadata

Enhance document processing with domain metadata:
```python
# Add to document processing
def add_domain_metadata(document, domain_type):
    metadata = {
        "domain": domain_type,
        "document_type": detect_document_type(document),
        "authority_level": assess_authority(document),
        "date_published": extract_date(document),
        "source_credibility": assess_credibility(document)
    }
    return metadata
```

#### 5. Evaluation for Your Domain

Create domain-specific test questions:
```json
{
  "medical_questions": [
    {
      "question": "What are the contraindications for ACE inhibitors?",
      "expected_topics": ["ACE inhibitors", "contraindications", "side effects"],
      "difficulty": "intermediate"
    }
  ],
  "legal_questions": [
    {
      "question": "What constitutes breach of contract?",
      "expected_topics": ["contract law", "breach", "remedies"],
      "difficulty": "basic"
    }
  ]
}
```

---

## Configuration and Customization

### Environment Variables

Create `.env` file for configuration:
```bash
# Elasticsearch Configuration
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=your_domain_docs

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
LLM_MODEL=qwen3:4b
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Processing Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_TOKENS=4096
BATCH_SIZE=32

# Retrieval Configuration
SIMILARITY_TOP_K=12
RERANK_TOP_N=4
TEMPERATURE=0.1

# Performance Configuration
USE_GPU=true
MAX_WORKERS=4
CACHE_SIZE=1000
```

### Model Customization

#### Switching LLM Models
```python
# In your scripts, change the model:
Settings.llm = Ollama(model="mistral:7b", request_timeout=300.0)
# Or:
Settings.llm = Ollama(model="llama2:13b", request_timeout=300.0)
# Or:
Settings.llm = Ollama(model="codellama:7b", request_timeout=300.0)
```

#### Switching Embedding Models
```python
# Different embedding models for different domains:
# General purpose (default):
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Better for scientific/technical content:
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/allenai-specter"
)

# Better for code:
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Chunking Strategy Customization

```python
# Modify the node parser in build script:
from llama_index.core.node_parser import HierarchicalNodeParser

# Conservative chunking (more context, slower)
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 1024, 512],  # Larger chunks
    chunk_overlap=100
)

# Aggressive chunking (less context, faster)
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[1024, 256, 128],   # Smaller chunks
    chunk_overlap=25
)

# Domain-specific chunking
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[1536, 384, 192],   # Balanced approach
    chunk_overlap=50,
    include_metadata=True,
    include_prev_next_rel=True
)
```

---

## Operations and Maintenance

### Daily Operations

#### System Health Checks
```bash
# Check Elasticsearch health
curl -X GET "localhost:9200/_cluster/health?pretty"

# Check disk usage
docker system df

# Check Ollama status
curl http://localhost:11434/api/tags

# Check index statistics
python 3_inspect_elasticsearch.py
```

#### Log Monitoring
```bash
# Elasticsearch logs
docker-compose -f docker-compose-elasticsearch.yml logs elasticsearch

# System resource usage
# Windows:
Get-Process | Where-Object {$_.ProcessName -like "*elasticsearch*"}
# Linux:
top -p $(pgrep elasticsearch)
```

### Maintenance Tasks

#### Weekly Maintenance
```bash
# Clean up Docker
docker system prune -f

# Update Ollama models
ollama pull qwen3:4b

# Backup Elasticsearch data
docker-compose -f docker-compose-elasticsearch.yml exec elasticsearch \
    tar czf /tmp/backup.tar.gz /usr/share/elasticsearch/data
```

#### Monthly Maintenance
```bash
# Update Python packages
pip list --outdated
pip install --upgrade llama-index-core

# Reindex if schema changes
python 1_build_database_elasticsearch.py --rebuild

# Performance analysis
python batch_test_rag.py
```

### Backup and Recovery

#### Backup Strategy
```bash
# 1. Backup Elasticsearch data
mkdir -p backups/$(date +%Y-%m-%d)
docker cp elasticsearch-rag:/usr/share/elasticsearch/data backups/$(date +%Y-%m-%d)/

# 2. Backup configuration files
cp docker-compose-elasticsearch.yml backups/$(date +%Y-%m-%d)/
cp requirements.txt backups/$(date +%Y-%m-%d)/
cp .env backups/$(date +%Y-%m-%d)/

# 3. Backup source documents
tar czf backups/$(date +%Y-%m-%d)/documents.tar.gz data_large/
```

#### Recovery Process
```bash
# 1. Stop services
docker-compose -f docker-compose-elasticsearch.yml down

# 2. Restore data
docker cp backups/YYYY-MM-DD/data elasticsearch-rag:/usr/share/elasticsearch/

# 3. Restart services
docker-compose -f docker-compose-elasticsearch.yml up -d

# 4. Verify restoration
curl http://localhost:9200/advanced_docs_elasticsearch_v2/_count
```

---

## Performance Optimization

### Hardware Optimization

#### CPU Optimization
```python
# Adjust worker processes based on CPU cores
import os
os.environ["OMP_NUM_THREADS"] = "4"  # Set to your CPU core count
os.environ["MKL_NUM_THREADS"] = "4"
```

#### Memory Optimization
```yaml
# In docker-compose-elasticsearch.yml
environment:
  - "ES_JAVA_OPTS=-Xms4g -Xmx8g"  # Adjust based on available RAM
```

#### GPU Optimization
```python
# Force GPU usage
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="cuda",
    batch_size=64  # Increase for better GPU utilization
)
```

### Software Optimization

#### Query Performance
```python
# Optimize retrieval parameters
retriever = index.as_retriever(
    similarity_top_k=8,      # Reduce from 12 for speed
    vector_store_query_mode="hybrid",
    alpha=0.5               # Balance vector vs text search
)

# Disable reranking for speed (if acceptable)
# query_engine = RetrieverQueryEngine.from_args(
#     retriever=retriever,
#     # node_postprocessors=[]  # Remove reranker
# )
```

#### Batch Processing
```python
# Process documents in batches
def process_documents_in_batches(documents, batch_size=50):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        yield batch
```

### Elasticsearch Optimization

#### Index Settings
```python
# Add to build script
index_settings = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "refresh_interval": "30s",
    "index.max_result_window": 50000
}
```

#### Query Optimization
```python
# Optimize search parameters
search_params = {
    "size": 10,
    "timeout": "30s",
    "_source": ["content", "metadata"],
    "track_total_hits": False
}
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Elasticsearch Won't Start
**Symptoms**: Connection refused on port 9200
```bash
# Check if port is in use
netstat -an | findstr :9200  # Windows
netstat -tuln | grep 9200    # Linux

# Check Docker status
docker ps
docker logs elasticsearch-rag

# Solutions:
# 1. Restart Docker
docker-compose -f docker-compose-elasticsearch.yml restart

# 2. Clean restart
docker-compose -f docker-compose-elasticsearch.yml down -v
docker-compose -f docker-compose-elasticsearch.yml up -d

# 3. Check memory limits
# Increase Docker memory allocation in Docker Desktop
```

#### 2. Out of Memory Errors
**Symptoms**: "OutOfMemoryError" or system freeze
```bash
# Solutions:
# 1. Reduce batch size
# Edit build script: batch_size=16 instead of 32

# 2. Increase system memory
# Edit docker-compose-elasticsearch.yml:
# ES_JAVA_OPTS=-Xms2g -Xmx4g

# 3. Use CPU instead of GPU
# Edit scripts: device="cpu"
```

#### 3. Ollama Connection Issues
**Symptoms**: "Connection refused on port 11434"
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Solutions:
# 1. Start Ollama service
ollama serve

# 2. Check model availability
ollama list
ollama pull qwen3:4b

# 3. Restart Ollama
# Windows: Restart from system tray
# Linux: sudo systemctl restart ollama
```

#### 4. Slow Query Performance
**Symptoms**: Queries take >10 seconds
```python
# Solutions:
# 1. Reduce similarity_top_k
similarity_top_k=6  # Instead of 12

# 2. Disable reranking temporarily
# Comment out reranker in query script

# 3. Check Elasticsearch performance
curl "localhost:9200/_cat/indices?v"
curl "localhost:9200/_nodes/stats/indices"
```

#### 5. CUDA/GPU Issues
**Symptoms**: CUDA out of memory or driver errors
```bash
# Check GPU status
nvidia-smi

# Solutions:
# 1. Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python 1_build_database_elasticsearch.py

# 2. Reduce batch size
# Edit embedding batch_size from 32 to 8

# 3. Update GPU drivers
# Download from nvidia.com
```

#### 6. Document Processing Errors
**Symptoms**: "Unable to parse document" errors
```python
# Check document format
file document.pdf
file document.docx

# Solutions:
# 1. Convert problematic documents
# Use LibreOffice or other tools

# 2. Skip corrupted files
# Add try-catch in document loading

# 3. Check file permissions
ls -la data_large/
```

### Diagnostic Scripts

#### System Health Check
```python
# health_check.py
import requests
import subprocess
import os

def check_elasticsearch():
    try:
        response = requests.get("http://localhost:9200")
        return response.status_code == 200
    except:
        return False

def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

def check_gpu():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        return result.returncode == 0
    except:
        return False

print(f"Elasticsearch: {'✓' if check_elasticsearch() else '✗'}")
print(f"Ollama: {'✓' if check_ollama() else '✗'}")
print(f"GPU: {'✓' if check_gpu() else '✗'}")
```

---

## Advanced Features

### Custom Retrievers

#### Multi-Modal Retrieval
```python
from llama_index.core.retrievers import BaseRetriever

class MultiModalRetriever(BaseRetriever):
    def __init__(self, vector_retriever, text_retriever, fusion_method="rrf"):
        self.vector_retriever = vector_retriever
        self.text_retriever = text_retriever
        self.fusion_method = fusion_method
    
    def _retrieve(self, query_bundle):
        vector_results = self.vector_retriever.retrieve(query_bundle)
        text_results = self.text_retriever.retrieve(query_bundle)
        return self._fuse_results(vector_results, text_results)
```

#### Time-Aware Retrieval
```python
class TimeAwareRetriever(BaseRetriever):
    def __init__(self, base_retriever, time_decay_factor=0.9):
        self.base_retriever = base_retriever
        self.time_decay_factor = time_decay_factor
    
    def _retrieve(self, query_bundle):
        results = self.base_retriever.retrieve(query_bundle)
        # Apply time decay based on document age
        for result in results:
            age_days = self._get_document_age(result)
            result.score *= (self.time_decay_factor ** (age_days / 365))
        return sorted(results, key=lambda x: x.score, reverse=True)
```

### Custom Query Engines

#### Citation-Enhanced Query Engine
```python
class CitationQueryEngine:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def query(self, query_str):
        context_nodes = self.retriever.retrieve(query_str)
        
        # Build context with citations
        context_with_citations = []
        for i, node in enumerate(context_nodes):
            citation = f"[{i+1}] {node.metadata.get('source', 'Unknown')}"
            context_with_citations.append(f"{node.text}\n\nSource: {citation}")
        
        prompt = self._build_citation_prompt(query_str, context_with_citations)
        return self.llm.complete(prompt)
```

#### Multi-Language Support
```python
class MultiLanguageQueryEngine:
    def __init__(self, retrievers_by_language, llm):
        self.retrievers = retrievers_by_language
        self.llm = llm
    
    def query(self, query_str):
        detected_language = self._detect_language(query_str)
        appropriate_retriever = self.retrievers.get(detected_language, self.retrievers['en'])
        
        context_nodes = appropriate_retriever.retrieve(query_str)
        return self._generate_response(query_str, context_nodes, detected_language)
```

### Evaluation and Metrics

#### Comprehensive Evaluation
```python
class RAGEvaluator:
    def __init__(self, query_engine, test_questions):
        self.query_engine = query_engine
        self.test_questions = test_questions
    
    def evaluate(self):
        results = []
        for question_data in self.test_questions:
            result = self._evaluate_single_question(question_data)
            results.append(result)
        
        return self._compute_aggregate_metrics(results)
    
    def _evaluate_single_question(self, question_data):
        start_time = time.time()
        response = self.query_engine.query(question_data['question'])
        response_time = time.time() - start_time
        
        return {
            'question': question_data['question'],
            'response': str(response),
            'response_time': response_time,
            'relevance_score': self._compute_relevance(response, question_data),
            'faithfulness_score': self._compute_faithfulness(response),
            'completeness_score': self._compute_completeness(response, question_data)
        }
```

---

## Production Deployment

### Docker Production Setup

#### Multi-Stage Dockerfile
```dockerfile
# Dockerfile
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash rag_user
USER rag_user

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "api_server.py"]
```

#### Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: elasticsearch-prod
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=true
      - ELASTIC_PASSWORD=your_secure_password
      - "ES_JAVA_OPTS=-Xms4g -Xmx4g"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
      - ./elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml
    ulimits:
      memlock:
        soft: -1
        hard: -1
    deploy:
      resources:
        limits:
          memory: 8g

  rag-api:
    build:
      context: .
      target: production
    container_name: rag-api-prod
    environment:
      - ELASTICSEARCH_HOST=elasticsearch
      - ELASTICSEARCH_PASSWORD=your_secure_password
      - OLLAMA_HOST=ollama
    ports:
      - "8000:8000"
    depends_on:
      - elasticsearch
      - ollama
    volumes:
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          memory: 4g
        reservations:
          memory: 2g

  ollama:
    image: ollama/ollama:latest
    container_name: ollama-prod
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    container_name: nginx-prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-api

volumes:
  elasticsearch_data:
  ollama_data:
```

### Monitoring and Logging

#### Application Logging
```python
# logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(log_level="INFO", log_file="rag_system.log"):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from dependencies
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)
```

#### Prometheus Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
QUERY_COUNT = Counter('rag_queries_total', 'Total number of queries')
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Query processing time')
ACTIVE_CONNECTIONS = Gauge('rag_active_connections', 'Active connections')
INDEX_SIZE = Gauge('rag_index_size_documents', 'Number of documents in index')

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        start_time = time.time()
        ACTIVE_CONNECTIONS.inc()
        QUERY_COUNT.inc()
        
        try:
            result = self.app(environ, start_response)
            return result
        finally:
            QUERY_DURATION.observe(time.time() - start_time)
            ACTIVE_CONNECTIONS.dec()
```

### Load Balancing and Scaling

#### Nginx Configuration
```nginx
# nginx.conf
upstream rag_backend {
    least_conn;
    server rag-api-1:8000 max_fails=3 fail_timeout=30s;
    server rag-api-2:8000 max_fails=3 fail_timeout=30s;
    server rag-api-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://rag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://rag_backend/health;
    }
}
```

### Security Considerations

#### Authentication and Authorization
```python
# auth.py
import jwt
import hashlib
from functools import wraps
from flask import request, jsonify

class AuthManager:
    def __init__(self, secret_key):
        self.secret_key = secret_key
    
    def require_auth(self, f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'Token missing'}), 401
            
            try:
                token = token.split(' ')[1]  # Remove 'Bearer '
                jwt.decode(token, self.secret_key, algorithms=['HS256'])
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Invalid token'}), 401
            
            return f(*args, **kwargs)
        return decorated
    
    def rate_limit(self, max_requests=100, window_minutes=60):
        # Implementation of rate limiting logic
        pass
```

---

## API Integration

### REST API Server

#### Flask API Implementation
```python
# api_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)

# Initialize RAG system (your existing code)
rag_system = initialize_rag_system()

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Quick system health check
        es_status = check_elasticsearch_health()
        ollama_status = check_ollama_health()
        
        return jsonify({
            'status': 'healthy' if es_status and ollama_status else 'unhealthy',
            'elasticsearch': es_status,
            'ollama': ollama_status,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_documents():
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        question = data['question']
        options = data.get('options', {})
        
        # Process query
        start_time = datetime.utcnow()
        response = rag_system.query(question, **options)
        end_time = datetime.utcnow()
        
        return jsonify({
            'question': question,
            'answer': str(response),
            'metadata': {
                'response_time_ms': (end_time - start_time).total_seconds() * 1000,
                'sources': getattr(response, 'source_nodes', []),
                'timestamp': start_time.isoformat()
            }
        })
        
    except Exception as e:
        logging.error(f"Query error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/documents', methods=['POST'])
def add_documents():
    """Add new documents to the index"""
    try:
        # Handle file uploads
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        results = []
        for file in files:
            if file.filename:
                # Save and process file
                result = process_uploaded_file(file)
                results.append(result)
        
        return jsonify({
            'message': f'Processed {len(results)} files',
            'results': results
        })
        
    except Exception as e:
        logging.error(f"Document upload error: {str(e)}")
        return jsonify({'error': 'Failed to process documents'}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics"""
    try:
        stats = {
            'document_count': get_document_count(),
            'index_size_mb': get_index_size(),
            'last_updated': get_last_update_time(),
            'query_count_today': get_daily_query_count(),
            'average_response_time_ms': get_avg_response_time()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=8000, debug=False)
```

### Client SDKs

#### Python Client
```python
# rag_client.py
import requests
import json
from typing import Optional, Dict, Any

class RAGClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def query(self, question: str, **options) -> Dict[str, Any]:
        """Query the RAG system"""
        url = f"{self.base_url}/query"
        payload = {'question': question, 'options': options}
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def add_documents(self, file_paths: list) -> Dict[str, Any]:
        """Add documents to the system"""
        url = f"{self.base_url}/documents"
        
        files = []
        for file_path in file_paths:
            files.append(('files', open(file_path, 'rb')))
        
        try:
            response = self.session.post(url, files=files)
            response.raise_for_status()
            return response.json()
        finally:
            for _, file_obj in files:
                file_obj.close()
    
    def get_health(self) -> Dict[str, Any]:
        """Check system health"""
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        url = f"{self.base_url}/statistics"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

# Usage example
if __name__ == "__main__":
    client = RAGClient("http://localhost:8000")
    
    # Check health
    health = client.get_health()
    print(f"System status: {health['status']}")
    
    # Query system
    result = client.query("What is machine learning?")
    print(f"Answer: {result['answer']}")
    
    # Get statistics
    stats = client.get_statistics()
    print(f"Documents: {stats['document_count']}")
```

#### JavaScript Client
```javascript
// rag-client.js
class RAGClient {
    constructor(baseUrl, apiKey = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.headers = {
            'Content-Type': 'application/json'
        };
        
        if (apiKey) {
            this.headers['Authorization'] = `Bearer ${apiKey}`;
        }
    }
    
    async query(question, options = {}) {
        const response = await fetch(`${this.baseUrl}/query`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ question, options })
        });
        
        if (!response.ok) {
            throw new Error(`Query failed: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    async addDocuments(files) {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        
        const response = await fetch(`${this.baseUrl}/documents`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    async getHealth() {
        const response = await fetch(`${this.baseUrl}/health`);
        
        if (!response.ok) {
            throw new Error(`Health check failed: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    async getStatistics() {
        const response = await fetch(`${this.baseUrl}/statistics`, {
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`Statistics failed: ${response.statusText}`);
        }
        
        return response.json();
    }
}

// Usage example
const client = new RAGClient('http://localhost:8000');

// Query with async/await
async function askQuestion() {
    try {
        const result = await client.query('What is machine learning?');
        console.log('Answer:', result.answer);
    } catch (error) {
        console.error('Error:', error.message);
    }
}
```

---

## Best Practices and Tips

### 1. Document Organization
- **Structure**: Organize documents by topic, authority level, and recency
- **Naming**: Use consistent, descriptive filenames
- **Metadata**: Include creation date, author, document type in filenames or metadata
- **Quality**: Ensure documents are well-formatted and free of OCR errors

### 2. Performance Optimization
- **Hardware**: Invest in SSD storage and adequate RAM
- **Batching**: Process documents in optimal batch sizes
- **Caching**: Implement caching for frequently accessed content
- **Monitoring**: Continuously monitor system performance

### 3. Quality Assurance
- **Testing**: Regularly test with domain-specific questions
- **Evaluation**: Use multiple metrics (relevance, faithfulness, completeness)
- **Feedback**: Implement feedback loops to improve responses
- **Updates**: Regularly update documents and retrain as needed

### 4. Security Best Practices
- **Access Control**: Implement proper authentication and authorization
- **Data Privacy**: Ensure sensitive documents are properly protected
- **Audit Logs**: Maintain comprehensive audit trails
- **Regular Updates**: Keep all components updated with security patches

### 5. Operational Excellence
- **Monitoring**: Implement comprehensive monitoring and alerting
- **Backups**: Maintain regular backups of all data and configurations
- **Documentation**: Keep detailed documentation of configurations and procedures
- **Disaster Recovery**: Have tested disaster recovery procedures

---

## Conclusion

This comprehensive manual provides everything needed to build, deploy, and maintain a subject matter expert RAG system for any domain. The system is designed to be:

- **Scalable**: From proof of concept to enterprise deployment
- **Flexible**: Easily adaptable to different domains and use cases
- **Robust**: Production-ready with monitoring and security features
- **Maintainable**: Clear documentation and operational procedures

For additional support or advanced customizations, refer to the individual component documentation and consider engaging with the open-source community around LlamaIndex, Elasticsearch, and Ollama.

Remember that building an effective RAG system is an iterative process. Start with the basic setup, evaluate performance with your specific documents and use cases, then gradually optimize and enhance the system based on your needs.
