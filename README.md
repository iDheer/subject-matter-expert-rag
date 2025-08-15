# SME (Subject Matter Expert) System

A sophisticated RAG (Retrieval Augmented Generation) system that provides intelligent document querying with conversational memory. Built with LlamaIndex, Elasticsearch, and FastAPI.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   UI Frontend   │───▶│   SME API        │───▶│  Elasticsearch  │
│   (Web/Mobile)  │    │   (FastAPI)      │    │  Vector Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Ollama LLM     │
                       │   (Qwen3:4b)     │
                       └──────────────────┘
```

## 📁 Project Structure

```
sme-system/
├── SME_1_build_elasticsearch_database.py    # Database builder
├── SME_2_query_elasticsearch_system.py      # CLI query system
├── SME_3_inspect_elasticsearch_database.py  # Database inspector
├── api_server.py                            # FastAPI REST API
├── test_api.py                              # API test suite
├── docker-compose-elasticsearch.yml         # Elasticsearch container
├── requirements.txt                         # Core dependencies
├── requirements_api.txt                     # API dependencies
├── data_large/                             # Document directory
├── elasticsearch_storage_v2/               # Generated storage
└── README.md                               # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Docker & Docker Compose**
- **Git** (optional)
- **8GB+ RAM** (16GB recommended)
- **GPU with CUDA** (optional but recommended)

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# API dependencies
pip install -r requirements_api.txt
```

### Step 2: Start Elasticsearch

```bash
# Start Elasticsearch container
docker-compose -f docker-compose-elasticsearch.yml up -d

# Verify it's running
curl http://localhost:9200
```

### Step 3: Install and Configure Ollama

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download

# Download the Qwen3:4b model
ollama pull qwen3:4b

# Verify installation
ollama list
```

### Step 4: Prepare Documents

```bash
# Create data directory
mkdir -p data_large

# Add your documents (supports .txt, .pdf, .docx, .md)
cp /path/to/your/documents/* data_large/

# Verify documents
ls data_large/
```

### Step 5: Build Database

```bash
# This will take 10-30 minutes depending on document size
python SME_1_build_elasticsearch_database.py
```

### Step 6: Start API Server

#### Option A: Native Python (Simple)
```bash
# Direct Python execution
python api_server.py

# Wait for: "🎉 SME API Server ready!"
```

#### Option B: Docker Container (Recommended for Production)
```bash
# Start API in Docker container
docker-compose -f docker-compose-api.yml up -d --build

# Check logs
docker logs sme-api -f

# Wait for: "🎉 SME API Server ready!"
```

#### Manual Startup (Any Platform)
```bash
# 1. Check prerequisites
curl http://localhost:9200  # Elasticsearch
curl http://localhost:11434/api/tags  # Ollama

# 2. Install API dependencies (if not using Docker)
pip install fastapi uvicorn[standard] pydantic python-multipart

# 3. Verify database exists
ls elasticsearch_storage_v2/  # Linux/Mac
dir elasticsearch_storage_v2\  # Windows

# 4A. Start with Python
python api_server.py

# 4B. Or start with Docker
docker-compose -f docker-compose-api.yml up -d --build
```

#### Troubleshooting Startup
If you get errors, check each prerequisite:

```bash
# Check Elasticsearch
python -c "import requests; print('ES:', requests.get('http://localhost:9200').status_code)"

# Check Ollama
python -c "import requests; print('Ollama:', requests.get('http://localhost:11434/api/tags').status_code)"

# Check database files
python SME_3_inspect_elasticsearch_database.py

# Check Python dependencies
python -c "import fastapi, llama_index; print('Dependencies OK')"
```

### Step 7: Test the System

```bash
# Run test suite
python test_api.py

# Or visit interactive docs
# http://localhost:8000/docs
```

## 🔧 Usage

### CLI Interface

```bash
# Interactive command-line interface
python SME_2_query_elasticsearch_system.py
```

**Available CLI commands:**
- Type questions normally for responses
- `/memory` - Toggle conversation memory
- `/clear` - Clear conversation history
- `/status` - Show memory status
- `/help` - Show help
- `exit` - Exit system

## 🔌 API Integration for UI Teams

### Quick Integration

**Base URL:** `http://your-server-ip:8000`

**Main Query Endpoint:** `POST /query`

**Minimal Integration Example:**
```javascript
const response = await fetch('http://your-server:8000/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        question: "What are the main topics?",
        use_memory: true
    })
});

const data = await response.json();

// Use the response
console.log(data.answer);    // "Based on the documents, the main topics include..."
console.log(data.sources);   // Array of supporting documents
console.log(data.memory_status); // "Memory enabled - 1 exchanges"
```

**Complete Response Structure:**
```json
{
    "answer": "Based on the documents, the main topics include...",
    "sources": [
        {
            "index": 1,
            "score": 0.8756,
            "file_name": "document1.pdf", 
            "node_type": "Leaf",
            "content_snippet": "The document discusses...",
            "node_id": "node_123"
        }
    ],
    "memory_status": "Memory enabled - 1 exchanges"
}
```

**UI Implementation:**
```javascript
// Complete integration example
async function askSME(question) {
    try {
        const response = await fetch('http://your-server:8000/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                use_memory: true
            })
        });
        
        const data = await response.json();
        
        // Display answer
        document.getElementById('answer').innerHTML = data.answer;
        
        // Display sources
        const sourcesHtml = data.sources.map(source => 
            `<div>📄 ${source.file_name} (Score: ${source.score})</div>`
        ).join('');
        document.getElementById('sources').innerHTML = sourcesHtml;
        
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
}
```

### All Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/query` | POST | **Main endpoint - Ask questions** |
| `/status` | GET | Check system health |
| `/memory/toggle` | POST | Toggle conversation memory |
| `/memory/clear` | POST | Clear conversation history |
| `/memory/status` | GET | Get memory status |
| `/docs` | GET | Interactive API documentation |

### API Interface

**Base URL:** `http://localhost:8000`

**Interactive Documentation:** `http://localhost:8000/docs`

#### Main Endpoints

##### Query SME System
```http
POST /query
Content-Type: application/json

{
    "question": "What are the main topics in the documents?",
    "use_memory": true,
    "stream": false
}
```

**Response:**
```json
{
    "answer": "Based on the documents, the main topics include...",
    "sources": [
        {
            "index": 1,
            "score": 0.8756,
            "file_name": "document1.pdf",
            "node_type": "Leaf",
            "content_snippet": "The document discusses...",
            "node_id": "node_123"
        }
    ],
    "memory_status": "Memory enabled - 1 exchanges"
}
```

##### System Status
```http
GET /status
```

**Response:**
```json
{
    "status": "ready",
    "memory_status": "Memory enabled - 3 exchanges",
    "elasticsearch_connected": true,
    "index_exists": true,
    "docstore_nodes": 8569
}
```

##### Memory Management
```http
# Toggle memory
POST /memory/toggle

# Clear memory
POST /memory/clear

# Check memory status
GET /memory/status
```

### JavaScript Integration Example

```javascript
async function askSME(question) {
    try {
        const response = await fetch('http://localhost:8000/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                use_memory: true
            })
        });
        
        const data = await response.json();
        
        console.log('Answer:', data.answer);
        console.log('Sources:', data.sources);
        
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
}

// Usage
askSME("What are the key concepts in the documents?");
```

## 🛠️ System Components

### 1. Document Ingestion (`SME_1_build_elasticsearch_database.py`)

**Features:**
- Hierarchical document chunking (2048, 512, 128 tokens)
- Vector embeddings with sentence-transformers
- Elasticsearch vector storage
- Parent-child relationship preservation

**Configuration:**
```python
# Chunk sizes for hierarchical parsing
chunk_sizes=[2048, 512, 128]
chunk_overlap=20

# Embedding model
model_name="sentence-transformers/all-mpnet-base-v2"

# Elasticsearch settings
ES_ENDPOINT = "http://localhost:9200"
INDEX_NAME = "advanced_docs_elasticsearch_v2"
```

### 2. Query System (`SME_2_query_elasticsearch_system.py`)

**Features:**
- AutoMergingRetriever for hierarchical retrieval
- Conversation memory management
- Response reranking with BGE-reranker-v2-m3
- Qwen3:4b LLM for response generation

### 3. API Server (`api_server.py`)

**Features:**
- FastAPI REST endpoints
- CORS support for web integration
- Conversation memory API
- Real-time response streaming
- Error handling and validation

### 4. Database Inspector (`SME_3_inspect_elasticsearch_database.py`)

**Features:**
- Elasticsearch index inspection
- Docstore integrity checking
- Hierarchy relationship validation
- Performance diagnostics

## ⚙️ Configuration

### Environment Variables

```bash
# GPU acceleration
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_GPU=1
export OLLAMA_GPU_LAYERS=35

# API settings
export SME_API_HOST=0.0.0.0
export SME_API_PORT=8000
```

### Model Configuration

**LLM Model:** Qwen3:4b via Ollama
- Download: `ollama pull qwen3:4b`
- Memory: ~4GB GPU/RAM
- Features: Thinking tags, multilingual

**Embedding Model:** sentence-transformers/all-mpnet-base-v2
- Dimensions: 768
- Language: English optimized
- Performance: High quality semantic search

**Reranker:** BAAI/bge-reranker-v2-m3
- Purpose: Result reranking
- Language: Multilingual support

### Elasticsearch Configuration

```yaml
# docker-compose-elasticsearch.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
```

## 🖥️ Platform-Specific Instructions

### Linux/Ubuntu Server
```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-pip docker.io docker-compose curl

# Install Python dependencies
pip3 install -r requirements.txt -r requirements_api.txt

# Start services
sudo docker-compose -f docker-compose-elasticsearch.yml up -d
ollama serve &
python3 api_server.py
```

### Windows Server
```powershell
# Install Python dependencies
pip install -r requirements.txt -r requirements_api.txt

# Start Elasticsearch (using Docker Desktop)
docker-compose -f docker-compose-elasticsearch.yml up -d

# Start Ollama (download from https://ollama.ai/download)
# ollama serve (runs automatically as service)

# Start API
python api_server.py
```

### CentOS/RHEL Server
```bash
# Install dependencies
sudo yum update
sudo yum install -y python3-pip docker docker-compose curl

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Follow Linux instructions above
```

### macOS
```bash
# Install dependencies
brew install python docker docker-compose

# Install Ollama
brew install ollama

# Follow Linux instructions above
```

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB
- Network: 1Gbps

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA with 8GB+ VRAM
- Storage: SSD with 50GB+
- Network: 1Gbps+

### Performance Tuning

**For Large Document Sets:**
```python
# Increase chunk sizes
chunk_sizes=[4096, 1024, 256]

# Increase retrieval count
similarity_top_k=20
top_n=8  # reranker
```

**For Better Speed:**
```python
# Reduce chunk sizes
chunk_sizes=[1024, 256, 64]

# Reduce retrieval count
similarity_top_k=8
top_n=3
```

**Elasticsearch Tuning:**
```yaml
environment:
  - "ES_JAVA_OPTS=-Xms4g -Xmx4g"  # Increase heap
  - bootstrap.memory_lock=true
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Elasticsearch Connection Failed
```bash
# Check if container is running
docker ps

# Restart Elasticsearch
docker-compose -f docker-compose-elasticsearch.yml restart

# Check logs
docker logs elasticsearch-rag
```

#### 2. Ollama Model Not Found
```bash
# Download model
ollama pull qwen3:4b

# Verify download
ollama list

# Check Ollama service
curl http://localhost:11434/api/tags
```

#### 3. Out of Memory Errors
```bash
# Reduce batch sizes in build script
batch_size = 50  # instead of 100

# Use CPU instead of GPU
device = "cpu"

# Reduce Elasticsearch heap
ES_JAVA_OPTS=-Xms512m -Xmx512m
```

#### 4. API Returns 503 Errors
```bash
# Check if database was built
ls elasticsearch_storage_v2/

# Rebuild database
python SME_1_build_elasticsearch_database.py

# Check API logs
python api_server.py  # Look for error messages
```

#### 5. Slow Query Performance
```bash
# Check document count
python SME_3_inspect_elasticsearch_database.py

# Optimize chunk sizes
# Reduce similarity_top_k in retriever
# Use GPU acceleration
```

### Debug Commands

```bash
# Check all services
curl http://localhost:9200  # Elasticsearch
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8000/status  # SME API

# Test components individually
python SME_3_inspect_elasticsearch_database.py
python test_api.py

# Check system resources
nvidia-smi  # GPU usage
htop  # CPU/RAM usage
df -h  # Disk usage
```

## 🔒 Security Considerations

### Production Deployment (No Scripts Required)

#### 1. Automated Service Setup (Linux)
```bash
# Create systemd service for SME API
sudo tee /etc/systemd/system/sme-api.service > /dev/null <<EOF
[Unit]
Description=SME API Server
After=network.target docker.service

[Service]
Type=simple
User=smeuser
WorkingDirectory=/opt/sme-system
Environment=PATH=/opt/sme-system/venv/bin
ExecStart=/opt/sme-system/venv/bin/python api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable sme-api
sudo systemctl start sme-api
```

#### 2. Manual Production Startup
```bash
# Navigate to project directory
cd /path/to/sme-system

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Start all services in order
docker-compose -f docker-compose-elasticsearch.yml up -d
ollama serve &
python api_server.py

# Keep running with screen/tmux (optional)
screen -S sme-api python api_server.py
# Detach with Ctrl+A, D
```

#### 3. Windows Service Setup
```powershell
# Create Windows service using NSSM (Non-Sucking Service Manager)
# Download NSSM from https://nssm.cc/download

# Install API as service
nssm install SME-API "C:\Python39\python.exe"
nssm set SME-API Parameters "C:\path\to\sme-system\api_server.py"
nssm set SME-API AppDirectory "C:\path\to\sme-system"

# Start service
nssm start SME-API
```

### Data Privacy

- Documents are stored locally in Elasticsearch
- No data sent to external services (except Ollama locally)
- Conversation memory can be disabled
- Clear memory option available

## 📈 Monitoring & Logging

### Health Checks

```bash
# Automated health check script
#!/bin/bash
curl -f http://localhost:8000/status || exit 1
curl -f http://localhost:9200/_cluster/health || exit 1
curl -f http://localhost:11434/api/tags || exit 1
```

### Log Locations

```bash
# API logs (when run as service)
journalctl -u sme-api -f

# Elasticsearch logs
docker logs elasticsearch-rag -f

# Ollama logs
journalctl -u ollama -f
```

### Metrics to Monitor

- Query response time
- Memory usage
- Elasticsearch cluster health
- API request rate
- Error rates

## 🔄 Updates & Maintenance

### Updating Documents

```bash
# Add new documents to data_large/
cp new_documents/* data_large/

# Rebuild database
python SME_1_build_elasticsearch_database.py
```

### Model Updates

```bash
# Update Ollama model
ollama pull qwen3:4b  # Gets latest version

# Update embedding model (automatic via HuggingFace)
# No manual update needed
```

### Backup & Recovery

```bash
# Backup Elasticsearch data
docker exec elasticsearch-rag tar czf /backup/es-data.tar.gz /usr/share/elasticsearch/data

# Backup docstore
tar czf sme-backup.tar.gz elasticsearch_storage_v2/

# Restore
tar xzf sme-backup.tar.gz
docker-compose restart
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repo-url>
cd sme-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt -r requirements_api.txt
```

### Code Structure

- **Core Logic:** SME_2_query_elasticsearch_system.py
- **API Layer:** api_server.py
- **Data Pipeline:** SME_1_build_elasticsearch_database.py
- **Utilities:** SME_3_inspect_elasticsearch_database.py

### Testing

```bash
# Run API tests
python test_api.py

# Manual testing
python SME_2_query_elasticsearch_system.py

# Integration testing
# Visit http://localhost:8000/docs
```

## 📞 Support

### Getting Help

1. **Check this README** for common issues
2. **Run diagnostic tools:**
   ```bash
   python SME_3_inspect_elasticsearch_database.py
   python test_api.py
   ```
3. **Check logs** for error messages
4. **Verify all services** are running

### System Requirements Verification

```bash
# Check Python version
python --version  # Should be 3.9+

# Check Docker
docker --version
docker-compose --version

# Check CUDA (if using GPU)
nvidia-smi

# Check disk space
df -h
```

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **LlamaIndex** - RAG framework
- **Elasticsearch** - Vector search engine
- **Ollama** - Local LLM serving
- **FastAPI** - API framework
- **HuggingFace** - Embedding models

---

**Happy querying! 🚀**

For questions or issues, please check the troubleshooting section or create an issue in the repository.