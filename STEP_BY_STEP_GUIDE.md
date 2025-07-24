# STEP_BY_STEP_GUIDE.md
# Complete Step-by-Step Guide: Running the Subject Matter Expert RAG System

## 🚀 QUICKEST WAY - Automated Setup

**Just run this one command:**
```bash
python RUN_COMPLETE_SYSTEM.py
```

This master script will automatically:
- Check prerequisites
- Install dependencies  
- Start Docker containers
- Build databases
- Create knowledge graph
- Generate visualizations
- Run system demo

**Estimated time:** 10-20 minutes

---

## 📋 MANUAL STEP-BY-STEP PROCESS

If you prefer to run each step manually or need to troubleshoot:

### Prerequisites Check
```bash
# Verify installations
python --version    # Should be 3.8+
docker --version    # Should be 20.10+
docker-compose --version
```

### Step 0: Prepare Your Documents
```bash
# Create data directory (if not exists)
mkdir data_large

# Add your documents to data_large/
# Supported formats: PDF, DOCX, TXT, MD, JSON
```

### Step 1: Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 2: Start Elasticsearch Database
```bash
# Start Elasticsearch container
docker-compose -f docker-compose-elasticsearch.yml up -d

# Wait for startup (30-60 seconds)
# Verify it's running
curl http://localhost:9200
```

### Step 3: Build Elasticsearch Database 
```bash
# Process documents and create searchable index
python 1_build_database_elasticsearch.py

# This will:
# - Load documents from data_large/
# - Create hierarchical chunks
# - Build vector embeddings
# - Store in Elasticsearch
```

### Step 4: Start Neo4j Database
```bash
# Start Neo4j container for knowledge graph
docker-compose -f docker-compose-neo4j.yml up -d

# Wait for startup (30 seconds)
# Access Neo4j Browser at http://localhost:7474
# Username: neo4j, Password: knowledge123
```

### Step 5: Build Knowledge Graph
```bash
# Analyze documents and create knowledge relationships
python 5_build_knowledge_graph.py

# This will:
# - Analyze document structure
# - Extract learning objectives
# - Create prerequisite relationships
# - Build hierarchical knowledge graph in Neo4j
```

### Step 6: Create Visualizations
```bash
# Generate interactive visualizations
python 7_visualize_knowledge_graph.py

# Creates:
# - knowledge_graph_network.html (Network diagram)
# - knowledge_graph_hierarchy.html (Sunburst chart)
# - knowledge_graph_analytics.html (Analytics dashboard)
# - learning_path.html (Learning path visualization)
```

### Step 7: Test the Complete System
```bash
# Run comprehensive system demo
python 8_demo_knowledge_graph.py

# This demonstrates:
# - RAG query capabilities
# - Knowledge graph exploration
# - Learning path recommendations
# - Prerequisite tracking
```

---

## 🎯 HOW TO USE YOUR SYSTEM

### 1. Query Documents (RAG System)
```bash
python 2_query_system_elasticsearch_hierarchy.py
```
- Ask questions about your documents
- Get contextual answers with sources
- Uses hierarchical retrieval for better accuracy

### 2. Explore Knowledge Graph
```bash
python 6_query_knowledge_graph.py
```
- Search for concepts and topics
- Get personalized learning paths
- Track your learning progress
- Find prerequisites for any topic

### 3. View Interactive Visualizations
Open these HTML files in your browser:
- `knowledge_graph_network.html` - Interactive network diagram
- `knowledge_graph_hierarchy.html` - Hierarchical tree view
- `knowledge_graph_analytics.html` - Analytics dashboard
- `learning_path.html` - Personal learning journey

### 4. Access Neo4j Browser
- URL: http://localhost:7474
- Username: `neo4j`
- Password: `knowledge123`
- Run custom Cypher queries to explore your knowledge graph

---

## 🔍 INSPECTION AND DEBUGGING

### Check Elasticsearch Index
```bash
python 3_inspect_elasticsearch.py
```
Shows:
- Number of documents indexed
- Index statistics
- Sample documents

### Check Knowledge Graph Structure
```bash
python 3_inspect_hierarchy.py
```
Shows:
- Knowledge graph statistics
- Node counts by type and level
- Relationship summaries

### Run Batch Tests
```bash
python batch_test_rag.py
```
- Tests system with predefined questions
- Measures response times and accuracy
- Generates performance reports

---

## 🛠️ TROUBLESHOOTING

### Common Issues:

#### 1. Elasticsearch Won't Start
```bash
# Check if already running
docker ps

# Restart if needed
docker-compose -f docker-compose-elasticsearch.yml restart

# Check logs
docker-compose -f docker-compose-elasticsearch.yml logs elasticsearch
```

#### 2. Out of Memory Errors
- Reduce batch size in build scripts
- Increase Docker memory allocation
- Use CPU instead of GPU if memory limited

#### 3. Neo4j Connection Issues
```bash
# Check if container is running
docker ps

# Restart Neo4j
docker-compose -f docker-compose-neo4j.yml restart

# Check logs
docker-compose -f docker-compose-neo4j.yml logs neo4j
```

#### 4. Ollama/LLM Issues
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Pull required model
ollama pull qwen3:4b

# Restart Ollama service if needed
```

#### 5. Missing Documents
- Ensure documents are in `data_large/` directory
- Supported formats: PDF, DOCX, TXT, MD, JSON
- Check file permissions

---

## 📊 FILE EXECUTION ORDER

Here's the complete execution sequence:

### Database Setup (Required)
1. `1_build_database_elasticsearch.py` ✅ **REQUIRED**
   - Processes documents
   - Creates Elasticsearch index
   
2. ~~`1_build_database_advanced.py`~~ ❌ **NOT NEEDED**
   - ChromaDB version (alternative approach)
   - Knowledge graph doesn't use this

### Knowledge Graph Pipeline
3. `4_knowledge_graph_analyzer.py` - Called automatically by step 5
4. `5_build_knowledge_graph.py` ✅ **REQUIRED FOR KNOWLEDGE GRAPH**
   - Builds Neo4j knowledge graph
5. `6_query_knowledge_graph.py` - Interactive exploration tool
6. `7_visualize_knowledge_graph.py` - Creates visualizations
7. `8_demo_knowledge_graph.py` - Comprehensive demo

### Query and Testing
8. `2_query_system_elasticsearch_hierarchy.py` - Main RAG interface
9. `3_inspect_elasticsearch.py` - Database inspection
10. `3_inspect_hierarchy.py` - Knowledge graph inspection
11. `batch_test_rag.py` - Batch testing

---

## 🎯 DIFFERENT USAGE SCENARIOS

### Scenario 1: Basic RAG Only (No Knowledge Graph)
```bash
# Minimal setup
docker-compose -f docker-compose-elasticsearch.yml up -d
python 1_build_database_elasticsearch.py
python 2_query_system_elasticsearch_hierarchy.py
```

### Scenario 2: Complete System with Knowledge Graph
```bash
# Full setup (recommended)
python RUN_COMPLETE_SYSTEM.py
```

### Scenario 3: Add Knowledge Graph to Existing RAG
```bash
# If you already have Elasticsearch running
docker-compose -f docker-compose-neo4j.yml up -d
python 5_build_knowledge_graph.py
python 7_visualize_knowledge_graph.py
```

---

## 📚 DOCUMENTATION

- **Complete Manual:** `COMPREHENSIVE_MANUAL.md` - Full technical documentation
- **Quick Start:** `KNOWLEDGE_GRAPH_QUICKSTART.md` - 5-minute setup guide
- **This Guide:** `STEP_BY_STEP_GUIDE.md` - Detailed execution steps

---

## 🎉 SUCCESS INDICATORS

You'll know everything is working when:

✅ **Elasticsearch:** `curl http://localhost:9200` returns JSON
✅ **Neo4j:** `http://localhost:7474` shows login page
✅ **RAG System:** Can ask questions and get answers with sources
✅ **Knowledge Graph:** Can explore learning paths and prerequisites
✅ **Visualizations:** HTML files open with interactive charts
✅ **Demo:** `8_demo_knowledge_graph.py` runs without errors

**Estimated total setup time:** 10-20 minutes depending on document size and hardware.

**Ready to start? Run:** `python RUN_COMPLETE_SYSTEM.py`
