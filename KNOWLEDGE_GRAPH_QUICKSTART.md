# 🧠 Enhanced Chapter-Based Knowledge Graph System - Quick Test Guide

## 🎯 Overview
This guide helps you quickly test the **Enhanced Chapter-Based Knowledge Graph System** that transforms traditional document chunks into intelligent chapter-based nodes with dynamic concept extraction and GPU acceleration.

## 🚀 What This System Does

### Key Features:
- **Dynamic Concept Extraction**: Intelligently extracts 10-30 concepts per chapter (not hardcoded!)
- **5 Extraction Methods**: Explicit objectives, key terms, technical concepts, section concepts, question concepts
- **GPU Acceleration**: CUDA optimization throughout for faster processing
- **Concept Clustering**: K-means clustering for better organization
- **Knowledge Graph**: Neo4j graph with concept relationships and prerequisites
- **Advanced Visualization**: 3D concept clusters, prerequisite networks, analytics dashboards
- **Separate Database**: Uses `gpu_chapter_knowledge_v1` - your original system stays intact

## 📋 Prerequisites

### Required Services:
1. **Elasticsearch** (port 9200)
2. **Neo4j** (port 7687, username: neo4j, password: knowledge123)  
3. **Ollama** (port 11434 with qwen3:8b model)

### Quick Service Setup:
```bash
# 1. Start Elasticsearch
docker-compose -f docker-compose-elasticsearch.yml up -d

# 2. Start Neo4j (install Neo4j Desktop or Docker)
# Set password to 'knowledge123' when prompted

# 3. Start Ollama and install model
ollama serve
ollama pull qwen3:8b
```

## 🔧 Installation

### 1. Install Python Dependencies:
```bash
pip install -r enhanced_requirements.txt
```

### 2. Install Additional NLP Models:
```bash
# Install spaCy model for enhanced relation extraction
python -m spacy download en_core_web_sm

# NLTK data will be downloaded automatically
```

## 🎮 Quick Test - Complete Workflow

### Option 1: Full Automated Test
```bash
python KNOWLEDGE_GRAPH_TESTER.py
```

### Option 2: Step-by-Step Testing
```bash
# 1. Build enhanced chapter database (with dynamic concept extraction)
python NEW_1_build_enhanced_chapter_database_gpu.py

# 2. Build knowledge graph with concept clustering  
python NEW_5_build_enhanced_knowledge_graph_gpu.py

# 3. Test enhanced query system
python NEW_6_query_enhanced_knowledge_graph_gpu.py

# 4. Generate visualizations
python NEW_7_visualize_enhanced_knowledge_graph_gpu.py
```

### Option 3: Master Control System
```bash
python NEW_master_enhanced_runner_gpu.py
```

## 📊 What You'll Get

### 1. Enhanced Chapter Database
- **Input**: Your documents in `data_large/`
- **Output**: 20-25 chapter nodes with 10-30 concepts each
- **Database**: `gpu_chapter_knowledge_v1` (separate from original)

### 2. Knowledge Graph
- **Nodes**: Chapters with concept clusters
- **Relationships**: Prerequisites, similarities, hierarchies
- **Storage**: Neo4j graph database

### 3. Interactive Query System
Commands available:
- `search <query>` - Semantic search across concepts
- `chapter <id>` - Explore chapter concepts and clusters
- `path <chapter_id>` - Find learning paths
- `recommend` - Get personalized recommendations

### 4. Advanced Visualizations
- 📊 Chapter overview with concept distribution
- 🔮 3D concept clusters (GPU-accelerated t-SNE)
- 🌐 Prerequisite network visualization
- 🔥 Difficulty heatmaps
- 📈 Learning analytics dashboard

## 🔍 Testing Examples

### Test Concept Extraction:
```python
# The system will dynamically extract concepts like:
# - "Machine learning requires understanding of statistics" (prerequisite relation)
# - "Neural networks" (key term)
# - "What is gradient descent?" (question-based concept)
# - "Supervised Learning Algorithms" (section concept)
# - "Implement backpropagation algorithm" (explicit objective)
```

### Test Knowledge Graph Queries:
```bash
# In the query system:
search neural networks           # Find all neural network concepts
chapter chapter_1               # Explore first chapter concepts  
path chapter_10 intermediate    # Find learning path to chapter 10
cluster cluster_ml_basics       # Explore concept cluster
```

### Test Visualizations:
```bash
# In the visualization system:
overview                        # Chapter overview dashboard
3d                             # 3D concept clusters  
network                        # Prerequisite network
heatmap                        # Difficulty distribution
report                         # Generate complete HTML report
```

## 📁 Output Files & Locations

### Database Storage:
- **Elasticsearch**: `gpu_chapter_knowledge_v1` index
- **Neo4j**: Enhanced graph database
- **JSON Files**: `./gpu_knowledge_graph_data/`
  - `enhanced_chapters.json`
  - `enhanced_concepts.json` 
  - `enhanced_stats.json`

### Visualizations:
- **HTML Reports**: `./enhanced_visualizations/`
- **Interactive Plots**: Automatically opened in browser
- **Summary Report**: `./enhanced_visualizations/index.html`

### Logs:
- **System Logs**: `enhanced_system.log`
- **Performance Metrics**: Displayed in terminal

## 🎯 Expected Results

### Performance (with GPU):
- **Chapter Processing**: ~30 seconds per chapter
- **Concept Extraction**: 10-30 concepts per chapter (dynamic)
- **Total Build Time**: ~10-15 minutes for complete system
- **Query Response**: <2 seconds

### Concept Quality Examples:
```json
{
  "concept": "Supervised learning algorithms require labeled training data",
  "type": "technical_concept",
  "difficulty": "intermediate", 
  "confidence": 0.85,
  "relations": ["prerequisite", "is_a"],
  "cluster": "machine_learning_basics"
}
```

## 🛠️ Troubleshooting

### Common Issues:

1. **GPU Memory Error**:
   ```bash
   # Reduce batch size in code or clear GPU cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

2. **Service Connection Failed**:
   ```bash
   # Check services
   curl http://localhost:9200        # Elasticsearch
   curl http://localhost:11434       # Ollama  
   # Neo4j browser: http://localhost:7474
   ```

3. **Import Errors**:
   ```bash
   pip install -r enhanced_requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **NLTK Data Missing**:
   ```python
   import nltk
   nltk.download('all')
   ```

### Performance Optimization:
- **GPU Memory**: Monitor with `nvidia-smi`
- **Batch Sizes**: Adjust in code based on GPU memory
- **ES Heap**: Increase for large datasets

## 📊 System Architecture

```
📄 Documents (data_large/)
       ⬇️
🔧 Enhanced Chapter Extractor (Dynamic concept extraction)
       ⬇️  
🗄️ Elasticsearch (gpu_chapter_knowledge_v1)
       ⬇️
🧠 Knowledge Graph Builder (Concept clustering + GPU)
       ⬇️
🕸️ Neo4j (Enhanced graph with relations)
       ⬇️
🎮 Query System + 📊 Visualizations
```

## 🔄 Concept Extraction Process

### Dynamic Extraction (NOT Hardcoded):
1. **Explicit Objectives**: Scans for "learn", "understand", "implement"
2. **Key Terms**: Uses POS tagging and TF-IDF scoring
3. **Technical Concepts**: Identifies domain-specific terms
4. **Section Concepts**: Extracts from headers and structure
5. **Question Concepts**: Finds Q&A patterns and examples

### Quality Control:
- **Deduplication**: Removes similar concepts
- **Confidence Scoring**: Ranks concepts by reliability
- **Minimum Threshold**: Ensures at least 10 quality concepts
- **Maximum Limit**: Caps at 30 to prevent overload

## 🎁 Bonus Features

### Enhanced Elasticsearch Relations:
```bash
python elasticsearch_enhanced_relations.py
```
- Advanced NLP pipelines for relation extraction
- Named entity recognition with spaCy + NLTK
- Semantic similarity using vector search
- Complex nested queries for prerequisites

### Learning Analytics:
- Progress tracking across concepts
- Difficulty progression analysis
- Personalized learning recommendations
- Knowledge gap identification

## 🚀 Next Steps After Testing

1. **Explore Interactive Systems**:
   - Query system for concept exploration
   - Visualization system for insights
   
2. **Customize for Your Domain**:
   - Modify concept extraction patterns
   - Add domain-specific relation types
   - Customize difficulty assessment

3. **Scale Up**:
   - Process larger document collections
   - Add more sophisticated clustering
   - Integrate with learning management systems

## 💡 Pro Tips

1. **Start Small**: Test with 2-3 documents first
2. **Monitor Resources**: Watch GPU/CPU usage during processing
3. **Check Logs**: Review `enhanced_system.log` for detailed info
4. **Explore Interactively**: Use the query system to understand extracted concepts
5. **Visualize Results**: Generate the visualization report to see concept relationships

---

**Ready to test? Run `python KNOWLEDGE_GRAPH_TESTER.py` and watch the magic happen! 🎉**
