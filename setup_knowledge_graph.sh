#!/bin/bash
# setup_knowledge_graph.sh - Setup script for Knowledge Graph functionality

echo "🎓 Setting up Knowledge Graph for Subject Matter Expert RAG System"
echo "=================================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"

# Check if Elasticsearch is running
if curl -s http://localhost:9200 > /dev/null; then
    echo "✅ Elasticsearch is running"
else
    echo "❌ Elasticsearch is not running. Starting Elasticsearch..."
    docker-compose -f docker-compose-elasticsearch.yml up -d
    echo "⏳ Waiting for Elasticsearch to start..."
    sleep 30
fi

# Start Neo4j
echo "🚀 Starting Neo4j for Knowledge Graph..."
docker-compose -f docker-compose-neo4j.yml up -d

echo "⏳ Waiting for Neo4j to start..."
sleep 30

# Check Neo4j connection
MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker exec neo4j-rag-kg cypher-shell -u neo4j -p knowledge123 "RETURN 1" > /dev/null 2>&1; then
        echo "✅ Neo4j is running and accessible"
        break
    else
        echo "⏳ Waiting for Neo4j... (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
        sleep 10
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Failed to connect to Neo4j after $MAX_RETRIES attempts"
    echo "Please check the logs: docker-compose -f docker-compose-neo4j.yml logs"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing additional Python dependencies for Knowledge Graph..."
pip install neo4j networkx matplotlib seaborn plotly

# Check if RAG system is indexed
echo "🔍 Checking if RAG system is indexed..."
if python -c "
from elasticsearch import Elasticsearch
es = Elasticsearch(['http://localhost:9200'])
try:
    count = es.count(index='advanced_docs_elasticsearch_v2')['count']
    print(f'Found {count} documents in RAG index')
    exit(0 if count > 0 else 1)
except:
    exit(1)
" 2>/dev/null; then
    echo "✅ RAG system is indexed"
else
    echo "❌ RAG system is not indexed. Please run:"
    echo "   python SME_1_build_elasticsearch_database.py"
    echo "   Then run this setup script again."
    exit 1
fi

echo ""
echo "🎉 Knowledge Graph setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Build chapter database & knowledge graph:"
echo "   python KG_ENHANCED_1_build_chapter_database_gpu.py"
echo "   python KG_ENHANCED_2_build_knowledge_graph_gpu.py"
echo ""
echo "2. Query the enhanced knowledge graph:"
echo "   python KG_ENHANCED_3_query_knowledge_graph_gpu.py"
echo ""
echo "3. Generate 3D visualizations:"
echo "   python KG_ENHANCED_4_visualize_knowledge_graph_gpu.py"
echo ""
echo "4. Or run everything automatically:"
echo "   python KG_ENHANCED_COMPLETE_RUNNER.py"
echo ""
echo "🌐 Access points:"
echo "   • Neo4j Browser: http://localhost:7474 (neo4j/knowledge123)"
echo "   • Elasticsearch: http://localhost:9200"
echo ""
echo "📚 Documentation:"
echo "   • See README.md for complete system overview"
echo "   • Enhanced_Knowledge_Graph_Setup_and_Testing.ipynb for interactive setup"
