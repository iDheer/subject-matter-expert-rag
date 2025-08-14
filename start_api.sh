#!/bin/bash

echo "🚀 Starting SME API Server..."

# Check if required files exist
if [ ! -f "SME_2_query_elasticsearch_system.py" ]; then
    echo "❌ SME_2_query_elasticsearch_system.py not found!"
    exit 1
fi

if [ ! -d "elasticsearch_storage_v2" ]; then
    echo "❌ elasticsearch_storage_v2 directory not found!"
    echo "   Please run SME_1_build_elasticsearch_database.py first"
    exit 1
fi

# Check if Elasticsearch is running
echo "🔍 Checking Elasticsearch connection..."
if ! curl -s http://localhost:9200/_cluster/health >/dev/null; then
    echo "❌ Elasticsearch not running on localhost:9200"
    echo "   Please start Elasticsearch first:"
    echo "   docker-compose -f docker-compose-elasticsearch.yml up -d"
    exit 1
fi

# Check if Ollama is running
echo "🔍 Checking Ollama connection..."
if ! curl -s http://localhost:11434/api/tags >/dev/null; then
    echo "❌ Ollama not running on localhost:11434"
    echo "   Please start Ollama first"
    exit 1
fi

# Install API requirements
echo "📦 Installing API requirements..."
pip install -r requirements_api.txt

# Start the API server
echo "🎉 Starting FastAPI server..."
echo "   Access the API at: http://localhost:8000"
echo "   API docs at: http://localhost:8000/docs"
echo "   Press Ctrl+C to stop"
echo

python api_server.py