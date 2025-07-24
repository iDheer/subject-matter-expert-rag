# setup_knowledge_graph.ps1 - PowerShell setup script for Windows

Write-Host "🎓 Setting up Knowledge Graph for Subject Matter Expert RAG System" -ForegroundColor Cyan
Write-Host "=================================================================="

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if Elasticsearch is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9200" -UseBasicParsing -TimeoutSec 5
    Write-Host "✅ Elasticsearch is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ Elasticsearch is not running. Starting Elasticsearch..." -ForegroundColor Yellow
    docker-compose -f docker-compose-elasticsearch.yml up -d
    Write-Host "⏳ Waiting for Elasticsearch to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
}

# Start Neo4j
Write-Host "🚀 Starting Neo4j for Knowledge Graph..." -ForegroundColor Cyan
docker-compose -f docker-compose-neo4j.yml up -d

Write-Host "⏳ Waiting for Neo4j to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check Neo4j connection
$maxRetries = 10
$retryCount = 0

while ($retryCount -lt $maxRetries) {
    try {
        docker exec neo4j-rag-kg cypher-shell -u neo4j -p knowledge123 "RETURN 1" | Out-Null
        Write-Host "✅ Neo4j is running and accessible" -ForegroundColor Green
        break
    }
    catch {
        $retryCount++
        Write-Host "⏳ Waiting for Neo4j... (attempt $retryCount/$maxRetries)" -ForegroundColor Yellow
        Start-Sleep -Seconds 10
    }
}

if ($retryCount -eq $maxRetries) {
    Write-Host "❌ Failed to connect to Neo4j after $maxRetries attempts" -ForegroundColor Red
    Write-Host "Please check the logs: docker-compose -f docker-compose-neo4j.yml logs" -ForegroundColor Red
    exit 1
}

# Install Python dependencies
Write-Host "📦 Installing additional Python dependencies for Knowledge Graph..." -ForegroundColor Cyan
pip install neo4j networkx matplotlib seaborn plotly

# Check if RAG system is indexed
Write-Host "🔍 Checking if RAG system is indexed..." -ForegroundColor Cyan
$pythonCheck = @"
from elasticsearch import Elasticsearch
es = Elasticsearch(['http://localhost:9200'])
try:
    count = es.count(index='advanced_docs_elasticsearch_v2')['count']
    print(f'Found {count} documents in RAG index')
    exit(0 if count > 0 else 1)
except:
    exit(1)
"@

$tempFile = [System.IO.Path]::GetTempFileName() + ".py"
$pythonCheck | Out-File -FilePath $tempFile -Encoding UTF8

try {
    python $tempFile
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ RAG system is indexed" -ForegroundColor Green
    } else {
        throw "No documents found"
    }
}
catch {
    Write-Host "❌ RAG system is not indexed. Please run:" -ForegroundColor Red
    Write-Host "   python SME_1_build_elasticsearch_database.py" -ForegroundColor Yellow
    Write-Host "   Then run this setup script again." -ForegroundColor Yellow
    Remove-Item $tempFile -ErrorAction SilentlyContinue
    exit 1
}
finally {
    Remove-Item $tempFile -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "🎉 Knowledge Graph setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Build the enhanced knowledge graph system:" -ForegroundColor White
Write-Host "   python KG_ENHANCED_COMPLETE_RUNNER.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Or run step by step:" -ForegroundColor White
Write-Host "   python KG_ENHANCED_1_build_chapter_database_gpu.py" -ForegroundColor Yellow
Write-Host "   python KG_ENHANCED_2_build_knowledge_graph_gpu.py" -ForegroundColor Yellow
Write-Host "   python KG_ENHANCED_3_query_knowledge_graph_gpu.py" -ForegroundColor Yellow
Write-Host "   python KG_ENHANCED_4_visualize_knowledge_graph_gpu.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Test the system:" -ForegroundColor White
Write-Host "   python KG_SYSTEM_TESTER.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "🌐 Access points:" -ForegroundColor Cyan
Write-Host "   • Neo4j Browser: http://localhost:7474 (neo4j/knowledge123)" -ForegroundColor White
Write-Host "   • Elasticsearch: http://localhost:9200" -ForegroundColor White
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "   • See COMPREHENSIVE_MANUAL.md for detailed instructions" -ForegroundColor White
Write-Host "   • Knowledge Graph section covers all features" -ForegroundColor White
