# 🚀 Quick Start Guide

**Get your SME system running in under 15 minutes!**

## Prerequisites ✅
- Windows 10+, macOS 10.15+, or Ubuntu 18.04+
- 8GB RAM minimum (16GB recommended)  
- 5GB free disk space
- Internet connection

## Step 1: Install Dependencies
```bash
# Install Python 3.11+ from python.org
# Install Docker Desktop from docker.com

# Verify installations
python --version
docker --version
```

## Step 2: Get the System
```bash
# Navigate to your SME directory
cd subject-matter-expert-rag

# Install Python packages
pip install -r requirements.txt

# For enhanced features (optional)
pip install -r enhanced_requirements.txt
```

## Step 3: Start Services
```bash
# Start Elasticsearch database
docker-compose -f docker-compose-elasticsearch.yml up -d

# Wait 30 seconds, then verify
curl http://localhost:9200

# Start Ollama AI service
ollama serve

# In another terminal, install AI model
ollama pull qwen3:4b-q4_0
```

## Step 4: Add Your Documents
```bash
# Create data directory
mkdir data_large

# Copy your files to data_large/
# Supported: PDF, DOCX, TXT, MD files
```

## Step 5: Build Database
```bash
# Process your documents (5-15 minutes)
python SME_1_build_elasticsearch_database.py

# Should see: "✅ Database built successfully!"
```

## Step 6: Start Asking Questions! 🎉
```bash
# Launch the interactive system
python SME_2_query_elasticsearch_system.py

# Try questions like:
# "What is our vacation policy?"
# "How do I reset a password?"
# "Explain the security procedures"
```

## Troubleshooting 🔧
- **System slow?** → See hardware optimization in SME_SETUP_MANUAL.md
- **Connection errors?** → Run `python check_system_health.py`
- **Need help?** → Check the comprehensive SME_SETUP_MANUAL.md

## Memory Features 🧠
- **Conversation memory**: Remembers context across questions
- **Commands**: `/memory`, `/clear`, `/status`, `/help`
- **Smart summarization**: Keeps relevant history

---
**🎯 That's it!** Your SME system is ready. For advanced features, scaling, and production deployment, see the complete SME_SETUP_MANUAL.md guide.
