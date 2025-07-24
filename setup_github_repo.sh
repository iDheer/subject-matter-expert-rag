#!/bin/bash
# GitHub Repository Setup Script for Subject Matter Expert RAG System

echo "🚀 Setting up GitHub Repository for Subject Matter Expert RAG System"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're already in a git repository
if [ -d ".git" ]; then
    print_warning "Already in a Git repository. Skipping git init."
else
    print_info "Initializing Git repository..."
    git init
    print_status "Git repository initialized"
fi

# Add files to git (respecting .gitignore)
print_info "Adding files to Git..."
git add .
print_status "Files added to staging area"

# Initial commit
print_info "Creating initial commit..."
git commit -m "Initial commit: Subject Matter Expert RAG System with Knowledge Graphs

Features:
- SME System with Elasticsearch and AutoMerging Retriever
- GPU-Enhanced Knowledge Graph System with Neo4j
- GPU memory optimization for RTX 4050 6GB
- Quantized model support (INT4/INT8)
- Interactive querying and visualization
- Comprehensive documentation and setup scripts"

print_status "Initial commit created"

# Repository setup instructions
echo ""
print_info "📋 Next Steps - Repository Setup:"
echo "=================================="

echo ""
echo "1️⃣  Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: subject-matter-expert-rag"
echo "   - Description: Advanced RAG system with GPU-enhanced knowledge graphs for document analysis and Q&A"
echo "   - Make it Public (recommended) or Private"
echo "   - Don't initialize with README, .gitignore, or license (we already have them)"

echo ""
echo "2️⃣  Connect your local repository to GitHub:"
echo "   Replace 'YOUR_USERNAME' with your actual GitHub username:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/subject-matter-expert-rag.git"
echo "   git branch -M main"
echo "   git push -u origin main"

echo ""
echo "3️⃣  Alternative: Use GitHub CLI (if installed):"
echo "   Replace 'YOUR_USERNAME' with your GitHub username:"
echo ""
echo "   gh repo create YOUR_USERNAME/subject-matter-expert-rag --public --description \"Advanced RAG system with GPU-enhanced knowledge graphs\""
echo "   git remote add origin https://github.com/YOUR_USERNAME/subject-matter-expert-rag.git"
echo "   git push -u origin main"

echo ""
echo "4️⃣  Repository Topics (add these on GitHub):"
echo "   - artificial-intelligence"
echo "   - rag"
echo "   - knowledge-graph"
echo "   - gpu-acceleration"
echo "   - elasticsearch"
echo "   - neo4j"
echo "   - llm"
echo "   - python"
echo "   - cuda"
echo "   - document-analysis"

echo ""
print_info "📝 Repository Description Template:"
echo "Advanced Retrieval-Augmented Generation (RAG) system combining Elasticsearch with GPU-accelerated Knowledge Graphs. Features automatic document processing, concept extraction, interactive querying, and memory-optimized deployment for consumer GPUs."

echo ""
print_status "Repository preparation complete!"
print_info "Your repository is ready to be pushed to GitHub."

# Additional git configuration suggestions
echo ""
echo "💡 Optional Git Configuration:"
echo "=============================="
echo "git config user.name \"Your Name\""
echo "git config user.email \"your.email@example.com\""
echo "git config init.defaultBranch main"

# Show current status
echo ""
print_info "📊 Current Repository Status:"
echo "==============================="
git status --short
echo ""
echo "Files ready to push: $(git ls-files | wc -l) files"
echo "Repository size: $(du -sh .git | cut -f1) (Git data)"
echo "Working directory size: $(du -sh --exclude=.git . | cut -f1)"

echo ""
print_status "Setup script completed successfully! 🎉"
