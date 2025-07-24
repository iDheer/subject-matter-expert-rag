# PowerShell script for GitHub Repository Setup
# Subject Matter Expert RAG System

Write-Host "Setting up GitHub Repository for Subject Matter Expert RAG System" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# Function to print colored output
function Write-Success {
    param($Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param($Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Info {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Success "Git is installed: $gitVersion"
} catch {
    Write-Error "Git is not installed. Please install Git first."
    Write-Host "Download from: https://git-scm.com/download/windows" -ForegroundColor Yellow
    exit 1
}

# Check if we're already in a git repository
if (Test-Path ".git") {
    Write-Warning "Already in a Git repository. Skipping git init."
} else {
    Write-Info "Initializing Git repository..."
    git init
    Write-Success "Git repository initialized"
}

# Add files to git (respecting .gitignore)
Write-Info "Adding files to Git..."
git add .
Write-Success "Files added to staging area"

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    # Initial commit
    Write-Info "Creating initial commit..."
    $commitMessage = "Initial commit: Subject Matter Expert RAG System

Features include SME System with Elasticsearch, GPU-Enhanced Knowledge Graph System, 
GPU memory optimization for RTX 4050 6GB, quantized model support, 
interactive querying and visualization, plus comprehensive documentation."
    
    git commit -m $commitMessage
    Write-Success "Initial commit created"
} else {
    Write-Warning "No changes to commit"
}

# Repository setup instructions
Write-Host ""
Write-Info "Next Steps - Repository Setup:"
Write-Host "==============================="

Write-Host ""
Write-Host "1. Create a new repository on GitHub:" -ForegroundColor White
Write-Host "   - Go to https://github.com/new" -ForegroundColor Gray
Write-Host "   - Repository name: subject-matter-expert-rag" -ForegroundColor Gray
Write-Host "   - Description: Advanced RAG system with GPU-enhanced knowledge graphs" -ForegroundColor Gray
Write-Host "   - Make it Public (recommended) or Private" -ForegroundColor Gray
Write-Host "   - Don't initialize with README, .gitignore, or license" -ForegroundColor Gray

Write-Host ""
Write-Host "2. Connect your local repository to GitHub:" -ForegroundColor White
Write-Host "   Replace 'YOUR_USERNAME' with your GitHub username:" -ForegroundColor Yellow
Write-Host ""
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/subject-matter-expert-rag.git" -ForegroundColor Cyan
Write-Host "   git branch -M main" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan

Write-Host ""
Write-Host "3. Alternative: Use GitHub CLI (if installed):" -ForegroundColor White
Write-Host ""
Write-Host "   gh repo create YOUR_USERNAME/subject-matter-expert-rag --public" -ForegroundColor Cyan
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/subject-matter-expert-rag.git" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan

Write-Host ""
Write-Host "4. Repository Topics (add these on GitHub):" -ForegroundColor White
$topics = @(
    "artificial-intelligence",
    "rag", 
    "knowledge-graph",
    "gpu-acceleration",
    "elasticsearch",
    "neo4j",
    "llm",
    "python",
    "cuda",
    "document-analysis"
)
foreach ($topic in $topics) {
    Write-Host "   - $topic" -ForegroundColor Gray
}

Write-Host ""
Write-Info "Repository Description Template:"
Write-Host "Advanced Retrieval-Augmented Generation (RAG) system combining Elasticsearch with GPU-accelerated Knowledge Graphs. Features automatic document processing, concept extraction, interactive querying, and memory-optimized deployment for consumer GPUs." -ForegroundColor Gray

Write-Host ""
Write-Success "Repository preparation complete!"
Write-Info "Your repository is ready to be pushed to GitHub."

# Show current status
Write-Host ""
Write-Info "Current Repository Status:"
Write-Host "=========================="
git status --short

$fileCount = (git ls-files | Measure-Object).Count
Write-Host ""
Write-Host "Files ready to push: $fileCount files" -ForegroundColor White

Write-Host ""
Write-Success "Setup script completed successfully!"

# Pause to let user read the output
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
