# 🚀 GitHub Repository Setup Commands

This document provides step-by-step commands to create a GitHub repository for the Subject Matter Expert RAG System.

## 📋 Prerequisites

1. **Git installed** on your system
2. **GitHub account** created
3. **GitHub CLI** (optional but recommended)

## 🛠️ Method 1: Using GitHub Web Interface (Recommended for beginners)

### Step 1: Prepare Local Repository

```bash
# Navigate to your project directory
cd c:\Users\inesh\Desktop\subject-matter-expert-rag

# Initialize git repository (if not already done)
git init

# Add all files to staging area
git add .

# Create initial commit
git commit -m "Initial commit: Subject Matter Expert RAG System

Features:
- SME System with Elasticsearch and AutoMerging Retriever  
- GPU-Enhanced Knowledge Graph System with Neo4j
- GPU memory optimization for RTX 4050 6GB
- Quantized model support (INT4/INT8)
- Interactive querying and visualization
- Comprehensive documentation and setup scripts"

# Set main branch as default
git branch -M main
```

### Step 2: Create Repository on GitHub

1. Go to **https://github.com/new**
2. Fill in repository details:
   - **Repository name**: `subject-matter-expert-rag`
   - **Description**: `Advanced RAG system with GPU-enhanced knowledge graphs for document analysis and Q&A`
   - **Visibility**: Public (recommended) or Private
   - **Do NOT initialize** with README, .gitignore, or license (we already have them)
3. Click **"Create repository"**

### Step 3: Connect Local to Remote

```bash
# Add remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/subject-matter-expert-rag.git

# Push to GitHub
git push -u origin main
```

## 🚀 Method 2: Using GitHub CLI (Faster)

### Step 1: Install GitHub CLI

**Windows:**
```powershell
winget install GitHub.cli
# or
scoop install gh
# or download from https://cli.github.com/
```

**macOS:**
```bash
brew install gh
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

### Step 2: Authenticate with GitHub

```bash
# Login to GitHub CLI
gh auth login

# Follow the prompts:
# - Choose GitHub.com
# - Choose HTTPS or SSH (HTTPS is easier)
# - Authenticate via web browser
```

### Step 3: Create Repository

```bash
# Navigate to project directory
cd c:\Users\inesh\Desktop\subject-matter-expert-rag

# Initialize git if not already done
git init
git add .
git commit -m "Initial commit: Subject Matter Expert RAG System with Knowledge Graphs"

# Create repository and push (replace YOUR_USERNAME)
gh repo create YOUR_USERNAME/subject-matter-expert-rag --public --description "Advanced RAG system with GPU-enhanced knowledge graphs for document analysis and Q&A"

# Set remote and push
git remote add origin https://github.com/YOUR_USERNAME/subject-matter-expert-rag.git
git branch -M main
git push -u origin main
```

## 🏷️ Method 3: Using SSH (For advanced users)

### Step 1: Set up SSH Keys

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add SSH key to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key to clipboard
# Linux:
cat ~/.ssh/id_ed25519.pub | xclip -selection clipboard
# macOS:
cat ~/.ssh/id_ed25519.pub | pbcopy
# Windows (Git Bash):
cat ~/.ssh/id_ed25519.pub | clip
```

### Step 2: Add SSH Key to GitHub

1. Go to **GitHub → Settings → SSH and GPG keys**
2. Click **"New SSH key"**
3. Paste your public key
4. Click **"Add SSH key"**

### Step 3: Create Repository with SSH

```bash
# Create repository using SSH
git remote add origin git@github.com:YOUR_USERNAME/subject-matter-expert-rag.git
git branch -M main  
git push -u origin main
```

## 📊 PowerShell Automation Script (Windows)

Save and run this PowerShell script:

```powershell
# GitHub Repository Setup Script
param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubUsername,
    
    [Parameter(Mandatory=$false)]
    [string]$RepositoryName = "subject-matter-expert-rag",
    
    [Parameter(Mandatory=$false)]
    [switch]$Private
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up GitHub repository..." -ForegroundColor Cyan

# Check if git is available
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git is not installed or not in PATH"
}

# Initialize repository
if (!(Test-Path ".git")) {
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
}

# Add files and commit
git add .
git commit -m "Initial commit: Subject Matter Expert RAG System with Knowledge Graphs"
git branch -M main

Write-Host "✅ Local repository prepared" -ForegroundColor Green

# Check if GitHub CLI is available
if (Get-Command gh -ErrorAction SilentlyContinue) {
    # Use GitHub CLI
    $visibility = if ($Private) { "--private" } else { "--public" }
    gh repo create "$GitHubUsername/$RepositoryName" $visibility --description "Advanced RAG system with GPU-enhanced knowledge graphs"
    git remote add origin "https://github.com/$GitHubUsername/$RepositoryName.git"
    git push -u origin main
    
    Write-Host "✅ Repository created and pushed using GitHub CLI" -ForegroundColor Green
} else {
    # Manual setup
    Write-Host "⚠️  GitHub CLI not found. Manual setup required:" -ForegroundColor Yellow
    Write-Host "1. Go to https://github.com/new" -ForegroundColor White
    Write-Host "2. Create repository: $RepositoryName" -ForegroundColor White
    Write-Host "3. Run these commands:" -ForegroundColor White
    Write-Host "   git remote add origin https://github.com/$GitHubUsername/$RepositoryName.git" -ForegroundColor Cyan
    Write-Host "   git push -u origin main" -ForegroundColor Cyan
}

Write-Host "🎉 Repository setup complete!" -ForegroundColor Green
```

Run it with:
```powershell
.\setup_repo.ps1 -GitHubUsername "YOUR_USERNAME"
```

## 🔧 Bash Automation Script (Linux/macOS)

Save and run this bash script:

```bash
#!/bin/bash
# GitHub Repository Setup Script

set -e

GITHUB_USERNAME="$1"
REPO_NAME="${2:-subject-matter-expert-rag}"
PRIVATE="${3:-false}"

if [ -z "$GITHUB_USERNAME" ]; then
    echo "Usage: $0 <github_username> [repo_name] [private]"
    echo "Example: $0 myusername subject-matter-expert-rag false"
    exit 1
fi

echo "🚀 Setting up GitHub repository..."

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed"
    exit 1
fi

# Initialize repository
if [ ! -d ".git" ]; then
    git init
    echo "✅ Git repository initialized"
fi

# Add files and commit
git add .
git commit -m "Initial commit: Subject Matter Expert RAG System with Knowledge Graphs"
git branch -M main

echo "✅ Local repository prepared"

# Check if GitHub CLI is available
if command -v gh &> /dev/null; then
    # Use GitHub CLI
    VISIBILITY="--public"
    if [ "$PRIVATE" = "true" ]; then
        VISIBILITY="--private"
    fi
    
    gh repo create "$GITHUB_USERNAME/$REPO_NAME" $VISIBILITY --description "Advanced RAG system with GPU-enhanced knowledge graphs"
    git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    git push -u origin main
    
    echo "✅ Repository created and pushed using GitHub CLI"
else
    # Manual setup
    echo "⚠️  GitHub CLI not found. Manual setup required:"
    echo "1. Go to https://github.com/new"
    echo "2. Create repository: $REPO_NAME"
    echo "3. Run these commands:"
    echo "   git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    echo "   git push -u origin main"
fi

echo "🎉 Repository setup complete!"
```

Run it with:
```bash
chmod +x setup_repo.sh
./setup_repo.sh YOUR_USERNAME
```

## 🏷️ Repository Configuration

### Add Topics (Labels) to Your Repository

After creating the repository, add these topics on GitHub:

1. Go to your repository page
2. Click the ⚙️ **Settings** gear next to "About"
3. Add these topics:
   - `artificial-intelligence`
   - `rag`
   - `knowledge-graph`
   - `gpu-acceleration`
   - `elasticsearch`
   - `neo4j`
   - `llm`
   - `python`
   - `cuda`
   - `document-analysis`
   - `retrieval-augmented-generation`
   - `quantization`

### Repository Settings

**Recommended settings:**
- ✅ **Issues** enabled
- ✅ **Discussions** enabled  
- ✅ **Wiki** enabled
- ✅ **Sponsorships** enabled (if you want)
- ✅ **Allow merge commits**
- ✅ **Allow squash merging**
- ❌ **Allow rebase merging** (optional)

### Branch Protection Rules

For main branch:
- ✅ **Require pull request reviews**
- ✅ **Require status checks**
- ✅ **Require branches to be up to date**
- ✅ **Include administrators**

## 📝 Post-Setup Checklist

After creating your repository:

- [ ] Repository is public/private as intended
- [ ] Topics/labels are added
- [ ] Repository description is set
- [ ] README.md displays correctly
- [ ] .gitignore is working (no unwanted files)
- [ ] All necessary files are included
- [ ] Links in documentation work
- [ ] Issues and Discussions are enabled
- [ ] Repository settings are configured

## 🔄 Maintaining Your Repository

### Regular Updates

```bash
# Keep your fork updated (if forked from another repo)
git remote add upstream https://github.com/original/repo.git
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# Regular maintenance
git fetch origin
git rebase origin/main
git push --force-with-lease origin feature-branch
```

### Release Management

```bash
# Create a release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Or use GitHub CLI
gh release create v1.0.0 --title "v1.0.0" --notes "Release notes here"
```

## 🎉 Success!

Your repository should now be live at:
`https://github.com/YOUR_USERNAME/subject-matter-expert-rag`

**Next steps:**
1. Share your repository link
2. Create your first issue or discussion
3. Invite collaborators
4. Set up continuous integration (GitHub Actions)
5. Create releases and documentation

Happy coding! 🚀
