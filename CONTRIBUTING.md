# Contributing to Subject Matter Expert RAG System

Thank you for your interest in contributing to the Subject Matter Expert RAG System! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug Reports** - Help us identify and fix issues
- ✨ **Feature Requests** - Suggest new functionality
- 📝 **Documentation** - Improve our docs and guides
- 🧪 **Testing** - Add or improve test coverage
- 🚀 **Performance** - Optimize existing features
- 🎨 **UI/UX** - Enhance user experience
- 🔧 **Infrastructure** - Improve development workflow

### Getting Started

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/subject-matter-expert-rag.git
   cd subject-matter-expert-rag
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   
   # Install dependencies
   pip install -r enhanced_requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   # or
   git checkout -b docs/improvement-description
   ```

## 📋 Development Guidelines

### Code Style

- **Python**: Follow PEP 8 guidelines
- **Line Length**: Maximum 88 characters (Black formatter default)
- **Imports**: Use absolute imports when possible
- **Documentation**: Include docstrings for all functions and classes

```python
def extract_concepts(text: str, model: str = "all-mpnet-base-v2") -> List[Dict]:
    """
    Extract concepts from text using NLP models.
    
    Args:
        text: Input text to analyze
        model: Name of the sentence-transformer model to use
        
    Returns:
        List of dictionaries containing concept information
        
    Raises:
        ValueError: If text is empty or model is not found
    """
    # Implementation here
    pass
```

### Project Structure

```
subject-matter-expert-rag/
├── SME_*.py                    # SME system modules
├── KG_ENHANCED_*.py           # Knowledge Graph modules
├── tests/                     # Test files
├── docs/                      # Documentation
├── data_large/               # Sample data (gitignored)
├── docker-compose-*.yml      # Docker configurations
└── setup_*.py/sh/ps1        # Setup scripts
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Modules**: Short, lowercase names

### GPU Development Guidelines

When working with GPU-accelerated features:

```python
import torch

def setup_gpu():
    """Setup GPU with proper error handling"""
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()  # Clear cache
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    return device
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_sme_system.py

# Run with coverage
python -m pytest --cov=. tests/

# Test GPU functionality
python vram_analyzer.py
python check_ollama_gpu.py
```

### Writing Tests

Create test files in the `tests/` directory:

```python
# tests/test_knowledge_graph.py
import pytest
from KG_ENHANCED_3_query_knowledge_graph_gpu import GPUAcceleratedEnhancedChapterQuerySystem

class TestKnowledgeGraph:
    def test_concept_extraction(self):
        """Test concept extraction functionality"""
        # Test implementation
        pass
        
    def test_gpu_memory_optimization(self):
        """Test GPU memory management"""
        # Test implementation
        pass
```

## 📝 Documentation

### Updating Documentation

- Update relevant `.md` files for feature changes
- Include code examples in docstrings
- Update README.md if adding new features
- Add inline comments for complex logic

### Documentation Structure

```
docs/
├── api/              # API documentation
├── tutorials/        # Step-by-step guides  
├── examples/         # Usage examples
└── troubleshooting/  # Common issues and solutions
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   ```
   - OS: Windows 11 / Ubuntu 22.04 / macOS 13
   - Python Version: 3.11.5
   - GPU: RTX 4050 6GB / CPU only
   - CUDA Version: 12.1 (if applicable)
   ```

2. **Steps to Reproduce**
   - Detailed steps to reproduce the issue
   - Expected vs. actual behavior
   - Screenshots if applicable

3. **Error Messages**
   ```
   Full error traceback or log output
   ```

4. **Additional Context**
   - Configuration files used
   - Data characteristics (if relevant)
   - Recent changes made

## ✨ Feature Requests

For new features, please include:

1. **Use Case**: Describe the problem this feature would solve
2. **Proposed Solution**: Your idea for implementation
3. **Alternatives**: Other solutions you've considered
4. **Impact**: Who would benefit from this feature

## 🔀 Pull Request Process

1. **Update Documentation**
   - Update README.md if needed
   - Add docstrings to new functions
   - Update relevant guides

2. **Add Tests**
   - Include tests for new functionality
   - Ensure existing tests still pass
   - Test GPU and CPU code paths

3. **Check Code Quality**
   ```bash
   # Format code
   black .
   
   # Check style
   flake8 .
   
   # Type checking (if configured)
   mypy .
   ```

4. **Update Dependencies**
   - Update `requirements.txt` if needed
   - Test installation from scratch

5. **Write Good Commit Messages**
   ```
   feat: Add GPU memory optimization for RTX 4050
   
   - Implement INT4 quantization support
   - Add VRAM usage analysis
   - Update documentation with memory requirements
   
   Fixes #123
   ```

6. **Create Pull Request**
   - Use a descriptive title
   - Reference related issues
   - Include testing instructions
   - Add screenshots for UI changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] GPU testing completed (if applicable)
- [ ] Documentation updated

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No breaking changes (or explicitly noted)
```

## 🏷️ Issue Labels

We use these labels to categorize issues:

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `gpu` - GPU-related functionality
- `memory` - Memory optimization
- `performance` - Performance improvements
- `question` - Further information requested

## 💬 Communication

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and ideas
- **Pull Request Comments** - Code-specific discussions

## 📜 Code of Conduct

### Our Standards

- **Be Respectful** - Treat everyone with respect and kindness
- **Be Inclusive** - Welcome contributions from everyone
- **Be Constructive** - Provide helpful feedback
- **Be Patient** - Help newcomers learn and grow

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or inflammatory comments
- Personal attacks
- Publishing private information

## 🎉 Recognition

Contributors will be recognized in:

- README.md acknowledgments
- Release notes
- GitHub contributors page

## 📞 Getting Help

If you need help:

1. Check existing documentation
2. Search GitHub issues
3. Create a new issue with the `question` label
4. Join GitHub Discussions

Thank you for contributing to the Subject Matter Expert RAG System! 🚀
