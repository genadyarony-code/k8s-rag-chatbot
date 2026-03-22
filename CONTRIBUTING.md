# Contributing to K8s RAG Chatbot

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates.

When you create a bug report, include:
- **Description** - Clear and descriptive title
- **Steps to Reproduce** - Detailed steps
- **Expected Behavior** - What you expected to happen
- **Actual Behavior** - What actually happened
- **Environment** - OS, Python version, Docker version
- **Logs** - Relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:
- **Use Case** - What problem does this solve?
- **Proposed Solution** - How would you implement it?
- **Alternatives** - What other approaches did you consider?

### Pull Requests

1. **Fork the repo** and create your branch from `master`
2. **Make your changes** with clear, descriptive commits
3. **Add tests** if you're adding functionality
4. **Update documentation** if needed
5. **Ensure tests pass** - Run `pytest tests/`
6. **Follow the code style** - This project uses Black for formatting
7. **Write a clear PR description** explaining what and why

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/k8s-rag-chatbot.git
cd k8s-rag-chatbot

# Create a virtual environment
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

## Code Style

- **Python**: Follow [PEP 8](https://pep8.org/)
- **Formatting**: Use [Black](https://black.readthedocs.io/) with default settings
- **Type Hints**: Use type hints wherever possible
- **Docstrings**: Use Google-style docstrings

```python
def example_function(param: str) -> dict:
    """One-line summary.
    
    More detailed explanation if needed.
    
    Args:
        param: Description of param
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param is invalid
    """
    pass
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_retrieval.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names that explain what is being tested

Example:

```python
def test_bm25_fallback_when_chroma_disabled():
    """BM25 should be used when ChromaDB is disabled via feature flag."""
    # Arrange
    settings.FF_USE_CHROMA = False
    
    # Act
    results = retrieve("kubernetes networking")
    
    # Assert
    assert results.retrieval_method == "bm25"
    assert len(results.chunks) > 0
```

## Commit Messages

Use clear, descriptive commit messages:

```
# Good
Add BM25 fallback for keyword queries
Fix session memory leak in multi-turn conversations
Update README with Docker setup instructions

# Bad
fix stuff
update
wip
```

Format:
```
<type>: <subject>

<optional body>

<optional footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Project Structure

Understanding the codebase:

```
src/
├── config/          # Settings and feature flags
├── ingestion/       # Document loading and indexing
│   ├── loaders/     # PDF, HTML, text loaders
│   ├── preprocessor.py  # Chunking and metadata
│   └── indexer.py   # ChromaDB and BM25 indexing
├── agent/           # LangGraph agent
│   ├── graph.py     # Agent workflow definition
│   ├── nodes.py     # Retrieve and generate nodes
│   ├── memory.py    # Session memory management
│   └── prompts.py   # LLM prompt templates
├── api/             # FastAPI backend
│   └── main.py      # API endpoints and lifespan
└── ui/              # Streamlit frontend
    └── app.py       # UI components
```

## Feature Flags

When adding new functionality that might fail in production, wrap it in a feature flag:

```python
# In src/config/settings.py
FF_USE_NEW_FEATURE: bool = Field(
    default=True,
    description="Enable new experimental feature"
)

# In your code
if settings.FF_USE_NEW_FEATURE:
    result = new_experimental_function()
else:
    result = stable_fallback_function()
```

## Documentation

- Update `README.md` for user-facing changes
- Update `PROJECT_MAP.md` for architectural decisions
- Add inline comments for complex logic
- Update docstrings when changing function signatures

## Questions?

- Open an issue with the `question` label
- Check existing issues and documentation first

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project and community

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
