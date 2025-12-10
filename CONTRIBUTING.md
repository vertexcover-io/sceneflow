# Contributing to SceneFlow

Thank you for your interest in contributing to SceneFlow! This document provides guidelines and instructions for contributing.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vertexcover-io/sceneflow.git
cd sceneflow
```

### 2. Install uv (if not already installed)

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install Dependencies

```bash
# Install all dependencies including dev dependencies
uv sync

# Or install in development mode
uv pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks (recommended)

```bash
uv add --dev pre-commit
uv run pre-commit install
```

## Project Structure

```
sceneflow/
├── src/sceneflow/          # Main package code
│   ├── __init__.py         # Package exports
│   ├── cli.py              # CLI implementation
│   ├── speech_detector.py  # VAD-based speech detection
│   ├── ranker.py           # Main ranking orchestrator
│   ├── extractors.py       # Feature extraction
│   ├── scorer.py           # Multi-stage scoring
│   ├── config.py           # Configuration
│   ├── models.py           # Data models
│   ├── normalizer.py       # Normalization utilities
│   ├── quality_gating.py   # Quality penalties
│   └── stability_analyzer.py  # Temporal stability
├── tests/                  # Test files
├── README.md              # User documentation
├── CONTRIBUTING.md        # This file
└── pyproject.toml         # Package configuration
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add docstrings to functions and classes
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run the CLI locally
uv run sceneflow /path/to/test/video.mp4 --verbose

# Run tests (when available)
uv run pytest tests/

# Test the package can be built
uv run python -m build
```

### 4. Format and Lint

```bash
# Format code with black
uv run black src/sceneflow/

# Lint with ruff
uv run ruff check src/sceneflow/

# Type check with mypy (optional)
uv run mypy src/sceneflow/
```

### 5. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add support for custom VAD models"
# or
git commit -m "fix: handle edge case in eye openness calculation"
# or
git commit -m "docs: update README with new examples"
```

Commit message prefixes:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- Clear description of changes
- Reference to any related issues
- Screenshots/examples if applicable

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints where appropriate
- Line length: 100 characters (configured in pyproject.toml)
- Use `black` for formatting
- Use `ruff` for linting

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
    """
    pass
```

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

## Adding New Features

### Speech Detection Features

If adding new speech detection methods:

1. Add method to `SpeechDetector` class in `speech_detector.py`
2. Ensure it returns `(timestamp, confidence)` tuple
3. Add documentation and examples
4. Update tests

### Visual Analysis Features

If adding new visual metrics:

1. Add extraction logic to `extractors.py`
2. Add scoring logic to `scorer.py`
3. Update `FrameFeatures` and `FrameScore` models in `models.py`
4. Add weight parameter to `RankingConfig` in `config.py`
5. Update documentation

### CLI Features

If adding CLI options:

1. Add parameter to `main()` function in `cli.py`
2. Use `cyclopts.Parameter()` for help text
3. Update README.md with new option
4. Test with `--help` flag

## Testing

### Writing Tests

Create test files in `tests/` directory:

```python
# tests/test_speech_detector.py
import pytest
from sceneflow.speech_detector import SpeechDetector

def test_speech_detector_initialization():
    detector = SpeechDetector(model_size="tiny")
    assert detector.model is not None
    assert detector.vad_model is not None
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=sceneflow

# Run specific test file
uv run pytest tests/test_speech_detector.py
```

## Documentation

### Updating README

- Keep examples up to date
- Add new features to feature list
- Update CLI options section if needed
- Ensure all code examples work

### Updating CLAUDE.md

- Document internal implementation details
- Update architecture descriptions
- Add notes about design decisions

## Reporting Issues

### Bug Reports

Include:

- Python version (`python --version`)
- SceneFlow version (`sceneflow --version`)
- Operating system
- Minimal code to reproduce the bug
- Error message/traceback
- Expected vs actual behavior

### Feature Requests

Include:

- Clear description of the feature
- Use case / motivation
- Example of how it would work
- Any relevant research or references

## Questions?

- Check existing issues on GitHub
- Open a discussion on GitHub

## License

By contributing to SceneFlow, you agree that your contributions will be licensed under the MIT License.
