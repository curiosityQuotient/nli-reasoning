# nli-reasoning

An investigation into using NLI with LLMs

## Project Structure

This project follows the modular structure outlined in AGENTS.md:

```
├── src/               # Core code that set the logic
│   ├── data/          # Data processing utilities
│   ├── models/        # Model configurations and utilities
│   ├── utils/         # General utilities
│   └── main.py        # Main training orchestration
├── tests/             # Unit and integration tests
├── AGENTS.md          # Agent constraints and guidelines
├── pyproject.toml     # Dependency and tool configurations
└── README.md          # This file
```

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Usage

```python
from src.main import main

# Run with default configuration
main()

# Run with custom configuration
config = {
    "wandb_api_key": "your-api-key"
}
main(config)
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
ruff format .
```