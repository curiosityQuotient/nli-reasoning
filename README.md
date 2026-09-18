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

Requires Python 3.11+ (Kaggle TPU runs use 3.11).

```bash
# Using uv (recommended)
uv venv --python 3.11
uv pip install -e '.[dev]'

# Or using pip
pip install -e '.[dev]'
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

### Running on Kaggle (TPU)

Kaggle's TPU image ships an older JAX (0.7.x) that Tunix refuses, so a fresh
install must upgrade it. Version caps in `pyproject.toml` keep that upgrade
safe — do **not** install `google-tunix[prod]` separately, and let one pip
invocation resolve everything:

```bash
pip install -e /kaggle/working/nli-reasoning "jax[tpu]>=0.10.2,<0.11"
```

The `jax[tpu]` constraint also pins a `libtpu` build matching jax 0.10.x.
Known-bad combinations (as of tunix 0.1.7 / flax 0.12.9):

- `jax 0.11.2` + `flax 0.12.9`: `AttributeError: module
  'jax.experimental.hijax' has no attribute 'HiPrimitive'` at `from flax
  import nnx`.

Quick sanity check before launching a run:

```bash
python -c "import src.main; print('imports ok')"
```

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