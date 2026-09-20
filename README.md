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
install must upgrade it. Run the one-shot bootstrap script from a notebook
cell — and **delete any older cell that installs `google-tunix[prod]`
directly**, since it resolves a broken jax/flax pair:

```bash
!bash /kaggle/working/nli-reasoning/scripts/kaggle_setup.sh
```

The script installs the repository together with the `jax[tpu]` constraint
(also pinning a matching `libtpu`) and verifies the full training stack
imports. Equivalent manual command:

```bash
pip install -e /kaggle/working/nli-reasoning "jax[tpu]>=0.10.2,<0.11"
```

Launch the run in a **fresh process**. A notebook kernel that was already
running before the setup script will not see the freshly installed editable
package. Either restart the kernel and use `from src.main import main`, or
simply run as a subprocess:

```bash
!cd /kaggle/working/nli-reasoning && python -m src.main
```

Known-bad combinations (as of tunix 0.1.7 / flax 0.12.9 / perfetto 0.58):

- `jax 0.11.2` + `flax 0.12.9`: `AttributeError: module
  'jax.experimental.hijax' has no attribute 'HiPrimitive'` at `from flax
  import nnx`.
- `perfetto >= 0.56` + protobuf 5.x runtime (Kaggle stock):
  `google.protobuf.runtime_version.VersionError: ... gencode 6.31.1
  runtime 5.29.5` at `import tunix`.

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