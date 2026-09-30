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

### Running on Kaggle (TPU or GPU)

The whole session is two notebook cells — and **delete any older cell that
installs `google-tunix[prod]` directly**, since it resolves a broken jax/flax
pair:

```bash
# Cell 1: install (idempotent, safe to re-run)
!rm -rf /kaggle/working/nli-reasoning
!git clone <your-repo-url> /kaggle/working/nli-reasoning
!bash /kaggle/working/nli-reasoning/scripts/kaggle_setup.sh
```

```bash
# Cell 2: train (fresh subprocess, streams logs)
!bash /kaggle/working/nli-reasoning/scripts/run_training.sh
```

Pick the accelerator under **Options > Accelerator** before starting the
session. The setup script detects what is actually attached and installs a
matching stack:

| Detected | Installed | Notes |
| --- | --- | --- |
| TPU (`/dev/accel*`) | `jax[tpu]` | pulls a matching `libtpu` |
| GPU (`nvidia-smi`) | `jax[cuda12]` | pulls a matching `jax-cuda12-plugin` |
| neither | `jax` (CPU) | smoke-test only; too slow to train |

Override detection with `NLI_ACCELERATOR`:

```bash
!NLI_ACCELERATOR=gpu bash /kaggle/working/nli-reasoning/scripts/kaggle_setup.sh
```

The script also **removes the base image's `jax_cuda12_plugin`**, which is
built against Kaggle's stock jaxlib 0.7.x. Left in place next to jaxlib
0.10.2 it still enumerates devices, so nothing looks wrong, but the first
real computation fails with
`Unexpected PJRT_FFI_UserData_Add_Args size: expected 48, got 40`.

Verification deliberately goes past imports: a mismatched PJRT plugin
imports and lists devices cleanly, so the script dispatches a small
computation and fails loudly if the result is wrong or the expected
accelerator is missing. It exits non-zero rather than letting a broken
environment reach a training run.

Launch the run in a **fresh process**. A notebook kernel that was already
running before the setup script will not see the freshly installed editable
package (PEP 660 path hooks register at interpreter startup). The simplest
reliable form is the second `!` cell above, which runs `python -m src.main`
from the repository root as a subprocess.

If you must drive it from a Python cell instead, put the repository root on
`sys.path` **first** — `src` is a regular package now, but a foreign `src`
directory earlier on the path still wins:

```python
import sys
sys.path.insert(0, "/kaggle/working/nli-reasoning")
from src.main import main
main()
```

Known-bad combinations (as of tunix 0.1.7 / flax 0.12.9 / perfetto 0.58):

- `jax 0.11.2` + `flax 0.12.9`: `AttributeError: module
  'jax.experimental.hijax' has no attribute 'HiPrimitive'` at `from flax
  import nnx`.
- `perfetto >= 0.56` + protobuf 5.x runtime (Kaggle stock):
  `google.protobuf.runtime_version.VersionError: ... gencode 6.31.1
  runtime 5.29.5` at `import tunix`.
- `jaxlib 0.10.2` + the image's stock `jax-cuda12-plugin 0.7.2`:
  `JaxRuntimeError: Unexpected PJRT_FFI_UserData_Add_Args size` on the
  first `jnp` call.

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