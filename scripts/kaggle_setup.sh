#!/usr/bin/env bash
# One-shot Kaggle TPU setup for this repository.
#
# Run this INSTEAD of any manual `pip install google-tunix[prod]` cell:
#
#   !bash /kaggle/working/nli-reasoning/scripts/kaggle_setup.sh
#
# Why: Kaggle's image ships jax 0.7.x, which Tunix refuses. Installing
# google-tunix on its own resolves the broken jax 0.11.2 / flax 0.12.9
# pair (jax 0.11.2 removed jax.experimental.hijax.HiPrimitive, which
# flax's NNX imports). Resolving the repository package and the jax[tpu]
# constraint together installs a coherent, tested stack:
# jax 0.10.x + flax 0.12.x + libtpu 0.0.42 + tunix 0.1.7 + qwix 0.1.8.

set -euo pipefail

REPO_DIR="${1:-/kaggle/working/nli-reasoning}"

if [ ! -f "${REPO_DIR}/pyproject.toml" ]; then
    echo "ERROR: repository not found at ${REPO_DIR}" >&2
    echo "Usage: bash scripts/kaggle_setup.sh [repo_dir]" >&2
    exit 1
fi

echo "Installing ${REPO_DIR} with pinned TPU-safe jax..."
pip install -e "${REPO_DIR}" "jax[tpu]>=0.10.2,<0.11"

echo "Verifying environment..."
python - <<'PYEOF'
import jax
import flax

jax_version = tuple(int(p) for p in jax.__version__.split(".")[:2])
if jax_version >= (0, 11):
    raise SystemExit(
        f"jax {jax.__version__} is too new for flax {flax.__version__}: "
        "jax 0.11.2 removed jax.experimental.hijax.HiPrimitive. "
        "Reinstall with: pip install -e <repo> 'jax[tpu]>=0.10.2,<0.11'"
    )

# Import the full training stack the way a run will.
import src.main  # noqa: F401
from tunix.rl import rl_cluster  # noqa: F401
from tunix.rl.grpo.grpo_learner import GRPOLearner  # noqa: F401
from tunix.models.gemma import model as gemma_model  # noqa: F401
from qwix import LoraProvider  # noqa: F401

print(f"Environment OK: jax {jax.__version__}, flax {flax.__version__}")
PYEOF

echo "Setup complete."
