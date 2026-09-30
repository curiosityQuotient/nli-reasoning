#!/usr/bin/env bash
# One-shot Kaggle setup for this repository.
#
# Run this INSTEAD of any manual `pip install google-tunix[prod]` cell:
#
#   !bash /kaggle/working/nli-reasoning/scripts/kaggle_setup.sh
#
# Two things this script exists to handle:
#
# 1. Kaggle's image ships jax 0.7.x, which Tunix refuses. Installing
#    google-tunix on its own resolves the broken jax 0.11.2 / flax 0.12.9
#    pair (jax 0.11.2 removed jax.experimental.hijax.HiPrimitive, which
#    flax's NNX imports). We pin jax 0.10.x + flax 0.12.x instead.
#
# 2. The image also ships a jax_cuda12_plugin built against its stock
#    jaxlib. Upgrading jaxlib without replacing that plugin leaves a PJRT
#    plugin whose API version predates the framework: it still enumerates
#    devices, so nothing looks wrong, but the first real computation dies
#    with "Unexpected PJRT_FFI_UserData_Add_Args size: expected 48, got
#    40". We drop the stale plugin and install one matching our jaxlib.

set -euo pipefail

REPO_DIR="${1:-/kaggle/working/nli-reasoning}"

if [ ! -f "${REPO_DIR}/pyproject.toml" ]; then
    echo "ERROR: repository not found at ${REPO_DIR}" >&2
    echo "Usage: bash scripts/kaggle_setup.sh [repo_dir]" >&2
    exit 1
fi

# The jax window is narrow and must match pyproject.toml: jax 0.11 removed
# HiPrimitive, and flax 0.12.9 (the only flax that tolerates it) requires
# jax>=0.11, so the 0.11.x line has no working flax.
JAX_SPEC=">=0.10.2,<0.11"

# Which accelerator is actually attached. Set NLI_ACCELERATOR=tpu|gpu|cpu
# to override detection.
detect_accelerator() {
    if [ -n "${NLI_ACCELERATOR:-}" ]; then
        echo "${NLI_ACCELERATOR}"
        return
    fi
    # TPU sessions expose libtpu character devices; GPU sessions expose
    # NVIDIA devices. Check TPU first: a TPU VM can also carry a GPU.
    if ls /dev/accel* >/dev/null 2>&1; then
        echo tpu
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo gpu
    else
        echo cpu
    fi
}

ACCELERATOR="$(detect_accelerator)"
echo "Detected accelerator: ${ACCELERATOR}"

# Drop the base image's PJRT plugins. On the GPU path the matching plugin is
# reinstalled below; on the TPU path removing it is the whole point, since a
# stale plugin that fails to load still poisons every later dispatch.
echo "Removing stale JAX plugins from the base image..."
pip uninstall -y jax-cuda12-plugin jax-cuda13-plugin >/dev/null 2>&1 || true

case "${ACCELERATOR}" in
    tpu)
        echo "Installing ${REPO_DIR} with pinned TPU-safe jax (jax[tpu])..."
        pip install -e "${REPO_DIR}" "jax[tpu]${JAX_SPEC}"
        ;;
    gpu)
        echo "Installing ${REPO_DIR} with pinned jax + matching CUDA plugin..."
        pip install -e "${REPO_DIR}" "jax[cuda12]${JAX_SPEC}"
        ;;
    *)
        echo "Installing ${REPO_DIR} (CPU; set NLI_ACCELERATOR=gpu or tpu to override)..."
        pip install -e "${REPO_DIR}" "jax${JAX_SPEC}"
        ;;
esac

echo "Verifying environment..."
NLI_EXPECTED_ACCELERATOR="${ACCELERATOR}" python - <<'PYEOF'
import os

import jax
import jax.numpy as jnp

import flax

jax_version = tuple(int(p) for p in jax.__version__.split(".")[:2])
if jax_version >= (0, 11):
    raise SystemExit(
        f"jax {jax.__version__} is too new for flax {flax.__version__}: "
        "jax 0.11.2 removed jax.experimental.hijax.HiPrimitive. "
        "Reinstall with: pip install -e <repo> 'jax[tpu]>=0.10.2,<0.11'"
    )

devices = jax.local_devices()
print(f"jax {jax.__version__}, flax {flax.__version__}")
print(f"default backend: {jax.default_backend()}")
print(f"devices ({len(devices)}): {devices}")

# Import the full training stack the way a run will.
import src.main  # noqa: E402,F401
from qwix import LoraProvider  # noqa: E402,F401
from tunix.models.gemma import model as gemma_model  # noqa: E402,F401
from tunix.rl import rl_cluster  # noqa: E402,F401
from tunix.rl.grpo.grpo_learner import GRPOLearner  # noqa: E402,F401

# Version and import checks pass even when the PJRT plugin underneath is
# mismatched, because a stale plugin still enumerates devices. Only actually
# dispatching a computation surfaces the ABI mismatch, so do that here rather
# than several minutes into a run.
side = 8
probe = jnp.ones((side, side), dtype=jnp.bfloat16)
# Every element of probe @ probe is `side`, and there are side**2 of them.
expected = float(side**3)
result = float((probe @ probe).sum())
if result != expected:
    raise SystemExit(f"Computation probe returned {result}, expected {expected}")

expected = os.environ.get("NLI_EXPECTED_ACCELERATOR", "")
kinds = {d.platform for d in devices}
if expected in ("tpu", "gpu") and expected not in kinds:
    raise SystemExit(
        f"Expected a {expected} device but JAX sees {sorted(kinds) or 'none'}. "
        f"Detected accelerator was '{expected}'. On Kaggle, set the session "
        "accelerator in the notebook options (Options > Accelerator), then "
        "re-run this script."
    )

print(f"Environment OK: jax {jax.__version__}, flax {flax.__version__}, {devices}")
PYEOF

echo "Setup complete."
