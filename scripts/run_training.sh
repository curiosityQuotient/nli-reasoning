#!/usr/bin/env bash
# Launch two-stage training in a fresh Python process.
#
# Use this from a Kaggle notebook cell (note the leading !):
#
#   !bash /kaggle/working/nli-reasoning/scripts/run_training.sh
#
# Why a subprocess: a notebook kernel that was already running before
# `kaggle_setup.sh` installed the editable package will not see it (PEP 660
# path hooks register at interpreter startup). Running the training as a
# subprocess always picks up the freshly installed package, and logs stream
# to the cell output.

set -euo pipefail

REPO_DIR="${1:-/kaggle/working/nli-reasoning}"

if [ ! -f "${REPO_DIR}/src/main.py" ]; then
    echo "ERROR: repository not found at ${REPO_DIR}" >&2
    echo "Usage: bash scripts/run_training.sh [repo_dir]" >&2
    exit 1
fi

cd "${REPO_DIR}"
export PYTHONUNBUFFERED=1
exec python -m src.main
