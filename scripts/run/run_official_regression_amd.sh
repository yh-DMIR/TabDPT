#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-benchmark_tabdpt_regression_amd.py}
OUT_DIR=${OUT_DIR:-result/TabDPT_official_regression}
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MODEL_PATH=${MODEL_PATH:-tabdpt1_1.safetensors}

${PYTHON} ${SCRIPT} \
  --model-path "${MODEL_PATH}" \
  --out-dir "${OUT_DIR}" \
  --workers "${WORKERS}" \
  --gpus "${GPUS}" \
  --inf-batch-size 512 \
  --normalizer standard \
  --clip-sigma 4.0 \
  --feature-reduction pca \
  --faiss-metric l2 \
  --n-ensembles 8 \
  --context-size 2048 \
  --seed 42 \
  --verbose
