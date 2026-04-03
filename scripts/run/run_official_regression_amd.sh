#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-benchmark_tabdpt_regression_amd.py}
OUT_DIR=${OUT_DIR:-result/TabDPT_official_regression}
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MODEL_PATH=${MODEL_PATH:-}

MODEL_ARGS=()
if [[ -n "${MODEL_PATH}" ]]; then
  MODEL_ARGS+=(--model-path "${MODEL_PATH}")
fi

${PYTHON} ${SCRIPT} \
  "${MODEL_ARGS[@]}" \
  --out-dir "${OUT_DIR}" \
  --workers "${WORKERS}" \
  --gpus "${GPUS}" \
  --device cuda:0 \
  --inf-batch-size 512 \
  --normalizer standard \
  --clip-sigma 4.0 \
  --feature-reduction pca \
  --faiss-metric l2 \
  --n-ensembles 8 \
  --context-size 2048 \
  --seed 42 \
  --verbose
