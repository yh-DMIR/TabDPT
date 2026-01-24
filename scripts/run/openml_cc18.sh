#!/usr/bin/env bash
set -e

# =========================
# 基本配置
# =========================
PYTHON=python
SCRIPT=benchmark_tabdpt_dynamic.py

DATA_ROOT=limix
OUT_ROOT=results/v1.1_dynamic
mkdir -p "${OUT_ROOT}"

# =========================
# 多进程时建议限制 CPU 线程争用
# =========================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# =========================
# GPU/worker 设置（8 卡 AMD）
# 如果你想指定具体用哪些卡，比如 0-7：
#   GPUS="0,1,2,3,4,5,6,7"
# =========================
WORKERS=8
GPUS="0,1,2,3,4,5,6,7"

# =========================
# TabDPT 权重（HuggingFace 自动下载）
# 你没有本地 ckpt，所以优先从 HF 下载
# =========================
HF_REPO="Layer6/TabDPT"
HF_FILENAME="tabdpt1_1.safetensors"
HF_REVISION="main"

# 可选：指定 HF cache（比如放到大盘上）
# HF_CACHE_DIR="/path/to/hf_cache"
HF_CACHE_DIR=""

# =========================
# 通用运行参数
# device 建议用 cuda:0（在每个 worker 里，它指的是该 worker 可见的那一张卡）
# ROCm 一般建议不开 use_flash / compile（默认就是 False）
# =========================
COMMON_ARGS="
  --workers ${WORKERS}
  --gpus ${GPUS}
  --device cuda:0

  --hf-repo ${HF_REPO}
  --hf-filename ${HF_FILENAME}
  --revision ${HF_REVISION}

  --inf-batch-size 512
  --normalizer standard
  --clip-sigma 4.0
  --feature-reduction pca
  --faiss-metric l2

  --n-ensembles 8
  --temperature 0.8
  --context-size 2048
  --permute-classes
  --seed 42

  --verbose
"

# 如果设置了 HF_CACHE_DIR，就加上 --cache-dir
if [[ -n "${HF_CACHE_DIR}" ]]; then
  COMMON_ARGS="${COMMON_ARGS} --cache-dir ${HF_CACHE_DIR}"
fi

# =========================
# 2️⃣ OpenML-CC18
# =========================
echo "===== Running OpenML-CC18 (TabDPT dynamic ${WORKERS} workers) ====="
${PYTHON} ${SCRIPT} \
  --root "${DATA_ROOT}/openml_cc18_csv" \
  --out-dir "${OUT_ROOT}/openml_cc18" \
  --all-out "${OUT_ROOT}/tabdpt_openml_cc18.ALL.csv" \
  --summary-txt "${OUT_ROOT}/tabdpt_openml_cc18.summary.txt" \
  ${COMMON_ARGS}



echo "✅ All datasets finished."
echo "Results saved in: ${OUT_ROOT}"
