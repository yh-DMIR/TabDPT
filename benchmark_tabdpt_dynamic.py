#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic multi-GPU benchmark runner for TabDPT (AMD/ROCm friendly).

Features (mirrors benchmark_tabicl_dynamic.py):
- Dynamic scheduling (work stealing): N worker processes pull datasets from a shared queue.
- Each worker binds to one GPU via HIP_VISIBLE_DEVICES=<gpu_id> and uses device "cuda:0".
- Writes per-worker CSVs, merges into one ALL CSV, and writes ONE global summary TXT:
  - discovered_pairs
  - processed_pairs
  - missing_test_datasets
  - failed_datasets
  - avg_accuracy_ok
  - avg_accuracy_ok_top_{27,63,154}

Differences vs TabICL runner:
- TabDPT weights are auto-downloaded from Hugging Face (no local ckpt required).
- Uses TabDPTClassifier and its ensemble inference parameters.

Example:
  python benchmark_tabdpt_dynamic.py \
    --root limix/tabzilla_csv \
    --out-dir results/tabdpt/v1.1/tabzilla_dynamic \
    --workers 8 \
    --gpus 0,1,2,3,4,5,6,7 \
    --hf-repo Layer6/TabDPT \
    --hf-filename tabdpt1_1.safetensors \
    --revision main \
    --n-ensembles 8 \
    --temperature 0.8 \
    --context-size 2048 \
    --permute-classes \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# -----------------------------
# Helpers: dataset discovery
# -----------------------------

TARGET_CANDIDATES = [
    "target", "label", "class", "y",
    "TARGET", "Label", "Class", "Y",
]


def sanitize_dataset_id(train_path: Path) -> str:
    m = re.search(r"(OpenML-ID-\d+)", str(train_path))
    return m.group(1) if m else train_path.parent.name


def find_dataset_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for train_path in root.rglob("*_train.csv"):
        test_path = train_path.with_name(train_path.name.replace("_train.csv", "_test.csv"))
        if test_path.exists():
            pairs.append((train_path, test_path))
    return sorted(pairs, key=lambda x: str(x[0]))


def find_missing_test_datasets(root: Path) -> List[str]:
    missing: List[str] = []
    for train_path in root.rglob("*_train.csv"):
        test_path = train_path.with_name(train_path.name.replace("_train.csv", "_test.csv"))
        if not test_path.exists():
            missing.append(sanitize_dataset_id(train_path))
    return sorted(set(missing))


def infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    # 1) common candidates
    for c in TARGET_CANDIDATES:
        if c in train_df.columns:
            return c
    # 2) column that exists only in train
    extra = [c for c in train_df.columns if c not in test_df.columns]
    if len(extra) == 1:
        return extra[0]
    # 3) fallback: last column
    return train_df.columns[-1]


def _default_all_out(out_dir: Path) -> Path:
    return out_dir / "tabdpt_results.ALL.csv"


def _default_summary_txt(out_dir: Path) -> Path:
    return out_dir / "tabdpt_results.summary.txt"


# -----------------------------
# HF download helpers
# -----------------------------

def download_tabdpt_weight(
    repo_id: str,
    filename: str,
    revision: str,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Download a single weight file from HF and return local path.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required for auto-download. "
            "Please: pip install -U huggingface_hub"
        ) from e

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
    )
    return str(Path(local_path).resolve())


# -----------------------------
# Result schema
# -----------------------------

@dataclass
class ResultRow:
    dataset_id: str
    n_train: int
    n_test: int
    n_features: int
    n_classes: Optional[int]
    accuracy: Optional[float]
    f1_weighted: Optional[float]
    logloss: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]


# -----------------------------
# Core evaluation (reuses one clf per worker)
# -----------------------------

def _to_numpy_features(df: pd.DataFrame) -> np.ndarray:
    # Keep exactly the same data content (no one-hot) — just convert to numpy.
    # TabDPT expects numeric; if dataset contains non-numeric, it will fail -> status=fail
    return df.to_numpy()


def run_one_dataset_with_clf(
    clf,
    train_csv: Path,
    test_csv: Path,
    predict_kwargs: Dict,
) -> ResultRow:
    dataset_id = sanitize_dataset_id(train_csv)

    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        target_col = infer_target_column(train_df, test_df)

        X_train_df = train_df.drop(columns=[target_col])
        y_train = train_df[target_col].to_numpy()

        if target_col in test_df.columns:
            X_test_df = test_df.drop(columns=[target_col])
            y_test = test_df[target_col].to_numpy()
        else:
            X_test_df = test_df
            y_test = None

        X_train = _to_numpy_features(X_train_df)
        X_test = _to_numpy_features(X_test_df)

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_s = time.time() - t0

        t1 = time.time()
        y_pred = clf.predict(X_test, **predict_kwargs)
        pred_s = time.time() - t1

        # Metrics
        if y_test is not None:
            acc = accuracy_score(y_test, y_pred)
            f1w = f1_score(y_test, y_pred, average="weighted")

            ll = None
            try:
                # Use ensemble proba when n_ensembles>1
                n_ens = int(predict_kwargs.get("n_ensembles", 8))
                temperature = float(predict_kwargs.get("temperature", 0.8))
                context_size = int(predict_kwargs.get("context_size", 2048))
                permute_classes = bool(predict_kwargs.get("permute_classes", True))
                seed = predict_kwargs.get("seed", None)

                if n_ens <= 1:
                    proba = clf.predict_proba(
                        X_test,
                        temperature=temperature,
                        context_size=context_size,
                        seed=seed,
                    )
                else:
                    proba = clf.ensemble_predict_proba(
                        X_test,
                        n_ensembles=n_ens,
                        temperature=temperature,
                        context_size=context_size,
                        permute_classes=permute_classes,
                        seed=seed,
                    )

                labels = np.unique(y_train)  # stable labels set
                ll = log_loss(y_test, proba, labels=labels)
            except Exception:
                ll = None

            n_classes = int(len(np.unique(y_train)))
        else:
            acc = f1w = ll = None
            n_classes = int(len(np.unique(y_train)))

        return ResultRow(
            dataset_id=dataset_id,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            n_features=int(X_train.shape[1]),
            n_classes=n_classes,
            accuracy=float(acc) if acc is not None else None,
            f1_weighted=float(f1w) if f1w is not None else None,
            logloss=float(ll) if ll is not None else None,
            fit_seconds=float(fit_s),
            predict_seconds=float(pred_s),
            status="ok",
            error=None,
        )

    except Exception as e:
        return ResultRow(
            dataset_id=dataset_id,
            n_train=0,
            n_test=0,
            n_features=0,
            n_classes=None,
            accuracy=None,
            f1_weighted=None,
            logloss=None,
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(e).__name__}: {e}",
        )


# -----------------------------
# Worker process (dynamic queue)
# -----------------------------

def worker_main(
    worker_id: int,
    gpu_id: int,
    task_queue,
    out_csv: str,
    clf_kwargs: Dict,
    predict_kwargs: Dict,
    verbose: bool,
):
    """
    Each worker:
    - bind to one GPU
    - create TabDPTClassifier once
    - pull tasks until sentinel None
    - write its own CSV at the end
    """
    try:
        # Bind GPU (AMD/ROCm)
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)

        # Reduce CPU contention
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # Import inside worker after setting HIP_VISIBLE_DEVICES
        from tabdpt import TabDPTClassifier  # noqa

        # In each worker, "cuda:0" refers to the one visible GPU
        if not clf_kwargs.get("device"):
            clf_kwargs["device"] = "cuda:0"

        clf = TabDPTClassifier(**clf_kwargs)

        rows: List[ResultRow] = []
        while True:
            item = task_queue.get()
            if item is None:
                break

            train_csv, test_csv = item
            row = run_one_dataset_with_clf(
                clf,
                Path(train_csv),
                Path(test_csv),
                predict_kwargs=predict_kwargs,
            )
            rows.append(row)

            if verbose:
                print(f"[worker {worker_id} | gpu {gpu_id}] [{row.status}] {row.dataset_id} acc={row.accuracy}")

        df = pd.DataFrame([asdict(r) for r in rows])
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    except Exception:
        # If a worker crashes, write a marker row for debugging
        err = traceback.format_exc()
        crash_df = pd.DataFrame([{
            "dataset_id": f"__WORKER_CRASH__{worker_id}",
            "n_train": 0,
            "n_test": 0,
            "n_features": 0,
            "n_classes": None,
            "accuracy": None,
            "f1_weighted": None,
            "logloss": None,
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "status": "fail",
            "error": err,
        }])
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        crash_df.to_csv(out_csv, index=False)
        if verbose:
            print(f"[worker {worker_id}] CRASHED:\n{err}")


# -----------------------------
# Summary writer
# -----------------------------

def write_summary_txt(
    out_txt: Path,
    root: Path,
    discovered_pairs: int,
    processed_pairs: int,
    missing_test_ids: List[str],
    failed_ids: List[str],
    avg_acc: Optional[float],
    topn_avgs: Dict[int, float],
):
    lines: List[str] = []
    lines.append(f"root: {root}")
    lines.append(f"discovered_pairs: {discovered_pairs}")
    lines.append(f"processed_pairs: {processed_pairs}")

    lines.append(f"missing_test_count: {len(missing_test_ids)}")
    if missing_test_ids:
        lines.append("missing_test_datasets: " + ", ".join(missing_test_ids))
    else:
        lines.append("missing_test_datasets: (none)")

    lines.append(f"failed_count: {len(failed_ids)}")
    if failed_ids:
        lines.append("failed_datasets: " + ", ".join(failed_ids))
    else:
        lines.append("failed_datasets: (none)")

    if avg_acc is None:
        lines.append("avg_accuracy_ok: (none)")
    else:
        lines.append(f"avg_accuracy_ok: {avg_acc:.6f}")

    for n in (27, 63, 154):
        if n in topn_avgs:
            lines.append(f"avg_accuracy_ok_top_{n}: {topn_avgs[n]:.6f}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True, help="Root folder containing *_train.csv and *_test.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-worker CSVs and merged results")
    ap.add_argument("--all-out", default=None, help="Path to merged ALL CSV (default: <out-dir>/tabdpt_results.ALL.csv)")
    ap.add_argument("--summary-txt", default=None, help="Path to ONE global summary txt (default: <out-dir>/tabdpt_results.summary.txt)")
    ap.add_argument("--workers", type=int, default=8, help="Number of worker processes (usually #GPUs)")
    ap.add_argument("--gpus", default=None, help="Comma-separated GPU ids to use (default: 0..workers-1)")

    # --- HF download (no local ckpt) ---
    ap.add_argument("--hf-repo", type=str, default="Layer6/TabDPT", help="HuggingFace repo id")
    ap.add_argument("--hf-filename", type=str, default="tabdpt1_1.safetensors", help="Weight filename in the repo")
    ap.add_argument("--revision", type=str, default="main", help="HF revision/branch/tag")
    ap.add_argument("--cache-dir", type=str, default=None, help="HF cache dir (optional)")

    # --- TabDPT model init params ---
    ap.add_argument("--device", default="cuda:0", help='Device string in workers (recommend: "cuda:0")')
    ap.add_argument("--inf-batch-size", type=int, default=512)
    ap.add_argument("--normalizer", type=str, default="standard")
    ap.add_argument("--missing-indicators", action="store_true")
    ap.add_argument("--clip-sigma", type=float, default=4.0)
    ap.add_argument("--feature-reduction", type=str, default="pca", choices=["pca", "subsample"])
    ap.add_argument("--faiss-metric", type=str, default="l2", choices=["l2", "ip"])

    # ROCm safer defaults: use_flash=False, compile=False (same风格参考你的cls_example)。
    ap.add_argument("--use-flash", action="store_true", help="Enable flash attention (default: False for ROCm safety)")
    ap.add_argument("--compile", action="store_true", help="Enable torch.compile (default: False for ROCm safety)")

    # --- TabDPT predict params (ensemble ICL) ---
    ap.add_argument("--n-ensembles", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--context-size", type=int, default=2048)
    ap.add_argument("--permute-classes", action="store_true", help="Enable class permutation (default: False unless set)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Reduce CPU contention (good default for multi-proc)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_out = Path(args.all_out) if args.all_out else _default_all_out(out_dir)
    summary_txt = Path(args.summary_txt) if args.summary_txt else _default_summary_txt(out_dir)

    root = Path(args.root)
    missing_test_ids = find_missing_test_datasets(root)
    pairs = find_dataset_pairs(root)
    discovered_pairs = len(pairs)

    if discovered_pairs == 0:
        empty_df = pd.DataFrame(columns=[f.name for f in ResultRow.__dataclass_fields__.values()])
        all_out.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(all_out, index=False)

        write_summary_txt(
            out_txt=summary_txt,
            root=root,
            discovered_pairs=0,
            processed_pairs=0,
            missing_test_ids=missing_test_ids,
            failed_ids=[],
            avg_acc=None,
            topn_avgs={},
        )
        print("No dataset pairs found. Wrote empty outputs.")
        return

    workers = int(args.workers)
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
        if len(gpu_ids) != workers:
            raise ValueError(f"--gpus must list exactly {workers} ids, got {len(gpu_ids)}")
    else:
        gpu_ids = list(range(workers))

    # 1) Download weights ONCE in main process, then pass local path to all workers
    weight_path = download_tabdpt_weight(
        repo_id=args.hf_repo,
        filename=args.hf_filename,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    # 2) Prepare model kwargs (mirrors your cls_example style: model_weight_path + device + use_flash/compile) :contentReference[oaicite:2]{index=2}
    clf_kwargs: Dict = dict(
        inf_batch_size=args.inf_batch_size,
        normalizer=args.normalizer if args.normalizer.lower() != "none" else None,
        missing_indicators=bool(args.missing_indicators),
        clip_sigma=float(args.clip_sigma),
        feature_reduction=args.feature_reduction,
        faiss_metric=args.faiss_metric,
        device=args.device,
        use_flash=bool(args.use_flash),
        compile=bool(args.compile),
        model_weight_path=weight_path,
        verbose=False,  # worker logs controlled by args.verbose
    )

    predict_kwargs: Dict = dict(
        n_ensembles=int(args.n_ensembles),
        temperature=float(args.temperature),
        context_size=int(args.context_size),
        permute_classes=bool(args.permute_classes),
        seed=int(args.seed) if args.seed is not None else None,
    )

    # --- multiprocessing setup ---
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    task_queue: mp.Queue = mp.Queue()

    # Enqueue all tasks
    for train_csv, test_csv in pairs:
        task_queue.put((str(train_csv), str(test_csv)))

    # Sentinels
    for _ in range(workers):
        task_queue.put(None)

    # Start workers
    procs: List[mp.Process] = []
    worker_csv_paths: List[Path] = []
    for wid in range(workers):
        w_csv = out_dir / f"worker_{wid}.csv"
        worker_csv_paths.append(w_csv)
        p = mp.Process(
            target=worker_main,
            args=(wid, gpu_ids[wid], task_queue, str(w_csv), dict(clf_kwargs), dict(predict_kwargs), args.verbose),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Wait
    for p in procs:
        p.join()

    # Merge per-worker CSVs
    dfs: List[pd.DataFrame] = []
    for w_csv in worker_csv_paths:
        if w_csv.exists():
            try:
                dfs.append(pd.read_csv(w_csv))
            except Exception:
                continue

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=[f.name for f in ResultRow.__dataclass_fields__.values()])

    all_out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(all_out, index=False)

    # Compute summary
    processed_pairs = int(len(all_df))

    if len(all_df):
        ok_df = all_df[(all_df["status"] == "ok") & all_df["accuracy"].notna()].copy()
    else:
        ok_df = pd.DataFrame(columns=["accuracy", "status", "dataset_id"])

    avg_acc = float(ok_df["accuracy"].mean()) if len(ok_df) > 0 else None

    # Top-N averages by highest accuracy
    cutoffs = (27, 63, 154)
    topn_avgs: Dict[int, float] = {}
    if len(ok_df) > 0:
        ok_sorted = ok_df.sort_values("accuracy", ascending=False, kind="mergesort")
        ok_count = len(ok_sorted)
        for n in cutoffs:
            if ok_count >= n:
                topn_avgs[n] = float(ok_sorted.head(n)["accuracy"].mean())

    failed_ids: List[str] = []
    if len(all_df):
        failed_ids = (
            all_df.loc[all_df["status"] == "fail", "dataset_id"]
            .dropna()
            .astype(str)
            .tolist()
        )
        failed_ids = sorted(set(failed_ids))

    write_summary_txt(
        out_txt=summary_txt,
        root=root,
        discovered_pairs=discovered_pairs,
        processed_pairs=processed_pairs,
        missing_test_ids=missing_test_ids,
        failed_ids=failed_ids,
        avg_acc=avg_acc,
        topn_avgs=topn_avgs,
    )

    # Print config for reproducibility
    print("\nSaved per-worker CSVs to:", str(out_dir))
    print("Saved merged ALL CSV to:", str(all_out))
    print("Saved summary TXT to:", str(summary_txt))
    print("\nTabDPT weight path:")
    print(weight_path)
    print("\nTabDPT kwargs:")
    print(json.dumps(clf_kwargs, indent=2, ensure_ascii=False))
    print("\nPredict kwargs:")
    print(json.dumps(predict_kwargs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
