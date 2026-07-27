#!/usr/bin/env python3
"""Semi-synthetic label-context support experiment on CELLxGENE embeddings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_cellxgene_embedding_audit import clean_string, read_embedding, write_csv  # noqa: E402
from run_cellxgene_representation_diagnostics import train_projector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--embedding-key", default="embeddings")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Geneformer")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--context-field", required=True)
    parser.add_argument("--methods", nargs="*", default=["erm_mlp", "label_context_reweight", "adv_context", "sabca"])
    parser.add_argument("--support-levels", nargs="*", type=float, default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-fair-var", type=float, default=0.5)
    parser.add_argument("--lambda-adv", type=float, default=0.25)
    parser.add_argument("--lambda-supcon", type=float, default=0.15)
    parser.add_argument("--lambda-consistency", type=float, default=0.15)
    parser.add_argument("--lambda-group-dro", type=float, default=1.0)
    parser.add_argument("--lambda-max-gap", type=float, default=0.5)
    parser.add_argument("--lambda-cond-mmd", type=float, default=0.2)
    parser.add_argument("--lambda-cond-coral", type=float, default=0.05)
    parser.add_argument("--sabca-min-group-size", type=int, default=20)
    parser.add_argument("--sabca-max-sample-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def normalize_metadata(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    out = df.copy()
    if label_column != "label":
        out = out.rename(columns={label_column: "label"})
    for column in out.columns:
        out[column] = out[column].astype(object).map(clean_string).astype(object)
    out["cell_index"] = pd.to_numeric(df["cell_index"], errors="raise").astype(int)
    return out


def context_gap(y_true: Sequence[object], y_pred: Sequence[object], context: Sequence[object]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=str), "pred": np.asarray(y_pred, dtype=str), "c": np.asarray(context, dtype=str)})
    for context_value, sub in frame.groupby("c"):
        if len(sub) == 0 or sub["y"].nunique() < 2:
            continue
        rows.append(
            {
                "context_value": context_value,
                "balanced_accuracy": float(balanced_accuracy_score(sub["y"], sub["pred"])),
                "n": int(len(sub)),
            }
        )
    if len(rows) < 2:
        return {
            "worst_context_ba": float("nan"),
            "best_context_ba": float("nan"),
            "gap": float("nan"),
            "worst_context_value": "",
            "best_context_value": "",
            "n_contexts_evaluated": len(rows),
        }
    df = pd.DataFrame(rows)
    best = df.loc[df["balanced_accuracy"].idxmax()]
    worst = df.loc[df["balanced_accuracy"].idxmin()]
    return {
        "worst_context_ba": float(worst["balanced_accuracy"]),
        "best_context_ba": float(best["balanced_accuracy"]),
        "gap": float(best["balanced_accuracy"] - worst["balanced_accuracy"]),
        "worst_context_value": str(worst["context_value"]),
        "best_context_value": str(best["context_value"]),
        "n_contexts_evaluated": int(len(df)),
    }


def support_coverage(y: Sequence[object], c: Sequence[object], min_n: int) -> Tuple[int, int, float]:
    frame = pd.DataFrame({"y": np.asarray(y, dtype=str), "c": np.asarray(c, dtype=str)})
    n_labels = frame["y"].nunique()
    n_contexts = frame["c"].nunique()
    total = int(n_labels * n_contexts)
    counts = frame.groupby(["y", "c"]).size()
    supported = int((counts >= min_n).sum())
    return total, supported, float(supported / total) if total else float("nan")


def choose_pairs(
    pairs: List[Tuple[str, str]],
    labels: Sequence[str],
    contexts: Sequence[str],
    target_fraction: float,
    rng: np.random.Generator,
) -> set:
    if target_fraction >= 0.999:
        return set(pairs)
    min_pairs = max(len(set(labels)), len(set(contexts)))
    target_n = max(min_pairs, int(round(target_fraction * len(pairs))))
    pair_arr = np.asarray(pairs, dtype=object)
    label_set = set(labels)
    context_set = set(contexts)
    for _ in range(500):
        chosen_idx = rng.choice(len(pair_arr), size=min(target_n, len(pair_arr)), replace=False)
        chosen = {tuple(pair_arr[i]) for i in chosen_idx}
        if {p[0] for p in chosen} >= label_set and {p[1] for p in chosen} >= context_set:
            return chosen
    chosen = set()
    for label in label_set:
        candidates = [p for p in pairs if p[0] == label]
        chosen.add(candidates[int(rng.integers(len(candidates)))])
    for context in context_set:
        candidates = [p for p in pairs if p[1] == context]
        chosen.add(candidates[int(rng.integers(len(candidates)))])
    remaining = [p for p in pairs if p not in chosen]
    while len(chosen) < target_n and remaining:
        pick = remaining.pop(int(rng.integers(len(remaining))))
        chosen.add(pick)
    return chosen


def subsample_by_support(
    y: np.ndarray,
    c: np.ndarray,
    target_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_fraction >= 0.999:
        return np.arange(len(y))
    frame = pd.DataFrame({"idx": np.arange(len(y)), "y": y.astype(str), "c": c.astype(str)})
    pairs = sorted({(row.y, row.c) for row in frame.itertuples(index=False)})
    chosen = choose_pairs(pairs, sorted(frame["y"].unique()), sorted(frame["c"].unique()), target_fraction, rng)
    mask = [(str(row.y), str(row.c)) in chosen for row in frame.itertuples(index=False)]
    return frame.loc[mask, "idx"].to_numpy(dtype=int)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metadata = normalize_metadata(pd.read_csv(args.metadata_csv), args.label_column)
    x = read_embedding(args.embedding_file, args.embedding_key).astype(np.float32)
    y = metadata["label"].astype(str).to_numpy()
    c = metadata[args.context_field].astype(str).to_numpy()
    groups = metadata["donor_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=args.seed)
    train_idx, test_idx = next(iter(splitter.split(x, y, groups)))
    scaler = StandardScaler()
    x_train_full = scaler.fit_transform(x[train_idx]).astype(np.float32)
    x_test = scaler.transform(x[test_idx]).astype(np.float32)
    y_train_full = y[train_idx]
    c_train_full = c[train_idx]
    y_test = y[test_idx]
    c_test = c[test_idx]

    rows: List[Dict[str, object]] = []
    for target_fraction in args.support_levels:
        local_idx = subsample_by_support(y_train_full, c_train_full, target_fraction, rng)
        x_train = x_train_full[local_idx]
        y_train = y_train_full[local_idx]
        c_train = c_train_full[local_idx]
        total_pairs, supported_pairs, actual_coverage = support_coverage(y_train, c_train, args.sabca_min_group_size)
        for method in args.methods:
            print(f"[support] target={target_fraction:.2f} actual={actual_coverage:.3f} method={method}", flush=True)
            pred, _, _, _, _ = train_projector(
                x_train,
                y_train,
                c_train,
                x_test,
                method,
                args,
                seed=args.seed + int(target_fraction * 1000) + len(rows),
            )
            overall = float(balanced_accuracy_score(y_test.astype(str), pred.astype(str)))
            gap = context_gap(y_test, pred, c_test)
            rows.append(
                {
                    "target_support_fraction": float(target_fraction),
                    "actual_support_coverage_ge20": actual_coverage,
                    "supported_pairs_ge20": supported_pairs,
                    "total_pairs": total_pairs,
                    "train_cells": int(len(local_idx)),
                    "test_cells": int(len(test_idx)),
                    "method": method,
                    "overall_ba": overall,
                    **gap,
                }
            )

    write_csv(args.output_dir / "support_curve_results.csv", rows)
    df = pd.DataFrame(rows)
    best_rows = []
    for target, sub in df.groupby("target_support_fraction"):
        best = sub.loc[sub["worst_context_ba"].astype(float).idxmax()]
        best_rows.append(best.to_dict())
    write_csv(args.output_dir / "support_curve_best_by_level.csv", best_rows)
    summary = {
        "model_name": args.model_name,
        "context_field": args.context_field,
        "methods": args.methods,
        "support_levels": args.support_levels,
        "n_rows": len(rows),
        "best_by_level": best_rows,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
