#!/usr/bin/env python3
"""HEST histology-to-gene context mitigation smoke test.

This is intentionally summary-only. It retrains lightweight regression heads on
frozen patch embeddings and reports average/worst-context correlations without
writing per-spot, per-gene predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_hest_image_stats_baseline import align_sample, choose_genes, read_manifest  # noqa: E402
from eval_context_predictions import safe_corr  # noqa: E402

DEFAULT_GROUP_FIELDS = ["split", "platform", "site", "organ", "disease", "study_id", "tissue_processing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-genes", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--max-spots-per-sample", type=int, default=2500)
    parser.add_argument("--mitigation-context-fields", nargs="*", default=["platform", "tissue_processing"])
    parser.add_argument("--group-fields", nargs="*", default=DEFAULT_GROUP_FIELDS)
    parser.add_argument("--min-pairs", type=int, default=20)
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["model", "method", "scope", "group_field", "group_value", "metric"]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cap_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=int)
    return np.sort(rng.choice(np.arange(n, dtype=int), size=max_n, replace=False))


def load_spot_table(
    rows: List[Dict[str, str]],
    genes: Sequence[str],
    raw_root: Path,
    feature_dir: Path,
    max_spots_per_sample: int,
    seed: int,
    group_fields: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    meta_parts: List[pd.DataFrame] = []
    status_rows: List[Dict[str, object]] = []

    for row in rows:
        sample_id = row["sample_id"]
        try:
            x, y_available, available_genes, barcodes = align_sample(sample_id, raw_root, feature_dir, genes)
        except Exception as exc:  # noqa: BLE001
            status_rows.append({"sample_id": sample_id, "status": "failed", "error": str(exc)})
            continue
        take = cap_indices(len(x), max_spots_per_sample, rng)
        x = x[take].astype(np.float32)
        y_available = y_available[take].astype(np.float32)
        barcodes = barcodes[take]
        y = np.full((len(x), len(genes)), np.nan, dtype=np.float32)
        for local_col, gene in enumerate(available_genes):
            target_col = gene_to_idx.get(str(gene))
            if target_col is not None:
                y[:, target_col] = y_available[:, local_col]
        meta = {
            "sample_id": [sample_id] * len(x),
            "spot_id": [str(value) for value in barcodes],
        }
        for field in group_fields:
            meta[field] = [row.get(field, "NA") or "NA"] * len(x)
        x_parts.append(x)
        y_parts.append(y)
        meta_parts.append(pd.DataFrame(meta))
        status_rows.append({"sample_id": sample_id, "status": "ok", "n_spots": int(len(x))})

    if not x_parts:
        raise SystemExit("No samples could be loaded.")
    return np.vstack(x_parts), np.vstack(y_parts), pd.concat(meta_parts, ignore_index=True)


def sample_weights(context: np.ndarray) -> np.ndarray:
    counts = Counter(context.astype(str))
    weights = np.asarray([1.0 / counts[str(value)] for value in context], dtype=np.float32)
    return weights / float(np.mean(weights))


def context_projection_matrix(x_train: np.ndarray, context_train: np.ndarray, seed: int) -> Optional[np.ndarray]:
    classes = sorted(set(context_train.astype(str)))
    if len(classes) < 2:
        return None
    class_to_idx = {value: idx for idx, value in enumerate(classes)}
    y_context = np.asarray([class_to_idx[value] for value in context_train.astype(str)], dtype=int)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed)
    clf.fit(x_train, y_context)
    coef = np.asarray(clf.coef_, dtype=np.float64)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    basis = coef[0:1] if coef.shape[0] == 1 else coef
    q, _ = np.linalg.qr(basis.T)
    norms = np.linalg.norm(q, axis=0)
    q = q[:, norms > 1e-8]
    if q.size == 0:
        return None
    return q.astype(np.float32)


def remove_projection(x: np.ndarray, basis: Optional[np.ndarray]) -> np.ndarray:
    if basis is None:
        return x
    return x - (x @ basis) @ basis.T


def fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    pred = np.full((len(x_eval), y_train.shape[1]), np.nan, dtype=np.float32)
    for gene_idx in range(y_train.shape[1]):
        keep = np.isfinite(y_train[:, gene_idx])
        if int(np.sum(keep)) < 20:
            continue
        model = Ridge(alpha=alpha)
        fit_kwargs = {}
        if weights is not None:
            fit_kwargs["sample_weight"] = weights[keep]
        model.fit(x_train[keep], y_train[keep, gene_idx], **fit_kwargs)
        pred[:, gene_idx] = model.predict(x_eval).astype(np.float32)
    return pred


def fit_calibrators(
    train_pred: np.ndarray,
    train_y: np.ndarray,
    context_train: np.ndarray,
    min_pairs: int,
) -> Tuple[List[Tuple[float, float]], Dict[Tuple[int, str], Tuple[float, float]]]:
    global_params: List[Tuple[float, float]] = []
    group_params: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for gene_idx in range(train_y.shape[1]):
        keep = np.isfinite(train_y[:, gene_idx]) & np.isfinite(train_pred[:, gene_idx])
        if int(np.sum(keep)) < min_pairs or float(np.std(train_pred[keep, gene_idx])) == 0.0:
            global_params.append((1.0, 0.0))
        else:
            slope, intercept = np.polyfit(train_pred[keep, gene_idx], train_y[keep, gene_idx], deg=1)
            global_params.append((float(slope), float(intercept)))
        for value in sorted(set(context_train.astype(str))):
            gkeep = keep & (context_train.astype(str) == value)
            if int(np.sum(gkeep)) < min_pairs or float(np.std(train_pred[gkeep, gene_idx])) == 0.0:
                continue
            slope, intercept = np.polyfit(train_pred[gkeep, gene_idx], train_y[gkeep, gene_idx], deg=1)
            group_params[(gene_idx, value)] = (float(slope), float(intercept))
    return global_params, group_params


def apply_calibration(
    pred: np.ndarray,
    context: np.ndarray,
    global_params: Sequence[Tuple[float, float]],
    group_params: Mapping[Tuple[int, str], Tuple[float, float]],
) -> np.ndarray:
    calibrated = np.array(pred, copy=True)
    context = context.astype(str)
    for gene_idx, (global_slope, global_intercept) in enumerate(global_params):
        for value in sorted(set(context)):
            mask = context == value
            slope, intercept = group_params.get((gene_idx, value), (global_slope, global_intercept))
            calibrated[mask, gene_idx] = calibrated[mask, gene_idx] * slope + intercept
    return calibrated.astype(np.float32)


def train_method(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    train_mask: np.ndarray,
    alpha: float,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train_raw = x[train_mask]
    x_all_raw = x
    x_train = scaler.fit_transform(x_train_raw)
    x_all = scaler.transform(x_all_raw)
    y_train = y[train_mask]

    if method == "erm":
        return fit_predict_ridge(x_train, y_train, x_all, alpha=alpha)

    op, field = method.split(":", 1)
    context_train = meta.loc[train_mask, field].astype(str).to_numpy()
    context_all = meta[field].astype(str).to_numpy()

    if op == "reweight":
        weights = sample_weights(context_train)
        return fit_predict_ridge(x_train, y_train, x_all, alpha=alpha, weights=weights)

    if op == "linear_debias":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            basis = context_projection_matrix(x_train, context_train, seed=seed)
        return fit_predict_ridge(
            remove_projection(x_train, basis),
            y_train,
            remove_projection(x_all, basis),
            alpha=alpha,
        )

    if op == "group_calibration":
        train_pred = fit_predict_ridge(x_train, y_train, x_train, alpha=alpha)
        all_pred = fit_predict_ridge(x_train, y_train, x_all, alpha=alpha)
        global_params, group_params = fit_calibrators(train_pred, y_train, context_train, min_pairs=50)
        return apply_calibration(all_pred, context_all, global_params, group_params)

    raise ValueError(f"Unknown method: {method}")


def group_metric_rows(
    pred: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    genes: Sequence[str],
    model_name: str,
    method: str,
    group_fields: Sequence[str],
    min_pairs: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    scopes = {
        "nontrain": meta["split"].astype(str).to_numpy() != "train",
        "all": np.ones(len(meta), dtype=bool),
    }
    for scope, scope_mask in scopes.items():
        groups: List[Tuple[str, str, np.ndarray]] = [("overall", "overall", scope_mask)]
        for field in group_fields:
            if field not in meta.columns:
                continue
            for value in sorted(meta.loc[scope_mask, field].astype(str).unique()):
                groups.append((field, value, scope_mask & (meta[field].astype(str).to_numpy() == value)))
        for field, value, mask in groups:
            gene_corrs = []
            for gene_idx, gene in enumerate(genes):
                keep = mask & np.isfinite(y[:, gene_idx]) & np.isfinite(pred[:, gene_idx])
                corr = safe_corr(y[keep, gene_idx], pred[keep, gene_idx], "pearson", min_pairs)
                if corr is not None and math.isfinite(corr):
                    gene_corrs.append(float(corr))
            if not gene_corrs:
                continue
            rows.append(
                {
                    "model": model_name,
                    "method": method.replace(":", "_"),
                    "scope": scope,
                    "group_field": field,
                    "group_value": value,
                    "n_spots": int(np.sum(mask)),
                    "n_samples": int(meta.loc[mask, "sample_id"].nunique()),
                    "n_genes": int(len(gene_corrs)),
                    "average_gene_pearson": float(np.mean(gene_corrs)),
                }
            )
    return rows


def summarize_methods(rows: Sequence[Mapping[str, object]], group_fields: Sequence[str]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    by_method = sorted({str(row["method"]) for row in rows})
    for method in by_method:
        method_rows = [row for row in rows if row["method"] == method and row["scope"] == "nontrain"]
        overall = next((row for row in method_rows if row["group_field"] == "overall"), None)
        for field in group_fields:
            candidates = [row for row in method_rows if row["group_field"] == field]
            if not candidates:
                continue
            worst = min(candidates, key=lambda row: float(row["average_gene_pearson"]))
            best = max(candidates, key=lambda row: float(row["average_gene_pearson"]))
            summary_rows.append(
                {
                    "method": method,
                    "context_field": field,
                    "overall_nontrain_average_gene_pearson": overall.get("average_gene_pearson") if overall else "NA",
                    "best_group": best["group_value"],
                    "best_average_gene_pearson": best["average_gene_pearson"],
                    "worst_group": worst["group_value"],
                    "worst_average_gene_pearson": worst["average_gene_pearson"],
                    "best_minus_worst": float(best["average_gene_pearson"]) - float(worst["average_gene_pearson"]),
                }
            )
    return summary_rows


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest_csv)
    choose_args = argparse.Namespace(
        min_train_samples_per_gene=5,
        min_eval_samples_per_gene=2,
        max_genes=args.max_genes,
    )
    genes = choose_genes(rows, args.raw_root, choose_args)
    x, y, meta = load_spot_table(
        rows=rows,
        genes=genes,
        raw_root=args.raw_root,
        feature_dir=args.feature_dir,
        max_spots_per_sample=args.max_spots_per_sample,
        seed=args.seed,
        group_fields=args.group_fields,
    )
    train_mask = meta["split"].astype(str).to_numpy() == "train"
    methods = ["erm"]
    for field in args.mitigation_context_fields:
        if field in meta.columns and meta.loc[train_mask, field].astype(str).nunique() >= 2:
            methods.extend([f"reweight:{field}", f"linear_debias:{field}", f"group_calibration:{field}"])

    metric_rows: List[Dict[str, object]] = []
    for idx, method in enumerate(methods):
        pred = train_method(method, x, y, meta, train_mask, alpha=args.alpha, seed=args.seed + idx)
        metric_rows.extend(group_metric_rows(pred, y, meta, genes, args.model_name, method, args.group_fields, args.min_pairs))

    summary_rows = summarize_methods(metric_rows, args.group_fields)
    write_csv(args.outdir / "method_group_metrics.csv", metric_rows)
    write_csv(args.outdir / "method_summary.csv", summary_rows)
    summary = {
        "model_name": args.model_name,
        "manifest_csv": str(args.manifest_csv),
        "feature_dir": str(args.feature_dir),
        "n_spots": int(len(meta)),
        "n_train_spots": int(np.sum(train_mask)),
        "n_genes": int(len(genes)),
        "genes": list(genes),
        "methods": [method.replace(":", "_") for method in methods],
        "max_spots_per_sample": int(args.max_spots_per_sample),
        "method_summary": summary_rows,
    }
    with (args.outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
