#!/usr/bin/env python3
"""Run a first-pass HEST histology-to-gene baseline from fixed image statistics."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_MANIFEST = Path("data/metadata/hest/selected_metadata_manifest.csv")
DEFAULT_RAW_ROOT = Path("data/raw/hest")
DEFAULT_FEATURE_DIR = Path("data/embeddings/hest/image_stats")
DEFAULT_OUTDIR = Path("outputs/hest/image_stats_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-genes", type=int, default=64)
    parser.add_argument("--min-train-samples-per-gene", type=int, default=5)
    parser.add_argument("--min-eval-samples-per-gene", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_features(sample_id: str, feature_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = feature_dir / f"{sample_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Feature file missing: {path}")
    data = np.load(path, allow_pickle=True)
    return data["features"].astype(np.float32), data["barcodes"].astype(str)


def first_gene_indices(var_names: Sequence[str]) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for idx, gene in enumerate(map(str, var_names)):
        if gene not in index:
            index[gene] = idx
    return index


def get_gene_scores(sample_ids: Sequence[str], raw_root: Path) -> Tuple[Counter, Counter]:
    presence = Counter()
    abundance = Counter()
    for sample_id in sample_ids:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = ad.read_h5ad(raw_root / "st" / f"{sample_id}.h5ad", backed="r")
        seen = set(map(str, a.var_names))
        presence.update(seen)
        if "total_counts" in a.var.columns:
            totals = np.asarray(a.var["total_counts"]).reshape(-1)
            for gene, value in zip(map(str, a.var_names), totals):
                if gene and np.isfinite(value):
                    abundance[gene] += float(value)
    return presence, abundance


def choose_genes(rows: List[Dict[str, str]], raw_root: Path, args: argparse.Namespace) -> List[str]:
    train_ids = [row["sample_id"] for row in rows if row["split"] == "train"]
    eval_ids = [row["sample_id"] for row in rows if row["split"] != "train"]
    train_presence, abundance = get_gene_scores(train_ids, raw_root)
    eval_presence, _ = get_gene_scores(eval_ids, raw_root)
    candidates = [
        gene for gene, count in train_presence.items()
        if count >= args.min_train_samples_per_gene
        and eval_presence.get(gene, 0) >= args.min_eval_samples_per_gene
    ]
    candidates.sort(key=lambda gene: (train_presence[gene], eval_presence[gene], abundance[gene], gene), reverse=True)
    return candidates[: args.max_genes]


def expression_for_genes(sample_id: str, raw_root: Path, genes: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ad.read_h5ad(raw_root / "st" / f"{sample_id}.h5ad")
    gene_to_idx = first_gene_indices(a.var_names)
    available = [gene for gene in genes if gene in gene_to_idx]
    if not available:
        return np.empty((a.n_obs, 0), dtype=np.float32), np.asarray([], dtype=object)
    indices = [gene_to_idx[gene] for gene in available]
    x = a.X[:, indices]
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    totals = np.asarray(a.obs.get("total_counts", x.sum(axis=1))).reshape(-1).astype(np.float32)
    totals = np.maximum(totals, 1.0)
    x = np.log1p(x / totals[:, None] * 10000.0).astype(np.float32)
    return x, np.asarray(available, dtype=object)


def align_sample(sample_id: str, raw_root: Path, feature_dir: Path, genes: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, feature_barcodes = load_features(sample_id, feature_dir)
    y, available_genes = expression_for_genes(sample_id, raw_root, genes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ad.read_h5ad(raw_root / "st" / f"{sample_id}.h5ad", backed="r")
    obs_to_idx = {str(barcode): idx for idx, barcode in enumerate(a.obs_names)}
    feature_keep = []
    expression_keep = []
    aligned_barcodes = []
    for idx, barcode in enumerate(feature_barcodes):
        expression_idx = obs_to_idx.get(str(barcode))
        if expression_idx is None:
            continue
        feature_keep.append(idx)
        expression_keep.append(expression_idx)
        aligned_barcodes.append(str(barcode))
    if not feature_keep:
        raise ValueError(f"No aligned barcodes for {sample_id}")
    return (
        features[feature_keep],
        y[expression_keep],
        available_genes,
        np.asarray(aligned_barcodes, dtype=object),
    )


def write_predictions(
    path: Path,
    rows: List[Dict[str, str]],
    genes: Sequence[str],
    models: Dict[str, object],
    raw_root: Path,
    feature_dir: Path,
) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    per_split = Counter()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "spot_id", "gene", "y_true", "y_pred"])
        writer.writeheader()
        for row in rows:
            sample_id = row["sample_id"]
            x, y, available_genes, barcodes = align_sample(sample_id, raw_root, feature_dir, genes)
            gene_to_col = {gene: idx for idx, gene in enumerate(available_genes)}
            for gene in genes:
                model = models.get(gene)
                col = gene_to_col.get(gene)
                if model is None or col is None:
                    continue
                pred = np.asarray(model.predict(x), dtype=np.float32)
                true = y[:, col].astype(np.float32)
                for barcode, y_true, y_pred in zip(barcodes, true, pred):
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "spot_id": barcode,
                            "gene": gene,
                            "y_true": f"{float(y_true):.6g}",
                            "y_pred": f"{float(y_pred):.6g}",
                        }
                    )
                total_rows += len(barcodes)
                per_split[row["split"]] += len(barcodes)
    return {"prediction_rows": total_rows, "prediction_rows_by_split": dict(per_split)}


def train_models(
    rows: List[Dict[str, str]],
    genes: Sequence[str],
    raw_root: Path,
    feature_dir: Path,
    alpha: float,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    train_by_gene_x: Dict[str, List[np.ndarray]] = defaultdict(list)
    train_by_gene_y: Dict[str, List[np.ndarray]] = defaultdict(list)
    train_sample_counts = Counter()

    for row in train_rows:
        sample_id = row["sample_id"]
        x, y, available_genes, _ = align_sample(sample_id, raw_root, feature_dir, genes)
        gene_to_col = {gene: idx for idx, gene in enumerate(available_genes)}
        for gene in genes:
            col = gene_to_col.get(gene)
            if col is None:
                continue
            train_by_gene_x[gene].append(x)
            train_by_gene_y[gene].append(y[:, col])
            train_sample_counts[gene] += 1

    models: Dict[str, object] = {}
    model_summary: Dict[str, object] = {}
    for gene in genes:
        chunks_x = train_by_gene_x.get(gene, [])
        chunks_y = train_by_gene_y.get(gene, [])
        if not chunks_x:
            continue
        x_train = np.vstack(chunks_x).astype(np.float32)
        y_train = np.concatenate(chunks_y).astype(np.float32)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x_train, y_train)
        models[gene] = model
        model_summary[gene] = {
            "train_rows": int(len(y_train)),
            "train_samples": int(train_sample_counts[gene]),
            "target_mean": float(np.mean(y_train)),
            "target_nonzero_rate": float(np.mean(y_train > 0)),
        }
    return models, model_summary


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    rows = read_manifest(args.manifest_csv)
    genes = choose_genes(rows, args.raw_root, args)
    if not genes:
        raise SystemExit("No target genes passed coverage filters")

    models, model_summary = train_models(rows, genes, args.raw_root, args.feature_dir, args.alpha)
    args.outdir.mkdir(parents=True, exist_ok=True)
    pred_path = args.outdir / "predictions.csv"
    prediction_summary = write_predictions(pred_path, rows, genes, models, args.raw_root, args.feature_dir)
    summary = {
        "manifest_csv": str(args.manifest_csv),
        "raw_root": str(args.raw_root),
        "feature_dir": str(args.feature_dir),
        "output_predictions": str(pred_path),
        "alpha": args.alpha,
        "selected_genes": genes,
        "n_selected_genes": len(genes),
        "n_trained_genes": len(models),
        "model_summary": model_summary,
    }
    summary.update(prediction_summary)
    with (args.outdir / "baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({k: summary[k] for k in ["n_selected_genes", "n_trained_genes", "prediction_rows", "output_predictions"]}, indent=2))


if __name__ == "__main__":
    main()
