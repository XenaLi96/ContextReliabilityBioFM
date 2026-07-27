#!/usr/bin/env python3
"""Extract lightweight CELLxGENE baseline embeddings for the FM audit pipeline.

This is not a foundation model extractor. It creates a reproducible HVG-PCA/SVD
embedding from the exact selected cell manifest used by the CELLxGENE context
experiments, so FM embeddings can be compared against a simple baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--selected-cells-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-genes-csv", type=Path, default=None)
    parser.add_argument("--n-top-genes", type=int, default=512)
    parser.add_argument("--n-components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def load_expression_subset(adata: ad.AnnData, row_indices: np.ndarray) -> sparse.csr_matrix:
    row_indices = np.asarray(row_indices, dtype=int)
    order = np.argsort(row_indices)
    sorted_rows = row_indices[order]
    x_sorted = sparse.csr_matrix(adata.X[sorted_rows, :])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return x_sorted[inverse, :]


def select_variable_genes(x: sparse.csr_matrix, n_top_genes: int) -> np.ndarray:
    means = np.asarray(x.mean(axis=0)).ravel()
    second = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    variances = second - means * means
    expressed = np.asarray((x > 0).sum(axis=0)).ravel()
    variances[expressed < 5] = -np.inf
    n_top = min(int(n_top_genes), int(np.isfinite(variances).sum()))
    if n_top <= 0:
        raise ValueError("No genes passed the minimum expression filter.")
    top = np.argpartition(variances, -n_top)[-n_top:]
    return top[np.argsort(variances[top])[::-1]]


def load_feature_indices(path: Path | None, x_all: sparse.csr_matrix, n_top_genes: int) -> np.ndarray:
    if path is None:
        return select_variable_genes(x_all, n_top_genes)
    df = pd.read_csv(path)
    if "feature_matrix_index" not in df.columns:
        raise ValueError(f"{path} has no feature_matrix_index column")
    indices = pd.to_numeric(df["feature_matrix_index"], errors="raise").astype(int).to_numpy()
    if len(indices) == 0:
        raise ValueError(f"{path} contains no feature rows")
    return indices[:n_top_genes]


def feature_rows(adata: ad.AnnData, feature_idx: np.ndarray) -> List[Dict[str, object]]:
    var = adata.var.iloc[feature_idx].copy()
    rows: List[Dict[str, object]] = []
    for rank, (matrix_idx, (_, row)) in enumerate(zip(feature_idx, var.iterrows()), start=1):
        rows.append(
            {
                "rank": rank,
                "feature_matrix_index": int(matrix_idx),
                "feature_id": row.get("feature_id", ""),
                "feature_name": row.get("feature_name", ""),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected_cells_csv)
    if "cell_index" not in selected.columns:
        raise ValueError("selected cells CSV must contain cell_index")
    adata = ad.read_h5ad(args.h5ad, backed="r")
    row_indices = pd.to_numeric(selected["cell_index"], errors="raise").astype(int).to_numpy()
    x_all = load_expression_subset(adata, row_indices)
    feature_idx = load_feature_indices(args.feature_genes_csv, x_all, args.n_top_genes)
    x_hvg = x_all[:, feature_idx].tocsr().astype(np.float32)

    scaler = StandardScaler(with_mean=False)
    x_scaled = scaler.fit_transform(x_hvg)
    n_components = min(args.n_components, x_scaled.shape[1] - 1, x_scaled.shape[0] - 1)
    if n_components < 2:
        raise ValueError(f"n_components too small after bounds: {n_components}")
    svd = TruncatedSVD(n_components=n_components, random_state=args.seed)
    embeddings = svd.fit_transform(x_scaled).astype(np.float32)

    selected.to_csv(args.output_dir / "metadata.csv", index=False)
    pd.DataFrame(feature_rows(adata, feature_idx)).to_csv(args.output_dir / "feature_genes.csv", index=False)
    np.savez_compressed(args.output_dir / "hvg_pca_embeddings.npz", embeddings=embeddings)

    summary = {
        "h5ad": str(args.h5ad),
        "selected_cells_csv": str(args.selected_cells_csv),
        "feature_genes_csv": str(args.feature_genes_csv) if args.feature_genes_csv else None,
        "n_cells": int(embeddings.shape[0]),
        "n_input_genes": int(len(feature_idx)),
        "n_components": int(embeddings.shape[1]),
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        "outputs": {
            "metadata": str(args.output_dir / "metadata.csv"),
            "feature_genes": str(args.output_dir / "feature_genes.csv"),
            "embeddings": str(args.output_dir / "hvg_pca_embeddings.npz"),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
