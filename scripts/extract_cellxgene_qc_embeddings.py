#!/usr/bin/env python3
"""Extract CELLxGENE QC/count-depth feature embeddings.

This is a low-level control representation for context probes. It intentionally
uses only library-size, detected-gene, mitochondrial, and sparsity statistics,
not gene identity or foundation-model embeddings.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cells", type=int, default=0)
    return parser.parse_args()


def gene_names(adata: ad.AnnData) -> np.ndarray:
    for column in ["feature_name", "gene_name", "gene_symbols"]:
        if column in adata.var.columns:
            return adata.var[column].astype(str).to_numpy()
    return np.asarray(adata.var_names.astype(str))


def load_rows(adata: ad.AnnData, row_indices: np.ndarray) -> sparse.csr_matrix:
    order = np.argsort(row_indices)
    sorted_rows = row_indices[order]
    x_sorted = sparse.csr_matrix(adata.X[sorted_rows, :])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return x_sorted[inverse, :].tocsr()


def qc_features(x: sparse.csr_matrix, mito_mask: np.ndarray) -> tuple[np.ndarray, List[str], List[Dict[str, float]]]:
    x = x.tocsr().astype(np.float32)
    totals = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    detected = np.diff(x.indptr).astype(np.float32)
    mito_counts = np.asarray(x[:, mito_mask].sum(axis=1)).ravel().astype(np.float32) if mito_mask.any() else np.zeros_like(totals)
    max_counts = np.asarray(x.max(axis=1).toarray()).ravel().astype(np.float32)
    mean_nonzero = np.divide(totals, detected, out=np.zeros_like(totals), where=detected > 0)
    pct_mito = np.divide(mito_counts, totals, out=np.zeros_like(totals), where=totals > 0)
    pct_max_gene = np.divide(max_counts, totals, out=np.zeros_like(totals), where=totals > 0)
    sparsity = 1.0 - detected / max(1, x.shape[1])

    features = np.stack(
        [
            np.log1p(totals),
            np.log1p(detected),
            pct_mito,
            pct_max_gene,
            np.log1p(mean_nonzero),
            sparsity,
        ],
        axis=1,
    ).astype(np.float32)
    names = [
        "log1p_total_counts",
        "log1p_detected_genes",
        "pct_mito_counts",
        "pct_top_gene_counts",
        "log1p_mean_nonzero_counts",
        "sparsity",
    ]
    rows = [
        {"feature": name, "mean": float(features[:, idx].mean()), "std": float(features[:, idx].std())}
        for idx, name in enumerate(names)
    ]
    return features, names, rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata_csv)
    if args.max_cells and args.max_cells > 0:
        metadata = metadata.head(args.max_cells).copy()
    if "cell_index" not in metadata.columns:
        raise ValueError("metadata CSV must contain cell_index")

    adata = ad.read_h5ad(args.h5ad, backed="r")
    row_indices = pd.to_numeric(metadata["cell_index"], errors="raise").astype(int).to_numpy()
    x = load_rows(adata, row_indices)
    names = gene_names(adata)
    mito_mask = np.char.upper(names.astype(str)).astype(str)
    mito_mask = np.asarray([name.startswith("MT-") or name.startswith("MT_") for name in mito_mask], dtype=bool)
    embeddings, feature_names, feature_summary = qc_features(x, mito_mask)

    metadata.to_csv(args.output_dir / "metadata.csv", index=False)
    np.savez_compressed(args.output_dir / "qc_count_depth_embeddings.npz", embeddings=embeddings)
    (args.output_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    summary = {
        "h5ad": str(args.h5ad),
        "metadata_csv": str(args.metadata_csv),
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "mitochondrial_gene_count": int(mito_mask.sum()),
        "feature_summary": feature_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
