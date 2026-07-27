#!/usr/bin/env python3
"""Extract Geneformer-V1-10M frozen cell embeddings for selected CELLxGENE cells.

This avoids depending on the Geneformer Python package. It downloads or reads:
- ctheodoris/Geneformer/Geneformer-V1-10M
- geneformer/gene_dictionaries_30m token and median dictionaries

Tokenization follows Geneformer's V1 rank-value encoding: counts are normalized
by total cell counts, scaled to 10,000, divided by the Genecorpus gene median,
then genes are sorted by the resulting value. V1 uses no CLS/EOS special token,
so embeddings are mean-pooled over non-padding token hidden states.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--selected-cells-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--read-chunk-size", type=int, default=128)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--model-input-size", type=int, default=2048)
    parser.add_argument("--target-sum", type=float, default=10_000.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def ensure_geneformer_files(model_root: Path | None, cache_dir: Path) -> Path:
    if model_root is not None:
        return model_root
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id="ctheodoris/Geneformer",
            cache_dir=str(cache_dir),
            allow_patterns=[
                "Geneformer-V1-10M/*",
                "geneformer/gene_dictionaries_30m/*",
            ],
        )
    )


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def normalize_ensembl(value: object) -> str:
    text = str(value).strip().upper()
    return text.split(".")[0]


def build_gene_index(
    adata: ad.AnnData,
    token_dict: Dict[str, int],
    median_dict: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if "feature_id" in adata.var.columns:
        gene_ids = adata.var["feature_id"].map(normalize_ensembl).to_numpy()
    else:
        gene_ids = pd.Series(adata.var.index).map(normalize_ensembl).to_numpy()
    keep_positions: List[int] = []
    tokens: List[int] = []
    medians: List[float] = []
    for idx, gene_id in enumerate(gene_ids):
        if gene_id in token_dict and gene_id in median_dict:
            keep_positions.append(idx)
            tokens.append(int(token_dict[gene_id]))
            medians.append(float(median_dict[gene_id]))
    if not keep_positions:
        raise ValueError("No genes overlap the Geneformer token and median dictionaries.")
    return (
        np.asarray(keep_positions, dtype=int),
        np.asarray(tokens, dtype=np.int64),
        np.asarray(medians, dtype=np.float32),
        int(len(set(gene_ids))),
    )


def load_expression_rows(adata: ad.AnnData, row_indices: np.ndarray) -> sparse.csr_matrix:
    row_indices = np.asarray(row_indices, dtype=int)
    order = np.argsort(row_indices)
    sorted_rows = row_indices[order]
    x_sorted = sparse.csr_matrix(adata.X[sorted_rows, :])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return x_sorted[inverse, :]


def tokenize_chunk(
    x_chunk: sparse.csr_matrix,
    gene_positions: np.ndarray,
    gene_tokens: np.ndarray,
    gene_medians: np.ndarray,
    model_input_size: int,
    target_sum: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_chunk = sparse.csr_matrix(x_chunk)
    n_counts = np.asarray(x_chunk.sum(axis=1)).ravel().astype(np.float32)
    n_counts[n_counts <= 0] = 1.0
    x_gene = sparse.csr_matrix(x_chunk[:, gene_positions])

    input_ids = np.zeros((x_gene.shape[0], model_input_size), dtype=np.int64)
    attention_mask = np.zeros((x_gene.shape[0], model_input_size), dtype=np.int64)
    lengths = np.zeros(x_gene.shape[0], dtype=np.int32)

    for row_i in range(x_gene.shape[0]):
        row = x_gene.getrow(row_i)
        if row.nnz == 0:
            continue
        scaled = row.data.astype(np.float32) / n_counts[row_i] * target_sum
        scaled = scaled / gene_medians[row.indices]
        order = np.argsort(-scaled)
        token_ids = gene_tokens[row.indices][order][:model_input_size]
        length = len(token_ids)
        input_ids[row_i, :length] = token_ids
        attention_mask[row_i, :length] = 1
        lengths[row_i] = length
    return input_ids, attention_mask, lengths


def batched_indices(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / denom


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = ensure_geneformer_files(args.model_root, args.hf_cache_dir)
    model_dir = repo_root / "Geneformer-V1-10M"
    dict_dir = repo_root / "geneformer" / "gene_dictionaries_30m"
    token_dict = load_pickle(dict_dir / "token_dictionary_gc30M.pkl")
    median_dict = load_pickle(dict_dir / "gene_median_dictionary_gc30M.pkl")

    selected = pd.read_csv(args.selected_cells_csv)
    if args.max_cells and args.max_cells > 0:
        selected = selected.head(args.max_cells).copy()
    row_indices = pd.to_numeric(selected["cell_index"], errors="raise").astype(int).to_numpy()

    adata = ad.read_h5ad(args.h5ad, backed="r")
    gene_positions, gene_tokens, gene_medians, n_unique_gene_ids = build_gene_index(adata, token_dict, median_dict)
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=False)
    model.eval().to(args.device)

    embeddings: List[np.ndarray] = []
    token_lengths: List[np.ndarray] = []
    t0 = time.time()
    total_chunks = int(np.ceil(len(row_indices) / args.read_chunk_size))
    for chunk_i, (read_start, read_end) in enumerate(batched_indices(len(row_indices), args.read_chunk_size), start=1):
        print(
            f"[Geneformer] chunk {chunk_i}/{total_chunks}: cells {read_start}:{read_end}",
            flush=True,
        )
        x_chunk = load_expression_rows(adata, row_indices[read_start:read_end])
        input_ids, attention_mask, lengths = tokenize_chunk(
            x_chunk,
            gene_positions=gene_positions,
            gene_tokens=gene_tokens,
            gene_medians=gene_medians,
            model_input_size=args.model_input_size,
            target_sum=args.target_sum,
        )
        token_lengths.append(lengths)
        chunk_embs: List[np.ndarray] = []
        with torch.inference_mode():
            for batch_start, batch_end in batched_indices(input_ids.shape[0], args.batch_size):
                ids = torch.tensor(input_ids[batch_start:batch_end], dtype=torch.long, device=args.device)
                mask = torch.tensor(attention_mask[batch_start:batch_end], dtype=torch.long, device=args.device)
                out = model(input_ids=ids, attention_mask=mask)
                pooled = mean_pool(out.last_hidden_state, mask)
                chunk_embs.append(pooled.detach().cpu().numpy().astype(np.float32))
        embeddings.append(np.concatenate(chunk_embs, axis=0))
        done = read_end
        elapsed = max(time.time() - t0, 1e-6)
        rate = done / elapsed
        remaining = (len(row_indices) - done) / rate if rate > 0 else float("nan")
        print(
            f"[Geneformer] completed {done}/{len(row_indices)} cells; "
            f"{rate:.2f} cells/s; eta {remaining / 60:.1f} min",
            flush=True,
        )

    emb = np.concatenate(embeddings, axis=0)
    lengths = np.concatenate(token_lengths, axis=0)
    selected.to_csv(args.output_dir / "metadata.csv", index=False)
    np.savez_compressed(args.output_dir / "geneformer_v1_embeddings.npz", embeddings=emb)
    pd.DataFrame({"cell_index": selected["cell_index"].astype(int), "token_length": lengths}).to_csv(
        args.output_dir / "token_lengths.csv", index=False
    )
    summary = {
        "model": "ctheodoris/Geneformer/Geneformer-V1-10M",
        "repo_root": str(repo_root),
        "h5ad": str(args.h5ad),
        "selected_cells_csv": str(args.selected_cells_csv),
        "n_cells": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
        "n_h5ad_unique_gene_ids": n_unique_gene_ids,
        "n_geneformer_overlap_genes": int(len(gene_positions)),
        "mean_token_length": float(np.mean(lengths)),
        "median_token_length": float(np.median(lengths)),
        "device": args.device,
        "outputs": {
            "metadata": str(args.output_dir / "metadata.csv"),
            "embeddings": str(args.output_dir / "geneformer_v1_embeddings.npz"),
            "token_lengths": str(args.output_dir / "token_lengths.csv"),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
