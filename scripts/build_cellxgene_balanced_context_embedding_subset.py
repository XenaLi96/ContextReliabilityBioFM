#!/usr/bin/env python3
"""Build a label-context balanced subset of a CELLxGENE embedding matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-field", required=True)
    parser.add_argument("--context-values", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--max-cells-per-label-context", type=int, default=300)
    parser.add_argument("--min-cells-per-label-context", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args()


def read_embedding(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if "embeddings" not in loaded:
        raise KeyError(f"{path} does not contain key 'embeddings'")
    return np.asarray(loaded["embeddings"], dtype=np.float32)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata_csv)
    embeddings = read_embedding(args.embedding_file)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(f"metadata rows {len(metadata)} != embedding rows {embeddings.shape[0]}")

    keep = metadata[args.context_field].astype(str).isin(args.context_values) & metadata["label"].astype(str).isin(args.labels)
    candidate = metadata[keep].copy()
    selected_indices: List[int] = []
    group_rows: List[Dict[str, object]] = []
    for context_value in args.context_values:
        for label in args.labels:
            mask = (candidate[args.context_field].astype(str) == context_value) & (candidate["label"].astype(str) == label)
            indices = candidate.index[mask].to_numpy()
            n_available = int(len(indices))
            if n_available < args.min_cells_per_label_context:
                group_rows.append(
                    {
                        "context_value": context_value,
                        "label": label,
                        "n_available": n_available,
                        "n_selected": 0,
                        "skipped": True,
                    }
                )
                continue
            n_select = min(args.max_cells_per_label_context, n_available)
            sampled = rng.choice(indices, size=n_select, replace=False)
            selected_indices.extend(int(i) for i in sampled)
            group_rows.append(
                {
                    "context_value": context_value,
                    "label": label,
                    "n_available": n_available,
                    "n_selected": int(n_select),
                    "skipped": False,
                }
            )
    selected_indices = sorted(selected_indices)
    subset_meta = metadata.iloc[selected_indices].copy().reset_index(drop=True)
    subset_embeddings = embeddings[selected_indices].astype(np.float32)

    subset_meta.to_csv(args.output_dir / "metadata.csv", index=False)
    pd.DataFrame(group_rows).to_csv(args.output_dir / "group_counts.csv", index=False)
    np.savez_compressed(args.output_dir / "geneformer_v1_embeddings.npz", embeddings=subset_embeddings)
    summary = {
        "metadata_csv": str(args.metadata_csv),
        "embedding_file": str(args.embedding_file),
        "context_field": args.context_field,
        "context_values": args.context_values,
        "labels": args.labels,
        "n_cells": int(len(subset_meta)),
        "embedding_shape": [int(subset_embeddings.shape[0]), int(subset_embeddings.shape[1])],
        "max_cells_per_label_context": int(args.max_cells_per_label_context),
        "min_cells_per_label_context": int(args.min_cells_per_label_context),
        "group_counts": group_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
