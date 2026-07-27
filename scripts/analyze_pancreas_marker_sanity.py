#!/usr/bin/env python3
"""Marker sanity check for acinar cells miscalled as ductal in pancreas."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ACINAR = ["PRSS1", "PRSS2", "REG1A", "AMY2A"]
DUCTAL = ["KRT8", "KRT18", "KRT19", "KRT7"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=Path("data/raw/cellxgene/pancreas.h5ad"),
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path(
            "data/cellxgene_support_calibrated_formal/pancreas_assay/"
            "scgpt_continual/assay"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pancreas_marker_sanity"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        args.prediction_root.glob(
            "seed_*/*/leave_one_context_predictions.csv"
        )
    )
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame.loc[
            frame["method"].eq("erm_mlp")
            & frame["assay"].astype(str).eq("microwell-seq")
            & frame["true_label"].astype(str).eq("acinar cell")
        ].copy()
        frame["seed"] = int(path.parts[-3].split("_")[-1])
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["prediction_group"] = np.where(
        predictions["pred_label"].astype(str).str.contains("ductal", case=False),
        "acinar_miscalled_ductal",
        np.where(
            predictions["pred_label"].astype(str).str.contains("acinar", case=False),
            "acinar_retained",
            "acinar_called_other",
        ),
    )

    adata = ad.read_h5ad(args.h5ad, backed="r")
    gene_to_index = {
        str(gene): int(index)
        for index, gene in enumerate(adata.var["feature_name"].astype(str))
    }
    genes = ACINAR + DUCTAL
    missing = [gene for gene in genes if gene not in gene_to_index]
    if missing:
        raise ValueError(f"Missing marker genes: {missing}")
    unique_cells = np.sort(predictions["cell_index"].astype(int).unique())
    x_all = adata.X[unique_cells, :]
    library_size = np.asarray(x_all.sum(axis=1)).ravel()
    x = np.asarray(
        adata.X[unique_cells, [gene_to_index[gene] for gene in genes]].todense()
    )
    normalized = np.log1p(1e4 * x / np.maximum(library_size[:, None], 1.0))
    marker = pd.DataFrame(normalized, columns=genes)
    marker["cell_index"] = unique_cells
    marker["acinar_module"] = marker[ACINAR].mean(axis=1)
    marker["ductal_module"] = marker[DUCTAL].mean(axis=1)
    marker["acinar_minus_ductal"] = (
        marker["acinar_module"] - marker["ductal_module"]
    )
    joined = predictions.merge(marker, on="cell_index", how="left", validate="many_to_one")

    per_donor = (
        joined.groupby(["seed", "donor_id", "prediction_group"], as_index=False)
        .agg(
            n_cells=("cell_index", "size"),
            acinar_module=("acinar_module", "mean"),
            ductal_module=("ductal_module", "mean"),
            acinar_minus_ductal=("acinar_minus_ductal", "mean"),
        )
    )
    summary = (
        per_donor.groupby("prediction_group", as_index=False)
        .agg(
            n_donor_seed=("donor_id", "size"),
            median_cells=("n_cells", "median"),
            median_acinar_module=("acinar_module", "median"),
            median_ductal_module=("ductal_module", "median"),
            median_acinar_minus_ductal=("acinar_minus_ductal", "median"),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output_dir / "cell_marker_scores.csv", index=False)
    per_donor.to_csv(args.output_dir / "donor_marker_scores.csv", index=False)
    summary.to_csv(args.output_dir / "marker_sanity_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
