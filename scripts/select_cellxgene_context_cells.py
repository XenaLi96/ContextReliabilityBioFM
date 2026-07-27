#!/usr/bin/env python3
"""Select CELLxGENE cells with label-context support for embedding audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_cellxgene_patient_context import (  # noqa: E402
    DEFAULT_CONTEXT_FIELDS,
    build_metadata_summary,
    choose_label_values,
    clean_string,
    donor_ok,
    label_context_audit,
    parse_age_group,
    sample_cells,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--label-values", nargs="*", default=[])
    parser.add_argument("--max-labels", type=int, default=6)
    parser.add_argument("--min-donors-per-label", type=int, default=3)
    parser.add_argument("--min-cells-per-label", type=int, default=1000)
    parser.add_argument("--max-cells-per-label", type=int, default=12000)
    parser.add_argument("--max-cells-per-donor-label", type=int, default=200)
    parser.add_argument("--context-fields", nargs="*", default=[*DEFAULT_CONTEXT_FIELDS, "disease"])
    parser.add_argument("--include-unknown-sex", action="store_true")
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs.copy()
    obs["cell_index"] = np.arange(adata.n_obs, dtype=int)
    for column in obs.columns:
        if column == "cell_index":
            continue
        obs[column] = obs[column].astype(object).map(clean_string).astype(object)
    if "donor_id" not in obs.columns:
        raise SystemExit("Input h5ad has no donor_id column.")
    if args.label_column not in obs.columns:
        raise SystemExit(f"Input h5ad has no label column: {args.label_column}")
    obs["age_group"] = obs.get("development_stage", pd.Series(["unknown"] * len(obs))).map(parse_age_group)

    keep = obs["donor_id"].map(donor_ok)
    keep &= obs[args.label_column].map(clean_string) != "unknown"
    if "sex" in obs.columns and not args.include_unknown_sex:
        keep &= obs["sex"].isin(["female", "male"])
    filtered = obs.loc[keep].copy()

    label_values = choose_label_values(
        filtered,
        args.label_column,
        args.label_values,
        args.max_labels,
        args.min_donors_per_label,
        args.min_cells_per_label,
    )
    if len(label_values) < 2:
        raise SystemExit(f"Need at least two labels after filtering; got {label_values}")
    filtered = filtered[filtered[args.label_column].isin(label_values)].copy()
    sampled = sample_cells(
        filtered,
        args.label_column,
        args.max_cells_per_donor_label,
        args.max_cells_per_label,
        args.seed,
    )
    sampled["cell_index"] = pd.to_numeric(sampled["cell_index"], errors="raise").astype(int)
    sampled = sampled.sort_values("cell_index").reset_index(drop=True)

    selected_columns = ["cell_index", "donor_id", args.label_column]
    selected_columns.extend([field for field in args.context_fields if field in sampled.columns])
    selected = sampled[selected_columns].copy().rename(columns={args.label_column: "label"})
    selected.to_csv(args.output_dir / "selected_cells.csv", index=False)

    association_rows, context_count_rows = label_context_audit(sampled, args.label_column, args.context_fields)
    write_csv(args.output_dir / "label_context_association.csv", association_rows)
    write_csv(args.output_dir / "label_context_counts.csv", context_count_rows)

    summary = {
        "h5ad": str(args.h5ad),
        "label_column": args.label_column,
        "label_values": label_values,
        "seed": int(args.seed),
        "max_cells_per_label": int(args.max_cells_per_label),
        "max_cells_per_donor_label": int(args.max_cells_per_donor_label),
        "filtered": build_metadata_summary(filtered, args.label_column, args.context_fields),
        "sampled": build_metadata_summary(sampled, args.label_column, args.context_fields),
        "outputs": {
            "selected_cells": str(args.output_dir / "selected_cells.csv"),
            "label_context_association": str(args.output_dir / "label_context_association.csv"),
            "label_context_counts": str(args.output_dir / "label_context_counts.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
