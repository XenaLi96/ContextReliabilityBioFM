#!/usr/bin/env python3
"""Build the appendix table for donor-level biological-consequence metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


METHOD_ORDER = {"ERM": 0, "LC-Reweight": 1, "SCA-Align": 2, "GroupDRO": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/donor_abundance_consequence"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/paper_tables/donor_abundance_consequence_table.csv"))
    parser.add_argument("--output-tex", type=Path, default=Path("tables/donor_abundance_rows.tex"))
    return parser.parse_args()


def task_label(row: pd.Series) -> str:
    model = "Geneformer" if "geneformer" in str(row["model"]).lower() else "scGPT"
    task = str(row["task"])
    tissue = "bone marrow" if task == "bone_marrow" or task.startswith("bone_marrow") else task.replace("_assay", "").replace("_dataset", "").replace("_", " ")
    context = "dataset" if str(row["context_field"]) == "dataset_id" else str(row["context_field"])
    return f"{model} / {tissue} / {context}"


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.input_dir / "aggregate_summary.csv")
    summary = summary.loc[summary["split_type"] == "leave_one_context"].copy()
    summary["task_label"] = summary.apply(task_label, axis=1)
    disease_path = args.input_dir / "differential_abundance_direction_summary.csv"
    if disease_path.exists() and disease_path.stat().st_size > 0:
        disease = pd.read_csv(disease_path)
        if not disease.empty:
            keys = ["task", "model", "context_field", "split_type", "method", "method_label"]
            disease = disease.groupby(keys, sort=False)["direction_match_rate"].mean().reset_index()
            summary = summary.merge(disease, on=keys, how="left")
    if "direction_match_rate" not in summary.columns:
        summary["direction_match_rate"] = float("nan")
    summary["method_order"] = summary["method_label"].map(METHOD_ORDER)
    summary = summary.sort_values(["task_label", "method_order"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    lines: List[str] = []
    for row in summary.itertuples(index=False):
        direction = "--" if pd.isna(row.direction_match_rate) else f"{row.direction_match_rate:.2f}"
        lines.append(
            f"{row.task_label} & {row.method_label} & "
            f"{row.donor_abundance_mae_mean:.3f} $\\pm$ {row.donor_abundance_mae_std:.3f} & "
            f"{row.mean_label_spearman_across_donors_mean:.3f} & "
            f"{row.mean_donor_rank_stability_mean:.3f} & "
            f"{row.worst_context_abundance_mae_mean:.3f} $\\pm$ {row.worst_context_abundance_mae_std:.3f} & "
            f"{direction} \\\\"
        )
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_csv} ({len(summary)} rows)")
    print(f"wrote {args.output_tex}")


if __name__ == "__main__":
    main()
