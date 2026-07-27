#!/usr/bin/env python3
"""Build and validate the complete support-gated cross-tissue formal matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


MODELS = ["geneformer_v1", "scgpt_continual", "scvi_style_vae"]
METHODS = ["erm_mlp", "label_context_reweight", "sca_lite", "group_dro"]
SPLITS = ["patient_level_cv", "leave_one_context"]
CONTEXT_SUFFIX = {"assay": "assay", "dataset_id": "dataset"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/cellxgene_support_calibrated_formal"),
    )
    parser.add_argument(
        "--eligibility-csv",
        type=Path,
        default=Path("data/tissue_mitigation_replication_selection/all_eligible_tasks.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cellxgene_cross_tissue_formal_matrix"),
    )
    parser.add_argument("--expected-seeds", type=int, default=5)
    return parser.parse_args()


def task_name(tissue: str, context: str) -> str:
    try:
        suffix = CONTEXT_SUFFIX[context]
    except KeyError as exc:
        raise ValueError(f"Unsupported eligible context: {context}") from exc
    return f"{tissue}_{suffix}"


def load_matrix(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligibility = pd.read_csv(args.eligibility_csv)
    if "eligible" in eligibility.columns:
        eligibility = eligibility.loc[eligibility["eligible"].astype(bool)].copy()
    eligibility = eligibility.drop_duplicates(["tissue", "context"])
    metric_rows: List[Dict[str, object]] = []
    delta_rows: List[Dict[str, object]] = []

    for eligible in eligibility.to_dict(orient="records"):
        tissue = str(eligible["tissue"])
        context = str(eligible["context"])
        task = task_name(tissue, context)
        for model in MODELS:
            run_root = args.input_root / task / model / context
            aggregate_path = run_root / "summary" / "aggregate_gaps.csv"
            delta_path = run_root / "paired_vs_erm" / "paired_delta_ci.csv"
            if not aggregate_path.is_file():
                raise FileNotFoundError(f"Missing formal aggregate: {aggregate_path}")
            if not delta_path.is_file():
                raise FileNotFoundError(f"Missing paired CI: {delta_path}")

            aggregate = pd.read_csv(aggregate_path)
            aggregate = aggregate.loc[
                aggregate["method"].isin(METHODS)
                & aggregate["split_type"].isin(SPLITS)
                & aggregate["summary_metric"].isin(["gap", "worst_value"])
            ].copy()
            pivot = aggregate.pivot_table(
                index=["split_type", "method", "n_seeds"],
                columns="summary_metric",
                values=["mean", "std"],
                aggfunc="first",
            )
            pivot.columns = [f"{metric}_{stat}" for stat, metric in pivot.columns]
            pivot = pivot.reset_index()
            for row in pivot.to_dict(orient="records"):
                metric_rows.append(
                    {
                        "tissue": tissue,
                        "task": task,
                        "model": model,
                        "context_field": context,
                        "support_coverage": float(eligible["support_coverage"]),
                        "erm_audit_leave_one_gap": float(eligible["erm_leave_one_gap"]),
                        **row,
                        "source_csv": str(aggregate_path),
                    }
                )

            deltas = pd.read_csv(delta_path)
            deltas = deltas.loc[
                deltas["method"].isin(METHODS)
                & deltas["split_type"].isin(SPLITS)
                & deltas["delta_metric"].isin(["gap_delta_vs_erm", "worst_delta_vs_erm"])
            ].copy()
            for row in deltas.to_dict(orient="records"):
                delta_rows.append(
                    {
                        "tissue": tissue,
                        "task": task,
                        "model": model,
                        "context_field": context,
                        **row,
                        "source_csv": str(delta_path),
                    }
                )

    return pd.DataFrame.from_records(metric_rows), pd.DataFrame.from_records(delta_rows)


def validate(matrix: pd.DataFrame, deltas: pd.DataFrame, expected_seeds: int) -> Dict[str, int]:
    expected_pairs = 5
    expected_matrix_rows = expected_pairs * len(MODELS) * len(SPLITS) * len(METHODS)
    expected_delta_rows = expected_matrix_rows * 2
    if len(matrix) != expected_matrix_rows:
        raise ValueError(f"Expected {expected_matrix_rows} matrix rows, found {len(matrix)}")
    if len(deltas) != expected_delta_rows:
        raise ValueError(f"Expected {expected_delta_rows} paired-delta rows, found {len(deltas)}")
    if set(matrix["n_seeds"].astype(int)) != {expected_seeds}:
        raise ValueError(f"Matrix contains non-{expected_seeds}-seed rows: {sorted(matrix['n_seeds'].unique())}")
    if set(deltas["n_seeds"].astype(int)) != {expected_seeds}:
        raise ValueError(f"Delta table contains non-{expected_seeds}-seed rows: {sorted(deltas['n_seeds'].unique())}")
    key_columns = ["tissue", "model", "context_field", "split_type", "method"]
    if bool(matrix.duplicated(key_columns).any()):
        raise ValueError("Duplicate task/model/context/split/method rows in formal matrix")
    return {
        "n_tissue_context_pairs": int(matrix[["tissue", "context_field"]].drop_duplicates().shape[0]),
        "n_models": int(matrix["model"].nunique()),
        "n_methods": int(matrix["method"].nunique()),
        "n_splits": int(matrix["split_type"].nunique()),
        "n_matrix_rows": int(len(matrix)),
        "n_paired_delta_rows": int(len(deltas)),
        "n_seeds_per_row": int(expected_seeds),
    }


def main() -> None:
    args = parse_args()
    matrix, deltas = load_matrix(args)
    counts = validate(matrix, deltas, args.expected_seeds)
    matrix = matrix.sort_values(
        ["split_type", "context_field", "tissue", "model", "method"]
    ).reset_index(drop=True)
    deltas = deltas.sort_values(
        ["split_type", "context_field", "tissue", "model", "method", "delta_metric"]
    ).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output_dir / "formal_matrix.csv", index=False)
    deltas.to_csv(args.output_dir / "paired_delta_ci.csv", index=False)
    summary = {
        "input_root": str(args.input_root),
        "eligibility_csv": str(args.eligibility_csv),
        **counts,
        "outputs": {
            "matrix": str(args.output_dir / "formal_matrix.csv"),
            "paired_delta_ci": str(args.output_dir / "paired_delta_ci.csv"),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
