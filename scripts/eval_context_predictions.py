#!/usr/bin/env python3
"""Evaluate long-form predictions with ContextShift-Bio subgroup metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_GROUP_FIELDS = ["platform", "site", "organ", "disease", "study_id", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--spot-col", default="spot_id")
    parser.add_argument("--gene-col", default="gene")
    parser.add_argument("--true-col", default="y_true")
    parser.add_argument("--pred-col", default="y_pred")
    parser.add_argument("--group-fields", nargs="*", default=DEFAULT_GROUP_FIELDS)
    parser.add_argument("--nonzero-threshold", type=float, default=0.0)
    parser.add_argument("--min-pairs", type=int, default=3)
    return parser.parse_args()


def safe_corr(x: Iterable[float], y: Iterable[float], method: str, min_pairs: int) -> Optional[float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if len(x_arr) < min_pairs:
        return None
    if method == "spearman":
        x_arr = pd.Series(x_arr).rank(method="average").to_numpy()
        y_arr = pd.Series(y_arr).rank(method="average").to_numpy()
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def clean_metric(value: Optional[float]) -> object:
    if value is None or not math.isfinite(value):
        return "NA"
    return round(float(value), 6)


def load_joined(args: argparse.Namespace) -> pd.DataFrame:
    preds = pd.read_csv(args.predictions_csv)
    required = {args.sample_col, args.gene_col, args.true_col, args.pred_col}
    missing = sorted(required - set(preds.columns))
    if missing:
        raise SystemExit(f"Missing prediction columns: {missing}")

    manifest = pd.read_csv(args.manifest_csv)
    keep_cols = [args.sample_col] + [field for field in args.group_fields if field in manifest.columns]
    manifest = manifest[keep_cols].drop_duplicates(args.sample_col)
    return preds.merge(manifest, on=args.sample_col, how="left")


def gene_metrics(
    df: pd.DataFrame,
    args: argparse.Namespace,
    group_field: str,
    group_value: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for gene, gene_df in df.groupby(args.gene_col):
        pearson = safe_corr(gene_df[args.true_col], gene_df[args.pred_col], "pearson", args.min_pairs)
        spearman = safe_corr(gene_df[args.true_col], gene_df[args.pred_col], "spearman", args.min_pairs)
        rows.append(
            {
                "group_field": group_field,
                "group_value": group_value,
                "gene": gene,
                "n_pairs": len(gene_df),
                "pearson": clean_metric(pearson),
                "spearman": clean_metric(spearman),
            }
        )
    return rows


def cell_corr(df: pd.DataFrame, args: argparse.Namespace, method: str) -> Optional[float]:
    if args.spot_col not in df.columns:
        return None
    values: List[float] = []
    for _, spot_df in df.groupby([args.sample_col, args.spot_col]):
        corr = safe_corr(spot_df[args.true_col], spot_df[args.pred_col], method, args.min_pairs)
        if corr is not None:
            values.append(corr)
    return float(np.mean(values)) if values else None


def group_summary(
    df: pd.DataFrame,
    args: argparse.Namespace,
    group_field: str,
    group_value: str,
    gene_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    pearsons = [row["pearson"] for row in gene_rows if isinstance(row["pearson"], float)]
    spearmans = [row["spearman"] for row in gene_rows if isinstance(row["spearman"], float)]
    true_nonzero = np.asarray(df[args.true_col], dtype=float) > args.nonzero_threshold
    pred_nonzero = np.asarray(df[args.pred_col], dtype=float) > args.nonzero_threshold
    true_rate = float(np.mean(true_nonzero)) if len(true_nonzero) else float("nan")
    pred_rate = float(np.mean(pred_nonzero)) if len(pred_nonzero) else float("nan")
    return {
        "group_field": group_field,
        "group_value": group_value,
        "n_rows": len(df),
        "n_samples": int(df[args.sample_col].nunique()),
        "n_genes": int(df[args.gene_col].nunique()),
        "average_gene_pearson": clean_metric(float(np.mean(pearsons)) if pearsons else None),
        "average_gene_spearman": clean_metric(float(np.mean(spearmans)) if spearmans else None),
        "cell_pearson": clean_metric(cell_corr(df, args, "pearson")),
        "cell_spearman": clean_metric(cell_corr(df, args, "spearman")),
        "true_nonzero_rate": clean_metric(true_rate),
        "pred_nonzero_rate": clean_metric(pred_rate),
        "nonzero_calibration_gap": clean_metric(abs(pred_rate - true_rate)),
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    df = load_joined(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_gene_rows: List[Dict[str, object]] = []
    all_group_rows: List[Dict[str, object]] = []

    overall_gene_rows = gene_metrics(df, args, "ALL", "ALL")
    all_gene_rows.extend(overall_gene_rows)
    all_group_rows.append(group_summary(df, args, "ALL", "ALL", overall_gene_rows))

    for group_field in args.group_fields:
        if group_field not in df.columns:
            continue
        for group_value, group_df in df.groupby(group_field, dropna=False):
            value = "NA" if pd.isna(group_value) else str(group_value)
            rows = gene_metrics(group_df, args, group_field, value)
            all_gene_rows.extend(rows)
            all_group_rows.append(group_summary(group_df, args, group_field, value, rows))

    summary: Dict[str, object] = {
        "predictions_csv": str(args.predictions_csv),
        "manifest_csv": str(args.manifest_csv),
        "n_rows": len(df),
        "n_samples": int(df[args.sample_col].nunique()),
        "n_genes": int(df[args.gene_col].nunique()),
        "overall": all_group_rows[0],
        "worst_group_by_field": {},
    }
    for field in args.group_fields:
        candidates = [
            row for row in all_group_rows
            if row["group_field"] == field and isinstance(row["average_gene_pearson"], float)
        ]
        if candidates:
            worst = min(candidates, key=lambda row: row["average_gene_pearson"])
            summary["worst_group_by_field"][field] = worst

    write_csv(
        args.outdir / "gene_metrics.csv",
        all_gene_rows,
        ["group_field", "group_value", "gene", "n_pairs", "pearson", "spearman"],
    )
    write_csv(
        args.outdir / "group_metrics.csv",
        all_group_rows,
        [
            "group_field",
            "group_value",
            "n_rows",
            "n_samples",
            "n_genes",
            "average_gene_pearson",
            "average_gene_spearman",
            "cell_pearson",
            "cell_spearman",
            "true_nonzero_rate",
            "pred_nonzero_rate",
            "nonzero_calibration_gap",
        ],
    )
    with (args.outdir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
