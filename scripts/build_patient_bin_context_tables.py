#!/usr/bin/env python3
"""Build support-aware age/sex context-bin tables for the paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ADULT_AGE_BINS = {"adult_18_39", "adult_40_59", "adult_60_plus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_tables"))
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--min-label-context-n", type=int, default=20)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def clean_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt_num(value: object, digits: int = 3) -> str:
    value = clean_float(value)
    if math.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def balanced_accuracy(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    true = pd.Series(y_true).astype(str)
    pred = pd.Series(y_pred).astype(str)
    labels = sorted(true.unique())
    recalls: List[float] = []
    for label in labels:
        mask = true == label
        denom = int(mask.sum())
        if denom:
            recalls.append(float((pred[mask] == label).sum() / denom))
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def context_metrics(pred: pd.DataFrame, context_col: str) -> Dict[str, object]:
    groups = []
    for context_value, sub in pred.groupby(context_col, dropna=False):
        if len(sub) == 0:
            continue
        groups.append(
            {
                "context_value": str(context_value),
                "balanced_accuracy": balanced_accuracy(sub["true_label"], sub["pred_label"]),
                "n_cells": int(len(sub)),
                "n_donors": int(sub["donor_id"].astype(str).nunique()) if "donor_id" in sub else np.nan,
                "n_labels": int(sub["true_label"].astype(str).nunique()),
            }
        )
    if len(groups) < 2:
        return {}
    group_df = pd.DataFrame(groups)
    best = group_df.loc[group_df["balanced_accuracy"].astype(float).idxmax()]
    worst = group_df.loc[group_df["balanced_accuracy"].astype(float).idxmin()]
    overall = balanced_accuracy(pred["true_label"], pred["pred_label"])
    return {
        "overall_ba": overall,
        "best_context_value": best["context_value"],
        "best_ba": clean_float(best["balanced_accuracy"]),
        "worst_context_value": worst["context_value"],
        "worst_ba": clean_float(worst["balanced_accuracy"]),
        "gap": clean_float(best["balanced_accuracy"]) - clean_float(worst["balanced_accuracy"]),
        "n_cells": int(len(pred)),
        "n_donors": int(pred["donor_id"].astype(str).nunique()) if "donor_id" in pred else np.nan,
        "n_context_values": int(group_df["context_value"].nunique()),
        "bin_counts": "; ".join(
            f"{row.context_value}:{int(row.n_cells)}"
            for row in group_df.sort_values("context_value").itertuples(index=False)
        ),
    }


def label_context_support(pred: pd.DataFrame, context_col: str, min_n: int) -> Tuple[int, int, float]:
    table = pred.groupby(["true_label", context_col], dropna=False).size().reset_index(name="n")
    n_labels = pred["true_label"].astype(str).nunique()
    n_contexts = pred[context_col].astype(str).nunique()
    total = int(n_labels * n_contexts)
    supported = int((table["n"] >= min_n).sum())
    coverage = float(supported / total) if total else float("nan")
    return total, supported, coverage


def cramers_v(pred: pd.DataFrame, context_col: str) -> float:
    try:
        from scipy.stats import chi2_contingency
    except Exception:
        return float("nan")
    contingency = pd.crosstab(pred["true_label"].astype(str), pred[context_col].astype(str))
    if contingency.empty or min(contingency.shape) < 2:
        return float("nan")
    chi2 = chi2_contingency(contingency, correction=False)[0]
    n = contingency.values.sum()
    denom = n * (min(contingency.shape) - 1)
    return float(math.sqrt(chi2 / denom)) if denom else float("nan")


def bootstrap_ci(
    pred: pd.DataFrame,
    context_col: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    pred = pred.reset_index(drop=True)
    if "donor_id" not in pred.columns or pred["donor_id"].astype(str).nunique() < 2:
        return {}
    donor_ids = pred["donor_id"].astype(str).unique()
    donor_to_idx = {
        donor: idx.to_numpy()
        for donor, idx in pred.groupby(pred["donor_id"].astype(str)).groups.items()
    }
    values = {"overall_ba": [], "worst_ba": [], "gap": []}
    for _ in range(n_bootstrap):
        sampled = rng.choice(donor_ids, size=len(donor_ids), replace=True)
        idx = np.concatenate([donor_to_idx[donor] for donor in sampled])
        boot = pred.iloc[idx]
        metrics = context_metrics(boot, context_col)
        if not metrics:
            continue
        for key in values:
            values[key].append(clean_float(metrics[key]))
    out: Dict[str, Tuple[float, float]] = {}
    for key, vals in values.items():
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            out[key] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return out


def collect_single_cell(n_bootstrap: int, seed: int, min_n: int) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for pred_path in sorted(Path("data").glob("cellxgene_*_embedding_audit/scgpt_continual_assay_erm/predictions.csv")):
        tissue = pred_path.parent.parent.name
        tissue = tissue.replace("cellxgene_", "").replace("_embedding_audit", "")
        pred = read_csv(pred_path)
        if pred.empty:
            continue
        pred = pred[pred["method"].astype(str) == "erm"].copy()
        for context_col in ["age_group", "sex"]:
            if context_col not in pred.columns:
                continue
            sub = pred.dropna(subset=[context_col]).copy()
            sub[context_col] = sub[context_col].astype(str)
            variants = [(context_col, sub)]
            if context_col == "age_group":
                adult = sub[sub[context_col].isin(ADULT_AGE_BINS)].copy()
                variants.append(("age_adult_only", adult))
            for context_name, context_pred in variants:
                if context_pred[context_col].astype(str).nunique() < 2:
                    continue
                metrics = context_metrics(context_pred, context_col)
                if not metrics:
                    continue
                ci = bootstrap_ci(context_pred, context_col, n_bootstrap=n_bootstrap, rng=rng)
                total_pairs, supported_pairs, support_coverage = label_context_support(context_pred, context_col, min_n=min_n)
                rows.append(
                    {
                        "domain": "single_cell",
                        "dataset": f"CELLxGENE:{tissue}",
                        "embedding": "scGPT",
                        "context": context_name,
                        "split": "patient-CV donor bootstrap",
                        "overall_ba": metrics["overall_ba"],
                        "overall_ba_ci_low": ci.get("overall_ba", (float("nan"), float("nan")))[0],
                        "overall_ba_ci_high": ci.get("overall_ba", (float("nan"), float("nan")))[1],
                        "worst_ba": metrics["worst_ba"],
                        "worst_ba_ci_low": ci.get("worst_ba", (float("nan"), float("nan")))[0],
                        "worst_ba_ci_high": ci.get("worst_ba", (float("nan"), float("nan")))[1],
                        "gap": metrics["gap"],
                        "gap_ci_low": ci.get("gap", (float("nan"), float("nan")))[0],
                        "gap_ci_high": ci.get("gap", (float("nan"), float("nan")))[1],
                        "best_context_value": metrics["best_context_value"],
                        "worst_context_value": metrics["worst_context_value"],
                        "n_cells": metrics["n_cells"],
                        "n_donors": metrics["n_donors"],
                        "n_context_values": metrics["n_context_values"],
                        "label_context_pairs": total_pairs,
                        "supported_label_context_pairs_ge20": supported_pairs,
                        "support_coverage_ge20": support_coverage,
                        "label_context_cramers_v": cramers_v(context_pred, context_col),
                        "bin_counts": metrics["bin_counts"],
                        "source_file": str(pred_path),
                    }
                )
    return rows


def collect_tcga() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path, dataset in [
        (Path("data/tcga_luad_molecular_context_suite/combined_context_summary.csv"), "TCGA LUAD"),
        (Path("data/tcga_lgg_idh_context_suite/combined_context_summary.csv"), "TCGA LGG"),
    ]:
        df = read_csv(path)
        if df.empty:
            continue
        sub = df[df["context_field"].astype(str).str.contains("age_group|sex", regex=True)].copy()
        for _, row in sub.iterrows():
            rows.append(
                {
                    "domain": "pathology",
                    "dataset": dataset,
                    "embedding": row["model"],
                    "task": row["gene"],
                    "context": row["context_field"],
                    "split": "repeated patient/slide CV",
                    "overall_ba": clean_float(row["overall_ba"]),
                    "overall_ba_ci_low": clean_float(row["overall_ba_ci_low"]),
                    "overall_ba_ci_high": clean_float(row["overall_ba_ci_high"]),
                    "worst_ba": clean_float(row["worst_group_ba"]),
                    "worst_ba_ci_low": clean_float(row["worst_group_ba_ci_low"]),
                    "worst_ba_ci_high": clean_float(row["worst_group_ba_ci_high"]),
                    "gap": clean_float(row["best_minus_worst_ba"]),
                    "gap_ci_low": clean_float(row["best_minus_worst_ba_ci_low"]),
                    "gap_ci_high": clean_float(row["best_minus_worst_ba_ci_high"]),
                    "directional_gap": clean_float(row["directional_gap"]),
                    "directional_gap_ci_low": clean_float(row["directional_gap_ci_low"]),
                    "directional_gap_ci_high": clean_float(row["directional_gap_ci_high"]),
                    "n_rows_with_embeddings": int(clean_float(row["n_rows_with_embeddings"])),
                    "n_predictions": int(clean_float(row["n_predictions"])),
                    "source_file": str(path),
                }
            )
    return rows


def collect_geneformer_leave_one() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for context, run_name in [
        ("age_group", "geneformer_v1_age_group_clinical"),
        ("sex", "geneformer_v1_sex_clinical"),
    ]:
        path = Path("data/cellxgene_bone_marrow_embedding_mitigation_clinical_context") / run_name
        subgroup = read_csv(path / "subgroup_gaps.csv")
        metrics = read_csv(path / "subgroup_metrics.csv")
        leave = read_csv(path / "leave_one_context_gaps.csv")
        for split_name, df in [("patient-CV", subgroup), ("leave-one-context", leave)]:
            sub = df[
                (df["method"].astype(str) == "erm_mlp")
                & (df["context_field"].astype(str) == context)
                & (df["metric"].astype(str) == "balanced_accuracy")
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            overall = float("nan")
            if split_name == "patient-CV" and not metrics.empty:
                overall_row = metrics[
                    (metrics["method"].astype(str) == "erm_mlp")
                    & (metrics["split_type"].astype(str) == "patient_level_cv")
                    & (metrics["context_field"].astype(str) == "overall")
                ]
                if not overall_row.empty:
                    overall = clean_float(overall_row.iloc[0]["balanced_accuracy"])
            rows.append(
                {
                    "domain": "single_cell",
                    "dataset": "CELLxGENE:bone_marrow",
                    "embedding": "Geneformer",
                    "context": context,
                    "split": split_name,
                    "overall_ba": overall,
                    "worst_ba": clean_float(row["worst_value"]),
                    "gap": clean_float(row["gap"]),
                    "best_context_value": row["best_context_value"],
                    "worst_context_value": row["worst_context_value"],
                    "source_file": str(path),
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def summarize(single_cell_rows: List[Dict[str, object]], tcga_rows: List[Dict[str, object]], geneformer_rows: List[Dict[str, object]]) -> Dict[str, object]:
    sc = pd.DataFrame(single_cell_rows)
    tcga = pd.DataFrame(tcga_rows)
    gf = pd.DataFrame(geneformer_rows)
    summary: Dict[str, object] = {
        "single_cell_rows": len(single_cell_rows),
        "tcga_rows": len(tcga_rows),
        "geneformer_leave_one_rows": len(geneformer_rows),
    }
    if not sc.empty:
        for context in ["age_group", "age_adult_only", "sex"]:
            sub = sc[sc["context"] == context]
            if not sub.empty:
                max_row = sub.loc[sub["gap"].astype(float).idxmax()]
                summary[f"single_cell_{context}_max_gap"] = float(max_row["gap"])
                summary[f"single_cell_{context}_max_gap_dataset"] = str(max_row["dataset"])
                summary[f"single_cell_{context}_max_gap_worst_bin"] = str(max_row["worst_context_value"])
                summary[f"single_cell_{context}_median_gap"] = float(sub["gap"].astype(float).median())
    if not tcga.empty:
        for context_name, mask in [
            ("age", tcga["context"].astype(str).str.contains("age_group")),
            ("sex", tcga["context"].astype(str) == "sex"),
        ]:
            sub = tcga[mask]
            if not sub.empty:
                max_row = sub.loc[sub["gap"].astype(float).idxmax()]
                summary[f"tcga_{context_name}_max_gap"] = float(max_row["gap"])
                summary[f"tcga_{context_name}_max_gap_task"] = f"{max_row['dataset']} {max_row['task']} {max_row['embedding']}"
                summary[f"tcga_{context_name}_stable_gap_count_ci_low_gt_0"] = int((sub["gap_ci_low"].astype(float) > 0).sum())
                summary[f"tcga_{context_name}_n_rows"] = int(len(sub))
    if not gf.empty:
        for context in ["age_group", "sex"]:
            sub = gf[(gf["context"] == context) & (gf["split"] == "leave-one-context")]
            if not sub.empty:
                row = sub.iloc[0]
                summary[f"geneformer_leave_one_{context}_gap"] = float(row["gap"])
                summary[f"geneformer_leave_one_{context}_worst_bin"] = str(row["worst_context_value"])
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    single_cell_rows = collect_single_cell(args.n_bootstrap, args.seed, args.min_label_context_n)
    tcga_rows = collect_tcga()
    geneformer_rows = collect_geneformer_leave_one()
    write_csv(args.output_dir / "single_cell_demographic_context_bins.csv", single_cell_rows)
    write_csv(args.output_dir / "tcga_demographic_context_bins.csv", tcga_rows)
    write_csv(args.output_dir / "geneformer_bone_marrow_demographic_leave_one.csv", geneformer_rows)
    summary = summarize(single_cell_rows, tcga_rows, geneformer_rows)
    (args.output_dir / "demographic_context_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
