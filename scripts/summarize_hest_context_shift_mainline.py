#!/usr/bin/env python3
"""Summarize HEST-51 site/platform context shift and mitigation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DEFAULT_BIAS_ROOT = Path("outputs/hest/bias_suite")
DEFAULT_MITIGATION_ROOT = Path("outputs/hest/context_mitigation")
DEFAULT_OUTDIR = Path("data/hest51_context_shift_mainline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias-root", type=Path, default=DEFAULT_BIAS_ROOT)
    parser.add_argument("--mitigation-root", type=Path, default=DEFAULT_MITIGATION_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def run_parts(run_dir: Path) -> tuple[str, str]:
    name = run_dir.name
    if "_" not in name:
        return name, "unknown"
    model, split = name.split("_", 1)
    return model, split


def as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def collect_representation(bias_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in bias_root.iterdir() if path.is_dir()):
        model, split = run_parts(run_dir)
        suite = run_dir / "bias_suite"
        probe_path = suite / "sample_cv_probe_results.csv"
        knn_path = suite / "sample_knn_enrichment.csv"
        if not probe_path.exists():
            continue
        probe = pd.read_csv(probe_path)
        knn = pd.read_csv(knn_path) if knn_path.exists() else pd.DataFrame()
        knn_by_field = {
            str(row["field"]): row.to_dict()
            for _, row in knn.iterrows()
            if str(row.get("status", "")) == "ok"
        }
        for _, row in probe.iterrows():
            field = str(row.get("field", ""))
            payload = {
                "model": model,
                "split_strategy": split,
                "field": field,
                "status": row.get("status", ""),
                "n_samples": row.get("n_samples", ""),
                "n_classes": row.get("n_classes", ""),
                "probe_balanced_accuracy": row.get("balanced_accuracy_mean", ""),
                "probe_balanced_accuracy_std": row.get("balanced_accuracy_std", ""),
                "class_counts": row.get("class_counts", ""),
                "knn_enrichment_over_baseline": "",
                "knn_mean_same_group": "",
                "knn_majority_baseline": "",
            }
            if field in knn_by_field:
                knn_row = knn_by_field[field]
                payload["knn_enrichment_over_baseline"] = knn_row.get("enrichment_over_baseline", "")
                payload["knn_mean_same_group"] = knn_row.get("mean_knn_same_group", "")
                payload["knn_majority_baseline"] = knn_row.get("majority_class_baseline", "")
            rows.append(payload)
    return pd.DataFrame(rows)


def collect_prediction_gaps(bias_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in bias_root.iterdir() if path.is_dir()):
        model, split = run_parts(run_dir)
        summary_path = run_dir / "prediction_audit" / "metrics_summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        overall = summary.get("overall", {})
        overall_pearson = overall.get("average_gene_pearson", "")
        for field, worst in sorted(summary.get("worst_group_by_field", {}).items()):
            worst_pearson = worst.get("average_gene_pearson", "")
            gap = ""
            overall_float = as_float(overall_pearson)
            worst_float = as_float(worst_pearson)
            if overall_float is not None and worst_float is not None:
                gap = overall_float - worst_float
            rows.append(
                {
                    "model": model,
                    "split_strategy": split,
                    "context_field": field,
                    "overall_average_gene_pearson": overall_pearson,
                    "worst_group": worst.get("group_value", ""),
                    "worst_average_gene_pearson": worst_pearson,
                    "overall_minus_worst": gap,
                    "worst_n_samples": worst.get("n_samples", ""),
                    "worst_n_genes": worst.get("n_genes", ""),
                    "worst_nonzero_calibration_gap": worst.get("nonzero_calibration_gap", ""),
                }
            )
    return pd.DataFrame(rows)


def collect_mitigation(mitigation_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in mitigation_root.iterdir() if path.is_dir()):
        model, split = run_parts(run_dir)
        path = run_dir / "method_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        erms = {
            str(row["context_field"]): row.to_dict()
            for _, row in df[df["method"].astype(str) == "erm"].iterrows()
        }
        for _, row in df.iterrows():
            context = str(row["context_field"])
            erm = erms.get(context, {})
            overall = as_float(row.get("overall_nontrain_average_gene_pearson"))
            worst = as_float(row.get("worst_average_gene_pearson"))
            gap = as_float(row.get("best_minus_worst"))
            erm_overall = as_float(erm.get("overall_nontrain_average_gene_pearson"))
            erm_worst = as_float(erm.get("worst_average_gene_pearson"))
            erm_gap = as_float(erm.get("best_minus_worst"))
            rows.append(
                {
                    "model": model,
                    "split_strategy": split,
                    "method": row.get("method", ""),
                    "context_field": context,
                    "overall_nontrain_average_gene_pearson": row.get("overall_nontrain_average_gene_pearson", ""),
                    "worst_group": row.get("worst_group", ""),
                    "worst_average_gene_pearson": row.get("worst_average_gene_pearson", ""),
                    "best_group": row.get("best_group", ""),
                    "best_minus_worst": row.get("best_minus_worst", ""),
                    "overall_delta_vs_erm": "" if overall is None or erm_overall is None else overall - erm_overall,
                    "worst_delta_vs_erm": "" if worst is None or erm_worst is None else worst - erm_worst,
                    "gap_delta_vs_erm": "" if gap is None or erm_gap is None else gap - erm_gap,
                    "erm_worst_group": erm.get("worst_group", ""),
                    "erm_worst_average_gene_pearson": erm.get("worst_average_gene_pearson", ""),
                    "erm_best_minus_worst": erm.get("best_minus_worst", ""),
                }
            )
    return pd.DataFrame(rows)


def headline(rep: pd.DataFrame, pred: pd.DataFrame, mit: pd.DataFrame) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    key_fields = {"platform", "site", "study_id", "tissue_processing", "split"}
    rep_ok = rep[rep["status"].astype(str) == "ok"].copy() if not rep.empty else pd.DataFrame()
    if not rep_ok.empty:
        rep_ok["probe_ba_num"] = pd.to_numeric(rep_ok["probe_balanced_accuracy"], errors="coerce")
        payload["top_context_probe"] = (
            rep_ok[rep_ok["field"].isin(key_fields)]
            .sort_values("probe_ba_num", ascending=False)
            .head(8)
            .drop(columns=["probe_ba_num"], errors="ignore")
            .to_dict(orient="records")
        )

    if not pred.empty:
        focus = pred[pred["context_field"].isin(["split", "platform", "site", "study_id", "tissue_processing"])]
        payload["downstream_worst_contexts"] = (
            focus.sort_values("overall_minus_worst", ascending=False)
            .head(12)
            .to_dict(orient="records")
        )

    if not mit.empty:
        mit_focus = mit[mit["context_field"].isin(["split", "platform", "site", "study_id", "tissue_processing"])]
        mit_focus = mit_focus[mit_focus["method"].astype(str) != "erm"].copy()
        if not mit_focus.empty:
            mit_focus["worst_delta_num"] = pd.to_numeric(mit_focus["worst_delta_vs_erm"], errors="coerce")
            mit_focus["gap_delta_num"] = pd.to_numeric(mit_focus["gap_delta_vs_erm"], errors="coerce")
            payload["best_worst_group_improvements"] = (
                mit_focus.sort_values("worst_delta_num", ascending=False)
                .head(12)
                .drop(columns=["worst_delta_num", "gap_delta_num"], errors="ignore")
                .to_dict(orient="records")
            )
            payload["best_gap_reductions"] = (
                mit_focus.sort_values("gap_delta_num", ascending=True)
                .head(12)
                .drop(columns=["worst_delta_num", "gap_delta_num"], errors="ignore")
                .to_dict(orient="records")
            )
    return payload


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rep = collect_representation(args.bias_root)
    pred = collect_prediction_gaps(args.bias_root)
    mit = collect_mitigation(args.mitigation_root)

    rep.to_csv(args.outdir / "representation_context_leakage.csv", index=False)
    pred.to_csv(args.outdir / "downstream_context_gaps.csv", index=False)
    mit.to_csv(args.outdir / "mitigation_deltas.csv", index=False)
    head = headline(rep, pred, mit)
    head.update(
        {
            "bias_root": str(args.bias_root),
            "mitigation_root": str(args.mitigation_root),
            "outputs": {
                "representation_context_leakage_csv": str(args.outdir / "representation_context_leakage.csv"),
                "downstream_context_gaps_csv": str(args.outdir / "downstream_context_gaps.csv"),
                "mitigation_deltas_csv": str(args.outdir / "mitigation_deltas.csv"),
            },
        }
    )
    with (args.outdir / "headline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(head, handle, indent=2, sort_keys=True)
    print(json.dumps({"outdir": str(args.outdir), "n_representation_rows": len(rep), "n_prediction_gap_rows": len(pred), "n_mitigation_rows": len(mit)}, indent=2))


if __name__ == "__main__":
    main()
