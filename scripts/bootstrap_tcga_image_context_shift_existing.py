#!/usr/bin/env python3
"""Bootstrap CIs from existing TCGA image context-shift leave-one predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("data/tcga_image_context_shift"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tcga_image_context_shift_bootstrap_ci"))
    parser.add_argument("--context-fields", nargs="*", default=["site", "primary_diagnosis"])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--min-subgroup-n", type=int, default=10)
    parser.add_argument("--methods", nargs="*", default=["erm"])
    parser.add_argument("--metrics", nargs="*", default=["balanced_accuracy"])
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_values(y_true, y_pred) -> Dict[str, float]:
    true = np.asarray(y_true, dtype=str)
    pred = np.asarray(y_pred, dtype=str)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
    }


def context_gap(df: pd.DataFrame, context_field: str, metric: str, min_n: int) -> Dict[str, object] | None:
    rows = []
    for value, group in df.groupby(context_field, sort=True):
        if len(group) < min_n or group["true_label"].astype(str).nunique() < 2:
            continue
        metrics = metric_values(group["true_label"], group["pred_label"])
        rows.append({"context_value": str(value), "n": int(len(group)), **metrics})
    if len(rows) < 2:
        return None
    best = max(rows, key=lambda row: float(row[metric]))
    worst = min(rows, key=lambda row: float(row[metric]))
    overall = metric_values(df["true_label"], df["pred_label"])
    return {
        "overall": float(overall[metric]),
        "best_context_value": best["context_value"],
        "best_value": float(best[metric]),
        "worst_context_value": worst["context_value"],
        "worst_value": float(worst[metric]),
        "best_minus_worst": float(best[metric]) - float(worst[metric]),
        "overall_minus_worst": float(overall[metric]) - float(worst[metric]),
        "n_context_values": int(len(rows)),
    }


def ci(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"boot_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    return {
        "boot_mean": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def bootstrap_group_quantities(df: pd.DataFrame, context_field: str, metric: str, n_bootstrap: int, min_n: int, seed: int) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    patient_col = "patient_id" if "patient_id" in df.columns else "sample_key"
    patients = df[patient_col].astype(str).drop_duplicates().to_numpy()
    groups = {patient: group for patient, group in df.groupby(df[patient_col].astype(str), sort=False)}
    values: Dict[str, List[float]] = {
        "overall": [],
        "worst_value": [],
        "best_minus_worst": [],
        "overall_minus_worst": [],
    }
    for _ in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([groups[patient] for patient in sampled], ignore_index=True)
        gap = context_gap(sample, context_field, metric, min_n)
        if gap is not None:
            for quantity in values:
                values[quantity].append(float(gap[quantity]))
    return values


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    run_rows = []
    for run_idx, result_dir in enumerate(sorted(path for path in args.result_root.iterdir() if path.is_dir()), start=1):
        summary_path = result_dir / "summary.json"
        pred_path = result_dir / "leave_one_context_predictions.csv"
        if not summary_path.exists() or not pred_path.exists() or pred_path.stat().st_size == 0:
            continue
        summary = read_json(summary_path)
        pred = pd.read_csv(pred_path)
        task = str(summary.get("label_column", ""))
        model = str(summary.get("model_name", ""))
        run_rows.append({"task": task, "model": model, "result_dir": str(result_dir), "n_predictions": int(len(pred))})
        for method in sorted(set(args.methods) & set(pred["method"].astype(str).unique())):
            method_df = pred[pred["method"].astype(str) == method].copy()
            for context_field in args.context_fields:
                field_df = method_df[method_df["context_field"].astype(str) == context_field].copy()
                if field_df.empty or context_field not in field_df.columns:
                    continue
                field_df[context_field] = field_df[context_field].fillna("unknown").astype(str)
                for metric in args.metrics:
                    observed = context_gap(field_df, context_field, metric, args.min_subgroup_n)
                    if observed is None:
                        continue
                    boot_by_quantity = bootstrap_group_quantities(
                        field_df,
                        context_field,
                        metric,
                        args.n_bootstrap,
                        args.min_subgroup_n,
                        args.seed + run_idx * 1000,
                    )
                    for quantity in ["overall", "worst_value", "best_minus_worst", "overall_minus_worst"]:
                        values = boot_by_quantity[quantity]
                        rows.append(
                            {
                                "task": task,
                                "model": model,
                                "method": method,
                                "result_dir": str(result_dir),
                                "context_field": context_field,
                                "metric": metric,
                                "quantity": quantity,
                                "observed": float(observed[quantity]),
                                "best_context_value": observed["best_context_value"],
                                "worst_context_value": observed["worst_context_value"],
                                "n_context_values": int(observed["n_context_values"]),
                                "n_bootstrap": int(len(values)),
                                **ci(values),
                            }
                        )
    pd.DataFrame(run_rows).to_csv(args.output_dir / "runs.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "leave_one_context_bootstrap_ci.csv", index=False)
    headline = out[
        (out["method"].astype(str) == "erm")
        & (out["metric"].astype(str) == "balanced_accuracy")
        & (out["quantity"].astype(str).isin(["best_minus_worst", "worst_value"]))
    ].copy()
    headline.to_csv(args.output_dir / "headline_erm_balanced_accuracy_ci.csv", index=False)
    summary = {
        "result_root": str(args.result_root),
        "output_dir": str(args.output_dir),
        "n_runs": int(len(run_rows)),
        "n_ci_rows": int(len(out)),
        "n_bootstrap_requested": int(args.n_bootstrap),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
