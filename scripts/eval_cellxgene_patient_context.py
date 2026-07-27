#!/usr/bin/env python3
"""Run a patient-context probe on CELLxGENE-style h5ad files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_CONTEXT_FIELDS = ["sex", "age_group", "dataset_id", "assay"]
BAD_DONOR_VALUES = {"", "nan", "none", "unknown", "pooled", "allcells"}
ORDINAL_DECADE = {
    "first": 0,
    "second": 10,
    "third": 20,
    "fourth": 30,
    "fifth": 40,
    "sixth": 50,
    "seventh": 60,
    "eighth": 70,
    "ninth": 80,
    "tenth": 90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column", default="disease")
    parser.add_argument("--label-values", nargs="*", default=[])
    parser.add_argument("--max-labels", type=int, default=4)
    parser.add_argument("--min-donors-per-label", type=int, default=3)
    parser.add_argument("--min-cells-per-label", type=int, default=1000)
    parser.add_argument("--max-cells-per-label", type=int, default=12000)
    parser.add_argument("--max-cells-per-donor-label", type=int, default=200)
    parser.add_argument("--n-top-genes", type=int, default=1024)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--context-fields", nargs="*", default=DEFAULT_CONTEXT_FIELDS)
    parser.add_argument("--leave-one-context-fields", nargs="*", default=[])
    parser.add_argument("--min-holdout-cells", type=int, default=200)
    parser.add_argument("--min-holdout-labels", type=int, default=2)
    parser.add_argument("--include-unknown-sex", action="store_true")
    return parser.parse_args()


def clean_string(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def donor_ok(value: object) -> bool:
    text = clean_string(value).lower()
    return text not in BAD_DONOR_VALUES and not text.startswith("allcells")


def parse_age_group(stage: object) -> str:
    text = clean_string(stage).lower()
    if text in {"unknown", "na", "nan"}:
        return "unknown"
    if "post-fertilization" in text or "prenatal" in text or "fetal" in text:
        return "prenatal"
    match = re.search(r"(\d+)-year-old", text)
    if match:
        age = int(match.group(1))
        if age < 18:
            return "child"
        if age < 40:
            return "adult_18_39"
        if age < 60:
            return "adult_40_59"
        return "adult_60_plus"
    for word, decade_start in ORDINAL_DECADE.items():
        if f"{word} decade" in text:
            if decade_start < 18:
                return "child"
            if decade_start < 40:
                return "adult_18_39"
            if decade_start < 60:
                return "adult_40_59"
            return "adult_60_plus"
    if "adult" in text:
        return "adult_unspecified"
    return "unknown"


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "fold",
        "split_type",
        "context_field",
        "context_value",
        "cell_index",
        "donor_id",
        "true_label",
        "pred_label",
        "feature_name",
        "feature_id",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_label_values(
    df: pd.DataFrame,
    label_column: str,
    explicit_values: Sequence[str],
    max_labels: int,
    min_donors_per_label: int,
    min_cells_per_label: int,
) -> List[str]:
    if explicit_values:
        return list(explicit_values)
    rows: List[Dict[str, object]] = []
    for label, label_df in df.groupby(label_column, dropna=False):
        rows.append(
            {
                "label": clean_string(label),
                "n_cells": int(len(label_df)),
                "n_donors": int(label_df["donor_id"].nunique()),
            }
        )
    candidates = [
        row
        for row in rows
        if row["label"] != "unknown"
        and int(row["n_donors"]) >= min_donors_per_label
        and int(row["n_cells"]) >= min_cells_per_label
    ]
    candidates.sort(key=lambda row: (-int(row["n_cells"]), str(row["label"])))
    return [str(row["label"]) for row in candidates[:max_labels]]


def sample_cells(
    df: pd.DataFrame,
    label_column: str,
    max_cells_per_donor_label: int,
    max_cells_per_label: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_parts: List[pd.DataFrame] = []
    for _, group in df.groupby([label_column, "donor_id"], sort=True, observed=True):
        if len(group) > max_cells_per_donor_label:
            take = rng.choice(group.index.to_numpy(), size=max_cells_per_donor_label, replace=False)
            sampled_parts.append(group.loc[np.sort(take)])
        else:
            sampled_parts.append(group)
    sampled = pd.concat(sampled_parts, axis=0).sort_index()

    capped_parts: List[pd.DataFrame] = []
    for _, group in sampled.groupby(label_column, sort=True, observed=True):
        if len(group) > max_cells_per_label:
            take = rng.choice(group.index.to_numpy(), size=max_cells_per_label, replace=False)
            capped_parts.append(group.loc[np.sort(take)])
        else:
            capped_parts.append(group)
    return pd.concat(capped_parts, axis=0).sort_index()


def infer_fold_count(df: pd.DataFrame, label_column: str, requested: int) -> int:
    donor_labels = df[["donor_id", label_column]].drop_duplicates()
    min_donors = int(donor_labels.groupby(label_column, observed=True)["donor_id"].nunique().min())
    return max(2, min(requested, min_donors))


def load_expression_subset(adata: ad.AnnData, row_indices: np.ndarray) -> sparse.csr_matrix:
    row_indices = np.asarray(row_indices, dtype=int)
    order = np.argsort(row_indices)
    sorted_rows = row_indices[order]
    x_sorted = adata.X[sorted_rows, :]
    x_sorted = sparse.csr_matrix(x_sorted)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return x_sorted[inverse, :]


def select_variable_genes(x: sparse.csr_matrix, n_top_genes: int) -> np.ndarray:
    means = np.asarray(x.mean(axis=0)).ravel()
    second = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    variances = second - means * means
    expressed = np.asarray((x > 0).sum(axis=0)).ravel()
    variances[expressed < 5] = -np.inf
    n_top = min(int(n_top_genes), int(np.isfinite(variances).sum()))
    if n_top <= 0:
        raise ValueError("No genes passed the minimum expression filter.")
    top = np.argpartition(variances, -n_top)[-n_top:]
    return top[np.argsort(variances[top])[::-1]]


def make_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def metric_row(y_true: Sequence[str], y_pred: Sequence[str], prefix: Mapping[str, object]) -> Dict[str, object]:
    y_true_arr = np.asarray(y_true, dtype=str)
    y_pred_arr = np.asarray(y_pred, dtype=str)
    labels = sorted(np.unique(y_true_arr))
    recalls = []
    for label in labels:
        mask = y_true_arr == label
        if np.any(mask):
            recalls.append(float(np.mean(y_pred_arr[mask] == label)))
    return {
        **dict(prefix),
        "n": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro")),
    }


def summarize_predictions(
    pred_df: pd.DataFrame,
    context_fields: Sequence[str],
    min_group_n: int = 20,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    gaps: List[Dict[str, object]] = []

    overall = metric_row(
        pred_df["true_label"].astype(str),
        pred_df["pred_label"].astype(str),
        {"split_type": "patient_level_cv", "context_field": "overall", "context_value": "overall"},
    )
    rows.append(overall)

    for field in context_fields:
        if field not in pred_df.columns:
            continue
        field_rows: List[Dict[str, object]] = []
        for value, group in pred_df.groupby(field, dropna=False):
            if len(group) < min_group_n or group["true_label"].nunique() < 2:
                continue
            row = metric_row(
                group["true_label"].astype(str),
                group["pred_label"].astype(str),
                {
                    "split_type": "patient_level_cv",
                    "context_field": field,
                    "context_value": clean_string(value),
                },
            )
            rows.append(row)
            field_rows.append(row)
        if len(field_rows) >= 2:
            for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
                values = [float(row[metric]) for row in field_rows if math.isfinite(float(row[metric]))]
                if len(values) >= 2:
                    best = max(field_rows, key=lambda row: float(row[metric]))
                    worst = min(field_rows, key=lambda row: float(row[metric]))
                    gaps.append(
                        {
                            "split_type": "patient_level_cv",
                            "context_field": field,
                            "metric": metric,
                            "best_context_value": best["context_value"],
                            "worst_context_value": worst["context_value"],
                            "best_value": float(best[metric]),
                            "worst_value": float(worst[metric]),
                            "gap": float(best[metric]) - float(worst[metric]),
                        }
                    )
    return rows, gaps


def summarize_metric_gaps(metric_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    gap_rows: List[Dict[str, object]] = []
    fields = sorted({str(row["context_field"]) for row in metric_rows})
    for field in fields:
        field_rows = [row for row in metric_rows if str(row["context_field"]) == field]
        if len(field_rows) < 2:
            continue
        for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
            valid = [row for row in field_rows if metric in row and math.isfinite(float(row[metric]))]
            if len(valid) < 2:
                continue
            best = max(valid, key=lambda row: float(row[metric]))
            worst = min(valid, key=lambda row: float(row[metric]))
            gap_rows.append(
                {
                    "split_type": best.get("split_type", "leave_one_context"),
                    "context_field": field,
                    "metric": metric,
                    "best_context_value": best["context_value"],
                    "worst_context_value": worst["context_value"],
                    "best_value": float(best[metric]),
                    "worst_value": float(worst[metric]),
                    "gap": float(best[metric]) - float(worst[metric]),
                }
            )
    return gap_rows


def run_leave_one_context(
    x: sparse.csr_matrix,
    sampled: pd.DataFrame,
    y: np.ndarray,
    context_fields: Sequence[str],
    min_holdout_cells: int,
    min_holdout_labels: int,
    seed: int,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    metric_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    groups = sampled["donor_id"].astype(str).to_numpy()

    for field in context_fields:
        if field not in sampled.columns:
            skipped_rows.append({"context_field": field, "context_value": "ALL", "reason": "missing_field"})
            continue
        for context_value in sorted(sampled[field].dropna().astype(str).unique()):
            test_mask = sampled[field].astype(str).to_numpy() == context_value
            test_idx = np.flatnonzero(test_mask)
            train_idx = np.flatnonzero(~test_mask)
            prefix = {"split_type": "leave_one_context", "context_field": field, "context_value": context_value}
            if len(test_idx) < min_holdout_cells:
                skipped_rows.append({**prefix, "reason": "too_few_holdout_cells", "n_test_cells": int(len(test_idx))})
                continue
            if len(np.unique(y[test_idx])) < min_holdout_labels:
                skipped_rows.append(
                    {
                        **prefix,
                        "reason": "too_few_holdout_labels",
                        "n_test_cells": int(len(test_idx)),
                        "n_test_labels": int(len(np.unique(y[test_idx]))),
                    }
                )
                continue
            if len(np.unique(y[train_idx])) < 2:
                skipped_rows.append({**prefix, "reason": "too_few_train_labels", "n_train_cells": int(len(train_idx))})
                continue
            model = make_classifier(seed + len(metric_rows) + 1000)
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            metric_rows.append(
                metric_row(
                    y[test_idx],
                    pred,
                    {
                        **prefix,
                        "n_train_cells": int(len(train_idx)),
                        "n_test_cells": int(len(test_idx)),
                        "n_train_donors": int(pd.Series(groups[train_idx]).nunique()),
                        "n_test_donors": int(pd.Series(groups[test_idx]).nunique()),
                        "n_train_labels": int(len(np.unique(y[train_idx]))),
                        "n_test_labels": int(len(np.unique(y[test_idx]))),
                    },
                )
            )
            test_meta = sampled.iloc[test_idx]
            for local_i, (_, meta_row) in enumerate(test_meta.iterrows()):
                row = {
                    "split_type": "leave_one_context",
                    "context_field": field,
                    "context_value": context_value,
                    "cell_index": int(meta_row["cell_index"]),
                    "donor_id": meta_row["donor_id"],
                    "true_label": str(y[test_idx][local_i]),
                    "pred_label": str(pred[local_i]),
                }
                for meta_field in sorted(set(context_fields) | set(DEFAULT_CONTEXT_FIELDS) | {"disease"}):
                    if meta_field in test_meta.columns:
                        row[meta_field] = meta_row[meta_field]
                pred_rows.append(row)
    return metric_rows, summarize_metric_gaps(metric_rows), pred_rows, skipped_rows


def build_metadata_summary(df: pd.DataFrame, label_column: str, context_fields: Sequence[str]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "n_cells": int(len(df)),
        "n_donors": int(df["donor_id"].nunique()),
        "label_column": label_column,
        "label_counts": df[label_column].value_counts().to_dict(),
        "label_donor_counts": df.groupby(label_column, observed=True)["donor_id"].nunique().to_dict(),
    }
    context_summary: Dict[str, object] = {}
    for field in context_fields:
        if field in df.columns:
            context_summary[field] = df[field].value_counts().head(20).to_dict()
    summary["context_counts"] = context_summary
    return summary


def cramers_v(table: pd.DataFrame) -> float:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = float(chi2_contingency(table, correction=False)[0])
    n = float(table.to_numpy().sum())
    if n <= 0:
        return float("nan")
    phi2 = chi2 / n
    r, k = table.shape
    denom = min(k - 1, r - 1)
    return float(math.sqrt(phi2 / denom)) if denom > 0 else float("nan")


def label_context_audit(
    df: pd.DataFrame,
    label_column: str,
    context_fields: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    association_rows: List[Dict[str, object]] = []
    count_rows: List[Dict[str, object]] = []
    for field in context_fields:
        if field not in df.columns:
            continue
        table = pd.crosstab(df[field], df[label_column])
        association_rows.append(
            {
                "context_field": field,
                "n_context_values": int(table.shape[0]),
                "n_labels": int(table.shape[1]),
                "cramers_v": cramers_v(table),
            }
        )
        for context_value, group in df.groupby(field, sort=True, observed=True):
            for label, label_group in group.groupby(label_column, sort=True, observed=True):
                count_rows.append(
                    {
                        "context_field": field,
                        "context_value": clean_string(context_value),
                        "label": clean_string(label),
                        "n_cells": int(len(label_group)),
                        "n_donors": int(label_group["donor_id"].nunique()),
                    }
                )
    return association_rows, count_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs.copy()
    obs["cell_index"] = np.arange(adata.n_obs, dtype=int)
    for column in obs.columns:
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
    if sampled[args.label_column].nunique() < 2:
        raise SystemExit("Sampled data has fewer than two labels.")

    row_indices = sampled["cell_index"].to_numpy(dtype=int)
    x_all = load_expression_subset(adata, row_indices)
    top_gene_idx = select_variable_genes(x_all, args.n_top_genes)
    x = x_all[:, top_gene_idx].tocsr()

    var = adata.var.iloc[top_gene_idx].copy()
    feature_rows = []
    for matrix_idx, (_, row) in enumerate(var.iterrows()):
        feature_rows.append(
            {
                "rank": matrix_idx + 1,
                "feature_matrix_index": int(top_gene_idx[matrix_idx]),
                "feature_id": row.get("feature_id", ""),
                "feature_name": row.get("feature_name", ""),
            }
        )
    write_csv(args.output_dir / "feature_genes.csv", feature_rows)

    y = sampled[args.label_column].astype(str).to_numpy()
    groups = sampled["donor_id"].astype(str).to_numpy()
    n_folds = infer_fold_count(sampled, args.label_column, args.n_folds)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = make_classifier(args.seed + fold)
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        fold_rows.append(
            metric_row(
                y[test_idx],
                pred,
                {
                    "fold": fold,
                    "split_type": "patient_level_cv",
                    "n_train_cells": int(len(train_idx)),
                    "n_test_cells": int(len(test_idx)),
                    "n_train_donors": int(pd.Series(groups[train_idx]).nunique()),
                    "n_test_donors": int(pd.Series(groups[test_idx]).nunique()),
                },
            )
        )
        test_meta = sampled.iloc[test_idx]
        for local_i, (_, meta_row) in enumerate(test_meta.iterrows()):
            row = {
                "fold": fold,
                "cell_index": int(meta_row["cell_index"]),
                "donor_id": meta_row["donor_id"],
                "true_label": str(y[test_idx][local_i]),
                "pred_label": str(pred[local_i]),
            }
            for field in args.context_fields:
                if field in test_meta.columns:
                    row[field] = meta_row[field]
            pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    subgroup_rows, gap_rows = summarize_predictions(pred_df, args.context_fields)
    association_rows, context_count_rows = label_context_audit(sampled, args.label_column, args.context_fields)
    leave_one_metrics: List[Dict[str, object]] = []
    leave_one_gaps: List[Dict[str, object]] = []
    leave_one_predictions: List[Dict[str, object]] = []
    leave_one_skipped: List[Dict[str, object]] = []
    if args.leave_one_context_fields:
        leave_one_metrics, leave_one_gaps, leave_one_predictions, leave_one_skipped = run_leave_one_context(
            x=x,
            sampled=sampled,
            y=y,
            context_fields=args.leave_one_context_fields,
            min_holdout_cells=args.min_holdout_cells,
            min_holdout_labels=args.min_holdout_labels,
            seed=args.seed,
        )

    selected_manifest = sampled[
        ["cell_index", "donor_id", args.label_column, *[f for f in args.context_fields if f in sampled.columns]]
    ].copy()
    selected_manifest = selected_manifest.rename(columns={args.label_column: "label"})
    selected_manifest.to_csv(args.output_dir / "selected_cells.csv", index=False)
    write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "predictions.csv", pred_rows)
    write_csv(args.output_dir / "subgroup_metrics.csv", subgroup_rows)
    write_csv(args.output_dir / "subgroup_gaps.csv", gap_rows)
    write_csv(args.output_dir / "label_context_association.csv", association_rows)
    write_csv(args.output_dir / "label_context_counts.csv", context_count_rows)
    write_csv(args.output_dir / "leave_one_context_metrics.csv", leave_one_metrics)
    write_csv(args.output_dir / "leave_one_context_gaps.csv", leave_one_gaps)
    write_csv(args.output_dir / "leave_one_context_predictions.csv", leave_one_predictions)
    write_csv(args.output_dir / "leave_one_context_skipped.csv", leave_one_skipped)

    summary = {
        "h5ad": str(args.h5ad),
        "label_column": args.label_column,
        "label_values": label_values,
        "n_folds": int(n_folds),
        "n_top_genes": int(len(top_gene_idx)),
        "filtered": build_metadata_summary(filtered, args.label_column, args.context_fields),
        "sampled": build_metadata_summary(sampled, args.label_column, args.context_fields),
        "overall": subgroup_rows[0] if subgroup_rows else {},
        "subgroup_gaps": gap_rows,
        "label_context_association": association_rows,
        "leave_one_context_gaps": leave_one_gaps,
        "leave_one_context_skipped": leave_one_skipped,
        "outputs": {
            "selected_cells": str(args.output_dir / "selected_cells.csv"),
            "predictions": str(args.output_dir / "predictions.csv"),
            "subgroup_metrics": str(args.output_dir / "subgroup_metrics.csv"),
            "subgroup_gaps": str(args.output_dir / "subgroup_gaps.csv"),
            "label_context_association": str(args.output_dir / "label_context_association.csv"),
            "label_context_counts": str(args.output_dir / "label_context_counts.csv"),
            "leave_one_context_metrics": str(args.output_dir / "leave_one_context_metrics.csv"),
            "leave_one_context_gaps": str(args.output_dir / "leave_one_context_gaps.csv"),
            "leave_one_context_predictions": str(args.output_dir / "leave_one_context_predictions.csv"),
            "leave_one_context_skipped": str(args.output_dir / "leave_one_context_skipped.csv"),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
