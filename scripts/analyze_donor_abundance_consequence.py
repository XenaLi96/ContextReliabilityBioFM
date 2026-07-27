#!/usr/bin/env python3
"""Translate cell-level errors into donor-level abundance reconstruction error.

The embedding cohorts were sampled with per-donor-label and per-label caps.
Directly counting their predictions would therefore distort abundance.  This
analysis uses the full source h5ad to recover each donor's true counts for the
selected labels, estimates a prediction distribution within each sampled
donor/true-label stratum, and propagates that distribution to the full counts.
This is a sampling-corrected reconstruction of the annotated cell composition,
not a causal disease-abundance analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MAIN_METHODS = ["erm_mlp", "label_context_reweight", "sca_lite", "group_dro"]
METHOD_LABELS = {
    "erm_mlp": "ERM",
    "label_context_reweight": "LC-Reweight",
    "sca_lite": "SCA-Align",
    "group_dro": "GroupDRO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=Path("data/cellxgene_support_calibrated_formal"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/donor_abundance_consequence"),
    )
    parser.add_argument("--min-unit-coverage", type=float, default=0.9)
    parser.add_argument("--min-donors-correlation", type=int, default=5)
    parser.add_argument("--min-donors-disease-group", type=int, default=5)
    return parser.parse_args()


def clean(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def mode_or_mixed(values: pd.Series) -> str:
    cleaned = values.astype(object).map(clean)
    counts = cleaned.value_counts()
    if counts.empty:
        return "unknown"
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "mixed"
    return str(counts.index[0])


def parse_run_path(path: Path, root: Path) -> Dict[str, object]:
    parts = path.relative_to(root).parts
    seed_index = next(index for index, value in enumerate(parts) if value.startswith("seed_"))
    model = parts[seed_index - 2]
    context = parts[seed_index - 1]
    task = parts[seed_index - 3] if seed_index >= 3 else "bone_marrow"
    return {
        "task": task,
        "model": model,
        "context_field": context,
        "seed": int(parts[seed_index].replace("seed_", "")),
        "run_dir": path.parent,
    }


def discover_runs(root: Path) -> pd.DataFrame:
    rows = [parse_run_path(path, root) for path in sorted(root.glob("**/seed_*/*/predictions.csv"))]
    if not rows:
        raise FileNotFoundError(f"No formal prediction artifacts found under {root}")
    return pd.DataFrame.from_records(rows)


def embedding_source_summary(run_dir: Path) -> Dict[str, object]:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        run_summary = json.load(handle)
    embedding_file = Path(run_summary["embedding_file"])
    with (embedding_file.parent / "summary.json").open("r", encoding="utf-8") as handle:
        embedding_summary = json.load(handle)
    return {**run_summary, "h5ad": embedding_summary["h5ad"]}


def source_counts(
    h5ad_path: Path,
    sampled_metadata: pd.DataFrame,
    context_field: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    labels = sorted(sampled_metadata["label"].astype(str).unique())
    donors = set(sampled_metadata["donor_id"].astype(str).unique())
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    label_column = "cell_type" if "cell_type" in obs.columns else "label"
    required = {"donor_id", label_column, context_field}
    missing = sorted(required - set(obs.columns))
    if missing:
        raise ValueError(f"{h5ad_path} lacks source columns {missing}")
    columns = ["donor_id", label_column, context_field]
    if "disease" in obs.columns:
        columns.append("disease")
    obs = obs[columns].copy()
    for column in columns:
        obs[column] = obs[column].astype(object).map(clean)
    obs = obs.loc[obs["donor_id"].isin(donors) & obs[label_column].isin(labels)].copy()
    obs = obs.rename(columns={label_column: "true_label", context_field: "context_value"})

    donor_counts = (
        obs.groupby(["donor_id", "true_label"], sort=True)
        .size()
        .reset_index(name="n_source_cells")
    )
    donor_context_counts = (
        obs.groupby(["donor_id", "context_value", "true_label"], sort=True)
        .size()
        .reset_index(name="n_source_cells")
    )
    if "disease" in sampled_metadata.columns:
        donor_disease = (
            sampled_metadata.groupby("donor_id", sort=True)["disease"]
            .agg(mode_or_mixed)
            .reset_index()
        )
    elif "disease" in obs.columns:
        donor_disease = obs.groupby("donor_id", sort=True)["disease"].agg(mode_or_mixed).reset_index()
    else:
        donor_disease = pd.DataFrame({"donor_id": sorted(donors), "disease": "unknown"})
    return donor_counts, donor_context_counts, donor_disease, labels


def prediction_distribution(pred: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    columns = list(group_columns) + ["true_label", "pred_label"]
    counts = pred.groupby(columns, sort=True).size().reset_index(name="n_sampled_predictions")
    totals = counts.groupby(list(group_columns) + ["true_label"], sort=False)["n_sampled_predictions"].transform("sum")
    counts["prediction_probability"] = counts["n_sampled_predictions"] / totals
    return counts


def reconstruct_units(
    predictions: pd.DataFrame,
    source: pd.DataFrame,
    labels: Sequence[str],
    group_columns: Sequence[str],
    min_coverage: float,
) -> pd.DataFrame:
    distribution = prediction_distribution(predictions, group_columns)
    weighted = distribution.merge(source, on=list(group_columns) + ["true_label"], how="inner")
    weighted["estimated_predicted_cells"] = weighted["prediction_probability"] * weighted["n_source_cells"]

    covered_strata = weighted[list(group_columns) + ["true_label"]].drop_duplicates()
    covered_true = source.merge(covered_strata, on=list(group_columns) + ["true_label"], how="inner")
    full_totals = source.groupby(list(group_columns), sort=False)["n_source_cells"].sum().rename("full_source_cells")
    covered_totals = covered_true.groupby(list(group_columns), sort=False)["n_source_cells"].sum().rename("covered_source_cells")
    coverage = pd.concat([full_totals, covered_totals], axis=1).fillna(0).reset_index()
    coverage["reconstruction_coverage"] = coverage["covered_source_cells"] / coverage["full_source_cells"]

    true_counts = covered_true.rename(columns={"true_label": "label", "n_source_cells": "true_cells"})
    true_counts = true_counts.groupby(list(group_columns) + ["label"], sort=False)["true_cells"].sum().reset_index()
    predicted_counts = (
        weighted.rename(columns={"pred_label": "label"})
        .groupby(list(group_columns) + ["label"], sort=False)["estimated_predicted_cells"]
        .sum()
        .reset_index(name="predicted_cells")
    )

    units = covered_true[list(group_columns)].drop_duplicates().copy()
    label_frame = pd.DataFrame({"label": list(labels)})
    units["_key"] = 1
    label_frame["_key"] = 1
    grid = units.merge(label_frame, on="_key").drop(columns="_key")
    grid = grid.merge(true_counts, on=list(group_columns) + ["label"], how="left")
    grid = grid.merge(predicted_counts, on=list(group_columns) + ["label"], how="left")
    grid = grid.merge(coverage, on=list(group_columns), how="left")
    grid[["true_cells", "predicted_cells"]] = grid[["true_cells", "predicted_cells"]].fillna(0.0)
    grid = grid.loc[grid["reconstruction_coverage"] >= float(min_coverage)].copy()
    denominator = grid.groupby(list(group_columns), sort=False)["true_cells"].transform("sum")
    grid["true_abundance"] = grid["true_cells"] / denominator
    grid["predicted_abundance"] = grid["predicted_cells"] / denominator
    grid["signed_error"] = grid["predicted_abundance"] - grid["true_abundance"]
    grid["absolute_error"] = grid["signed_error"].abs()
    return grid


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 2 or np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    return float(spearmanr(x_arr, y_arr).statistic)


def abundance_summary(
    donor_rows: pd.DataFrame,
    context_rows: pd.DataFrame,
    min_donors_correlation: int,
) -> Dict[str, object]:
    rank_values = []
    for _, donor in donor_rows.groupby("donor_id", sort=False):
        rank_values.append(safe_spearman(donor["true_abundance"], donor["predicted_abundance"]))
    label_values = []
    for _, label in donor_rows.groupby("label", sort=False):
        if label["donor_id"].nunique() >= int(min_donors_correlation):
            label_values.append(safe_spearman(label["true_abundance"], label["predicted_abundance"]))
    context_mae = context_rows.groupby("context_value", sort=False)["absolute_error"].mean()
    valid_rank = np.asarray([value for value in rank_values if math.isfinite(value)], dtype=float)
    valid_label = np.asarray([value for value in label_values if math.isfinite(value)], dtype=float)
    return {
        "n_donors": int(donor_rows["donor_id"].nunique()),
        "donor_abundance_mae": float(donor_rows["absolute_error"].mean()),
        "mean_label_spearman_across_donors": float(np.mean(valid_label)) if len(valid_label) else float("nan"),
        "mean_donor_rank_stability": float(np.mean(valid_rank)) if len(valid_rank) else float("nan"),
        "worst_context_abundance_mae": float(context_mae.max()) if len(context_mae) else float("nan"),
        "worst_context_value": str(context_mae.idxmax()) if len(context_mae) else "",
        "mean_reconstruction_coverage": float(donor_rows.groupby("donor_id")["reconstruction_coverage"].first().mean()),
    }


def disease_direction_rows(
    donor_rows: pd.DataFrame,
    donor_disease: pd.DataFrame,
    min_group_donors: int,
) -> List[Dict[str, object]]:
    merged = donor_rows.merge(donor_disease, on="donor_id", how="left")
    donor_groups = merged[["donor_id", "disease"]].drop_duplicates()
    counts = donor_groups["disease"].value_counts()
    if int(counts.get("normal", 0)) < int(min_group_donors):
        return []
    non_normal = counts.drop(labels=["normal"], errors="ignore")
    non_normal = non_normal.loc[(non_normal.index != "unknown") & (non_normal.index != "mixed")]
    non_normal = non_normal.loc[non_normal >= int(min_group_donors)]
    if non_normal.empty:
        return []
    comparison = str(non_normal.sort_values(ascending=False).index[0])
    rows: List[Dict[str, object]] = []
    for label, label_rows in merged.groupby("label", sort=True):
        means = label_rows.loc[label_rows["disease"].isin(["normal", comparison])].groupby("disease")[["true_abundance", "predicted_abundance"]].mean()
        if set(means.index) != {"normal", comparison}:
            continue
        true_delta = float(means.loc[comparison, "true_abundance"] - means.loc["normal", "true_abundance"])
        predicted_delta = float(means.loc[comparison, "predicted_abundance"] - means.loc["normal", "predicted_abundance"])
        rows.append(
            {
                "label": str(label),
                "comparison": f"{comparison} - normal",
                "n_normal_donors": int(counts["normal"]),
                "n_comparison_donors": int(counts[comparison]),
                "true_delta": true_delta,
                "predicted_delta": predicted_delta,
                "direction_match": bool(np.sign(true_delta) == np.sign(predicted_delta)),
            }
        )
    return rows


def add_prefix(frame: pd.DataFrame, prefix: Mapping[str, object]) -> pd.DataFrame:
    result = frame.copy()
    for key, value in reversed(list(prefix.items())):
        result.insert(0, key, value)
    return result


def aggregate_seed_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "model", "context_field", "split_type", "method", "method_label"]
    metrics = [
        "donor_abundance_mae",
        "mean_label_spearman_across_donors",
        "mean_donor_rank_stability",
        "worst_context_abundance_mae",
        "mean_reconstruction_coverage",
    ]
    rows: List[Dict[str, object]] = []
    for group_key, group in per_seed.groupby(keys, sort=True):
        row = dict(zip(keys, group_key))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        worst = group["worst_context_value"].astype(str).value_counts()
        row["modal_worst_context_value"] = str(worst.index[0]) if len(worst) else ""
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.formal_root)
    donor_outputs: List[pd.DataFrame] = []
    context_outputs: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    disease_outputs: List[pd.DataFrame] = []

    task_keys = ["task", "model", "context_field"]
    for task_key, task_runs in runs.groupby(task_keys, sort=True):
        representative_dir = Path(task_runs.iloc[0]["run_dir"])
        source_summary = embedding_source_summary(representative_dir)
        sampled_metadata = pd.read_csv(source_summary["metadata_csv"])
        donor_source, donor_context_source, donor_disease, labels = source_counts(
            Path(source_summary["h5ad"]), sampled_metadata, str(task_key[2])
        )

        for run in task_runs.to_dict(orient="records"):
            run_dir = Path(run["run_dir"])
            split_files = {
                "patient_level_cv": run_dir / "predictions.csv",
                "leave_one_context": run_dir / "leave_one_context_predictions.csv",
            }
            for split_type, path in split_files.items():
                predictions = pd.read_csv(path)
                predictions = predictions.loc[predictions["method"].isin(MAIN_METHODS)].copy()
                for method, method_predictions in predictions.groupby("method", sort=True):
                    donor_rows = reconstruct_units(
                        method_predictions,
                        donor_source,
                        labels,
                        group_columns=["donor_id"],
                        min_coverage=args.min_unit_coverage,
                    )
                    context_predictions = method_predictions.copy()
                    if "context_value" not in context_predictions.columns:
                        context_predictions["context_value"] = context_predictions[str(task_key[2])].astype(str)
                    context_rows = reconstruct_units(
                        context_predictions,
                        donor_context_source,
                        labels,
                        group_columns=["donor_id", "context_value"],
                        min_coverage=args.min_unit_coverage,
                    )
                    prefix = {
                        "task": str(task_key[0]),
                        "model": str(task_key[1]),
                        "context_field": str(task_key[2]),
                        "seed": int(run["seed"]),
                        "split_type": split_type,
                        "method": str(method),
                        "method_label": METHOD_LABELS[str(method)],
                    }
                    donor_outputs.append(add_prefix(donor_rows, prefix))
                    context_outputs.append(add_prefix(context_rows, prefix))
                    summary_rows.append({**prefix, **abundance_summary(donor_rows, context_rows, args.min_donors_correlation)})
                    disease = disease_direction_rows(donor_rows, donor_disease, args.min_donors_disease_group)
                    if disease:
                        disease_outputs.append(add_prefix(pd.DataFrame.from_records(disease), prefix))

    donor_table = pd.concat(donor_outputs, ignore_index=True)
    context_table = pd.concat(context_outputs, ignore_index=True)
    per_seed_summary = pd.DataFrame.from_records(summary_rows)
    aggregate = aggregate_seed_summary(per_seed_summary)
    disease_table = pd.concat(disease_outputs, ignore_index=True) if disease_outputs else pd.DataFrame()
    if disease_table.empty:
        disease_summary = pd.DataFrame()
    else:
        disease_summary = (
            disease_table.groupby(
                ["task", "model", "context_field", "split_type", "method", "method_label", "comparison"],
                sort=True,
            )
            .agg(
                n_seeds=("seed", "nunique"),
                n_label_seed_comparisons=("direction_match", "size"),
                direction_match_rate=("direction_match", "mean"),
                mean_absolute_true_delta=("true_delta", lambda values: float(np.mean(np.abs(values)))),
                mean_absolute_predicted_delta=("predicted_delta", lambda values: float(np.mean(np.abs(values)))),
            )
            .reset_index()
        )
    context_summary = (
        context_table.groupby(
            ["task", "model", "context_field", "seed", "split_type", "method", "method_label", "context_value"],
            sort=True,
        )["absolute_error"]
        .mean()
        .reset_index(name="context_abundance_mae")
    )
    celltype_context = (
        context_table.groupby(
            ["task", "model", "context_field", "seed", "split_type", "method", "method_label", "context_value", "label"],
            sort=True,
        )[["true_abundance", "predicted_abundance", "signed_error", "absolute_error"]]
        .mean()
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    donor_table.to_csv(args.output_dir / "donor_label_abundance.csv", index=False)
    context_table.to_csv(args.output_dir / "donor_context_label_abundance.csv", index=False)
    context_summary.to_csv(args.output_dir / "context_abundance_summary.csv", index=False)
    celltype_context.to_csv(args.output_dir / "celltype_context_abundance_error.csv", index=False)
    per_seed_summary.to_csv(args.output_dir / "per_seed_summary.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_summary.csv", index=False)
    disease_table.to_csv(args.output_dir / "differential_abundance_direction.csv", index=False)
    disease_summary.to_csv(args.output_dir / "differential_abundance_direction_summary.csv", index=False)
    protocol = {
        "formal_root": str(args.formal_root),
        "min_unit_coverage": float(args.min_unit_coverage),
        "min_donors_correlation": int(args.min_donors_correlation),
        "min_donors_disease_group": int(args.min_donors_disease_group),
        "sampling_correction": "propagate within-donor/true-label prediction distributions to full h5ad source counts",
        "interpretation_boundary": "annotated selected-label composition reconstruction; not a causal disease-abundance analysis",
    }
    with (args.output_dir / "analysis_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
