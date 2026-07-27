#!/usr/bin/env python3
"""Systematic clustered-bootstrap/FDR summary for context audits.

The analysis reuses saved out-of-fold predictions.  For each
dataset/model/context/regime row it freezes the observed best and worst context
bins, resamples independent donors or patients within those bins, and estimates
the best-minus-worst balanced-accuracy confidence interval.  Statistical
testing uses label-stratified donor/patient-block permutation with shared
cluster ranks across labels.  The max-minus-min statistic is recomputed in
every replicate, retaining adaptive best/worst-bin selection.  These p-values
are BH-adjusted across eligible rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


FOUNDATION_MODELS = {
    "geneformer_v1",
    "scgpt_continual",
    "UNI",
    "CONCH",
    "Virchow2",
    "H-optimus0",
}


def strip_prefix_suffix(value: str, prefix: str, suffix: str) -> str:
    if value.startswith(prefix):
        value = value[len(prefix) :]
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/systematic_audit_robustness"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--min-donors", type=int, default=5)
    parser.add_argument("--support-coverage", type=float, default=0.8)
    parser.add_argument("--pathology-min-patients-per-label", type=int, default=5)
    return parser.parse_args()


def balanced_accuracy(frame: pd.DataFrame) -> float:
    recalls = []
    for label, group in frame.groupby("true_label", sort=False):
        if len(group):
            recalls.append(float((group["pred_label"].astype(str) == str(label)).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def intended_context(run_name: str) -> str | None:
    if run_name.endswith("_assay_erm") or run_name.endswith("_assay"):
        return "assay"
    if run_name.endswith("_dataset_erm") or run_name.endswith("_dataset_id_erm"):
        return "dataset_id"
    return None


def model_from_run(run_name: str, context: str) -> str:
    suffixes = (
        f"_{context}_erm",
        "_dataset_erm" if context == "dataset_id" else "",
        f"_{context}",
        "_erm",
    )
    value = run_name
    for suffix in suffixes:
        if suffix and value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def support_stats(
    frame: pd.DataFrame,
    context: str,
    cluster: str,
    min_samples: int,
    min_clusters: int,
) -> Dict[str, object]:
    counts = (
        frame.groupby(["true_label", context], observed=True)
        .agg(n_samples=("true_label", "size"), n_clusters=(cluster, "nunique"))
        .reset_index()
    )
    counts["cell_ok"] = counts["n_samples"].ge(min_samples)
    counts["donor_ok"] = counts["n_clusters"].ge(min_clusters)
    counts["pair_ok_raw"] = counts["cell_ok"] & counts["donor_ok"]
    connected = counts.loc[counts["pair_ok_raw"]].groupby("true_label")[context].nunique()
    connected_labels = set(connected[connected.ge(2)].index.astype(str))
    counts["pair_ok"] = counts["pair_ok_raw"] & counts["true_label"].astype(str).isin(connected_labels)
    lookup = {
        (str(row.true_label), str(getattr(row, context))): bool(row.pair_ok)
        for row in counts.itertuples(index=False)
    }
    eligible = np.asarray(
        [
            lookup.get((str(y), str(c)), False)
            for y, c in zip(frame["true_label"], frame[context])
        ],
        dtype=bool,
    )
    eligible_contexts = counts.loc[counts["pair_ok"], context].astype(str).nunique()
    return {
        "support_coverage": float(eligible.mean()) if len(eligible) else 0.0,
        "eligible_pairs": int(counts["pair_ok"].sum()),
        "total_observed_pairs": int(len(counts)),
        "eligible_labels": int(len(connected_labels)),
        "eligible_contexts": int(eligible_contexts),
        "min_observed_clusters": int(counts["n_clusters"].min()) if len(counts) else 0,
    }


def context_ba(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(context, sort=True):
        score = balanced_accuracy(group)
        if np.isfinite(score) and group["true_label"].nunique() >= 2:
            rows.append(
                {
                    "context_value": str(value),
                    "ba": score,
                    "n": int(len(group)),
                    "n_clusters": int(group["_cluster"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def cluster_class_arrays(
    frame: pd.DataFrame, labels: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    clusters = sorted(frame["_cluster"].astype(str).unique())
    cluster_index = {value: idx for idx, value in enumerate(clusters)}
    label_index = {value: idx for idx, value in enumerate(labels)}
    totals = np.zeros((len(clusters), len(labels)), dtype=np.float64)
    correct = np.zeros_like(totals)
    for row in frame[["_cluster", "true_label", "pred_label"]].itertuples(index=False):
        i = cluster_index[str(row[0])]
        j = label_index[str(row[1])]
        totals[i, j] += 1.0
        correct[i, j] += float(str(row[1]) == str(row[2]))
    return correct, totals


def bootstrap_ba(
    correct: np.ndarray,
    totals: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.full(draws, np.nan, dtype=np.float64)
    n_clusters = correct.shape[0]
    if n_clusters == 0:
        return out
    batch = 256
    for start in range(0, draws, batch):
        size = min(batch, draws - start)
        sampled = rng.integers(0, n_clusters, size=(size, n_clusters))
        sampled_correct = correct[sampled].sum(axis=1)
        sampled_totals = totals[sampled].sum(axis=1)
        valid = sampled_totals > 0
        recalls = np.divide(
            sampled_correct,
            sampled_totals,
            out=np.full_like(sampled_correct, np.nan),
            where=valid,
        )
        out[start : start + size] = np.nanmean(recalls, axis=1)
    return out


def bootstrap_gap(
    frame: pd.DataFrame,
    context: str,
    best: str,
    worst: str,
    draws: int,
    seed: int,
) -> Dict[str, float]:
    left = frame.loc[frame[context].astype(str).eq(best)].copy()
    right = frame.loc[frame[context].astype(str).eq(worst)].copy()
    labels = sorted(set(left["true_label"].astype(str)) | set(right["true_label"].astype(str)))
    lc, lt = cluster_class_arrays(left, labels)
    rc, rt = cluster_class_arrays(right, labels)
    rng = np.random.default_rng(seed)
    left_ba = bootstrap_ba(lc, lt, draws, rng)
    right_ba = bootstrap_ba(rc, rt, draws, rng)
    gap = left_ba - right_ba
    gap = gap[np.isfinite(gap)]
    if not len(gap):
        return {"ci_low": np.nan, "ci_high": np.nan, "n_bootstrap": 0}
    return {
        "ci_low": float(np.quantile(gap, 0.025)),
        "ci_high": float(np.quantile(gap, 0.975)),
        "n_bootstrap": int(len(gap)),
    }


def clustered_gap_test(
    frame: pd.DataFrame,
    context: str,
    observed_gap: float,
    draws: int,
    seed: int,
) -> Dict[str, float]:
    """Label-stratified donor/patient-block permutation test."""
    rng = np.random.default_rng(seed)
    labels = sorted(frame["true_label"].astype(str).unique())
    contexts = [
        str(value)
        for value, group in frame.groupby(context, sort=True)
        if group["true_label"].nunique() >= 2
    ]
    clusters = sorted(frame["_cluster"].astype(str).unique())
    if len(contexts) < 2 or not clusters:
        return {"p_value": np.nan, "n_null": 0, "test_method": "unavailable"}
    cluster_index = {value: idx for idx, value in enumerate(clusters)}
    context_index = {value: idx for idx, value in enumerate(contexts)}
    blocks_by_label = []
    local = frame.loc[frame[context].astype(str).isin(contexts)].copy()
    local["_correct"] = local["true_label"].astype(str).eq(
        local["pred_label"].astype(str)
    )
    for label in labels:
        blocks = (
            local.loc[local["true_label"].astype(str).eq(label)]
            .groupby(["_cluster", context], as_index=False)
            .agg(total=("_correct", "size"), correct=("_correct", "sum"))
        )
        if blocks.empty:
            continue
        blocks_by_label.append(
            {
                "cluster": np.asarray(
                    [cluster_index[str(value)] for value in blocks["_cluster"]],
                    dtype=int,
                ),
                "context": np.asarray(
                    [context_index[str(value)] for value in blocks[context]],
                    dtype=int,
                ),
                "total": blocks["total"].to_numpy(dtype=float),
                "correct": blocks["correct"].to_numpy(dtype=float),
            }
        )
    null_gaps = np.full(draws, np.nan, dtype=np.float64)
    batch = 256
    for start in range(0, draws, batch):
        size = min(batch, draws - start)
        shared_cluster_rank = rng.random((size, len(clusters)))
        perturbed_correct = np.zeros(
            (size, len(contexts), len(blocks_by_label)),
            dtype=np.float64,
        )
        perturbed_totals = np.zeros_like(perturbed_correct)
        for label_index, block in enumerate(blocks_by_label):
            rank = shared_cluster_rank[:, block["cluster"]]
            rank = rank + 1e-9 * rng.random(rank.shape)
            order = np.argsort(rank, axis=1)
            assigned = block["context"][order]
            membership = np.eye(len(contexts), dtype=np.float64)[assigned]
            perturbed_correct[:, :, label_index] = np.einsum(
                "brc,r->bc",
                membership,
                block["correct"],
            )
            perturbed_totals[:, :, label_index] = np.einsum(
                "brc,r->bc",
                membership,
                block["total"],
            )
        recalls = np.divide(
            perturbed_correct,
            perturbed_totals,
            out=np.full_like(perturbed_correct, np.nan),
            where=perturbed_totals > 0,
        )
        scores = np.nanmean(recalls, axis=2)
        null_gaps[start : start + size] = (
            np.nanmax(scores, axis=1) - np.nanmin(scores, axis=1)
        )
    null_gaps = null_gaps[np.isfinite(null_gaps)]
    if not len(null_gaps):
        return {"p_value": np.nan, "n_null": 0, "test_method": "unavailable"}
    p_value = (1.0 + np.sum(null_gaps >= float(observed_gap))) / (
        len(null_gaps) + 1.0
    )
    return {
        "p_value": float(p_value),
        "n_null": int(len(null_gaps)),
        "test_method": "label_stratified_cluster_block_permutation",
    }


def prepare_frame(
    path: Path,
    context: str,
    regime: str,
    cluster: str,
    method: str,
) -> pd.DataFrame:
    usecols = None
    frame = pd.read_csv(path, usecols=usecols)
    if "method" in frame.columns:
        frame = frame.loc[frame["method"].astype(str).eq(method)].copy()
    if regime == "observed_context":
        frame = frame.loc[frame["split_type"].astype(str).eq("patient_level_cv")].copy()
    elif "context_field" in frame.columns:
        frame = frame.loc[frame["context_field"].astype(str).eq(context)].copy()
    if context not in frame.columns and "context_value" in frame.columns:
        frame[context] = frame["context_value"].astype(str)
    required = {"true_label", "pred_label", context, cluster}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.dropna(subset=list(required)).copy()
    frame["_cluster"] = frame[cluster].astype(str)
    return frame


def audit_row(
    frame: pd.DataFrame,
    *,
    domain: str,
    dataset: str,
    model: str,
    context: str,
    regime: str,
    source: str,
    min_samples: int,
    min_clusters: int,
    support_threshold: float,
    draws: int,
    seed: int,
) -> Dict[str, object] | None:
    if frame.empty or frame[context].nunique() < 2:
        return None
    support = support_stats(frame, context, "_cluster", min_samples, min_clusters)
    metrics = context_ba(frame, context)
    if len(metrics) < 2:
        return None
    best_row = metrics.loc[metrics["ba"].idxmax()]
    worst_row = metrics.loc[metrics["ba"].idxmin()]
    observed_gap = float(best_row["ba"] - worst_row["ba"])
    boot = bootstrap_gap(
        frame,
        context,
        str(best_row["context_value"]),
        str(worst_row["context_value"]),
        draws,
        seed,
    )
    test = clustered_gap_test(
        frame,
        context,
        observed_gap,
        draws,
        seed + 1000003,
    )
    return {
        "domain": domain,
        "dataset": dataset,
        "model": model,
        "foundation_model": model in FOUNDATION_MODELS,
        "context_axis": context,
        "deployment_regime": regime,
        "n_samples": int(len(frame)),
        "n_clusters": int(frame["_cluster"].nunique()),
        "n_contexts": int(frame[context].nunique()),
        "best_context": str(best_row["context_value"]),
        "worst_context": str(worst_row["context_value"]),
        "best_context_ba": float(best_row["ba"]),
        "worst_context_ba": float(worst_row["ba"]),
        "context_gap": observed_gap,
        "gap_ci_low": boot["ci_low"],
        "gap_ci_high": boot["ci_high"],
        "p_value": test["p_value"],
        "p_value_method": test["test_method"],
        "n_bootstrap": boot["n_bootstrap"],
        "n_null": test["n_null"],
        "support_eligible": bool(
            support["support_coverage"] >= support_threshold
            and support["eligible_contexts"] >= 2
        ),
        **support,
        "source_predictions": source,
    }


def collect_single_cell(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    index = 0
    for run_dir in sorted(Path("data").glob("cellxgene_*_embedding_audit/*")):
        if not run_dir.is_dir() or "smoke" in run_dir.name:
            continue
        context = intended_context(run_dir.name)
        if context is None:
            continue
        tissue = strip_prefix_suffix(
            run_dir.parent.name, "cellxgene_", "_embedding_audit"
        )
        model = model_from_run(run_dir.name, context)
        for regime, filename in (
            ("observed_context", "predictions.csv"),
            ("unseen_context", "leave_one_context_predictions.csv"),
        ):
            path = run_dir / filename
            if not path.exists():
                continue
            frame = prepare_frame(path, context, regime, "donor_id", "erm")
            row = audit_row(
                frame,
                domain="single_cell",
                dataset=f"CELLxGENE:{tissue}",
                model=model,
                context=context,
                regime=regime,
                source=str(path),
                min_samples=args.min_cells,
                min_clusters=args.min_donors,
                support_threshold=args.support_coverage,
                draws=args.n_bootstrap,
                seed=args.seed + index,
            )
            if row:
                rows.append(row)
                index += 1
    return rows


def pathology_patient_frame(path: Path, context: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["method"].astype(str).eq("erm")
        & frame["context_field"].astype(str).eq(context)
    ].copy()
    if frame.empty:
        return frame
    patient = (
        frame.groupby(["task", "model", context, "patient_id"], as_index=False)
        .agg(
            true_label=("true_label", lambda x: x.astype(str).mode().iloc[0]),
            pred_label=("pred_label", lambda x: x.astype(str).mode().iloc[0]),
        )
    )
    patient["_cluster"] = patient["patient_id"].astype(str)
    return patient


def collect_pathology(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    index = 10000
    for path in sorted(Path("data/tcga_image_context_shift").glob("*/leave_one_context_predictions.csv")):
        for context in ("site", "primary_diagnosis"):
            frame = pathology_patient_frame(path, context)
            if frame.empty:
                continue
            task = str(frame["task"].iloc[0])
            model = str(frame["model"].iloc[0])
            row = audit_row(
                frame,
                domain="pathology",
                dataset=f"TCGA:{task}",
                model=model,
                context=context,
                regime="unseen_context",
                source=str(path),
                min_samples=args.pathology_min_patients_per_label,
                min_clusters=args.pathology_min_patients_per_label,
                support_threshold=args.support_coverage,
                draws=args.n_bootstrap,
                seed=args.seed + index,
            )
            if row:
                rows.append(row)
                index += 1
    return rows


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def grouped_summary(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[frame["support_eligible"]].copy()
    rows = []
    groupings: Iterable[Tuple[str, Sequence[str]]] = (
        ("overall", []),
        ("modality", ["domain"]),
        ("context_axis", ["context_axis"]),
        ("deployment_regime", ["deployment_regime"]),
        ("modality_x_regime", ["domain", "deployment_regime"]),
    )
    for scope, columns in groupings:
        grouped = [((), eligible)] if not columns else eligible.groupby(list(columns), dropna=False)
        for key, group in grouped:
            key = key if isinstance(key, tuple) else (key,)
            base = {"scope": scope}
            for column, value in zip(columns, key):
                base[column] = value
            rows.append(
                {
                    **base,
                    "eligible_rows": int(len(group)),
                    "significant_rows_fdr_0_05": int(group["significant_fdr_0_05"].sum()),
                    "significant_fraction": float(group["significant_fdr_0_05"].mean())
                    if len(group)
                    else np.nan,
                    "median_gap": float(group["context_gap"].median()) if len(group) else np.nan,
                    "median_ci_width": float(
                        (group["gap_ci_high"] - group["gap_ci_low"]).median()
                    )
                    if len(group)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def donor_threshold_summary(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for threshold in (3, 5, 8):
        eligibility = []
        for record in frame.itertuples(index=False):
            path = Path(str(record.source_predictions))
            if record.domain == "single_cell":
                cluster = "donor_id"
                method = "erm"
                min_samples = args.min_cells
                source = prepare_frame(
                    path,
                    str(record.context_axis),
                    str(record.deployment_regime),
                    cluster,
                    method,
                )
            else:
                source = pathology_patient_frame(path, str(record.context_axis))
                cluster = "_cluster"
                min_samples = args.pathology_min_patients_per_label
            support = support_stats(
                source,
                str(record.context_axis),
                cluster,
                min_samples,
                threshold,
            )
            eligibility.append(
                bool(
                    support["support_coverage"] >= args.support_coverage
                    and support["eligible_contexts"] >= 2
                )
            )
        sub = frame.loc[np.asarray(eligibility, dtype=bool)]
        rows.append(
            {
                "d_min": threshold,
                "eligible_rows": int(len(sub)),
                "support_coverage_mean": float(sub["support_coverage"].mean()) if len(sub) else np.nan,
                "median_gap": float(sub["context_gap"].median()) if len(sub) else np.nan,
                "median_ci_width": float(
                    (sub["gap_ci_high"] - sub["gap_ci_low"]).median()
                )
                if len(sub)
                else np.nan,
                "tp53_site_50_86_claim_status": "support met"
                if threshold <= 8
                else "support not met",
                "pancreas_acinar_claim_status": "support met"
                if threshold <= 3
                else "support not met",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_single_cell(args) + collect_pathology(args)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No audit rows were collected")
    frame["q_value"] = np.nan
    eligible = frame["support_eligible"] & frame["p_value"].notna()
    frame.loc[eligible, "q_value"] = bh_adjust(frame.loc[eligible, "p_value"])
    frame["significant_fdr_0_05"] = frame["support_eligible"] & frame["q_value"].le(0.05)
    frame = frame.sort_values(
        ["domain", "dataset", "model", "context_axis", "deployment_regime"]
    ).reset_index(drop=True)
    frame.to_csv(args.output_dir / "audit_row_bootstrap_fdr.csv", index=False)
    summary = grouped_summary(frame)
    summary.to_csv(args.output_dir / "audit_failure_summary.csv", index=False)
    donor = donor_threshold_summary(frame, args)
    donor.to_csv(args.output_dir / "donor_threshold_sensitivity.csv", index=False)

    eligible_frame = frame.loc[frame["support_eligible"]]
    foundation = eligible_frame.loc[eligible_frame["foundation_model"]]
    payload = {
        "n_bootstrap": args.n_bootstrap,
        "support_rule": {
            "single_cell_min_cells": args.min_cells,
            "min_donors": args.min_donors,
            "pathology_min_patients_per_label": args.pathology_min_patients_per_label,
            "coverage": args.support_coverage,
        },
        "all_representations": {
            "eligible_rows": int(len(eligible_frame)),
            "significant_rows_fdr_0_05": int(eligible_frame["significant_fdr_0_05"].sum()),
            "median_gap": float(eligible_frame["context_gap"].median()),
        },
        "foundation_models": {
            "eligible_rows": int(len(foundation)),
            "significant_rows_fdr_0_05": int(foundation["significant_fdr_0_05"].sum()),
            "median_gap": float(foundation["context_gap"].median()) if len(foundation) else np.nan,
        },
        "note": (
            "Best and worst bins are frozen only for clustered-bootstrap confidence "
            "intervals. P-values use label-stratified donor/patient-block permutation "
            "with shared cluster ranks across labels and recompute the adaptive "
            "max-minus-min statistic. BH correction is "
            "applied jointly across support-eligible evaluations."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
