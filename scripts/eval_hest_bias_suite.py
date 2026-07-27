#!/usr/bin/env python3
"""HEST context-shift and shortcut bias audit suite.

The suite is intentionally lightweight: it consumes frozen patch embeddings and
optionally long-form histology-to-gene predictions produced by the existing
HEST baseline script. It reports representation shortcut probes, conditional
shortcut probes, kNN context enrichment, downstream subgroup gaps, bootstrap CIs,
and a simple group-calibration mitigation on predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MANIFEST = Path("data/metadata/hest/selected_metadata_manifest.csv")
DEFAULT_SOURCE_METADATA = Path("data/metadata/hest/HEST_v1_3_0.csv")
DEFAULT_FEATURE_DIR = Path("data/embeddings/hest/uni")
DEFAULT_OUTDIR = Path("outputs/hest/bias_suite")

BASE_FIELDS = [
    "platform",
    "site",
    "study_id",
    "organ",
    "disease",
    "species",
    "tissue_processing",
    "preservation_method",
    "magnification_group",
    "resolution_bin",
    "spot_diameter_bin",
    "pixel_size_bin",
    "nb_genes_bin",
    "spots_under_tissue_bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-metadata-csv", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--max-spots-per-sample", type=int, default=2000)
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument("--max-cv-folds", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260609)
    return parser.parse_args()


def clean_label(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA" or text.lower() == "nan":
        return None
    return text


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def clean_metric(value: Optional[float]) -> object:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return round(float(value), 6)


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise SystemExit(f"Manifest lacks sample_id: {path}")
    return df


def enrich_manifest(manifest: pd.DataFrame, source_metadata_csv: Path) -> pd.DataFrame:
    df = manifest.copy()
    if source_metadata_csv.is_file():
        source = pd.read_csv(source_metadata_csv)
        source_cols = [
            "id",
            "inter_spot_dist",
            "spot_diameter",
            "spots_under_tissue",
            "preservation_method",
            "nb_genes",
            "pixel_size_um_embedded",
            "pixel_size_um_estimated",
            "magnification",
            "subseries",
            "tissue",
            "data_publication_date",
        ]
        source_cols = [col for col in source_cols if col in source.columns]
        source = source[source_cols].rename(columns={"id": "source_metadata_id"})
        if "source_metadata_id" in df.columns:
            df = df.merge(source, on="source_metadata_id", how="left", suffixes=("", "_source"))
        else:
            source = source.rename(columns={"source_metadata_id": "sample_id"})
            df = df.merge(source, on="sample_id", how="left", suffixes=("", "_source"))

    if "preservation_method" in df.columns and "tissue_processing" in df.columns:
        df["tissue_processing"] = df["tissue_processing"].where(
            df["tissue_processing"].map(clean_label).notna(),
            df["preservation_method"],
        )

    for col, out_col, bins in [
        ("inter_spot_dist", "resolution_bin", 3),
        ("spot_diameter", "spot_diameter_bin", 3),
        ("pixel_size_um_estimated", "pixel_size_bin", 3),
        ("nb_genes", "nb_genes_bin", 3),
        ("spots_under_tissue", "spots_under_tissue_bin", 3),
    ]:
        if col in df.columns:
            df[out_col] = quantile_bin(df[col], bins=bins, prefix=out_col.replace("_bin", ""))

    if "magnification" in df.columns:
        df["magnification_group"] = df["magnification"].map(clean_label)
    return df


def quantile_bin(values: pd.Series, bins: int, prefix: str) -> pd.Series:
    numeric = values.map(safe_float)
    valid = numeric[np.isfinite(numeric)]
    out = pd.Series(["NA"] * len(values), index=values.index, dtype=object)
    if valid.nunique() < 2:
        return out
    try:
        codes = pd.qcut(valid, q=min(bins, valid.nunique()), labels=False, duplicates="drop")
        out.loc[valid.index] = [f"{prefix}_q{int(code) + 1}" for code in codes]
    except ValueError:
        median = float(valid.median())
        out.loc[valid.index] = np.where(valid <= median, f"{prefix}_low", f"{prefix}_high")
    return out


def load_sample_feature(sample_id: str, feature_dir: Path, max_spots: int, seed: int) -> Optional[np.ndarray]:
    path = feature_dir / f"{sample_id}.npz"
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    x = data["features"].astype(np.float32)
    if x.shape[0] > max_spots:
        rng = np.random.default_rng(seed + (abs(hash(sample_id)) % 100000))
        idx = np.sort(rng.choice(x.shape[0], size=max_spots, replace=False))
        x = x[idx]
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


def load_sample_table(manifest: pd.DataFrame, feature_dir: Path, max_spots: int, seed: int) -> Tuple[np.ndarray, pd.DataFrame, List[Dict[str, object]]]:
    features: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []
    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        x = load_sample_feature(sample_id, feature_dir, max_spots, seed)
        if x is None:
            status_rows.append({"sample_id": sample_id, "status": "missing_feature"})
            continue
        features.append(x)
        rows.append(row.to_dict())
        status_rows.append({"sample_id": sample_id, "status": "ok", "feature_dim": int(x.shape[0])})
    if not features:
        raise SystemExit(f"No feature files found in {feature_dir}")
    return np.vstack(features).astype(np.float32), pd.DataFrame(rows), status_rows


def label_vector(meta: pd.DataFrame, field: str) -> Tuple[np.ndarray, np.ndarray]:
    labels = meta[field].map(clean_label) if field in meta.columns else pd.Series([None] * len(meta))
    keep = labels.notna().to_numpy()
    return keep, labels.fillna("NA").astype(str).to_numpy()


def can_probe(y: np.ndarray, min_classes: int, min_class_count: int) -> bool:
    counts = Counter(y)
    kept = [count for count in counts.values() if count >= min_class_count]
    return len(kept) >= min_classes


def cv_probe(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> Dict[str, object]:
    counts = Counter(y)
    usable_classes = sorted([label for label, count in counts.items() if count >= args.min_class_count])
    keep = np.isin(y, usable_classes)
    y_keep = y[keep]
    x_keep = x[keep]
    if not can_probe(y_keep, args.min_classes, args.min_class_count):
        return {
            "status": "skipped_insufficient_classes",
            "n_samples": int(len(y_keep)),
            "n_classes": int(len(set(y_keep))),
        }
    min_count = min(Counter(y_keep).values())
    n_splits = min(args.max_cv_folds, min_count)
    if n_splits < 2:
        return {
            "status": "skipped_insufficient_folds",
            "n_samples": int(len(y_keep)),
            "n_classes": int(len(set(y_keep))),
        }

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1),
    )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    scores = []
    for train_idx, test_idx in splitter.split(x_keep, y_keep):
        model.fit(x_keep[train_idx], y_keep[train_idx])
        pred = model.predict(x_keep[test_idx])
        scores.append(float(balanced_accuracy_score(y_keep[test_idx], pred)))
    return {
        "status": "ok",
        "n_samples": int(len(y_keep)),
        "n_classes": int(len(set(y_keep))),
        "n_splits": int(n_splits),
        "balanced_accuracy_mean": clean_metric(float(np.mean(scores))),
        "balanced_accuracy_std": clean_metric(float(np.std(scores))),
        "class_counts": ";".join(f"{key}:{counts[key]}" for key in sorted(usable_classes)),
    }


def knn_enrichment(x: np.ndarray, y: np.ndarray, k: int) -> Dict[str, object]:
    if len(y) <= k or len(set(y)) < 2:
        return {"status": "skipped", "n_samples": int(len(y)), "n_classes": int(len(set(y)))}
    x_scaled = StandardScaler().fit_transform(x)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(y))).fit(x_scaled)
    indices = nbrs.kneighbors(x_scaled, return_distance=False)[:, 1:]
    local_same = np.mean(y[indices] == y[:, None], axis=1)
    base = max(Counter(y).values()) / len(y)
    return {
        "status": "ok",
        "n_samples": int(len(y)),
        "n_classes": int(len(set(y))),
        "k": int(indices.shape[1]),
        "mean_knn_same_group": clean_metric(float(np.mean(local_same))),
        "majority_class_baseline": clean_metric(float(base)),
        "enrichment_over_baseline": clean_metric(float(np.mean(local_same) - base)),
    }


def representation_audits(x: np.ndarray, meta: pd.DataFrame, args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    probe_rows: List[Dict[str, object]] = []
    knn_rows: List[Dict[str, object]] = []
    conditional_rows: List[Dict[str, object]] = []

    for field in BASE_FIELDS:
        if field not in meta.columns:
            continue
        keep, y = label_vector(meta, field)
        y_keep = y[keep]
        x_keep = x[keep]
        probe = cv_probe(x_keep, y_keep, args)
        probe.update({"field": field, "audit": "sample_cv_probe"})
        probe_rows.append(probe)
        knn = knn_enrichment(x_keep, y_keep, args.knn_k)
        knn.update({"field": field, "audit": "sample_knn_enrichment"})
        knn_rows.append(knn)

    for target, condition in [
        ("site", "organ"),
        ("site", "disease"),
        ("platform", "organ"),
        ("platform", "disease"),
        ("platform", "site"),
        ("organ", "platform"),
        ("disease", "platform"),
    ]:
        if target not in meta.columns or condition not in meta.columns:
            continue
        cond_values = meta[condition].map(clean_label)
        for cond_value in sorted(set(v for v in cond_values if v is not None)):
            cond_mask = cond_values == cond_value
            keep, y = label_vector(meta.loc[cond_mask].reset_index(drop=True), target)
            if keep.sum() < args.min_classes * args.min_class_count:
                continue
            probe = cv_probe(x[cond_mask.to_numpy()][keep], y[keep], args)
            probe.update(
                {
                    "target_field": target,
                    "condition_field": condition,
                    "condition_value": cond_value,
                    "audit": "conditional_sample_cv_probe",
                }
            )
            conditional_rows.append(probe)
    return probe_rows, knn_rows, conditional_rows


def safe_corr(x: Iterable[float], y: Iterable[float], min_pairs: int = 3) -> Optional[float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if len(x_arr) < min_pairs or float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def prediction_group_metrics(preds: pd.DataFrame, manifest: pd.DataFrame, group_fields: Sequence[str]) -> List[Dict[str, object]]:
    df = preds.merge(manifest.drop_duplicates("sample_id"), on="sample_id", how="left")
    rows: List[Dict[str, object]] = []
    rows.append(prediction_summary_for_group(df, "ALL", "ALL"))
    for field in group_fields:
        if field not in df.columns:
            continue
        for value, group_df in df.groupby(field, dropna=False):
            label = "NA" if pd.isna(value) else str(value)
            if label == "NA":
                continue
            rows.append(prediction_summary_for_group(group_df, field, label))
    return rows


def prediction_summary_for_group(df: pd.DataFrame, group_field: str, group_value: str) -> Dict[str, object]:
    gene_scores = []
    for _, gene_df in df.groupby("gene"):
        corr = safe_corr(gene_df["y_true"], gene_df["y_pred"])
        if corr is not None:
            gene_scores.append(corr)
    true_nonzero = np.asarray(df["y_true"], dtype=float) > 0
    pred_nonzero = np.asarray(df["y_pred"], dtype=float) > 0
    return {
        "group_field": group_field,
        "group_value": group_value,
        "n_rows": int(len(df)),
        "n_samples": int(df["sample_id"].nunique()),
        "n_genes": int(df["gene"].nunique()),
        "average_gene_pearson": clean_metric(float(np.mean(gene_scores)) if gene_scores else None),
        "true_nonzero_rate": clean_metric(float(np.mean(true_nonzero)) if len(true_nonzero) else None),
        "pred_nonzero_rate": clean_metric(float(np.mean(pred_nonzero)) if len(pred_nonzero) else None),
        "nonzero_calibration_gap": clean_metric(float(abs(np.mean(true_nonzero) - np.mean(pred_nonzero))) if len(true_nonzero) else None),
    }


def gap_rows(group_rows: List[Dict[str, object]], fields: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for field in fields:
        values = [
            row for row in group_rows
            if row["group_field"] == field and isinstance(row["average_gene_pearson"], float)
        ]
        if not values:
            continue
        worst = min(values, key=lambda row: row["average_gene_pearson"])
        best = max(values, key=lambda row: row["average_gene_pearson"])
        rows.append(
            {
                "group_field": field,
                "n_groups": len(values),
                "best_group": best["group_value"],
                "best_average_gene_pearson": best["average_gene_pearson"],
                "worst_group": worst["group_value"],
                "worst_average_gene_pearson": worst["average_gene_pearson"],
                "best_minus_worst_gap": clean_metric(float(best["average_gene_pearson"]) - float(worst["average_gene_pearson"])),
            }
        )
    return rows


def bootstrap_gaps(preds: pd.DataFrame, manifest: pd.DataFrame, fields: Sequence[str], iters: int, seed: int) -> List[Dict[str, object]]:
    if iters <= 0:
        return []
    rng = np.random.default_rng(seed)
    sample_ids = np.asarray(sorted(preds["sample_id"].unique()))
    rows = []
    for field in fields:
        if field not in manifest.columns:
            continue
        gaps = []
        for _ in range(iters):
            boot_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
            boot = preds[preds["sample_id"].isin(boot_ids)]
            metrics = prediction_group_metrics(boot, manifest, [field])
            gaps_for_field = gap_rows(metrics, [field])
            if gaps_for_field and isinstance(gaps_for_field[0]["best_minus_worst_gap"], float):
                gaps.append(float(gaps_for_field[0]["best_minus_worst_gap"]))
        if gaps:
            rows.append(
                {
                    "group_field": field,
                    "n_bootstrap": len(gaps),
                    "gap_mean": clean_metric(float(np.mean(gaps))),
                    "gap_ci_low": clean_metric(float(np.quantile(gaps, 0.025))),
                    "gap_ci_high": clean_metric(float(np.quantile(gaps, 0.975))),
                }
            )
    return rows


def group_calibration(preds: pd.DataFrame, manifest: pd.DataFrame, fields: Sequence[str]) -> List[Dict[str, object]]:
    df = preds.merge(manifest.drop_duplicates("sample_id"), on="sample_id", how="left")
    if "split" not in df.columns:
        return []
    train = df[df["split"].astype(str) == "train"].copy()
    eval_df = df[df["split"].astype(str) != "train"].copy()
    if train.empty or eval_df.empty:
        return []
    rows = []
    base = prediction_summary_for_group(eval_df, "ALL", "ALL")
    rows.append(
        {
            "calibration_field": "none",
            "eval_average_gene_pearson": base["average_gene_pearson"],
            "eval_nonzero_calibration_gap": base["nonzero_calibration_gap"],
            "status": "baseline",
        }
    )
    for field in fields:
        if field not in df.columns:
            continue
        correction = (
            train.assign(residual=train["y_true"].astype(float) - train["y_pred"].astype(float))
            .groupby(["gene", field], dropna=False)["residual"]
            .mean()
            .to_dict()
        )
        global_correction = train.assign(residual=train["y_true"].astype(float) - train["y_pred"].astype(float)).groupby("gene")["residual"].mean().to_dict()
        adjusted = eval_df.copy()
        deltas = []
        for _, row in adjusted.iterrows():
            key = (row["gene"], row[field])
            deltas.append(float(correction.get(key, global_correction.get(row["gene"], 0.0))))
        adjusted["y_pred"] = adjusted["y_pred"].astype(float) + np.asarray(deltas, dtype=float)
        summary = prediction_summary_for_group(adjusted, "ALL", "ALL")
        rows.append(
            {
                "calibration_field": field,
                "eval_average_gene_pearson": summary["average_gene_pearson"],
                "eval_nonzero_calibration_gap": summary["nonzero_calibration_gap"],
                "status": "ok",
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = enrich_manifest(read_manifest(args.manifest_csv), args.source_metadata_csv)
    manifest.to_csv(args.outdir / "enriched_manifest.csv", index=False)

    x, sample_meta, feature_status = load_sample_table(
        manifest=manifest,
        feature_dir=args.feature_dir,
        max_spots=args.max_spots_per_sample,
        seed=args.seed,
    )
    probe_rows, knn_rows, conditional_rows = representation_audits(x, sample_meta, args)
    write_csv(args.outdir / "sample_cv_probe_results.csv", probe_rows)
    write_csv(args.outdir / "sample_knn_enrichment.csv", knn_rows)
    write_csv(args.outdir / "conditional_probe_results.csv", conditional_rows)
    write_csv(args.outdir / "feature_status.csv", feature_status)

    prediction_outputs: Dict[str, object] = {}
    group_fields = [field for field in BASE_FIELDS if field in manifest.columns]
    if args.predictions_csv and args.predictions_csv.is_file():
        preds = pd.read_csv(args.predictions_csv)
        for col in ["y_true", "y_pred"]:
            preds[col] = pd.to_numeric(preds[col], errors="coerce")
        group_metric_rows = prediction_group_metrics(preds, manifest, group_fields)
        pred_gap_rows = gap_rows(group_metric_rows, group_fields)
        ci_rows = bootstrap_gaps(preds, manifest, group_fields, args.bootstrap_iters, args.seed)
        calibration_rows = group_calibration(preds, manifest, group_fields)
        write_csv(args.outdir / "prediction_group_metrics.csv", group_metric_rows)
        write_csv(args.outdir / "prediction_group_gaps.csv", pred_gap_rows)
        write_csv(args.outdir / "prediction_gap_bootstrap_ci.csv", ci_rows)
        write_csv(args.outdir / "group_calibration_mitigation.csv", calibration_rows)
        prediction_outputs = {
            "prediction_group_metrics": len(group_metric_rows),
            "prediction_group_gaps": len(pred_gap_rows),
            "bootstrap_rows": len(ci_rows),
            "calibration_rows": len(calibration_rows),
        }

    summary = {
        "model_name": args.model_name,
        "manifest_csv": str(args.manifest_csv),
        "source_metadata_csv": str(args.source_metadata_csv),
        "feature_dir": str(args.feature_dir),
        "predictions_csv": str(args.predictions_csv) if args.predictions_csv else None,
        "n_samples_with_features": int(len(sample_meta)),
        "sample_feature_dim": int(x.shape[1]),
        "probe_rows": len(probe_rows),
        "knn_rows": len(knn_rows),
        "conditional_probe_rows": len(conditional_rows),
        "prediction_outputs": prediction_outputs,
    }
    with (args.outdir / "bias_suite_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
