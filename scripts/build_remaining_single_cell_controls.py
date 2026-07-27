#!/usr/bin/env python3
"""Build remaining reviewer-control tables for representative single-cell rows."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    model_label: str
    context: str
    context_label: str
    run_dir: Path


RUNS = [
    RunSpec("geneformer_v1", "Geneformer", "assay", "assay", Path("data/cellxgene_bone_marrow_embedding_audit/geneformer_v1_assay_erm")),
    RunSpec("geneformer_v1", "Geneformer", "dataset_id", "dataset", Path("data/cellxgene_bone_marrow_embedding_audit/geneformer_v1_dataset_erm")),
    RunSpec("scgpt_continual", "scGPT", "assay", "assay", Path("data/cellxgene_bone_marrow_embedding_audit/scgpt_continual_assay_erm")),
    RunSpec("scgpt_continual", "scGPT", "dataset_id", "dataset", Path("data/cellxgene_bone_marrow_embedding_audit/scgpt_continual_dataset_erm")),
    RunSpec("scvi_style_vae", "scVI-style", "assay", "assay", Path("data/cellxgene_bone_marrow_embedding_audit/scvi_style_vae_assay_erm")),
    RunSpec("scvi_style_vae", "scVI-style", "dataset_id", "dataset", Path("data/cellxgene_bone_marrow_embedding_audit/scvi_style_vae_dataset_erm")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/reviewer_control_tables"))
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--min-group-cells", type=int, default=20)
    parser.add_argument("--min-probe-cells", type=int, default=200)
    parser.add_argument("--max-probe-classes", type=int, default=30)
    parser.add_argument("--mlp-max-iter", type=int, default=20)
    parser.add_argument("--probe-random-projection-components", type=int, default=64)
    parser.add_argument("--mlp-max-balanced-cells-per-class", type=int, default=300)
    parser.add_argument("--skip-mlp", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_embedding(path: Path, key: str = "embeddings") -> np.ndarray:
    if path.suffix.lower() == ".npz":
        loaded = np.load(path)
        if key not in loaded:
            raise KeyError(f"{key!r} not found in {path}; available keys: {loaded.files}")
        arr = loaded[key]
    elif path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        raise ValueError(f"Unsupported embedding file: {path}")
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"Embedding must be 2D, got {out.shape} from {path}")
    return out


def ci(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan")}
    return {"ci_low": float(np.percentile(arr, 2.5)), "ci_high": float(np.percentile(arr, 97.5))}


def balanced_accuracy(y_true: Sequence[object], y_pred: Sequence[object], weights: np.ndarray | None = None) -> float:
    y_true_arr = np.asarray(y_true, dtype=str)
    y_pred_arr = np.asarray(y_pred, dtype=str)
    if weights is None:
        weights_arr = np.ones(len(y_true_arr), dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
    recalls: List[float] = []
    for label in sorted(np.unique(y_true_arr[weights_arr > 0])):
        mask = (y_true_arr == label) & (weights_arr > 0)
        denom = float(weights_arr[mask].sum())
        if denom > 0:
            recalls.append(float((weights_arr[mask] * (y_pred_arr[mask] == label)).sum() / denom))
    return float(np.mean(recalls)) if recalls else float("nan")


def metric_row(y_true: Sequence[object], y_pred: Sequence[object], prefix: Mapping[str, object]) -> Dict[str, object]:
    y_true_arr = np.asarray(y_true, dtype=str)
    y_pred_arr = np.asarray(y_pred, dtype=str)
    labels = sorted(np.unique(y_true_arr))
    return {
        **dict(prefix),
        "n": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": balanced_accuracy(y_true_arr, y_pred_arr),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro")),
    }


def context_ba_rows(
    df: pd.DataFrame,
    context_col: str,
    min_group_cells: int,
    weights: np.ndarray | None = None,
) -> List[Dict[str, object]]:
    y_true = df["true_label"].astype(str).to_numpy()
    y_pred = df["pred_label"].astype(str).to_numpy()
    values = df[context_col].astype(str).to_numpy()
    if weights is None:
        weights = np.ones(len(df), dtype=float)
    rows: List[Dict[str, object]] = []
    for value in sorted(pd.unique(values)):
        mask = values == value
        active = mask & (weights > 0)
        if float(weights[mask].sum()) < min_group_cells or len(np.unique(y_true[active])) < 2:
            continue
        rows.append(
            {
                "context_value": value,
                "n_cells": int(mask.sum()),
                "weighted_n_cells": float(weights[mask].sum()),
                "n_donors": int(df.loc[mask, "donor_id"].astype(str).nunique()),
                "balanced_accuracy": balanced_accuracy(y_true[mask], y_pred[mask], weights[mask]),
            }
        )
    return rows


def gap_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    valid = [row for row in rows if math.isfinite(float(row["balanced_accuracy"]))]
    if len(valid) < 2:
        return {
            "best_bin": "",
            "worst_bin": "",
            "best_bin_ba": float("nan"),
            "worst_bin_ba": float("nan"),
            "gap": float("nan"),
            "worst_bin_n_cells": 0,
            "worst_bin_n_donors": 0,
        }
    best = max(valid, key=lambda row: float(row["balanced_accuracy"]))
    worst = min(valid, key=lambda row: float(row["balanced_accuracy"]))
    return {
        "best_bin": str(best["context_value"]),
        "worst_bin": str(worst["context_value"]),
        "best_bin_ba": float(best["balanced_accuracy"]),
        "worst_bin_ba": float(worst["balanced_accuracy"]),
        "gap": float(best["balanced_accuracy"]) - float(worst["balanced_accuracy"]),
        "worst_bin_n_cells": int(worst["n_cells"]),
        "worst_bin_n_donors": int(worst["n_donors"]),
    }


def donor_bootstrap_summary(
    df: pd.DataFrame,
    context_col: str,
    min_group_cells: int,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    observed_rows = context_ba_rows(df, context_col, min_group_cells)
    observed = gap_summary(observed_rows)
    overall_observed = balanced_accuracy(df["true_label"].astype(str), df["pred_label"].astype(str))

    donors = pd.Index(df["donor_id"].astype(str).unique())
    donor_codes = pd.Categorical(df["donor_id"].astype(str), categories=donors).codes
    rng = np.random.default_rng(seed)
    overall_draws: List[float] = []
    worst_draws: List[float] = []
    gap_draws: List[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(donors), size=len(donors))
        donor_weights = np.bincount(sampled, minlength=len(donors)).astype(float)
        weights = donor_weights[donor_codes]
        overall_draws.append(balanced_accuracy(df["true_label"].astype(str), df["pred_label"].astype(str), weights))
        boot_rows = context_ba_rows(df, context_col, min_group_cells, weights)
        boot_gap = gap_summary(boot_rows)
        worst_draws.append(float(boot_gap["worst_bin_ba"]))
        gap_draws.append(float(boot_gap["gap"]))

    overall_ci = ci(overall_draws)
    worst_ci = ci(worst_draws)
    gap_ci = ci(gap_draws)
    return {
        **observed,
        "average_ba": overall_observed,
        "average_ba_ci_low": overall_ci["ci_low"],
        "average_ba_ci_high": overall_ci["ci_high"],
        "worst_bin_ba_ci_low": worst_ci["ci_low"],
        "worst_bin_ba_ci_high": worst_ci["ci_high"],
        "gap_ci_low": gap_ci["ci_low"],
        "gap_ci_high": gap_ci["ci_high"],
        "n_bootstrap": int(n_bootstrap),
        "n_donors": int(len(donors)),
    }


def prediction_frames(run: RunSpec) -> List[Dict[str, object]]:
    frames: List[Dict[str, object]] = []
    patient = pd.read_csv(run.run_dir / "predictions.csv")
    frames.append(
        {
            "split_type": "patient_level_cv",
            "split_label": "patient-CV",
            "df": patient,
            "context_col": run.context,
        }
    )
    leave_one_path = run.run_dir / "leave_one_context_predictions.csv"
    if leave_one_path.exists():
        leave_one = pd.read_csv(leave_one_path)
        leave_one = leave_one[leave_one["context_field"].astype(str) == run.context].copy()
        if not leave_one.empty:
            frames.append(
                {
                    "split_type": "leave_one_context",
                    "split_label": "leave-one",
                    "df": leave_one,
                    "context_col": "context_value",
                }
            )
    return frames


def build_representative_ci(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    seed = args.seed
    for run in RUNS:
        for frame in prediction_frames(run):
            summary = donor_bootstrap_summary(
                frame["df"],
                str(frame["context_col"]),
                args.min_group_cells,
                args.n_bootstrap,
                seed,
            )
            seed += 1
            rows.append(
                {
                    "domain": "single_cell",
                    "dataset": "CELLxGENE:bone_marrow",
                    "task": "cell type annotation",
                    "model": run.model_key,
                    "model_label": run.model_label,
                    "context": run.context,
                    "context_label": run.context_label,
                    "split_type": frame["split_type"],
                    "split_label": frame["split_label"],
                    "n_cells": int(len(frame["df"])),
                    **summary,
                    "source_dir": str(run.run_dir),
                }
            )
    return pd.DataFrame(rows)


def same_context_train_support(df: pd.DataFrame, context_col: str, context_value: str, split_type: str) -> Dict[str, float]:
    if split_type != "patient_level_cv" or "fold" not in df.columns:
        return {
            "same_context_train_cells_weighted": 0.0 if split_type == "leave_one_context" else float("nan"),
            "same_context_train_cells_min": 0.0 if split_type == "leave_one_context" else float("nan"),
        }
    values: List[float] = []
    weights: List[float] = []
    context = df[context_col].astype(str)
    for fold, fold_df in df.groupby("fold", sort=True):
        test_n = int(((fold_df[context_col].astype(str)) == context_value).sum())
        train_n = int(((df["fold"] != fold) & (context == context_value)).sum())
        if test_n:
            values.append(float(train_n))
            weights.append(float(test_n))
    if not values:
        return {"same_context_train_cells_weighted": float("nan"), "same_context_train_cells_min": float("nan")}
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    return {
        "same_context_train_cells_weighted": float((values_arr * weights_arr).sum() / weights_arr.sum()),
        "same_context_train_cells_min": float(values_arr.min()),
    }


def matched_reference_distribution(
    df: pd.DataFrame,
    context_col: str,
    reference_value: str,
    n_target: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    ref = df[df[context_col].astype(str) == reference_value].copy()
    if ref.empty or n_target <= 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    y_true = ref["true_label"].astype(str).to_numpy()
    y_pred = ref["pred_label"].astype(str).to_numpy()
    draws = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(ref), size=n_target)
        draws[i] = balanced_accuracy(y_true[idx], y_pred[idx])
    return draws


def build_size_matched_controls(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    seed = args.seed + 1000
    for run in RUNS:
        for frame in prediction_frames(run):
            df = frame["df"]
            context_col = str(frame["context_col"])
            summary = gap_summary(context_ba_rows(df, context_col, args.min_group_cells))
            best_value = str(summary["best_bin"])
            worst_value = str(summary["worst_bin"])
            if not best_value or not worst_value:
                continue
            worst = df[df[context_col].astype(str) == worst_value].copy()
            best = df[df[context_col].astype(str) == best_value].copy()
            matched = matched_reference_distribution(df, context_col, best_value, len(worst), args.n_bootstrap, seed)
            seed += 1
            matched_ci = ci(matched)
            worst_train = same_context_train_support(df, context_col, worst_value, str(frame["split_type"]))
            best_train = same_context_train_support(df, context_col, best_value, str(frame["split_type"]))
            rows.append(
                {
                    "domain": "single_cell",
                    "dataset": "CELLxGENE:bone_marrow",
                    "task": "cell type annotation",
                    "model": run.model_key,
                    "model_label": run.model_label,
                    "context": run.context,
                    "context_label": run.context_label,
                    "split_type": frame["split_type"],
                    "split_label": frame["split_label"],
                    "worst_bin": worst_value,
                    "reference_bin": best_value,
                    "worst_bin_n_cells": int(len(worst)),
                    "reference_bin_n_cells": int(len(best)),
                    "worst_bin_n_donors": int(worst["donor_id"].astype(str).nunique()),
                    "reference_bin_n_donors": int(best["donor_id"].astype(str).nunique()),
                    "worst_bin_ba": float(summary["worst_bin_ba"]),
                    "reference_bin_observed_ba": balanced_accuracy(best["true_label"].astype(str), best["pred_label"].astype(str)),
                    "reference_size_matched_ba_mean": float(np.nanmean(matched)) if len(matched) else float("nan"),
                    "reference_size_matched_ba_ci_low": matched_ci["ci_low"],
                    "reference_size_matched_ba_ci_high": matched_ci["ci_high"],
                    "size_matched_reference_minus_worst": float(np.nanmean(matched) - float(summary["worst_bin_ba"])) if len(matched) else float("nan"),
                    "p_size_matched_reference_le_worst": float(np.mean(matched <= float(summary["worst_bin_ba"]))) if len(matched) else float("nan"),
                    "worst_same_context_train_cells_weighted": worst_train["same_context_train_cells_weighted"],
                    "worst_same_context_train_cells_min": worst_train["same_context_train_cells_min"],
                    "reference_same_context_train_cells_weighted": best_train["same_context_train_cells_weighted"],
                    "reference_same_context_train_cells_min": best_train["same_context_train_cells_min"],
                    "n_bootstrap": int(args.n_bootstrap),
                    "source_dir": str(run.run_dir),
                }
            )
    return pd.DataFrame(rows)


def build_probe_majority_baselines() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run in RUNS:
        probe = pd.read_csv(run.run_dir / "context_probe_results.csv")
        probe = probe[probe["context_field"].astype(str) == run.context].copy()
        if probe.empty:
            continue
        probe["majority_balanced_accuracy"] = 1.0 / pd.to_numeric(probe["n_classes"], errors="coerce")
        rows.append(
            {
                "domain": "single_cell",
                "dataset": "CELLxGENE:bone_marrow",
                "model": run.model_key,
                "model_label": run.model_label,
                "context": run.context,
                "context_label": run.context_label,
                "n_folds": int(probe["fold"].nunique()),
                "n_classes_min": int(pd.to_numeric(probe["n_classes"], errors="coerce").min()),
                "n_classes_max": int(pd.to_numeric(probe["n_classes"], errors="coerce").max()),
                "linear_probe_ba_mean": float(probe["balanced_accuracy"].astype(float).mean()),
                "linear_probe_ba_min": float(probe["balanced_accuracy"].astype(float).min()),
                "linear_probe_ba_max": float(probe["balanced_accuracy"].astype(float).max()),
                "majority_ba_mean": float(probe["majority_balanced_accuracy"].mean()),
                "linear_minus_majority_ba": float(
                    probe["balanced_accuracy"].astype(float).mean() - probe["majority_balanced_accuracy"].mean()
                ),
                "source_dir": str(run.run_dir),
            }
        )
    return pd.DataFrame(rows)


def infer_fold_count(metadata: pd.DataFrame, label_column: str, requested: int) -> int:
    donor_labels = metadata[["donor_id", label_column]].drop_duplicates()
    min_donors = int(donor_labels.groupby(label_column, observed=True)["donor_id"].nunique().min())
    return max(2, min(requested, min_donors))


def balanced_oversample_indices(y: np.ndarray, rng: np.random.Generator, max_per_class: int) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    target = int(min(counts.max(), max_per_class))
    parts: List[np.ndarray] = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        if len(idx) >= target:
            idx = rng.choice(idx, size=target, replace=False)
        else:
            idx = rng.choice(idx, size=target, replace=True)
        parts.append(idx)
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def probe_capacity_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if args.skip_mlp:
        return pd.DataFrame(rows)

    for run in RUNS:
        saved_probe = pd.read_csv(run.run_dir / "context_probe_results.csv")
        saved_probe = saved_probe[saved_probe["context_field"].astype(str) == run.context].copy()
        for _, probe_row in saved_probe.iterrows():
            rows.append(
                {
                    "domain": "single_cell",
                    "dataset": "CELLxGENE:bone_marrow",
                    "model": run.model_key,
                    "model_label": run.model_label,
                    "context": run.context,
                    "context_label": run.context_label,
                    "fold": int(probe_row["fold"]),
                    "probe_type": "linear_logistic_saved",
                    "training_balance": "class_weight_balanced",
                    "n_classes": int(probe_row["n_classes"]),
                    "n_train_cells": int(probe_row["n_train_cells"]),
                    "n_test_cells": int(probe_row["n_test_cells"]),
                    "probe_feature_preprocessing": "full_embedding_saved",
                    "mlp_max_balanced_cells_per_class": -1,
                    "accuracy": float(probe_row["accuracy"]),
                    "balanced_accuracy": float(probe_row["balanced_accuracy"]),
                    "macro_f1": float(probe_row["macro_f1"]),
                    "n": int(probe_row["n"]),
                    "source_dir": str(run.run_dir),
                }
            )

        summary = read_json(run.run_dir / "summary.json")
        metadata = pd.read_csv(str(summary["metadata_csv"]))
        for column in metadata.columns:
            if column != "cell_index":
                metadata[column] = metadata[column].astype(str)
        x_arr = read_embedding(Path(str(summary["embedding_file"])))
        values = metadata[run.context].astype(str).to_numpy()
        counts = pd.Series(values).value_counts()
        keep_values = counts[counts >= args.min_probe_cells].index.tolist()
        if len(keep_values) < 2:
            continue
        if len(keep_values) > args.max_probe_classes:
            keep_values = counts.head(args.max_probe_classes).index.tolist()
        keep = np.isin(values, keep_values)
        x_keep = x_arr[keep]
        preprocessing = "raw"
        if 0 < args.probe_random_projection_components < x_keep.shape[1]:
            projector = GaussianRandomProjection(
                n_components=args.probe_random_projection_components,
                random_state=20260612,
            )
            x_keep = projector.fit_transform(x_keep).astype(np.float32)
            preprocessing = f"gaussian_random_projection_{args.probe_random_projection_components}"
        y_keep = LabelEncoder().fit_transform(values[keep])
        groups = metadata.loc[keep, "donor_id"].astype(str).to_numpy()
        n_folds = infer_fold_count(metadata.loc[keep].copy(), run.context, int(summary.get("n_folds", 2)))
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=20260612)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x_keep, y_keep, groups), start=1):
            if len(np.unique(y_keep[train_idx])) < 2 or len(np.unique(y_keep[test_idx])) < 2:
                continue
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_keep[train_idx]).astype(np.float32)
            x_test = scaler.transform(x_keep[test_idx]).astype(np.float32)
            mlp_rng = np.random.default_rng(args.seed + fold + len(rows))
            balanced_idx = balanced_oversample_indices(
                y_keep[train_idx],
                mlp_rng,
                max_per_class=int(args.mlp_max_balanced_cells_per_class),
            )
            mlp = MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=args.mlp_max_iter,
                early_stopping=False,
                batch_size=256,
                n_iter_no_change=10,
                random_state=args.seed + fold,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                mlp.fit(x_train[balanced_idx], y_keep[train_idx][balanced_idx])
            mlp_pred = mlp.predict(x_test)
            rows.append(
                metric_row(
                    y_keep[test_idx],
                    mlp_pred,
                    {
                        "domain": "single_cell",
                        "dataset": "CELLxGENE:bone_marrow",
                        "model": run.model_key,
                        "model_label": run.model_label,
                        "context": run.context,
                        "context_label": run.context_label,
                        "fold": fold,
                        "probe_type": "balanced_mlp",
                        "n_classes": int(len(np.unique(y_keep))),
                        "n_train_cells": int(len(train_idx)),
                        "n_test_cells": int(len(test_idx)),
                        "probe_feature_preprocessing": preprocessing,
                        "mlp_max_balanced_cells_per_class": int(args.mlp_max_balanced_cells_per_class),
                        "training_balance": "class_balanced_subsample",
                        "mlp_n_iter": int(getattr(mlp, "n_iter_", -1)),
                        "mlp_convergence_warnings": int(sum(isinstance(w.message, ConvergenceWarning) for w in caught)),
                        "source_dir": str(run.run_dir),
                    },
                )
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    grouped_rows: List[Dict[str, object]] = []
    for keys, group in raw.groupby(["model", "model_label", "context", "context_label", "probe_type"], sort=True):
        model, model_label, context, context_label, probe_type = keys
        first = group.iloc[0]
        mlp_iter = pd.to_numeric(group.get("mlp_n_iter", pd.Series([-1])), errors="coerce").fillna(-1)
        mlp_warnings = pd.to_numeric(group.get("mlp_convergence_warnings", pd.Series([0])), errors="coerce").fillna(0)
        grouped_rows.append(
            {
                "domain": "single_cell",
                "dataset": "CELLxGENE:bone_marrow",
                "model": model,
                "model_label": model_label,
                "context": context,
                "context_label": context_label,
                "probe_type": probe_type,
                "training_balance": first.get("training_balance", ""),
                "probe_feature_preprocessing": first.get("probe_feature_preprocessing", ""),
                "mlp_max_balanced_cells_per_class": int(first.get("mlp_max_balanced_cells_per_class", -1)),
                "n_folds": int(group["fold"].nunique()),
                "n_classes": int(group["n_classes"].max()),
                "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
                "balanced_accuracy_min": float(group["balanced_accuracy"].min()),
                "balanced_accuracy_max": float(group["balanced_accuracy"].max()),
                "macro_f1_mean": float(group["macro_f1"].mean()),
                "accuracy_mean": float(group["accuracy"].mean()),
                "mlp_n_iter_max": int(mlp_iter.max()),
                "mlp_convergence_warnings": int(mlp_warnings.sum()),
                "source_dir": str(first["source_dir"]),
            }
        )
    summary = pd.DataFrame(grouped_rows)
    pivot = summary.pivot_table(
        index=["model", "context"],
        columns="probe_type",
        values="balanced_accuracy_mean",
        aggfunc="first",
    )
    deltas = []
    for (model, context), row in pivot.iterrows():
        if "balanced_mlp" in row and "linear_logistic_saved" in row:
            deltas.append(
                {
                    "model": model,
                    "context": context,
                    "mlp_minus_linear_ba": float(row["balanced_mlp"] - row["linear_logistic_saved"]),
                }
            )
    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        summary = summary.merge(delta_df, on=["model", "context"], how="left")
    return summary


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "single_cell_representative_donor_ci.csv": build_representative_ci(args),
        "single_cell_train_size_matched_controls.csv": build_size_matched_controls(args),
        "single_cell_probe_majority_baselines.csv": build_probe_majority_baselines(),
        "single_cell_probe_capacity_sensitivity.csv": probe_capacity_rows(args),
    }
    summary: Dict[str, object] = {
        "n_bootstrap": int(args.n_bootstrap),
        "mlp_max_iter": int(args.mlp_max_iter),
        "probe_random_projection_components": int(args.probe_random_projection_components),
        "mlp_max_balanced_cells_per_class": int(args.mlp_max_balanced_cells_per_class),
        "runs": [str(run.run_dir) for run in RUNS],
        "tables": {},
    }
    for filename, df in outputs.items():
        path = args.output_dir / filename
        write_table(df, path)
        summary["tables"][filename] = {"rows": int(len(df)), "path": str(path)}
    (args.output_dir / "remaining_single_cell_controls_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
