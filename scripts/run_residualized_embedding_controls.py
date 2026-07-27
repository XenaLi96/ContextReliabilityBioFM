#!/usr/bin/env python3
"""Run split-safe residualized embedding controls.

The artifact control regresses frozen embeddings on low-level covariates inside
each train split, then reruns context probes and downstream context-gap audits.
The label/composition controls regress embeddings on the downstream label and
are only used for context probes, because label prediction after removing the
label signal is not a meaningful downstream task.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_tcga_embedding_mitigation import load_embeddings as load_tcga_embeddings  # noqa: E402


CONTROL_TYPES_FOR_PROBES = ["none", "artifact", "label", "artifact_label"]
CONTROL_TYPES_FOR_DOWNSTREAM = ["none", "artifact"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cellxgene", "tcga"], required=True)
    parser.add_argument("--cellxgene-summary-json", type=Path)
    parser.add_argument("--tcga-summary-json", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--embedding-key", default="embeddings")
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--embedding-suffix")
    parser.add_argument("--artifact-file", type=Path)
    parser.add_argument("--artifact-key", default="embeddings")
    parser.add_argument("--artifact-features-dir", type=Path)
    parser.add_argument("--artifact-embedding-suffix", default="_image_stats_embedding.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--label-column")
    parser.add_argument("--path-column", default="slide_file_name")
    parser.add_argument("--donor-column", default="donor_id")
    parser.add_argument("--cell-index-column", default="cell_index")
    parser.add_argument("--context-fields", nargs="*")
    parser.add_argument("--leave-one-context-fields", nargs="*")
    parser.add_argument("--n-folds", type=int)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-probe-cells", type=int, default=200)
    parser.add_argument("--max-probe-classes", type=int, default=30)
    parser.add_argument("--min-holdout-cells", type=int, default=200)
    parser.add_argument("--min-holdout-labels", type=int, default=2)
    return parser.parse_args()


def clean_string(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "control_type",
        "method",
        "fold",
        "split_type",
        "context_field",
        "context_value",
        "metric",
        "sample_key",
        "cell_index",
        "donor_id",
        "true_label",
        "pred_label",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fill_from_summary(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "cellxgene" and args.cellxgene_summary_json:
        summary = read_json(args.cellxgene_summary_json)
        args.metadata_csv = args.metadata_csv or Path(str(summary["metadata_csv"]))
        args.embedding_file = args.embedding_file or Path(str(summary["embedding_file"]))
        args.model_name = args.model_name or str(summary.get("model_name", args.cellxgene_summary_json.parent.name))
        args.label_column = args.label_column or "label"
        args.context_fields = args.context_fields or ["sex", "age_group", "dataset_id", "assay", "disease"]
        mitigation = str(summary.get("mitigation_context_field", "assay"))
        args.leave_one_context_fields = args.leave_one_context_fields or [mitigation]
        args.n_folds = args.n_folds or int(summary.get("n_folds", 2))
    if args.mode == "tcga" and args.tcga_summary_json:
        summary = read_json(args.tcga_summary_json)
        args.metadata_csv = args.metadata_csv or Path(str(summary["metadata_csv"]))
        args.features_dir = args.features_dir or Path(str(summary["features_dir"]))
        args.embedding_suffix = args.embedding_suffix or str(summary["embedding_suffix"])
        args.model_name = args.model_name or str(summary.get("model_name", args.tcga_summary_json.parent.name))
        args.label_column = args.label_column or str(summary["label_column"])
        args.context_fields = args.context_fields or list(summary.get("context_fields", ["site", "primary_diagnosis", "platform"]))
        args.leave_one_context_fields = args.leave_one_context_fields or list(args.context_fields)
        args.n_folds = args.n_folds or int(summary.get("probe_splits", 3))
    if args.mode == "cellxgene" and args.artifact_file is None:
        if args.metadata_csv and "bone_marrow" in str(args.metadata_csv):
            args.artifact_file = Path("data/cellxgene_bone_marrow_embeddings/qc_count_depth/qc_count_depth_embeddings.npz")
        elif args.metadata_csv and "lymph_node" in str(args.metadata_csv):
            args.artifact_file = Path("data/cellxgene_lymph_node_embeddings/qc_count_depth/qc_count_depth_embeddings.npz")
    if args.mode == "tcga" and args.artifact_features_dir is None:
        label = str(args.label_column or "")
        if label in {"TP53_status", "KRAS_status"}:
            args.artifact_features_dir = Path("data/tcga_image_stats_features/image_stats_luad_p16/features")
        elif label == "IDH_status":
            args.artifact_features_dir = Path("data/tcga_image_stats_features/image_stats_lgg_p16/features")
    return args


def read_matrix(path: Path, key: str) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        if key not in payload:
            raise KeyError(f"Key {key!r} not found in {path}; available={payload.files}")
        arr = payload[key]
    elif path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".csv":
        arr = pd.read_csv(path).to_numpy()
    else:
        raise ValueError(f"Unsupported matrix format: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix at {path}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values in {path}")
    return arr


def sample_key_from_path(value: str) -> str:
    return Path(value).stem


def normalize_metadata(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        out[column] = out[column].astype(object).map(clean_string).astype(object)
    if args.mode == "cellxgene":
        if args.label_column and args.label_column != "label":
            out = out.rename(columns={args.label_column: "label"})
        if args.donor_column != "donor_id":
            out = out.rename(columns={args.donor_column: "donor_id"})
        if args.cell_index_column != "cell_index":
            out = out.rename(columns={args.cell_index_column: "cell_index"})
        out["cell_index"] = pd.to_numeric(df[args.cell_index_column], errors="raise").astype(int)
    return out


def load_cellxgene(args: argparse.Namespace) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if args.metadata_csv is None or args.embedding_file is None or args.artifact_file is None:
        raise ValueError("CELLxGENE mode needs metadata, embedding, and artifact files")
    metadata = normalize_metadata(pd.read_csv(args.metadata_csv), args)
    x = read_matrix(args.embedding_file, args.embedding_key)
    artifact = read_matrix(args.artifact_file, args.artifact_key)
    if x.shape[0] != len(metadata):
        raise ValueError(f"Embedding rows {x.shape[0]} != metadata rows {len(metadata)}")
    if artifact.shape[0] != len(metadata):
        raise ValueError(f"Artifact rows {artifact.shape[0]} != metadata rows {len(metadata)}")
    return metadata, x, artifact


def load_tcga_artifacts(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    if args.artifact_features_dir is None:
        raise ValueError("TCGA mode needs --artifact-features-dir")
    rows: List[np.ndarray] = []
    for sample_key in df["sample_key"].astype(str):
        path = args.artifact_features_dir / f"{sample_key}{args.artifact_embedding_suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact embedding for {sample_key}: {path}")
        payload = np.load(path, allow_pickle=True)
        rows.append(np.asarray(payload["embedding"], dtype=np.float32))
    return np.vstack(rows).astype(np.float32)


def load_tcga(args: argparse.Namespace) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if args.metadata_csv is None or args.features_dir is None or args.embedding_suffix is None:
        raise ValueError("TCGA mode needs metadata, features dir, and embedding suffix")
    metadata = pd.read_csv(args.metadata_csv)
    for column in metadata.columns:
        metadata[column] = metadata[column].astype(object).map(clean_string).astype(object)
    for field in args.context_fields or []:
        if field in metadata.columns:
            metadata[field] = metadata[field].astype(object).map(clean_string).astype(object)
    try:
        df = load_tcga_embeddings(metadata, args.features_dir, args.path_column, args.embedding_suffix)
    except ValueError:
        fallback = tcga_shared_feature_fallback(args)
        if fallback is None:
            raise
        df = load_tcga_embeddings(metadata, fallback, args.path_column, args.embedding_suffix)
        args.features_dir = fallback
    x = np.vstack(df["embedding"].to_numpy()).astype(np.float32)
    artifact = load_tcga_artifacts(df, args)
    return df, x, artifact


def tcga_shared_feature_fallback(args: argparse.Namespace) -> Optional[Path]:
    label = str(args.label_column or "")
    suffix = str(args.embedding_suffix or "")
    if label in {"TP53_status", "KRAS_status"} and suffix == "_uni_embedding.npz":
        path = Path("data/tcga_shared_features/uni_luad_p16/features")
    elif label in {"TP53_status", "KRAS_status"} and suffix == "_conch_embedding.npz":
        path = Path("data/tcga_shared_features/conch_luad_p16/features")
    elif label == "IDH_status" and suffix == "_uni_embedding.npz":
        path = Path("data/tcga_shared_features/uni_lgg_p16/features")
    elif label == "IDH_status" and suffix == "_conch_embedding.npz":
        path = Path("data/tcga_shared_features/conch_lgg_p16/features")
    else:
        return None
    return path if path.exists() else None


def one_hot(values: Sequence[object], categories: Sequence[str]) -> np.ndarray:
    mapping = {value: idx for idx, value in enumerate(categories)}
    arr = np.zeros((len(values), len(categories)), dtype=np.float32)
    for i, value in enumerate(np.asarray(values, dtype=str)):
        idx = mapping.get(value)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr


def make_covariates(
    control_type: str,
    metadata_train: pd.DataFrame,
    metadata_eval: pd.DataFrame,
    artifact_train: np.ndarray,
    artifact_eval: np.ndarray,
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    train_parts: List[np.ndarray] = []
    eval_parts: List[np.ndarray] = []
    if control_type in {"artifact", "artifact_label"}:
        train_parts.append(artifact_train.astype(np.float32))
        eval_parts.append(artifact_eval.astype(np.float32))
    if control_type in {"label", "artifact_label"}:
        categories = sorted(metadata_train[label_column].astype(str).unique())
        train_parts.append(one_hot(metadata_train[label_column].astype(str), categories))
        eval_parts.append(one_hot(metadata_eval[label_column].astype(str), categories))
    if not train_parts:
        return (
            np.zeros((len(metadata_train), 0), dtype=np.float32),
            np.zeros((len(metadata_eval), 0), dtype=np.float32),
        )
    return np.hstack(train_parts).astype(np.float32), np.hstack(eval_parts).astype(np.float32)


def residualize(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    cov_train: np.ndarray,
    cov_eval: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if cov_train.shape[1] == 0:
        return x_train, x_eval
    scaler = StandardScaler()
    c_train = scaler.fit_transform(cov_train).astype(np.float32)
    c_eval = scaler.transform(cov_eval).astype(np.float32)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(c_train, x_train)
    return (
        (x_train - model.predict(c_train)).astype(np.float32),
        (x_eval - model.predict(c_eval)).astype(np.float32),
    )


def fit_predict_classifier(
    x_train: np.ndarray,
    y_train: Sequence[object],
    x_eval: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_eval_s = scaler.transform(x_eval)
    clf = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced", random_state=seed)
    clf.fit(x_train_s, np.asarray(y_train, dtype=str))
    return clf.predict(x_eval_s)


def metric_row(y_true: Sequence[object], y_pred: Sequence[object], prefix: Mapping[str, object]) -> Dict[str, object]:
    y_true_arr = np.asarray(y_true, dtype=str)
    y_pred_arr = np.asarray(y_pred, dtype=str)
    return {
        **dict(prefix),
        "n": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
    }


def infer_cellxgene_folds(metadata: pd.DataFrame, requested: int) -> int:
    donor_labels = metadata[["donor_id", "label"]].drop_duplicates()
    min_donors = int(donor_labels.groupby("label", observed=True)["donor_id"].nunique().min())
    return max(2, min(requested, min_donors))


def valid_context_values(values: np.ndarray, min_n: int, max_classes: int) -> List[str]:
    counts = pd.Series(values.astype(str)).value_counts()
    keep = counts[counts >= min_n].index.tolist()
    if len(keep) > max_classes:
        keep = counts.head(max_classes).index.tolist()
    return [str(v) for v in keep]


def run_context_probes(
    mode: str,
    metadata: pd.DataFrame,
    x: np.ndarray,
    artifact: np.ndarray,
    context_fields: Sequence[str],
    label_column: str,
    n_folds: int,
    seed: int,
    ridge_alpha: float,
    min_probe_cells: int,
    max_probe_classes: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    groups = metadata["donor_id"].astype(str).to_numpy() if mode == "cellxgene" and "donor_id" in metadata.columns else None
    for control_type in CONTROL_TYPES_FOR_PROBES:
        for field in context_fields:
            if field not in metadata.columns:
                skipped.append({"control_type": control_type, "context_field": field, "reason": "missing_field"})
                continue
            values = metadata[field].astype(str).to_numpy()
            keep_values = valid_context_values(values, min_probe_cells, max_probe_classes)
            if len(keep_values) < 2:
                skipped.append({"control_type": control_type, "context_field": field, "reason": "too_few_classes"})
                continue
            mask = np.isin(values, keep_values)
            y_text = values[mask]
            y_encoded = LabelEncoder().fit_transform(y_text)
            if mode == "cellxgene":
                group_keep = groups[mask] if groups is not None else np.arange(mask.sum()).astype(str)
                actual_folds = max(2, min(n_folds, int(pd.Series(group_keep).nunique())))
                splitter = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
                splits = splitter.split(x[mask], y_encoded, group_keep)
            else:
                min_count = int(np.bincount(y_encoded).min())
                actual_folds = min(n_folds, min_count)
                if actual_folds < 2:
                    skipped.append({"control_type": control_type, "context_field": field, "reason": "min_class_count_lt_2"})
                    continue
                splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
                splits = splitter.split(x[mask], y_encoded)
            global_idx = np.flatnonzero(mask)
            y_true_all: List[str] = []
            y_pred_all: List[str] = []
            for fold, (train_local, test_local) in enumerate(splits, start=1):
                train_idx = global_idx[train_local]
                test_idx = global_idx[test_local]
                train_meta = metadata.iloc[train_idx]
                test_meta = metadata.iloc[test_idx]
                cov_train, cov_test = make_covariates(
                    control_type,
                    train_meta,
                    test_meta,
                    artifact[train_idx],
                    artifact[test_idx],
                    label_column,
                )
                x_train_r, x_test_r = residualize(x[train_idx], x[test_idx], cov_train, cov_test, ridge_alpha)
                pred = fit_predict_classifier(x_train_r, train_meta[field].astype(str), x_test_r, seed + fold)
                y_true_all.extend(test_meta[field].astype(str).tolist())
                y_pred_all.extend(pred.astype(str).tolist())
            rows.append(
                metric_row(
                    y_true_all,
                    y_pred_all,
                    {
                        "control_type": control_type,
                        "split_type": "context_probe",
                        "context_field": field,
                        "context_value": "__all__",
                        "n_classes": int(len(keep_values)),
                        "n_folds": int(actual_folds),
                    },
                )
            )
    return rows, skipped


def append_prediction_rows(
    rows: List[Dict[str, object]],
    metadata: pd.DataFrame,
    indices: np.ndarray,
    y_true: Sequence[object],
    y_pred: Sequence[object],
    prefix: Mapping[str, object],
    context_fields: Sequence[str],
) -> None:
    sub = metadata.iloc[indices]
    for local_i, (_, meta_row) in enumerate(sub.iterrows()):
        row: Dict[str, object] = {
            **dict(prefix),
            "true_label": str(y_true[local_i]),
            "pred_label": str(y_pred[local_i]),
        }
        if "cell_index" in sub.columns:
            row["cell_index"] = int(meta_row["cell_index"])
        if "donor_id" in sub.columns:
            row["donor_id"] = meta_row["donor_id"]
        if "sample_key" in sub.columns:
            row["sample_key"] = meta_row["sample_key"]
        for field in context_fields:
            if field in sub.columns:
                row[field] = meta_row[field]
        rows.append(row)


def summarize_predictions(pred_rows: Sequence[Mapping[str, object]], context_fields: Sequence[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        return [], []
    metrics: List[Dict[str, object]] = []
    gaps: List[Dict[str, object]] = []
    group_cols = ["control_type", "split_type"]
    for (control_type, split_type), control_df in pred_df.groupby(group_cols, dropna=False):
        metrics.append(
            metric_row(
                control_df["true_label"].astype(str),
                control_df["pred_label"].astype(str),
                {
                    "control_type": control_type,
                    "method": "erm",
                    "split_type": split_type,
                    "context_field": "overall",
                    "context_value": "overall",
                },
            )
        )
        for field in context_fields:
            if field not in control_df.columns:
                continue
            field_rows: List[Dict[str, object]] = []
            for value, sub in control_df.groupby(field, dropna=False):
                if len(sub) < 20 or sub["true_label"].nunique() < 2:
                    continue
                row = metric_row(
                    sub["true_label"].astype(str),
                    sub["pred_label"].astype(str),
                    {
                        "control_type": control_type,
                        "method": "erm",
                        "split_type": split_type,
                        "context_field": field,
                        "context_value": clean_string(value),
                    },
                )
                metrics.append(row)
                field_rows.append(row)
            if len(field_rows) >= 2:
                for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
                    valid = [row for row in field_rows if math.isfinite(float(row[metric]))]
                    if len(valid) < 2:
                        continue
                    best = max(valid, key=lambda row: float(row[metric]))
                    worst = min(valid, key=lambda row: float(row[metric]))
                    gaps.append(
                        {
                            "control_type": control_type,
                            "method": "erm",
                            "split_type": split_type,
                            "context_field": field,
                            "metric": metric,
                            "best_context_value": best["context_value"],
                            "worst_context_value": worst["context_value"],
                            "best_value": float(best[metric]),
                            "worst_value": float(worst[metric]),
                            "gap": float(best[metric]) - float(worst[metric]),
                        }
                    )
    return metrics, gaps


def run_patient_cv(
    mode: str,
    metadata: pd.DataFrame,
    x: np.ndarray,
    artifact: np.ndarray,
    label_column: str,
    context_fields: Sequence[str],
    n_folds: int,
    seed: int,
    ridge_alpha: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    if mode != "cellxgene":
        return [], [], []
    y = metadata[label_column].astype(str).to_numpy()
    groups = metadata["donor_id"].astype(str).to_numpy()
    n_folds = infer_cellxgene_folds(metadata, n_folds)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred_rows: List[Dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        for control_type in CONTROL_TYPES_FOR_DOWNSTREAM:
            train_meta = metadata.iloc[train_idx]
            test_meta = metadata.iloc[test_idx]
            cov_train, cov_test = make_covariates(
                control_type,
                train_meta,
                test_meta,
                artifact[train_idx],
                artifact[test_idx],
                label_column,
            )
            x_train_r, x_test_r = residualize(x[train_idx], x[test_idx], cov_train, cov_test, ridge_alpha)
            pred = fit_predict_classifier(x_train_r, y[train_idx], x_test_r, seed + fold)
            append_prediction_rows(
                pred_rows,
                metadata,
                test_idx,
                y[test_idx],
                pred,
                {"control_type": control_type, "method": "erm", "fold": fold, "split_type": "patient_level_cv"},
                context_fields,
            )
    metrics, gaps = summarize_predictions(pred_rows, context_fields)
    return metrics, gaps, pred_rows


def eligible_holdout(y_test: pd.Series, y_train: pd.Series, min_labels: int) -> Tuple[bool, str]:
    test_counts = y_test.astype(str).value_counts()
    train_counts = y_train.astype(str).value_counts()
    if len(test_counts) < min_labels:
        return False, "holdout_has_too_few_labels"
    if len(train_counts) < min_labels:
        return False, "train_has_too_few_labels"
    missing = sorted(set(test_counts.index) - set(train_counts.index))
    if missing:
        return False, f"holdout_label_missing_in_train:{','.join(missing)}"
    return True, ""


def run_leave_one(
    metadata: pd.DataFrame,
    x: np.ndarray,
    artifact: np.ndarray,
    label_column: str,
    context_fields: Sequence[str],
    leave_one_fields: Sequence[str],
    seed: int,
    ridge_alpha: float,
    min_holdout_cells: int,
    min_holdout_labels: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    y = metadata[label_column].astype(str).to_numpy()
    pred_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    for field in leave_one_fields:
        if field not in metadata.columns:
            skipped.append({"context_field": field, "context_value": "ALL", "reason": "missing_field"})
            continue
        for value in sorted(metadata[field].astype(str).unique()):
            test_idx = np.flatnonzero(metadata[field].astype(str).to_numpy() == value)
            train_idx = np.flatnonzero(metadata[field].astype(str).to_numpy() != value)
            prefix = {"split_type": "leave_one_context", "context_field": field, "context_value": value}
            if len(test_idx) < min_holdout_cells:
                skipped.append({**prefix, "reason": "too_few_holdout_cells", "n_test": int(len(test_idx))})
                continue
            ok, reason = eligible_holdout(pd.Series(y[test_idx]), pd.Series(y[train_idx]), min_holdout_labels)
            if not ok:
                skipped.append({**prefix, "reason": reason, "n_test": int(len(test_idx))})
                continue
            for control_type in CONTROL_TYPES_FOR_DOWNSTREAM:
                train_meta = metadata.iloc[train_idx]
                test_meta = metadata.iloc[test_idx]
                cov_train, cov_test = make_covariates(
                    control_type,
                    train_meta,
                    test_meta,
                    artifact[train_idx],
                    artifact[test_idx],
                    label_column,
                )
                x_train_r, x_test_r = residualize(x[train_idx], x[test_idx], cov_train, cov_test, ridge_alpha)
                pred = fit_predict_classifier(x_train_r, y[train_idx], x_test_r, seed + len(metric_rows) + 1000)
                row = metric_row(
                    y[test_idx],
                    pred,
                    {
                        **prefix,
                        "control_type": control_type,
                        "method": "erm",
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                    },
                )
                metric_rows.append(row)
                append_prediction_rows(
                    pred_rows,
                    metadata,
                    test_idx,
                    y[test_idx],
                    pred,
                    {**prefix, "control_type": control_type, "method": "erm"},
                    context_fields,
                )
    gaps: List[Dict[str, object]] = []
    metric_df = pd.DataFrame(metric_rows)
    if not metric_df.empty:
        for (control_type, field), sub in metric_df.groupby(["control_type", "context_field"], dropna=False):
            if len(sub) < 2:
                continue
            for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
                vals = pd.to_numeric(sub[metric], errors="coerce")
                valid = sub[np.isfinite(vals)].copy()
                if len(valid) < 2:
                    continue
                best = valid.loc[pd.to_numeric(valid[metric], errors="coerce").idxmax()]
                worst = valid.loc[pd.to_numeric(valid[metric], errors="coerce").idxmin()]
                gaps.append(
                    {
                        "control_type": control_type,
                        "method": "erm",
                        "split_type": "leave_one_context",
                        "context_field": field,
                        "metric": metric,
                        "best_context_value": best["context_value"],
                        "worst_context_value": worst["context_value"],
                        "best_value": float(best[metric]),
                        "worst_value": float(worst[metric]),
                        "gap": float(best[metric]) - float(worst[metric]),
                    }
                )
    return metric_rows, gaps, pred_rows, skipped


def build_control_summary(
    probe_rows: Sequence[Mapping[str, object]],
    patient_gaps: Sequence[Mapping[str, object]],
    leave_gaps: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    contexts = sorted(
        {
            str(row["context_field"])
            for row in list(probe_rows) + list(patient_gaps) + list(leave_gaps)
            if str(row.get("context_field", "overall")) != "overall"
        }
    )
    controls = sorted({str(row["control_type"]) for row in list(probe_rows) + list(patient_gaps) + list(leave_gaps)})
    for control in controls:
        for context in contexts:
            probe = [
                row for row in probe_rows
                if str(row.get("control_type")) == control and str(row.get("context_field")) == context
            ]
            patient = [
                row for row in patient_gaps
                if str(row.get("control_type")) == control and str(row.get("context_field")) == context and str(row.get("metric")) == "balanced_accuracy"
            ]
            leave = [
                row for row in leave_gaps
                if str(row.get("control_type")) == control and str(row.get("context_field")) == context and str(row.get("metric")) == "balanced_accuracy"
            ]
            rows.append(
                {
                    "control_type": control,
                    "context_field": context,
                    "context_probe_ba": float(probe[0]["balanced_accuracy"]) if probe else float("nan"),
                    "patient_cv_ba_gap": float(patient[0]["gap"]) if patient else float("nan"),
                    "patient_cv_worst_ba": float(patient[0]["worst_value"]) if patient else float("nan"),
                    "leave_one_ba_gap": float(leave[0]["gap"]) if leave else float("nan"),
                    "leave_one_worst_ba": float(leave[0]["worst_value"]) if leave else float("nan"),
                }
            )
    return rows


def main() -> None:
    args = fill_from_summary(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.context_fields is None:
        args.context_fields = ["sex", "age_group", "dataset_id", "assay", "disease"] if args.mode == "cellxgene" else ["site", "primary_diagnosis", "platform"]
    if args.leave_one_context_fields is None:
        args.leave_one_context_fields = [args.context_fields[0]]
    if args.label_column is None:
        args.label_column = "label"
    if args.model_name is None:
        args.model_name = "embedding"
    if args.n_folds is None:
        args.n_folds = 2 if args.mode == "cellxgene" else 3

    metadata, x, artifact = load_cellxgene(args) if args.mode == "cellxgene" else load_tcga(args)
    if args.mode == "tcga":
        metadata["label"] = metadata[args.label_column].astype(str)
        label_column = args.label_column
    else:
        label_column = "label"

    probe_rows, probe_skipped = run_context_probes(
        args.mode,
        metadata,
        x,
        artifact,
        args.context_fields,
        label_column,
        args.n_folds,
        args.seed,
        args.ridge_alpha,
        args.min_probe_cells if args.mode == "cellxgene" else 10,
        args.max_probe_classes,
    )
    patient_metrics, patient_gaps, patient_preds = run_patient_cv(
        args.mode,
        metadata,
        x,
        artifact,
        label_column,
        args.context_fields,
        args.n_folds,
        args.seed,
        args.ridge_alpha,
    )
    leave_metrics, leave_gaps, leave_preds, leave_skipped = run_leave_one(
        metadata,
        x,
        artifact,
        label_column,
        args.context_fields,
        args.leave_one_context_fields,
        args.seed,
        args.ridge_alpha,
        args.min_holdout_cells if args.mode == "cellxgene" else 12,
        args.min_holdout_labels,
    )
    control_summary = build_control_summary(probe_rows, patient_gaps, leave_gaps)

    write_csv(args.output_dir / "context_probe_results.csv", probe_rows)
    write_csv(args.output_dir / "context_probe_skipped.csv", probe_skipped)
    write_csv(args.output_dir / "patient_cv_metrics.csv", patient_metrics)
    write_csv(args.output_dir / "patient_cv_gaps.csv", patient_gaps)
    write_csv(args.output_dir / "patient_cv_predictions.csv", patient_preds)
    write_csv(args.output_dir / "leave_one_context_metrics.csv", leave_metrics)
    write_csv(args.output_dir / "leave_one_context_gaps.csv", leave_gaps)
    write_csv(args.output_dir / "leave_one_context_predictions.csv", leave_preds)
    write_csv(args.output_dir / "leave_one_context_skipped.csv", leave_skipped)
    write_csv(args.output_dir / "residualized_control_summary.csv", control_summary)

    summary = {
        "mode": args.mode,
        "model_name": args.model_name,
        "metadata_csv": str(args.metadata_csv),
        "embedding_file": str(args.embedding_file) if args.embedding_file else None,
        "features_dir": str(args.features_dir) if args.features_dir else None,
        "embedding_suffix": args.embedding_suffix,
        "artifact_file": str(args.artifact_file) if args.artifact_file else None,
        "artifact_features_dir": str(args.artifact_features_dir) if args.artifact_features_dir else None,
        "label_column": args.label_column,
        "context_fields": list(args.context_fields),
        "leave_one_context_fields": list(args.leave_one_context_fields),
        "embedding_shape": [int(x.shape[0]), int(x.shape[1])],
        "artifact_shape": [int(artifact.shape[0]), int(artifact.shape[1])],
        "n_probe_rows": int(len(probe_rows)),
        "n_patient_gap_rows": int(len(patient_gaps)),
        "n_leave_gap_rows": int(len(leave_gaps)),
        "n_control_summary_rows": int(len(control_summary)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
