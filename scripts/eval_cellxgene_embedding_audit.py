#!/usr/bin/env python3
"""Audit CELLxGENE cell embeddings for context leakage and worst-context gaps.

This script evaluates a precomputed embedding matrix on the same CELLxGENE
cell-type context-shift task used by eval_cellxgene_context_aware.py.

Expected inputs:
- A metadata CSV with one row per cell and at least: cell_index, donor_id, label.
- An embedding file in .npy, .npz, or .csv format with the same row order as the
  metadata CSV. For .npz, the default key is "embeddings".

It reports:
- context probes: how well assay/dataset/donor/etc. can be predicted from the
  frozen embedding;
- patient-level CV downstream metrics and subgroup gaps;
- leave-one-context-out metrics;
- simple context-aware head mitigations using the selected mitigation context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2_contingency
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEFAULT_CONTEXT_FIELDS = ["sex", "age_group", "dataset_id", "assay"]
METHODS = ["erm", "label_context_reweight", "linear_debias", "group_dro_linear"]


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
        "method",
        "fold",
        "split_type",
        "context_field",
        "context_value",
        "metric",
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


def infer_fold_count(df: pd.DataFrame, label_column: str, requested: int) -> int:
    donor_labels = df[["donor_id", label_column]].drop_duplicates()
    min_donors = int(donor_labels.groupby(label_column, observed=True)["donor_id"].nunique().min())
    return max(2, min(requested, min_donors))


def metric_row(y_true: Sequence[object], y_pred: Sequence[object], prefix: Mapping[str, object]) -> Dict[str, object]:
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
                valid = [row for row in field_rows if math.isfinite(float(row[metric]))]
                if len(valid) < 2:
                    continue
                best = max(valid, key=lambda row: float(row[metric]))
                worst = min(valid, key=lambda row: float(row[metric]))
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


def summarize_method_predictions(
    pred_rows: Sequence[Mapping[str, object]],
    context_fields: Sequence[str],
    split_type: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    metrics: List[Dict[str, object]] = []
    gaps: List[Dict[str, object]] = []
    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        return metrics, gaps
    for method in sorted(pred_df["method"].astype(str).unique()):
        method_df = pred_df[pred_df["method"].astype(str) == method]
        method_metrics, method_gaps = summarize_predictions(method_df, context_fields)
        for row in method_metrics:
            row["method"] = method
            row["split_type"] = split_type
        for row in method_gaps:
            row["method"] = method
            row["split_type"] = split_type
        metrics.extend(method_metrics)
        gaps.extend(method_gaps)
    return metrics, gaps


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


def label_context_weights(y: np.ndarray, context: np.ndarray) -> np.ndarray:
    keys = [f"{label}::{ctx}" for label, ctx in zip(y, context.astype(str))]
    counts: Dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float32)
    return weights / float(np.mean(weights))


def context_projection_matrix(x_train: np.ndarray, context_train: np.ndarray, seed: int):
    classes = sorted(set(context_train.astype(str)))
    if len(classes) < 2:
        return None
    y_context = LabelEncoder().fit_transform(context_train.astype(str))
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed)
    clf.fit(x_train, y_context)
    coef = np.asarray(clf.coef_, dtype=np.float64)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    basis = coef[0:1] if coef.shape[0] == 1 else coef
    q, _ = np.linalg.qr(basis.T)
    norms = np.linalg.norm(q, axis=0)
    q = q[:, norms > 1e-8]
    if q.size == 0:
        return None
    return q.astype(np.float32)


def remove_projection(x: np.ndarray, basis) -> np.ndarray:
    if basis is None:
        return x
    return x - (x @ basis) @ basis.T


def fit_group_dro_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    context_train: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    import torch

    torch.manual_seed(seed)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(LabelEncoder().fit_transform(y_train.astype(str)), dtype=torch.long)
    group_keys = [f"{label}::{ctx}" for label, ctx in zip(y_train.astype(str), context_train.astype(str))]
    group_encoder = {key: idx for idx, key in enumerate(sorted(set(group_keys)))}
    group_ids = torch.tensor([group_encoder[key] for key in group_keys], dtype=torch.long)
    unique_groups = torch.unique(group_ids)

    model = torch.nn.Linear(x_train.shape[1], n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor)
        losses = loss_fn(logits, y_tensor)
        group_losses = torch.stack([losses[group_ids == group_id].mean() for group_id in unique_groups])
        objective = group_losses.max() + 0.05 * losses.mean()
        objective.backward()
        optimizer.step()
    model.eval()
    return model


def fit_predict_method(
    method: str,
    x_train_sparse: sparse.csr_matrix,
    y_train: np.ndarray,
    context_train: np.ndarray,
    x_test_sparse: sparse.csr_matrix,
    seed: int,
    n_classes: int,
    group_dro_epochs: int,
    group_dro_lr: float,
    group_dro_weight_decay: float,
) -> np.ndarray:
    scaler = StandardScaler(with_mean=False)
    x_train_scaled_sparse = scaler.fit_transform(x_train_sparse).astype(np.float32)
    x_test_scaled_sparse = scaler.transform(x_test_sparse).astype(np.float32)

    if method in {"erm", "label_context_reweight"}:
        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced" if method == "erm" else None,
            random_state=seed,
        )
        weights = label_context_weights(y_train, context_train) if method == "label_context_reweight" else None
        clf.fit(x_train_scaled_sparse, y_train, sample_weight=weights)
        return clf.predict(x_test_scaled_sparse)

    x_train_scaled = x_train_scaled_sparse.toarray()
    x_test_scaled = x_test_scaled_sparse.toarray()

    if method == "linear_debias":
        basis = context_projection_matrix(x_train_scaled, context_train, seed=seed)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed)
        clf.fit(remove_projection(x_train_scaled, basis), y_train)
        return clf.predict(remove_projection(x_test_scaled, basis))

    if method == "group_dro_linear":
        label_encoder = LabelEncoder()
        y_train_int = label_encoder.fit_transform(y_train.astype(str))
        model = fit_group_dro_linear(
            x_train_scaled,
            y_train.astype(str),
            context_train,
            n_classes=n_classes,
            seed=seed,
            epochs=group_dro_epochs,
            lr=group_dro_lr,
            weight_decay=group_dro_weight_decay,
        )
        import torch

        with torch.inference_mode():
            logits = model(torch.tensor(x_test_scaled, dtype=torch.float32))
            pred_int = logits.argmax(dim=1).cpu().numpy()
        _ = y_train_int
        return label_encoder.inverse_transform(pred_int)

    raise ValueError(f"Unknown method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="embedding")
    parser.add_argument("--embedding-key", default="embeddings")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--donor-column", default="donor_id")
    parser.add_argument("--cell-index-column", default="cell_index")
    parser.add_argument("--context-fields", nargs="*", default=[*DEFAULT_CONTEXT_FIELDS, "disease"])
    parser.add_argument("--mitigation-context-field", default="assay")
    parser.add_argument("--leave-one-context-fields", nargs="*", default=["assay"])
    parser.add_argument("--methods", nargs="*", default=["erm", "label_context_reweight", "linear_debias"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--min-probe-cells", type=int, default=200)
    parser.add_argument("--max-probe-classes", type=int, default=30)
    parser.add_argument("--min-holdout-cells", type=int, default=200)
    parser.add_argument("--min-holdout-labels", type=int, default=2)
    parser.add_argument("--group-dro-epochs", type=int, default=250)
    parser.add_argument("--group-dro-lr", type=float, default=0.02)
    parser.add_argument("--group-dro-weight-decay", type=float, default=1e-3)
    parser.add_argument("--svd-components", type=int, default=0)
    return parser.parse_args()


def read_embedding(path: Path, key: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        loaded = np.load(path)
        if key not in loaded:
            available = ", ".join(sorted(loaded.files))
            raise KeyError(f"Embedding key {key!r} not found in {path}; available: {available}")
        arr = loaded[key]
    elif suffix == ".csv":
        arr = pd.read_csv(path).to_numpy()
    else:
        raise ValueError(f"Unsupported embedding format: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Embedding matrix contains non-finite values")
    return arr


def maybe_svd(x: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    if n_components <= 0 or n_components >= x.shape[1]:
        return x
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    return svd.fit_transform(x).astype(np.float32)


def normalize_metadata(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = [args.label_column, args.donor_column, args.cell_index_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")
    out = df.copy()
    for column in out.columns:
        out[column] = out[column].astype(object).map(clean_string).astype(object)
    out[args.cell_index_column] = pd.to_numeric(df[args.cell_index_column], errors="raise").astype(int)
    if args.label_column != "label":
        out = out.rename(columns={args.label_column: "label"})
    if args.donor_column != "donor_id":
        out = out.rename(columns={args.donor_column: "donor_id"})
    if args.cell_index_column != "cell_index":
        out = out.rename(columns={args.cell_index_column: "cell_index"})
    return out


def append_prediction_rows(
    rows: List[Dict[str, object]],
    metadata: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str,
    prefix: Mapping[str, object],
    context_fields: Sequence[str],
) -> None:
    test_meta = metadata.iloc[indices]
    for local_i, (_, meta_row) in enumerate(test_meta.iterrows()):
        row = {
            **dict(prefix),
            "method": method,
            "cell_index": int(meta_row["cell_index"]),
            "donor_id": meta_row["donor_id"],
            "true_label": str(y_true[local_i]),
            "pred_label": str(y_pred[local_i]),
        }
        for field in sorted(set(context_fields) | set(DEFAULT_CONTEXT_FIELDS) | {"disease"}):
            if field in test_meta.columns:
                row[field] = meta_row[field]
        rows.append(row)


def run_context_probes(
    x: sparse.csr_matrix,
    metadata: pd.DataFrame,
    context_fields: Sequence[str],
    groups: np.ndarray,
    seed: int,
    requested_folds: int,
    min_probe_cells: int,
    max_probe_classes: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    for field in context_fields:
        if field not in metadata.columns:
            skipped.append({"stage": "context_probe", "context_field": field, "reason": "missing_field"})
            continue
        values = metadata[field].astype(str).to_numpy()
        counts = pd.Series(values).value_counts()
        keep_values = counts[counts >= min_probe_cells].index.tolist()
        if len(keep_values) < 2:
            skipped.append({"stage": "context_probe", "context_field": field, "reason": "too_few_classes"})
            continue
        if len(keep_values) > max_probe_classes:
            keep_values = counts.head(max_probe_classes).index.tolist()
        keep = np.isin(values, keep_values)
        x_keep = x[keep]
        y_keep = values[keep]
        group_keep = groups[keep]
        y_encoded = LabelEncoder().fit_transform(y_keep)
        n_folds = max(2, min(requested_folds, int(pd.Series(group_keep).nunique())))
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x_keep, y_encoded, group_keep), start=1):
            if len(np.unique(y_encoded[train_idx])) < 2 or len(np.unique(y_encoded[test_idx])) < 2:
                skipped.append(
                    {
                        "stage": "context_probe",
                        "context_field": field,
                        "fold": fold,
                        "reason": "too_few_labels_in_fold",
                    }
                )
                continue
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=seed + fold)
            model = clf
            scaler = StandardScaler(with_mean=False)
            train_x = scaler.fit_transform(x_keep[train_idx])
            test_x = scaler.transform(x_keep[test_idx])
            model.fit(train_x, y_encoded[train_idx])
            pred = model.predict(test_x)
            rows.append(
                metric_row(
                    y_encoded[test_idx],
                    pred,
                    {
                        "split_type": "context_probe",
                        "context_field": field,
                        "fold": fold,
                        "n_classes": int(len(np.unique(y_encoded))),
                        "n_train_cells": int(len(train_idx)),
                        "n_test_cells": int(len(test_idx)),
                    },
                )
            )
    return rows, skipped


def run_patient_cv(
    x: sparse.csr_matrix,
    metadata: pd.DataFrame,
    y: np.ndarray,
    context_values: np.ndarray,
    methods: Sequence[str],
    context_fields: Sequence[str],
    n_folds: int,
    seed: int,
    group_dro_epochs: int,
    group_dro_lr: float,
    group_dro_weight_decay: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    groups = metadata["donor_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    n_classes = int(len(np.unique(y)))

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        for method in methods:
            pred = fit_predict_method(
                method,
                x[train_idx],
                y[train_idx],
                context_values[train_idx],
                x[test_idx],
                seed=seed + fold,
                n_classes=n_classes,
                group_dro_epochs=group_dro_epochs,
                group_dro_lr=group_dro_lr,
                group_dro_weight_decay=group_dro_weight_decay,
            )
            fold_rows.append(
                metric_row(
                    y[test_idx],
                    pred,
                    {
                        "method": method,
                        "fold": fold,
                        "split_type": "patient_level_cv",
                        "n_train_cells": int(len(train_idx)),
                        "n_test_cells": int(len(test_idx)),
                        "n_train_donors": int(pd.Series(groups[train_idx]).nunique()),
                        "n_test_donors": int(pd.Series(groups[test_idx]).nunique()),
                    },
                )
            )
            append_prediction_rows(
                pred_rows,
                metadata,
                test_idx,
                y[test_idx],
                pred,
                method,
                {"fold": fold, "split_type": "patient_level_cv"},
                context_fields,
            )
    subgroup_rows, gap_rows = summarize_method_predictions(pred_rows, context_fields, "patient_level_cv")
    return fold_rows, subgroup_rows, gap_rows, pred_rows


def run_leave_one_context(
    x: sparse.csr_matrix,
    metadata: pd.DataFrame,
    y: np.ndarray,
    context_values: np.ndarray,
    methods: Sequence[str],
    context_fields: Sequence[str],
    leave_one_fields: Sequence[str],
    min_holdout_cells: int,
    min_holdout_labels: int,
    seed: int,
    group_dro_epochs: int,
    group_dro_lr: float,
    group_dro_weight_decay: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    metric_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    n_classes = int(len(np.unique(y)))
    groups = metadata["donor_id"].astype(str).to_numpy()

    for field in leave_one_fields:
        if field not in metadata.columns:
            skipped_rows.append({"context_field": field, "context_value": "ALL", "reason": "missing_field"})
            continue
        for context_value in sorted(metadata[field].dropna().astype(str).unique()):
            test_mask = metadata[field].astype(str).to_numpy() == context_value
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
            for method in methods:
                pred = fit_predict_method(
                    method,
                    x[train_idx],
                    y[train_idx],
                    context_values[train_idx],
                    x[test_idx],
                    seed=seed + len(metric_rows) + 1000,
                    n_classes=n_classes,
                    group_dro_epochs=group_dro_epochs,
                    group_dro_lr=group_dro_lr,
                    group_dro_weight_decay=group_dro_weight_decay,
                )
                metric_rows.append(
                    metric_row(
                        y[test_idx],
                        pred,
                        {
                            **prefix,
                            "method": method,
                            "n_train_cells": int(len(train_idx)),
                            "n_test_cells": int(len(test_idx)),
                            "n_train_donors": int(pd.Series(groups[train_idx]).nunique()),
                            "n_test_donors": int(pd.Series(groups[test_idx]).nunique()),
                            "n_train_labels": int(len(np.unique(y[train_idx]))),
                            "n_test_labels": int(len(np.unique(y[test_idx]))),
                        },
                    )
                )
                append_prediction_rows(pred_rows, metadata, test_idx, y[test_idx], pred, method, prefix, context_fields)

    gap_rows: List[Dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in metric_rows}):
        method_rows = [row for row in metric_rows if row["method"] == method]
        method_gaps = summarize_metric_gaps(method_rows)
        for row in method_gaps:
            row["method"] = method
        gap_rows.extend(method_gaps)
    return metric_rows, gap_rows, pred_rows, skipped_rows


def write_summary_json(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    unknown = [method for method in args.methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {METHODS}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = normalize_metadata(pd.read_csv(args.metadata_csv), args)
    embeddings = maybe_svd(read_embedding(args.embedding_file, args.embedding_key), args.svd_components, args.seed)
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Embedding row count {embeddings.shape[0]} does not match metadata rows {len(metadata)}"
        )
    if args.mitigation_context_field not in metadata.columns:
        raise ValueError(f"Missing mitigation context field: {args.mitigation_context_field}")

    x = sparse.csr_matrix(embeddings)
    y = metadata["label"].astype(str).to_numpy()
    context_values = metadata[args.mitigation_context_field].astype(str).to_numpy()
    groups = metadata["donor_id"].astype(str).to_numpy()
    n_folds = infer_fold_count(metadata, "label", args.n_folds)

    probe_rows, probe_skipped = run_context_probes(
        x=x,
        metadata=metadata,
        context_fields=args.context_fields,
        groups=groups,
        seed=args.seed,
        requested_folds=n_folds,
        min_probe_cells=args.min_probe_cells,
        max_probe_classes=args.max_probe_classes,
    )
    fold_rows, subgroup_rows, gap_rows, pred_rows = run_patient_cv(
        x=x,
        metadata=metadata,
        y=y,
        context_values=context_values,
        methods=args.methods,
        context_fields=args.context_fields,
        n_folds=n_folds,
        seed=args.seed,
        group_dro_epochs=args.group_dro_epochs,
        group_dro_lr=args.group_dro_lr,
        group_dro_weight_decay=args.group_dro_weight_decay,
    )
    leave_metrics, leave_gaps, leave_preds, leave_skipped = run_leave_one_context(
        x=x,
        metadata=metadata,
        y=y,
        context_values=context_values,
        methods=args.methods,
        context_fields=args.context_fields,
        leave_one_fields=args.leave_one_context_fields,
        min_holdout_cells=args.min_holdout_cells,
        min_holdout_labels=args.min_holdout_labels,
        seed=args.seed,
        group_dro_epochs=args.group_dro_epochs,
        group_dro_lr=args.group_dro_lr,
        group_dro_weight_decay=args.group_dro_weight_decay,
    )
    association_rows, context_count_rows = label_context_audit(metadata, "label", args.context_fields)

    write_csv(args.output_dir / "context_probe_results.csv", probe_rows)
    write_csv(args.output_dir / "context_probe_skipped.csv", probe_skipped)
    write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "predictions.csv", pred_rows)
    write_csv(args.output_dir / "subgroup_metrics.csv", subgroup_rows)
    write_csv(args.output_dir / "subgroup_gaps.csv", gap_rows)
    write_csv(args.output_dir / "label_context_association.csv", association_rows)
    write_csv(args.output_dir / "label_context_counts.csv", context_count_rows)
    write_csv(args.output_dir / "leave_one_context_metrics.csv", leave_metrics)
    write_csv(args.output_dir / "leave_one_context_gaps.csv", leave_gaps)
    write_csv(args.output_dir / "leave_one_context_predictions.csv", leave_preds)
    write_csv(args.output_dir / "leave_one_context_skipped.csv", leave_skipped)

    summary = {
        "model_name": args.model_name,
        "metadata_csv": str(args.metadata_csv),
        "embedding_file": str(args.embedding_file),
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "label_counts": metadata["label"].value_counts().to_dict(),
        "donor_count": int(metadata["donor_id"].nunique()),
        "methods": list(args.methods),
        "mitigation_context_field": args.mitigation_context_field,
        "n_folds": int(n_folds),
        "context_probe_rows": int(len(probe_rows)),
        "context_probe_skipped": probe_skipped,
        "patient_level_gaps": gap_rows,
        "leave_one_context_gaps": leave_gaps,
        "leave_one_context_skipped": leave_skipped,
    }
    write_summary_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
