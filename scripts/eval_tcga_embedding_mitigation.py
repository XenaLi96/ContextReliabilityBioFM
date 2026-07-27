#!/usr/bin/env python3
"""Evaluate simple context-gap mitigation on frozen TCGA FM embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


METHODS = [
    "erm",
    "label_context_reweight",
    "linear_debias",
    "group_dro_linear",
    "group_threshold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--embedding-suffix", required=True)
    parser.add_argument("--context-field", required=True)
    parser.add_argument("--label-column", default="study_id")
    parser.add_argument("--path-column", default="slide_file_name")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--group-dro-epochs", type=int, default=500)
    parser.add_argument("--group-dro-lr", type=float, default=0.02)
    parser.add_argument("--group-dro-weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def sample_key_from_path(value: str) -> str:
    return Path(value).stem


def load_embeddings(
    metadata_df: pd.DataFrame,
    features_dir: Path,
    path_column: str,
    embedding_suffix: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in metadata_df.iterrows():
        sample_key = sample_key_from_path(str(row[path_column]))
        npz_path = features_dir / f"{sample_key}{embedding_suffix}"
        if not npz_path.exists():
            continue
        payload = np.load(npz_path, allow_pickle=True)
        rows.append(
            {
                "sample_key": sample_key,
                "feature_path": str(npz_path),
                "embedding": payload["embedding"].astype(np.float32),
            }
        )
    if not rows:
        raise ValueError(f"No embedding files found under {features_dir}")
    emb_df = pd.DataFrame(rows)
    merged = metadata_df.copy()
    merged["sample_key"] = merged[path_column].astype(str).map(sample_key_from_path)
    return merged.merge(emb_df, on="sample_key", how="inner")


def metric_row(y_true: Sequence[str], y_pred: Sequence[str], prefix: Dict[str, object]) -> Dict[str, object]:
    return {
        **prefix,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def build_cv(y: np.ndarray, n_splits: int, n_repeats: int, seed: int):
    counts = np.bincount(y)
    actual_splits = min(n_splits, int(counts.min()))
    if actual_splits < 2:
        raise ValueError(f"Need at least 2 samples per class; counts={counts.tolist()}")
    if n_repeats > 1:
        return (
            RepeatedStratifiedKFold(
                n_splits=actual_splits,
                n_repeats=n_repeats,
                random_state=seed,
            ),
            actual_splits,
        )
    return StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed), actual_splits


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
    class_weight: Optional[str] = "balanced",
) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight=class_weight,
        random_state=seed,
    )
    clf.fit(x_train, y_train, sample_weight=sample_weight)
    return clf


def best_binary_threshold(y_true: np.ndarray, positive_scores: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return 0.5
    candidates = np.unique(np.concatenate([positive_scores, np.asarray([0.5], dtype=np.float64)]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        pred = (positive_scores >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def predict_group_threshold(
    clf: LogisticRegression,
    x_train: np.ndarray,
    y_train: np.ndarray,
    context_train: np.ndarray,
    x_test: np.ndarray,
    context_test: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    if n_classes != 2:
        return clf.predict(x_test)

    train_scores = clf.predict_proba(x_train)[:, 1]
    test_scores = clf.predict_proba(x_test)[:, 1]
    global_threshold = best_binary_threshold(y_train, train_scores)
    thresholds: Dict[str, float] = {}
    for context_value in sorted(set(context_train.astype(str))):
        mask = context_train.astype(str) == context_value
        thresholds[context_value] = best_binary_threshold(y_train[mask], train_scores[mask])

    pred = []
    for score, context_value in zip(test_scores, context_test.astype(str)):
        threshold = thresholds.get(context_value, global_threshold)
        pred.append(1 if score >= threshold else 0)
    return np.asarray(pred, dtype=int)


def label_context_weights(y: np.ndarray, context: np.ndarray) -> np.ndarray:
    keys = [f"{label}::{ctx}" for label, ctx in zip(y, context)]
    counts: Dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float32)
    return weights / weights.mean()


def context_projection_matrix(x_train: np.ndarray, context_train: np.ndarray, seed: int) -> Optional[np.ndarray]:
    context_encoder = LabelEncoder()
    c = context_encoder.fit_transform(context_train.astype(str))
    if len(context_encoder.classes_) < 2:
        return None

    clf = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_train, c)
    coef = np.asarray(clf.coef_, dtype=np.float64)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    if coef.shape[0] == 1:
        basis = coef[0:1]
    else:
        basis = coef

    # Orthonormalize context-predictive directions and project them out.
    q, _ = np.linalg.qr(basis.T)
    norms = np.linalg.norm(q, axis=0)
    q = q[:, norms > 1e-8]
    if q.size == 0:
        return None
    return q.astype(np.float32)


def remove_projection(x: np.ndarray, basis: Optional[np.ndarray]) -> np.ndarray:
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
) -> torch.nn.Module:
    torch.manual_seed(seed)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    group_keys = [f"{label}::{ctx}" for label, ctx in zip(y_train, context_train)]
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


def predict_method(
    method: str,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    context_train: np.ndarray,
    x_test_raw: np.ndarray,
    context_test: np.ndarray,
    seed: int,
    n_classes: int,
    group_dro_epochs: int,
    group_dro_lr: float,
    group_dro_weight_decay: float,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)

    if method == "erm":
        clf = fit_logistic(x_train, y_train, seed=seed, class_weight="balanced")
        return clf.predict(x_test)

    if method == "label_context_reweight":
        weights = label_context_weights(y_train, context_train)
        clf = fit_logistic(
            x_train,
            y_train,
            seed=seed,
            sample_weight=weights,
            class_weight=None,
        )
        return clf.predict(x_test)

    if method == "linear_debias":
        basis = context_projection_matrix(x_train, context_train, seed=seed)
        x_train_debiased = remove_projection(x_train, basis)
        x_test_debiased = remove_projection(x_test, basis)
        clf = fit_logistic(x_train_debiased, y_train, seed=seed, class_weight="balanced")
        return clf.predict(x_test_debiased)

    if method == "group_dro_linear":
        model = fit_group_dro_linear(
            x_train,
            y_train,
            context_train,
            n_classes=n_classes,
            seed=seed,
            epochs=group_dro_epochs,
            lr=group_dro_lr,
            weight_decay=group_dro_weight_decay,
        )
        with torch.inference_mode():
            logits = model(torch.tensor(x_test, dtype=torch.float32))
            return logits.argmax(dim=1).cpu().numpy()

    if method == "group_threshold":
        clf = fit_logistic(x_train, y_train, seed=seed, class_weight="balanced")
        return predict_group_threshold(
            clf=clf,
            x_train=x_train,
            y_train=y_train,
            context_train=context_train,
            x_test=x_test,
            context_test=context_test,
            n_classes=n_classes,
        )

    raise ValueError(f"Unknown method: {method}")


def summarize_predictions(
    pred_df: pd.DataFrame,
    model_name: str,
    context_field: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    metric_rows: List[Dict[str, object]] = []
    gap_rows: List[Dict[str, object]] = []
    for method in sorted(pred_df["method"].unique()):
        method_df = pred_df[pred_df["method"] == method]
        overall = metric_row(
            method_df["true_label"].astype(str),
            method_df["pred_label"].astype(str),
            {
                "model": model_name,
                "method": method,
                "context_field": context_field,
                "context_value": "overall",
            },
        )
        metric_rows.append(overall)
        subgroup_rows: List[Dict[str, object]] = []
        for value in sorted(method_df[context_field].astype(str).unique()):
            group = method_df[method_df[context_field].astype(str) == value]
            row = metric_row(
                group["true_label"].astype(str),
                group["pred_label"].astype(str),
                {
                    "model": model_name,
                    "method": method,
                    "context_field": context_field,
                    "context_value": value,
                },
            )
            metric_rows.append(row)
            subgroup_rows.append(row)

        for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
            values = [float(row[metric]) for row in subgroup_rows]
            gap_rows.append(
                {
                    "model": model_name,
                    "method": method,
                    "context_field": context_field,
                    "metric": metric,
                    "overall": float(overall[metric]),
                    "best_group": max(values),
                    "worst_group": min(values),
                    "best_minus_worst": max(values) - min(values),
                    "overall_minus_worst": float(overall[metric]) - min(values),
                }
            )
    return metric_rows, gap_rows


def run_cv(
    df: pd.DataFrame,
    model_name: str,
    label_column: str,
    context_field: str,
    methods: List[str],
    n_splits: int,
    n_repeats: int,
    seed: int,
    group_dro_epochs: int,
    group_dro_lr: float,
    group_dro_weight_decay: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    x = np.stack(df["embedding"].to_list(), axis=0)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_column].astype(str).to_numpy())
    labels = label_encoder.inverse_transform(y)
    context_values = df[context_field].astype(str).to_numpy()
    cv, actual_splits = build_cv(y, n_splits, n_repeats, seed)

    pred_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        for method in methods:
            pred_int = predict_method(
                method,
                x[train_idx],
                y[train_idx],
                context_values[train_idx],
                x[test_idx],
                context_values[test_idx],
                seed=seed + fold,
                n_classes=len(label_encoder.classes_),
                group_dro_epochs=group_dro_epochs,
                group_dro_lr=group_dro_lr,
                group_dro_weight_decay=group_dro_weight_decay,
            )
            pred_labels = label_encoder.inverse_transform(pred_int)
            true_labels = labels[test_idx]
            fold_rows.append(
                metric_row(
                    true_labels,
                    pred_labels,
                    {
                        "model": model_name,
                        "method": method,
                        "fold": fold,
                        "n_splits": actual_splits,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                    },
                )
            )
            for local_idx, sample_idx in enumerate(test_idx):
                pred_rows.append(
                    {
                        "model": model_name,
                        "method": method,
                        "fold": fold,
                        "sample_key": df.iloc[sample_idx]["sample_key"],
                        "patient_id": df.iloc[sample_idx].get("patient_id", ""),
                        "true_label": true_labels[local_idx],
                        "pred_label": pred_labels[local_idx],
                        context_field: context_values[sample_idx],
                    }
                )

    metric_rows, gap_rows = summarize_predictions(pd.DataFrame(pred_rows), model_name, context_field)
    return fold_rows, pred_rows, gap_rows + metric_rows


def split_summary(rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    gap_rows = [row for row in rows if "metric" in row]
    metric_rows = [row for row in rows if "metric" not in row]
    return metric_rows, gap_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["model", "method", "context_field", "context_value", "metric"]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact_summary(gap_rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for row in gap_rows:
        if row.get("metric") != "balanced_accuracy":
            continue
        rows.append(
            {
                "model": row["model"],
                "method": row["method"],
                "context_field": row["context_field"],
                "overall_balanced_accuracy": row["overall"],
                "worst_group_balanced_accuracy": row["worst_group"],
                "balanced_accuracy_gap": row["best_minus_worst"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["balanced_accuracy_gap"],
            -row["worst_group_balanced_accuracy"],
            -row["overall_balanced_accuracy"],
        )
    )
    return rows


def main() -> None:
    args = parse_args()
    unknown = [method for method in args.methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {METHODS}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_df = pd.read_csv(args.metadata_csv)
    df = load_embeddings(metadata_df, args.features_dir, args.path_column, args.embedding_suffix)
    df = df[df[args.label_column].notna() & df[args.context_field].notna()].copy()
    df = df[df[args.context_field].astype(str).str.len() > 0].copy()
    if len(df) < 4:
        raise ValueError("Too few rows after filtering labels/context.")

    fold_rows, pred_rows, rows = run_cv(
        df=df,
        model_name=args.model_name,
        label_column=args.label_column,
        context_field=args.context_field,
        methods=args.methods,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seed=args.seed,
        group_dro_epochs=args.group_dro_epochs,
        group_dro_lr=args.group_dro_lr,
        group_dro_weight_decay=args.group_dro_weight_decay,
    )
    metric_rows, gap_rows = split_summary(rows)
    compact_rows = compact_summary(gap_rows)

    write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "predictions.csv", pred_rows)
    write_csv(args.output_dir / "subgroup_metrics.csv", metric_rows)
    write_csv(args.output_dir / "subgroup_gaps.csv", gap_rows)
    write_csv(args.output_dir / "method_summary.csv", compact_rows)

    summary = {
        "metadata_csv": str(args.metadata_csv),
        "features_dir": str(args.features_dir),
        "model_name": args.model_name,
        "embedding_suffix": args.embedding_suffix,
        "label_column": args.label_column,
        "context_field": args.context_field,
        "methods": args.methods,
        "num_rows": int(len(df)),
        "label_counts": df[args.label_column].astype(str).value_counts().to_dict(),
        "context_counts": df[args.context_field].astype(str).value_counts().to_dict(),
        "method_summary": compact_rows,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
