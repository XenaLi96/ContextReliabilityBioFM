#!/usr/bin/env python3
"""Evaluate patient-context probes from TCGA slide embeddings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Directory containing *_uni_embedding.npz files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-column", default="slide_file_name")
    parser.add_argument("--age-column", default="age_at_index")
    parser.add_argument("--sex-column", default="sex")
    parser.add_argument("--age-threshold", type=float, default=60.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def sample_key_from_path(value: str) -> str:
    return Path(value).stem


def load_embeddings(metadata_df: pd.DataFrame, features_dir: Path, path_column: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in metadata_df.iterrows():
        sample_key = sample_key_from_path(str(row[path_column]))
        npz_path = features_dir / f"{sample_key}_uni_embedding.npz"
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


def numeric_series(values: Iterable[object]) -> np.ndarray:
    parsed: List[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            parsed.append(np.nan)
    return np.asarray(parsed, dtype=np.float32)


def classification_cv(y: np.ndarray, n_splits: int, n_repeats: int, seed: int):
    counts = np.bincount(y)
    min_count = int(counts.min())
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        raise ValueError(f"Need at least 2 examples per class; class counts={counts.tolist()}")
    if n_repeats > 1:
        return RepeatedStratifiedKFold(
            n_splits=actual_splits,
            n_repeats=n_repeats,
            random_state=seed,
        ), actual_splits
    return StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed), actual_splits


def safe_auc(y_true: np.ndarray, prob_positive: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) != 2:
        return None
    try:
        return float(roc_auc_score(y_true, prob_positive))
    except ValueError:
        return None


def knn_same_group_enrichment(
    df: pd.DataFrame,
    label_values: pd.Series,
    valid_labels: Tuple[str, str],
    k: int = 10,
) -> Dict[str, object]:
    task_df = df.loc[label_values.notna()].copy()
    task_df["label"] = label_values.loc[task_df.index].astype(str)
    task_df = task_df[task_df["label"].isin(valid_labels)].copy()
    if len(task_df) <= k:
        return {"status": "skipped_too_few_rows", "k": k, "n_samples": int(len(task_df))}
    x = np.stack(task_df["embedding"].to_list(), axis=0)
    y = task_df["label"].to_numpy()
    x_scaled = StandardScaler().fit_transform(x)
    indices = NearestNeighbors(n_neighbors=k + 1).fit(x_scaled).kneighbors(
        x_scaled, return_distance=False
    )[:, 1:]
    local_same = np.mean(y[indices] == y[:, None], axis=1)
    counts = task_df["label"].value_counts()
    majority_baseline = float(counts.max() / len(task_df))
    random_baseline = float(((counts / len(task_df)) ** 2).sum())
    return {
        "status": "ok",
        "k": k,
        "n_samples": int(len(task_df)),
        "mean_knn_same_group": float(np.mean(local_same)),
        "majority_class_baseline": majority_baseline,
        "random_label_baseline": random_baseline,
        "enrichment_over_majority": float(np.mean(local_same) - majority_baseline),
        "enrichment_over_random": float(np.mean(local_same) - random_baseline),
    }


def run_classification_probe(
    df: pd.DataFrame,
    task_name: str,
    label_values: pd.Series,
    output_classes: Tuple[str, str],
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    task_df = df.loc[label_values.notna()].copy()
    task_df["label"] = label_values.loc[task_df.index].astype(str)
    task_df = task_df[task_df["label"].isin(output_classes)].copy()
    if task_df.empty:
        raise ValueError(f"No usable rows for {task_name}")

    X = np.stack(task_df["embedding"].to_list(), axis=0)
    raw_y = task_df["label"].to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_y)
    class_names = list(encoder.classes_)

    cv, actual_splits = classification_cv(y, n_splits, n_repeats, seed)
    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    positive_idx = 1 if len(class_names) == 2 else None

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        classifier.fit(X[train_idx], y[train_idx])
        pred = classifier.predict(X[test_idx])
        proba = classifier.predict_proba(X[test_idx])
        auc = safe_auc(y[test_idx], proba[:, positive_idx]) if positive_idx is not None else None

        row: Dict[str, object] = {
            "task": task_name,
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_splits": int(actual_splits),
            "accuracy": float(accuracy_score(y[test_idx], pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
            "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
            "auroc": auc,
        }
        fold_rows.append(row)

        for local, sample_idx in enumerate(test_idx):
            record: Dict[str, object] = {
                "task": task_name,
                "fold": fold,
                "sample_key": task_df.iloc[sample_idx]["sample_key"],
                "patient_id": task_df.iloc[sample_idx].get("patient_id", ""),
                "true_label": raw_y[sample_idx],
                "pred_label": class_names[int(pred[local])],
            }
            for class_idx, class_name in enumerate(class_names):
                record[f"prob_{class_name}"] = float(proba[local, class_idx])
            pred_rows.append(record)

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "task": task_name,
        "n_samples": int(len(task_df)),
        "classes": class_names,
        "class_counts": task_df["label"].value_counts().sort_index().to_dict(),
        "n_splits": int(actual_splits),
        "n_repeats": int(n_repeats),
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
        "macro_f1_mean": float(fold_df["macro_f1"].mean()),
        "auroc_mean": float(fold_df["auroc"].dropna().mean()) if fold_df["auroc"].notna().any() else None,
        "balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=0)),
        "auroc_std": float(fold_df["auroc"].dropna().std(ddof=0)) if fold_df["auroc"].notna().any() else None,
    }
    return fold_rows, pred_rows, summary


def run_age_regression(
    df: pd.DataFrame,
    ages: np.ndarray,
    n_splits: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    mask = ~np.isnan(ages)
    task_df = df.loc[mask].copy()
    y = ages[mask].astype(np.float32)
    if len(task_df) < 3:
        raise ValueError("Need at least 3 age-labeled samples for age regression")

    X = np.stack(task_df["embedding"].to_list(), axis=0)
    actual_splits = min(n_splits, len(task_df))
    cv = KFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    regressor = Pipeline(steps=[("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        regressor.fit(X[train_idx], y[train_idx])
        pred = regressor.predict(X[test_idx])
        rmse = math.sqrt(float(mean_squared_error(y[test_idx], pred)))
        corr = float(np.corrcoef(y[test_idx], pred)[0, 1]) if len(test_idx) > 1 else float("nan")
        fold_rows.append(
            {
                "task": "age_regression",
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_splits": int(actual_splits),
                "mae": float(mean_absolute_error(y[test_idx], pred)),
                "rmse": rmse,
                "r2": float(r2_score(y[test_idx], pred)),
                "pearson": corr,
            }
        )
        for local, sample_idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "task": "age_regression",
                    "fold": fold,
                    "sample_key": task_df.iloc[sample_idx]["sample_key"],
                    "patient_id": task_df.iloc[sample_idx].get("patient_id", ""),
                    "true_age": float(y[sample_idx]),
                    "pred_age": float(pred[local]),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "task": "age_regression",
        "n_samples": int(len(task_df)),
        "n_splits": int(actual_splits),
        "mae_mean": float(fold_df["mae"].mean()),
        "rmse_mean": float(fold_df["rmse"].mean()),
        "r2_mean": float(fold_df["r2"].mean()),
        "pearson_mean": float(fold_df["pearson"].replace([np.inf, -np.inf], np.nan).dropna().mean()),
        "mae_std": float(fold_df["mae"].std(ddof=0)),
    }
    return fold_rows, pred_rows, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(args.metadata_csv)
    merged = load_embeddings(metadata_df, args.features_dir, args.path_column)
    ages = numeric_series(merged[args.age_column])

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []

    sex_labels = merged[args.sex_column].astype(str).str.lower()
    knn_summaries: List[Dict[str, object]] = []
    rows, preds, summary = run_classification_probe(
        df=merged,
        task_name="sex_binary",
        label_values=sex_labels,
        output_classes=("female", "male"),
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )
    fold_rows.extend(rows)
    pred_rows.extend(preds)
    summaries.append(summary)
    knn = knn_same_group_enrichment(merged, sex_labels, ("female", "male"))
    knn["task"] = "sex_binary"
    knn_summaries.append(knn)

    age_labels = pd.Series(
        np.where(
            np.isnan(ages),
            None,
            np.where(ages >= args.age_threshold, f"age_ge_{args.age_threshold:g}", f"age_lt_{args.age_threshold:g}"),
        ),
        index=merged.index,
    )
    rows, preds, summary = run_classification_probe(
        df=merged,
        task_name=f"age_binary_ge_{args.age_threshold:g}",
        label_values=age_labels,
        output_classes=(f"age_lt_{args.age_threshold:g}", f"age_ge_{args.age_threshold:g}"),
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seed=args.seed + 1,
    )
    fold_rows.extend(rows)
    pred_rows.extend(preds)
    summaries.append(summary)
    knn = knn_same_group_enrichment(
        merged,
        age_labels,
        (f"age_lt_{args.age_threshold:g}", f"age_ge_{args.age_threshold:g}"),
    )
    knn["task"] = f"age_binary_ge_{args.age_threshold:g}"
    knn_summaries.append(knn)

    rows, preds, summary = run_age_regression(
        df=merged,
        ages=ages,
        n_splits=args.n_splits,
        seed=args.seed + 2,
    )
    fold_rows.extend(rows)
    pred_rows.extend(preds)
    summaries.append(summary)

    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "task_summary.csv", index=False)
    pd.DataFrame(knn_summaries).to_csv(args.output_dir / "knn_enrichment.csv", index=False)

    cohort_summary = {
        "metadata_csv": str(args.metadata_csv),
        "features_dir": str(args.features_dir),
        "num_metadata_rows": int(len(metadata_df)),
        "num_rows_with_embeddings": int(len(merged)),
        "study_counts": merged.get("study_id", pd.Series(dtype=str)).value_counts().to_dict(),
        "sex_counts": sex_labels.value_counts().to_dict(),
        "age_labeled_count": int((~np.isnan(ages)).sum()),
        "age_threshold": float(args.age_threshold),
        "tasks": summaries,
        "knn_enrichment": knn_summaries,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(cohort_summary, handle, indent=2, sort_keys=True)
    print(json.dumps(cohort_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
