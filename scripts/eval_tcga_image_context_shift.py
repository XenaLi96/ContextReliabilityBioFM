#!/usr/bin/env python3
"""Evaluate TCGA image-FM context shift on site/cohort metadata.

This script is intentionally TCGA/pathology-specific. It uses frozen WSI FM
embeddings and tests whether image-derived heads are brittle under held-out
TCGA context values such as tissue-source site or primary diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_tcga_embedding_mitigation import (  # noqa: E402
    METHODS,
    load_embeddings,
    predict_method,
)


DEFAULT_METHODS = ["erm", "label_context_reweight", "linear_debias"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--embedding-suffix", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--path-column", default="slide_file_name")
    parser.add_argument(
        "--context-fields",
        nargs="*",
        default=["site", "primary_diagnosis", "platform"],
    )
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--min-context-n", type=int, default=12)
    parser.add_argument("--min-context-label-n", type=int, default=2)
    parser.add_argument("--probe-min-class-n", type=int, default=10)
    parser.add_argument("--probe-splits", type=int, default=3)
    parser.add_argument("--random-cv-splits", type=int, default=5)
    parser.add_argument("--random-cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--group-dro-epochs", type=int, default=200)
    parser.add_argument("--group-dro-lr", type=float, default=0.02)
    parser.add_argument("--group-dro-weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def clean_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("unknown").astype(str).str.strip()
    return cleaned.replace({"": "unknown", "nan": "unknown", "None": "unknown", "NA": "unknown"})


def metric_row(y_true: Sequence[str], y_pred: Sequence[str], prefix: Dict[str, object]) -> Dict[str, object]:
    return {
        **prefix,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "task",
        "model",
        "method",
        "split_type",
        "context_field",
        "context_value",
        "metric",
        "sample_key",
        "patient_id",
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


def make_probe_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def run_context_probes(
    df: pd.DataFrame,
    x: np.ndarray,
    context_fields: Sequence[str],
    model_name: str,
    label_column: str,
    probe_min_class_n: int,
    probe_splits: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    for field in context_fields:
        counts = df[field].astype(str).value_counts()
        kept_values = counts[counts >= probe_min_class_n].index.tolist()
        if len(kept_values) < 2:
            skipped.append(
                {
                    "stage": "context_probe",
                    "context_field": field,
                    "reason": "insufficient_classes_after_min_count",
                    "n_classes": int(len(kept_values)),
                    "class_counts": counts.to_dict(),
                }
            )
            continue
        mask = df[field].astype(str).isin(kept_values).to_numpy()
        y_text = df.loc[mask, field].astype(str).to_numpy()
        y = LabelEncoder().fit_transform(y_text)
        min_count = int(np.bincount(y).min())
        actual_splits = min(probe_splits, min_count)
        if actual_splits < 2:
            skipped.append(
                {
                    "stage": "context_probe",
                    "context_field": field,
                    "reason": "min_class_count_lt_2",
                    "n_classes": int(len(kept_values)),
                    "class_counts": counts.to_dict(),
                }
            )
            continue
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
        y_true_all: List[str] = []
        y_pred_all: List[str] = []
        classes = np.asarray(sorted(set(y_text)))
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(x[mask], y), start=1):
            clf = make_probe_classifier(seed + fold_id)
            clf.fit(x[mask][train_idx], y_text[train_idx])
            pred = clf.predict(x[mask][test_idx])
            y_true_all.extend(y_text[test_idx].tolist())
            y_pred_all.extend(pred.tolist())
        row = metric_row(
            y_true_all,
            y_pred_all,
            {
                "task": label_column,
                "model": model_name,
                "split_type": "context_probe",
                "context_field": field,
                "context_value": "__all__",
                "n_classes": int(len(classes)),
                "class_counts": ";".join(f"{k}:{int(counts[k])}" for k in kept_values),
            },
        )
        rows.append(row)
    return rows, skipped


def run_random_cv(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    label_column: str,
    random_cv_splits: int,
    random_cv_repeats: int,
    seed: int,
) -> List[Dict[str, object]]:
    counts = np.bincount(y)
    actual_splits = min(random_cv_splits, int(counts.min()))
    if actual_splits < 2:
        return []
    if random_cv_repeats > 1:
        cv = RepeatedStratifiedKFold(
            n_splits=actual_splits,
            n_repeats=random_cv_repeats,
            random_state=seed,
        )
    else:
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)

    true_all: List[str] = []
    pred_all: List[str] = []
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        clf = make_probe_classifier(seed + fold_id)
        clf.fit(x[train_idx], labels[train_idx])
        pred = clf.predict(x[test_idx])
        true_all.extend(labels[test_idx].tolist())
        pred_all.extend(pred.tolist())
    return [
        metric_row(
            true_all,
            pred_all,
            {
                "task": label_column,
                "model": model_name,
                "method": "erm",
                "split_type": "random_stratified_cv",
                "context_field": "overall",
                "context_value": "overall",
                "n_splits": int(actual_splits),
                "n_repeats": int(random_cv_repeats),
            },
        )
    ]


def is_eligible_holdout(
    test_labels: pd.Series,
    train_labels: pd.Series,
    min_context_label_n: int,
) -> Tuple[bool, str]:
    test_counts = test_labels.astype(str).value_counts()
    train_counts = train_labels.astype(str).value_counts()
    if len(test_counts) < 2:
        return False, "holdout_has_single_label"
    if test_counts.min() < min_context_label_n:
        return False, "holdout_label_count_below_min"
    if len(train_counts) < 2:
        return False, "train_has_single_label"
    missing_in_train = sorted(set(test_counts.index) - set(train_counts.index))
    if missing_in_train:
        return False, f"test_label_missing_in_train:{','.join(missing_in_train)}"
    return True, "ok"


def summarize_leave_one_predictions(
    pred_df: pd.DataFrame,
    model_name: str,
    label_column: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if pred_df.empty:
        return rows
    for (context_field, method), method_df in pred_df.groupby(["context_field", "method"], sort=True):
        overall = metric_row(
            method_df["true_label"].astype(str),
            method_df["pred_label"].astype(str),
            {
                "task": label_column,
                "model": model_name,
                "method": method,
                "split_type": "leave_one_context_out",
                "context_field": context_field,
                "context_value": "__pooled__",
            },
        )
        heldout_rows: List[Dict[str, object]] = []
        for context_value, group in method_df.groupby("context_value", sort=True):
            heldout_rows.append(
                metric_row(
                    group["true_label"].astype(str),
                    group["pred_label"].astype(str),
                    {
                        "task": label_column,
                        "model": model_name,
                        "method": method,
                        "split_type": "leave_one_context_out",
                        "context_field": context_field,
                        "context_value": context_value,
                    },
                )
            )
        for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
            values = [float(row[metric]) for row in heldout_rows]
            if not values:
                continue
            worst_idx = int(np.argmin(values))
            best_idx = int(np.argmax(values))
            rows.append(
                {
                    **overall,
                    "metric": metric,
                    "worst_context_value": heldout_rows[worst_idx]["context_value"],
                    "worst_context_metric": values[worst_idx],
                    "best_context_value": heldout_rows[best_idx]["context_value"],
                    "best_context_metric": values[best_idx],
                    "best_minus_worst": values[best_idx] - values[worst_idx],
                    "overall_minus_worst": float(overall[metric]) - values[worst_idx],
                    "n_holdout_contexts": int(len(values)),
                }
            )

    # Add deltas against ERM for easier mitigation reading.
    indexed = {
        (row["context_field"], row["metric"], row["method"]): row
        for row in rows
    }
    for row in rows:
        erm = indexed.get((row["context_field"], row["metric"], "erm"))
        if not erm:
            continue
        row["overall_delta_vs_erm"] = float(row[row["metric"]]) - float(erm[erm["metric"]])
        row["worst_delta_vs_erm"] = float(row["worst_context_metric"]) - float(
            erm["worst_context_metric"]
        )
        row["gap_delta_vs_erm"] = float(row["best_minus_worst"]) - float(
            erm["best_minus_worst"]
        )
    return rows


def run_leave_one_context(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    context_fields: Sequence[str],
    methods: Sequence[str],
    model_name: str,
    label_column: str,
    min_context_n: int,
    min_context_label_n: int,
    seed: int,
    group_dro_epochs: int,
    group_dro_lr: float,
    group_dro_weight_decay: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    pred_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    for field in context_fields:
        values = sorted(df[field].astype(str).unique())
        if len(values) < 2:
            skipped.append(
                {
                    "stage": "leave_one_context_out",
                    "context_field": field,
                    "reason": "context_constant_or_missing",
                    "n_classes": int(len(values)),
                    "class_counts": df[field].astype(str).value_counts().to_dict(),
                }
            )
            continue
        for context_value in values:
            test_mask = df[field].astype(str).to_numpy() == context_value
            n_test = int(test_mask.sum())
            if n_test < min_context_n:
                skipped.append(
                    {
                        "stage": "leave_one_context_out",
                        "context_field": field,
                        "context_value": context_value,
                        "reason": "holdout_n_below_min",
                        "n_test": n_test,
                    }
                )
                continue
            train_mask = ~test_mask
            ok, reason = is_eligible_holdout(
                df.loc[test_mask, label_column],
                df.loc[train_mask, label_column],
                min_context_label_n=min_context_label_n,
            )
            if not ok:
                skipped.append(
                    {
                        "stage": "leave_one_context_out",
                        "context_field": field,
                        "context_value": context_value,
                        "reason": reason,
                        "n_test": n_test,
                        "test_label_counts": df.loc[test_mask, label_column]
                        .astype(str)
                        .value_counts()
                        .to_dict(),
                    }
                )
                continue

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)
            context_train = df.loc[train_mask, field].astype(str).to_numpy()
            context_test = df.loc[test_mask, field].astype(str).to_numpy()
            for method_idx, method in enumerate(methods):
                pred_int = predict_method(
                    method=method,
                    x_train_raw=x[train_idx],
                    y_train=y[train_idx],
                    context_train=context_train,
                    x_test_raw=x[test_idx],
                    context_test=context_test,
                    seed=seed + method_idx,
                    n_classes=int(len(set(y.tolist()))),
                    group_dro_epochs=group_dro_epochs,
                    group_dro_lr=group_dro_lr,
                    group_dro_weight_decay=group_dro_weight_decay,
                )
                pred_labels = labels[pred_int]
                true_labels = labels[y[test_idx]]
                metric_rows.append(
                    metric_row(
                        true_labels,
                        pred_labels,
                        {
                            "task": label_column,
                            "model": model_name,
                            "method": method,
                            "split_type": "leave_one_context_out",
                            "context_field": field,
                            "context_value": context_value,
                            "n_train": int(len(train_idx)),
                        },
                    )
                )
                for local_idx, sample_idx in enumerate(test_idx):
                    row = df.iloc[int(sample_idx)]
                    pred_rows.append(
                        {
                            "task": label_column,
                            "model": model_name,
                            "method": method,
                            "split_type": "leave_one_context_out",
                            "context_field": field,
                            "context_value": context_value,
                            "sample_key": row["sample_key"],
                            "patient_id": row.get("patient_id", ""),
                            "true_label": true_labels[local_idx],
                            "pred_label": pred_labels[local_idx],
                            "site": row.get("site", ""),
                            "platform": row.get("platform", ""),
                            "primary_diagnosis": row.get("primary_diagnosis", ""),
                            "study_id": row.get("study_id", ""),
                        }
                    )
    summary_rows = summarize_leave_one_predictions(
        pd.DataFrame(pred_rows),
        model_name=model_name,
        label_column=label_column,
    )
    return pred_rows, metric_rows, skipped + summary_rows


def main() -> None:
    args = parse_args()
    unknown = [method for method in args.methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {METHODS}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_df = pd.read_csv(args.metadata_csv)
    df = load_embeddings(metadata_df, args.features_dir, args.path_column, args.embedding_suffix)

    required = [args.label_column, *args.context_fields]
    for column in required:
        if column not in df.columns:
            df[column] = "unknown"
        df[column] = clean_series(df[column])
    df = df[df[args.label_column] != "unknown"].copy().reset_index(drop=True)
    if len(df) < 4:
        raise ValueError("Too few rows after label/context filtering.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[args.label_column].astype(str).to_numpy())
    labels = label_encoder.classes_
    x = np.stack(df["embedding"].to_list(), axis=0)

    probe_rows, probe_skipped = run_context_probes(
        df=df,
        x=x,
        context_fields=args.context_fields,
        model_name=args.model_name,
        label_column=args.label_column,
        probe_min_class_n=args.probe_min_class_n,
        probe_splits=args.probe_splits,
        seed=args.seed,
    )
    random_cv_rows = run_random_cv(
        df=df,
        x=x,
        y=y,
        labels=labels[y],
        model_name=args.model_name,
        label_column=args.label_column,
        random_cv_splits=args.random_cv_splits,
        random_cv_repeats=args.random_cv_repeats,
        seed=args.seed,
    )
    pred_rows, leave_one_metric_rows, leave_one_rows_or_skips = run_leave_one_context(
        df=df,
        x=x,
        y=y,
        labels=labels,
        context_fields=args.context_fields,
        methods=args.methods,
        model_name=args.model_name,
        label_column=args.label_column,
        min_context_n=args.min_context_n,
        min_context_label_n=args.min_context_label_n,
        seed=args.seed,
        group_dro_epochs=args.group_dro_epochs,
        group_dro_lr=args.group_dro_lr,
        group_dro_weight_decay=args.group_dro_weight_decay,
    )
    leave_one_summary_rows = [
        row for row in leave_one_rows_or_skips if row.get("split_type") == "leave_one_context_out"
    ]
    leave_one_skipped = [
        row for row in leave_one_rows_or_skips if row.get("split_type") != "leave_one_context_out"
    ]
    skipped_rows = probe_skipped + leave_one_skipped

    write_csv(args.output_dir / "context_probe_results.csv", probe_rows)
    write_csv(args.output_dir / "random_cv_metrics.csv", random_cv_rows)
    write_csv(args.output_dir / "leave_one_context_predictions.csv", pred_rows)
    write_csv(args.output_dir / "leave_one_context_metrics.csv", leave_one_metric_rows)
    write_csv(args.output_dir / "leave_one_context_summary.csv", leave_one_summary_rows)
    write_csv(args.output_dir / "skipped_contexts.csv", skipped_rows)

    summary = {
        "metadata_csv": str(args.metadata_csv),
        "features_dir": str(args.features_dir),
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "embedding_suffix": args.embedding_suffix,
        "label_column": args.label_column,
        "context_fields": args.context_fields,
        "methods": args.methods,
        "n_rows_with_embeddings": int(len(df)),
        "label_counts": df[args.label_column].astype(str).value_counts().to_dict(),
        "context_counts": {
            field: df[field].astype(str).value_counts().to_dict() for field in args.context_fields
        },
        "n_probe_rows": int(len(probe_rows)),
        "n_random_cv_rows": int(len(random_cv_rows)),
        "n_leave_one_metric_rows": int(len(leave_one_metric_rows)),
        "n_leave_one_summary_rows": int(len(leave_one_summary_rows)),
        "n_skipped_rows": int(len(skipped_rows)),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
