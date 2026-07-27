#!/usr/bin/env python3
"""Run first-pass site/platform representation probes on HEST image features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_MANIFEST = Path("data/metadata/hest/selected_metadata_manifest.csv")
DEFAULT_FEATURE_DIR = Path("data/embeddings/hest/image_stats")
DEFAULT_OUTDIR = Path("outputs/hest/image_stats_representation_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fields", nargs="*", default=["platform", "organ", "disease", "site"])
    parser.add_argument("--max-spots-per-sample", type=int, default=1000)
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument("--min-test-classes", type=int, default=2)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_sample_features(sample_id: str, feature_dir: Path, max_spots: int, seed: int) -> np.ndarray:
    data = np.load(feature_dir / f"{sample_id}.npz", allow_pickle=True)
    x = data["features"].astype(np.float32)
    if x.shape[0] > max_spots:
        rng = np.random.default_rng(seed + abs(hash(sample_id)) % 100000)
        idx = np.sort(rng.choice(x.shape[0], max_spots, replace=False))
        x = x[idx]
    return x


def load_feature_table(manifest: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, pd.DataFrame]:
    chunks = []
    meta_rows = []
    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        x = load_sample_features(sample_id, args.feature_dir, args.max_spots_per_sample, args.seed)
        chunks.append(x)
        for _ in range(x.shape[0]):
            meta_rows.append(row.to_dict())
    return np.vstack(chunks), pd.DataFrame(meta_rows)


def run_probe(x: np.ndarray, meta: pd.DataFrame, field: str, args: argparse.Namespace) -> Dict[str, object]:
    keep = meta[field].notna() & (meta[field].astype(str) != "NA")
    x_keep = x[keep.to_numpy()]
    y = meta.loc[keep, field].astype(str).to_numpy()
    split = meta.loc[keep, "split"].astype(str).to_numpy()
    train_mask = split == "train"
    test_mask = split != "train"
    train_classes = sorted(set(y[train_mask]))
    test_classes = sorted(set(y[test_mask]))
    required_train_classes = max(args.min_classes, 2)
    if len(train_classes) < required_train_classes or len(set(test_classes) & set(train_classes)) < args.min_test_classes:
        return {
            "field": field,
            "status": "skipped_insufficient_classes",
            "train_classes": len(train_classes),
            "test_classes_seen_in_train": len(set(test_classes) & set(train_classes)),
        }
    test_seen = test_mask & np.isin(y, train_classes)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1),
    )
    model.fit(x_keep[train_mask], y[train_mask])
    pred = model.predict(x_keep[test_seen])
    y_test = y[test_seen]
    return {
        "field": field,
        "status": "ok",
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_seen.sum()),
        "train_classes": len(train_classes),
        "test_classes_seen_in_train": len(set(y_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 6),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 6),
    }


def knn_enrichment(x: np.ndarray, meta: pd.DataFrame, field: str, k: int) -> Dict[str, object]:
    keep = meta[field].notna() & (meta[field].astype(str) != "NA")
    x_keep = x[keep.to_numpy()]
    y = meta.loc[keep, field].astype(str).to_numpy()
    if x_keep.shape[0] <= k or len(set(y)) < 2:
        return {"field": field, "status": "skipped"}
    x_scaled = StandardScaler().fit_transform(x_keep)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(x_scaled)
    indices = nbrs.kneighbors(x_scaled, return_distance=False)[:, 1:]
    local_same = np.mean(y[indices] == y[:, None], axis=1)
    base = max(Counter(y).values()) / len(y)
    return {
        "field": field,
        "status": "ok",
        "k": k,
        "mean_knn_same_group": round(float(np.mean(local_same)), 6),
        "majority_class_baseline": round(float(base), 6),
        "enrichment_over_baseline": round(float(np.mean(local_same) - base), 6),
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest_csv)
    x, meta = load_feature_table(manifest, args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    probe_rows = [run_probe(x, meta, field, args) for field in args.fields if field in meta.columns]
    knn_rows = [knn_enrichment(x, meta, field, args.knn_k) for field in args.fields if field in meta.columns]
    shortcut_rows = []
    for row in probe_rows:
        field = row["field"]
        knn = next((item for item in knn_rows if item["field"] == field), {})
        shortcut_rows.append(
            {
                "field": field,
                "probe_balanced_accuracy": row.get("balanced_accuracy", "NA"),
                "knn_enrichment_over_baseline": knn.get("enrichment_over_baseline", "NA"),
                "status": row.get("status", "NA"),
            }
        )

    write_csv(args.outdir / "probe_results.csv", probe_rows, ["field", "status", "train_rows", "test_rows", "train_classes", "test_classes_seen_in_train", "accuracy", "balanced_accuracy", "macro_f1"])
    write_csv(args.outdir / "knn_enrichment.csv", knn_rows, ["field", "status", "k", "mean_knn_same_group", "majority_class_baseline", "enrichment_over_baseline"])
    write_csv(args.outdir / "context_shortcut_score.csv", shortcut_rows, ["field", "probe_balanced_accuracy", "knn_enrichment_over_baseline", "status"])
    summary = {
        "manifest_csv": str(args.manifest_csv),
        "feature_dir": str(args.feature_dir),
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "probe_results": probe_rows,
        "knn_enrichment": knn_rows,
    }
    with (args.outdir / "representation_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
