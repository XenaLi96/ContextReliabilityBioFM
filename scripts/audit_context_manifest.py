#!/usr/bin/env python3
"""Audit a ContextShift-Bio metadata manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

DEFAULT_MANIFEST = Path("data/metadata/hest/metadata_manifest.csv")
DEFAULT_AXES = Path("context_benchmark/context_axes.yaml")
DEFAULT_OUTDIR = Path("outputs/hest/manifest_audit")
MISSING = {"", "na", "n/a", "none", "null", "unknown", "nan"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--context-axes", type=Path, default=DEFAULT_AXES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def is_missing(value: str) -> bool:
    return (value or "").strip().lower() in MISSING


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def entropy(values: List[str]) -> float:
    kept = [value for value in values if not is_missing(value)]
    if not kept:
        return 0.0
    total = len(kept)
    counts = Counter(kept)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def effective_diversity(values: List[str]) -> float:
    return math.exp(entropy(values))


def missingness_report(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    total = len(rows)
    columns = rows[0].keys() if rows else []
    report: List[Dict[str, object]] = []
    for column in columns:
        missing = sum(is_missing(row.get(column, "")) for row in rows)
        report.append(
            {
                "column": column,
                "missing_count": missing,
                "missing_fraction": round(missing / total, 6) if total else 0,
                "unique_nonmissing": len({row.get(column, "") for row in rows if not is_missing(row.get(column, ""))}),
            }
        )
    return report


def axis_counts(rows: List[Dict[str, str]], axes: Iterable[str]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for axis in axes:
        counts = Counter(row.get(axis, "NA") or "NA" for row in rows)
        for value, count in counts.most_common():
            output.append({"axis": axis, "value": value, "count": count})
    return output


def split_counts(rows: List[Dict[str, str]], group_field: str) -> List[Dict[str, object]]:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        grouped[row.get(group_field, "NA")][row.get("split", "NA")] += 1
    output: List[Dict[str, object]] = []
    for group, counts in sorted(grouped.items()):
        payload = {"group_field": group_field, "group_value": group}
        payload.update(dict(counts))
        output.append(payload)
    return output


def leakage_risk(rows: List[Dict[str, str]]) -> Dict[str, object]:
    patient_to_splits: Dict[str, set] = defaultdict(set)
    sample_id_patient_rows = 0
    for row in rows:
        patient = row.get("patient_id", "NA")
        sample = row.get("sample_id", "NA")
        patient_to_splits[patient].add(row.get("split", "NA"))
        if patient == sample:
            sample_id_patient_rows += 1
    crossing = {
        patient: sorted(splits)
        for patient, splits in patient_to_splits.items()
        if len(splits - {"unassigned"}) > 1
    }
    return {
        "row_count": len(rows),
        "patients_crossing_splits": crossing,
        "patient_crossing_split_count": len(crossing),
        "rows_with_patient_id_equal_sample_id": sample_id_patient_rows,
        "risk_level": "high" if crossing or sample_id_patient_rows else "low",
    }


def load_axes(path: Path) -> Dict[str, object]:
    if yaml is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    args = parse_args()
    rows = read_csv(args.manifest_csv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    axes_cfg = load_axes(args.context_axes)
    required = axes_cfg.get("required_manifest_columns", []) if isinstance(axes_cfg, dict) else []

    missing_required = [column for column in required if rows and column not in rows[0]]
    core_axes = ["study_id", "site", "platform", "organ", "disease", "age_bin", "sex", "species", "split"]

    card = {
        "manifest_csv": str(args.manifest_csv),
        "context_axes": str(args.context_axes),
        "row_count": len(rows),
        "required_columns_missing": missing_required,
        "axis_effective_diversity": {
            axis: round(effective_diversity([row.get(axis, "") for row in rows]), 4)
            for axis in core_axes
        },
        "split_counts": dict(Counter(row.get("split", "NA") for row in rows)),
        "metric_values": sorted({row.get("metric", "NA") for row in rows}),
    }

    write_json(args.outdir / "dataset_card.json", card)
    write_csv(
        args.outdir / "missingness_report.csv",
        missingness_report(rows),
        ["column", "missing_count", "missing_fraction", "unique_nonmissing"],
    )
    write_csv(
        args.outdir / "context_axis_counts.csv",
        axis_counts(rows, core_axes),
        ["axis", "value", "count"],
    )
    write_csv(
        args.outdir / "site_platform_split_counts.csv",
        split_counts(rows, "platform") + split_counts(rows, "site"),
        ["group_field", "group_value", "train", "val", "test", "ood_test", "unassigned"],
    )
    write_json(args.outdir / "leakage_risk_report.json", leakage_risk(rows))
    print(json.dumps(card, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
