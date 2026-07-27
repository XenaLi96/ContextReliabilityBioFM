#!/usr/bin/env python3
"""Build a TCGA WSI manifest by joining cBioPortal mutations to local patient dirs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_tcga_cancer_type_manifest import make_row  # noqa: E402
from build_tcga_cbio_mutation_manifest import (  # noqa: E402
    add_age_groups,
    mutation_sets,
    patient_from_sample,
)
from build_tcga_organized_manifest import DEFAULT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sample-ids-json", type=Path, required=True)
    parser.add_argument("--mutations-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--genes", nargs="*", required=True)
    parser.add_argument(
        "--combined-label",
        action="append",
        default=[],
        help="Combined binary label as LABEL:GENE,GENE, e.g. IDH:IDH1,IDH2.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_combined_specs(specs: Sequence[str]) -> Dict[str, List[str]]:
    parsed: Dict[str, List[str]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --combined-label {spec}; expected LABEL:GENE,GENE")
        label, genes_raw = spec.split(":", 1)
        genes = [gene.strip().upper() for gene in genes_raw.split(",") if gene.strip()]
        if not label.strip() or not genes:
            raise ValueError(f"Invalid --combined-label {spec}; expected LABEL:GENE,GENE")
        parsed[label.strip().upper()] = genes
    return parsed


def patient_metadata_rows(
    root_dir: Path,
    project_id: str,
    sequenced_patients: Sequence[str],
) -> tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, object]] = []
    stats: Counter[str] = Counter()
    for patient_id in sorted(set(sequenced_patients)):
        stats["patients_seen"] += 1
        metadata_path = root_dir / patient_id / "metadata.json"
        try:
            has_metadata = metadata_path.is_file()
        except OSError:
            stats["unreadable_metadata_path"] += 1
            continue
        if not has_metadata:
            stats["missing_local_patient_dir_or_metadata"] += 1
            continue
        try:
            row = make_row(root_dir, metadata_path)
        except Exception:  # noqa: BLE001
            stats["bad_or_unreadable_case"] += 1
            continue
        if row is None:
            stats["no_usable_row"] += 1
            continue
        if str(row.get("study_id") or "") != project_id:
            stats["project_filtered"] += 1
            continue
        rows.append(row)
    return pd.DataFrame(rows), dict(stats)


def label_summary(rows: List[Dict[str, object]], labels: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_rows": len(rows),
        "sex_counts": dict(Counter(str(row.get("sex", "")) for row in rows)),
        "age_group_60_counts": dict(Counter(str(row.get("age_group_60", "")) for row in rows)),
        "age_group_70_counts": dict(Counter(str(row.get("age_group_70", "")) for row in rows)),
        "labels": {},
    }
    for label in labels:
        status_col = f"{label}_status"
        label_rows: Dict[str, Any] = {
            "status_counts": dict(Counter(str(row[status_col]) for row in rows)),
        }
        for age_field in ["age_group_60", "age_group_70"]:
            counts: Dict[str, int] = {}
            for row in rows:
                key = f"{row[status_col]} | {row.get('sex', '')} | {row.get(age_field, '')}"
                counts[key] = counts.get(key, 0) + 1
            label_rows[f"label_x_sex_x_{age_field}"] = dict(sorted(counts.items()))
        summary["labels"][label] = label_rows
    return summary


def main() -> None:
    args = parse_args()
    genes = [gene.upper() for gene in args.genes]
    combined = parse_combined_specs(args.combined_label)
    sample_ids = load_json(args.sample_ids_json)
    mutations = load_json(args.mutations_json)
    sequenced_patients = [patient_from_sample(sample_id) for sample_id in sample_ids]
    mutated = mutation_sets(mutations, genes)

    metadata, scan_stats = patient_metadata_rows(args.root_dir, args.project_id, sequenced_patients)
    metadata = add_age_groups(metadata)
    df = metadata[metadata["patient_id"].astype(str).isin(sequenced_patients)].copy()
    df = df.dropna(subset=["sex", "age_at_index"]).copy()
    df = df[df["sex"].astype(str).isin(["female", "male"])].copy()

    for gene in genes:
        df[f"{gene}_status"] = [
            f"{gene}_mut" if patient_id in mutated[gene] else f"{gene}_wt"
            for patient_id in df["patient_id"].astype(str)
        ]
        df[f"{gene}_mutated"] = [
            "1" if patient_id in mutated[gene] else "0"
            for patient_id in df["patient_id"].astype(str)
        ]
    for label, label_genes in combined.items():
        label_mutated = set().union(*(mutated[gene] for gene in label_genes))
        df[f"{label}_status"] = [
            f"{label}_mut" if patient_id in label_mutated else f"{label}_wt"
            for patient_id in df["patient_id"].astype(str)
        ]
        df[f"{label}_mutated"] = [
            "1" if patient_id in label_mutated else "0"
            for patient_id in df["patient_id"].astype(str)
        ]

    rows = df.to_dict(orient="records")
    label_names = [*genes, *combined.keys()]
    write_csv(args.output_csv, rows)
    summary = {
        "root_dir": str(args.root_dir),
        "sample_ids_json": str(args.sample_ids_json),
        "mutations_json": str(args.mutations_json),
        "project_id": args.project_id,
        "genes": genes,
        "combined_labels": combined,
        "n_sequenced_samples": len(sample_ids),
        "n_sequenced_patients": len(set(sequenced_patients)),
        "n_mutation_records": len(mutations),
        "mutated_patient_counts": {gene: len(mutated[gene]) for gene in genes},
        "scan_stats": scan_stats,
        "manifest": label_summary(rows, label_names),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
