#!/usr/bin/env python3
"""Join cBioPortal mutation labels to a TCGA WSI metadata manifest."""

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

from build_tcga_cancer_type_manifest import PROJECT_PREFIXES, make_row  # noqa: E402
from build_tcga_organized_manifest import DEFAULT_ROOT  # noqa: E402


GENES = ["EGFR", "KRAS", "STK11", "TP53"]
EXCLUDED_MUTATION_TYPES = {
    "Silent",
    "Intron",
    "IGR",
    "3'UTR",
    "5'UTR",
    "3'Flank",
    "5'Flank",
    "RNA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sample-ids-json", type=Path, required=True)
    parser.add_argument("--mutations-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--project-id", default="TCGA-LUAD")
    parser.add_argument("--genes", nargs="*", default=GENES)
    parser.add_argument(
        "--scan-root",
        action="store_true",
        help="Build metadata by scanning --root-dir instead of reading --metadata-csv.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


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


def patient_from_sample(sample_id: str) -> str:
    return "-".join(str(sample_id).split("-")[:3])


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gene_symbol(record: Mapping[str, Any]) -> str:
    gene = record.get("gene")
    if isinstance(gene, Mapping):
        return str(gene.get("hugoGeneSymbol") or "").upper()
    return str(record.get("hugoGeneSymbol") or "").upper()


def is_coding_mutation(record: Mapping[str, Any]) -> bool:
    mutation_type = str(record.get("mutationType") or record.get("variantClassification") or "")
    return mutation_type not in EXCLUDED_MUTATION_TYPES


def mutation_sets(mutations: Sequence[Mapping[str, Any]], genes: Sequence[str]) -> Dict[str, set[str]]:
    wanted = {gene.upper() for gene in genes}
    mutated: Dict[str, set[str]] = {gene.upper(): set() for gene in genes}
    for record in mutations:
        symbol = gene_symbol(record)
        if symbol not in wanted or not is_coding_mutation(record):
            continue
        patient_id = str(record.get("patientId") or patient_from_sample(str(record.get("sampleId") or "")))
        if patient_id:
            mutated[symbol].add(patient_id)
    return mutated


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ages = pd.to_numeric(out["age_at_index"], errors="coerce")
    out["age_group_65"] = ["age_ge_65" if age >= 65 else "age_lt_65" for age in ages]
    out["age_group_70"] = ["age_ge_70" if age >= 70 else "age_lt_70" for age in ages]
    return out


def candidate_case_dirs(root_dir: Path, project_id: str) -> List[Path]:
    prefixes = PROJECT_PREFIXES.get(project_id)
    if not prefixes:
        return sorted(root_dir.iterdir(), key=lambda path: path.name)
    seen: set[Path] = set()
    case_dirs: List[Path] = []
    for prefix in prefixes:
        for case_dir in sorted(root_dir.glob(f"{prefix}*"), key=lambda path: path.name):
            if case_dir in seen:
                continue
            seen.add(case_dir)
            case_dirs.append(case_dir)
    return case_dirs


def scan_project_metadata(
    root_dir: Path,
    project_id: str,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, object]] = []
    stats: Counter[str] = Counter()
    candidates = candidate_case_dirs(root_dir, project_id)
    stats["candidate_case_dirs"] = len(candidates)
    for index, case_dir in enumerate(candidates, start=1):
        if max_cases is not None and index > max_cases:
            break
        if not case_dir.is_dir():
            continue
        stats["case_dirs_seen"] += 1
        metadata_path = case_dir / "metadata.json"
        try:
            has_metadata = metadata_path.is_file()
        except OSError:
            stats["unreadable_metadata_path"] += 1
            continue
        if not has_metadata:
            stats["missing_metadata_json"] += 1
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


def nested_counts(rows: List[Dict[str, object]], genes: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_rows": len(rows),
        "sex_counts": dict(Counter(str(row.get("sex", "")) for row in rows)),
        "age_group_60_counts": dict(Counter(str(row.get("age_group_60", "")) for row in rows)),
        "age_group_70_counts": dict(Counter(str(row.get("age_group_70", "")) for row in rows)),
        "genes": {},
    }
    for gene in genes:
        status_col = f"{gene}_status"
        gene_rows: Dict[str, Any] = {
            "status_counts": dict(Counter(str(row[status_col]) for row in rows)),
            "label_x_sex_x_age_group_60": {},
            "label_x_sex_x_age_group_70": {},
        }
        for age_field in ["age_group_60", "age_group_70"]:
            counts: Dict[str, int] = {}
            for row in rows:
                key = f"{row[status_col]} | {row.get('sex', '')} | {row.get(age_field, '')}"
                counts[key] = counts.get(key, 0) + 1
            gene_rows[f"label_x_sex_x_{age_field}"] = dict(sorted(counts.items()))
        summary["genes"][gene] = gene_rows
    return summary


def main() -> None:
    args = parse_args()
    genes = [gene.upper() for gene in args.genes]
    sample_ids = load_json(args.sample_ids_json)
    mutations = load_json(args.mutations_json)
    sequenced_patients = {patient_from_sample(sample_id) for sample_id in sample_ids}
    mutated = mutation_sets(mutations, genes)

    scan_stats: Dict[str, int] = {}
    if args.scan_root:
        metadata, scan_stats = scan_project_metadata(args.root_dir, args.project_id, args.max_cases)
    elif args.metadata_csv:
        metadata = pd.read_csv(args.metadata_csv)
    else:
        raise ValueError("Provide --metadata-csv or use --scan-root.")
    metadata = add_age_groups(metadata)
    df = metadata[
        (metadata["study_id"].astype(str) == args.project_id)
        & (metadata["patient_id"].astype(str).isin(sequenced_patients))
    ].copy()
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

    rows = df.to_dict(orient="records")
    write_csv(args.output_csv, rows)
    summary = {
        "metadata_csv": str(args.metadata_csv) if args.metadata_csv else None,
        "root_dir": str(args.root_dir) if args.scan_root else None,
        "scan_root": bool(args.scan_root),
        "scan_stats": scan_stats,
        "sample_ids_json": str(args.sample_ids_json),
        "mutations_json": str(args.mutations_json),
        "project_id": args.project_id,
        "genes": genes,
        "n_sequenced_samples": len(sample_ids),
        "n_sequenced_patients": len(sequenced_patients),
        "n_mutation_records": len(mutations),
        "mutated_patient_counts": {gene: len(mutated[gene]) for gene in genes},
        "manifest": nested_counts(rows, genes),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
