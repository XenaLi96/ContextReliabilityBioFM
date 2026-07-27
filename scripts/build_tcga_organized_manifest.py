#!/usr/bin/env python3
"""Build a local TCGA organized-data manifest for context-bias pilots."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_ROOT = Path("data/raw/tcga")
SLIDE_DIRS = ("diagnostic_slide", "tissue_slide")
SLIDE_SUFFIXES = {".svs", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Stop after this many case directories have been inspected.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after this many slide rows have been emitted.",
    )
    parser.add_argument(
        "--one-slide-per-case",
        action="store_true",
        help="Keep the best-ranked slide per case for leakage-free pilots.",
    )
    parser.add_argument(
        "--project-id",
        action="append",
        default=None,
        help="Optional TCGA project filter, e.g. TCGA-LUAD. Can be repeated.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    if isinstance(payload, dict):
        return payload
    return {}


def first_dict(items: Any) -> Dict[str, Any]:
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0]
    return {}


def derive_age_bin(age: Any) -> str:
    try:
        value = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if value < 40:
        return "<40"
    if value < 50:
        return "40-49"
    if value < 60:
        return "50-59"
    if value < 70:
        return "60-69"
    if value < 80:
        return "70-79"
    return "80+"


def barcode_tss(case_submitter_id: str) -> str:
    parts = case_submitter_id.split("-")
    if len(parts) >= 2:
        return parts[1]
    return ""


def project_info(metadata: Dict[str, Any]) -> Dict[str, Any]:
    project = metadata.get("project") or {}
    if isinstance(project, dict):
        return project
    return {}


def slide_submitter_id(path: Path) -> str:
    return path.name.split(".", 1)[0]


def slide_rank(row: Dict[str, str]) -> tuple:
    slide_type_rank = 0 if row["slide_type"] == "diagnostic_slide" else 1
    name = row["slide_basename"]
    dx_rank = 0 if "-DX" in name else 1
    sample_rank = 0 if "-01" in name else 1
    return (slide_type_rank, dx_rank, sample_rank, name)


def iter_case_dirs(root: Path) -> Iterable[Path]:
    with os.scandir(root) as scan:
        for entry in sorted(scan, key=lambda item: item.name):
            if entry.is_dir(follow_symlinks=False):
                yield Path(entry.path)


def find_slide_paths(case_dir: Path) -> List[tuple[str, Path]]:
    slides: List[tuple[str, Path]] = []
    for slide_dir in SLIDE_DIRS:
        directory = case_dir / slide_dir
        if not directory.is_dir():
            continue
        try:
            entries = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        for path in entries:
            if path.is_file() and path.suffix.lower() in SLIDE_SUFFIXES:
                slides.append((slide_dir, path))
    return slides


def build_rows(
    root: Path,
    max_cases: Optional[int],
    max_rows: Optional[int],
    one_slide_per_case: bool,
    project_filter: Optional[set[str]],
) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    rows: List[Dict[str, str]] = []
    stats: Counter[str] = Counter()
    projects: Counter[str] = Counter()
    sexes: Counter[str] = Counter()

    for inspected, case_dir in enumerate(iter_case_dirs(root), start=1):
        if max_cases is not None and inspected > max_cases:
            break

        stats["case_dirs_inspected"] += 1
        clinical_path = case_dir / "clinical.json"
        metadata_path = case_dir / "metadata.json"
        if not clinical_path.is_file() or not metadata_path.is_file():
            stats["case_dirs_missing_json"] += 1
            continue

        try:
            clinical = load_json(clinical_path)
            metadata = load_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            stats["case_dirs_bad_json"] += 1
            continue

        project = project_info(metadata)
        project_id = str(project.get("project_id") or "")
        if project_filter and project_id not in project_filter:
            stats["case_dirs_project_filtered"] += 1
            continue

        slides = find_slide_paths(case_dir)
        if not slides:
            stats["case_dirs_without_slides"] += 1
            continue

        demographic = clinical.get("demographic") or {}
        diagnosis = first_dict(clinical.get("diagnoses"))
        case_submitter_id = str(clinical.get("submitter_id") or metadata.get("submitter_id") or case_dir.name)
        case_id = str(clinical.get("case_id") or metadata.get("case_id") or metadata.get("id") or "")
        sex = str(demographic.get("sex_at_birth") or demographic.get("gender") or "").lower()
        age = demographic.get("age_at_index")
        organ = str(clinical.get("primary_site") or diagnosis.get("tissue_or_organ_of_origin") or "")
        disease = str(project.get("name") or clinical.get("disease_type") or metadata.get("disease_type") or "")

        case_rows: List[Dict[str, str]] = []
        for slide_type, slide_path in slides:
            relpath = slide_path.relative_to(root)
            case_rows.append(
                {
                    "sample_id": slide_submitter_id(slide_path),
                    "patient_id": case_submitter_id,
                    "case_id": case_id,
                    "study_id": project_id,
                    "organ": organ,
                    "disease": disease,
                    "age_bin": derive_age_bin(age),
                    "age_at_index": "" if age is None else str(age),
                    "sex": sex or "unknown",
                    "platform": "TCGA-WSI",
                    "site": barcode_tss(case_submitter_id),
                    "split": "",
                    "metric": "",
                    "slide_type": slide_type,
                    "slide_file_name": str(relpath),
                    "slide_basename": slide_path.name,
                    "primary_diagnosis": str(diagnosis.get("primary_diagnosis") or ""),
                    "ajcc_pathologic_stage": str(diagnosis.get("ajcc_pathologic_stage") or ""),
                    "vital_status": str(demographic.get("vital_status") or ""),
                }
            )

        if one_slide_per_case:
            case_rows = [sorted(case_rows, key=slide_rank)[0]]

        for row in case_rows:
            rows.append(row)
            projects[row["study_id"]] += 1
            sexes[row["sex"]] += 1
            if max_rows is not None and len(rows) >= max_rows:
                stats["stopped_at_max_rows"] = 1
                summary = make_summary(root, rows, stats, projects, sexes)
                return rows, summary

    summary = make_summary(root, rows, stats, projects, sexes)
    return rows, summary


def make_summary(
    root: Path,
    rows: List[Dict[str, str]],
    stats: Counter[str],
    projects: Counter[str],
    sexes: Counter[str],
) -> Dict[str, Any]:
    patient_ids = {row["patient_id"] for row in rows}
    return {
        "root": str(root),
        "row_count": len(rows),
        "patient_count": len(patient_ids),
        "stats": dict(stats),
        "project_counts_top20": dict(projects.most_common(20)),
        "sex_counts": dict(sexes),
    }


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    project_filter = set(args.project_id) if args.project_id else None
    rows, summary = build_rows(
        root=args.root,
        max_cases=args.max_cases,
        max_rows=args.max_rows,
        one_slide_per_case=args.one_slide_per_case,
        project_filter=project_filter,
    )
    write_csv(args.output_csv, rows)
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
