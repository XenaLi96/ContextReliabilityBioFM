#!/usr/bin/env python3
"""Build a small balanced TCGA cancer-type classification manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from build_tcga_organized_manifest import (
    DEFAULT_ROOT,
    derive_age_bin,
    find_slide_paths,
    first_dict,
    load_json,
    project_info,
    slide_rank,
    slide_submitter_id,
)


DEFAULT_PROJECTS = ["TCGA-LUAD", "TCGA-LUSC", "TCGA-KIRC", "TCGA-THCA", "TCGA-GBM"]
PROJECT_PREFIXES = {
    "TCGA-LUAD": [
        "TCGA-05",
        "TCGA-17",
        "TCGA-35",
        "TCGA-44",
        "TCGA-49",
        "TCGA-50",
        "TCGA-53",
        "TCGA-55",
        "TCGA-69",
        "TCGA-71",
        "TCGA-73",
        "TCGA-75",
        "TCGA-86",
        "TCGA-91",
        "TCGA-95",
        "TCGA-97",
        "TCGA-99",
        "TCGA-J2",
        "TCGA-MP",
        "TCGA-NJ",
    ],
    "TCGA-LUSC": [
        "TCGA-18",
        "TCGA-21",
        "TCGA-22",
        "TCGA-33",
        "TCGA-39",
        "TCGA-52",
        "TCGA-56",
        "TCGA-60",
        "TCGA-63",
        "TCGA-66",
        "TCGA-77",
        "TCGA-85",
        "TCGA-92",
    ],
    "TCGA-KIRC": [
        "TCGA-A3",
        "TCGA-AK",
        "TCGA-B0",
        "TCGA-B4",
        "TCGA-B8",
        "TCGA-BP",
        "TCGA-CJ",
        "TCGA-CW",
        "TCGA-CZ",
        "TCGA-EU",
        "TCGA-MM",
    ],
    "TCGA-THCA": [
        "TCGA-BJ",
        "TCGA-CE",
        "TCGA-DJ",
        "TCGA-E3",
        "TCGA-E8",
        "TCGA-EL",
        "TCGA-EM",
        "TCGA-ET",
        "TCGA-FE",
        "TCGA-FK",
        "TCGA-FY",
        "TCGA-J8",
    ],
    "TCGA-GBM": [
        "TCGA-02",
        "TCGA-06",
        "TCGA-08",
        "TCGA-12",
        "TCGA-14",
        "TCGA-16",
        "TCGA-19",
        "TCGA-26",
        "TCGA-27",
        "TCGA-28",
        "TCGA-32",
        "TCGA-41",
        "TCGA-76",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--project-id", action="append", default=None)
    parser.add_argument("--per-project", type=int, default=20)
    parser.add_argument("--max-candidates-per-project", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260605)
    return parser.parse_args()


def find_project_metadata(root: Path, project_id: str, limit: int) -> List[Path]:
    paths: List[Path] = []
    prefixes = PROJECT_PREFIXES.get(project_id, [])
    for prefix in prefixes:
        for case_dir in sorted(root.glob(f"{prefix}*")):
            metadata_path = case_dir / "metadata.json"
            try:
                is_file = metadata_path.is_file()
            except OSError:
                continue
            if not is_file:
                continue
            try:
                metadata = load_json(metadata_path)
            except (OSError, json.JSONDecodeError):
                continue
            if str(project_info(metadata).get("project_id") or "") != project_id:
                continue
            paths.append(metadata_path)
            if len(paths) >= limit:
                return paths
    return paths


def make_row(root: Path, metadata_path: Path) -> Optional[Dict[str, str]]:
    case_dir = metadata_path.parent
    clinical_path = case_dir / "clinical.json"
    if not clinical_path.is_file():
        return None
    try:
        clinical = load_json(clinical_path)
        metadata = load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return None
    slides = find_slide_paths(case_dir)
    if not slides:
        return None

    project = project_info(metadata)
    demographic = clinical.get("demographic") or {}
    diagnosis = first_dict(clinical.get("diagnoses"))
    sex = str(demographic.get("sex_at_birth") or demographic.get("gender") or "").lower()
    age = demographic.get("age_at_index")
    if sex not in {"female", "male"}:
        return None
    try:
        age_value = float(age)
    except (TypeError, ValueError):
        return None

    slide_type, slide_path = sorted(
        [
            (
                slide_type,
                path,
                {
                    "slide_type": slide_type,
                    "slide_basename": path.name,
                },
            )
            for slide_type, path in slides
        ],
        key=lambda item: slide_rank(item[2]),
    )[0][:2]
    relpath = slide_path.relative_to(root)
    case_submitter_id = str(clinical.get("submitter_id") or metadata.get("submitter_id") or case_dir.name)
    project_id = str(project.get("project_id") or "")
    organ = str(clinical.get("primary_site") or diagnosis.get("tissue_or_organ_of_origin") or "")
    disease = str(project.get("name") or clinical.get("disease_type") or metadata.get("disease_type") or "")
    return {
        "sample_id": slide_submitter_id(slide_path),
        "patient_id": case_submitter_id,
        "case_id": str(clinical.get("case_id") or metadata.get("case_id") or metadata.get("id") or ""),
        "study_id": project_id,
        "organ": organ,
        "disease": disease,
        "age_bin": derive_age_bin(age_value),
        "age_at_index": str(age_value),
        "age_group_60": "age_ge_60" if age_value >= 60 else "age_lt_60",
        "sex": sex,
        "platform": "TCGA-WSI",
        "site": case_submitter_id.split("-")[1] if "-" in case_submitter_id else "",
        "split": "",
        "metric": "",
        "slide_type": slide_type,
        "slide_file_name": str(relpath),
        "slide_basename": slide_path.name,
        "primary_diagnosis": str(diagnosis.get("primary_diagnosis") or ""),
        "ajcc_pathologic_stage": str(diagnosis.get("ajcc_pathologic_stage") or ""),
        "vital_status": str(demographic.get("vital_status") or ""),
    }


def balanced_select(rows: List[Dict[str, str]], per_project: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    selected: List[Dict[str, str]] = []
    projects = sorted({row["study_id"] for row in rows})
    half = max(1, per_project // 2)
    for project_id in projects:
        project_rows = [row for row in rows if row["study_id"] == project_id]
        picked: List[Dict[str, str]] = []
        for sex in ["female", "male"]:
            sex_rows = [row for row in project_rows if row["sex"] == sex]
            rng.shuffle(sex_rows)
            picked.extend(sex_rows[:half])
        if len(picked) < per_project:
            picked_ids = {row["patient_id"] for row in picked}
            remainder = [row for row in project_rows if row["patient_id"] not in picked_ids]
            rng.shuffle(remainder)
            picked.extend(remainder[: per_project - len(picked)])
        selected.extend(sorted(picked[:per_project], key=lambda row: row["patient_id"]))
    selected.sort(key=lambda row: (row["study_id"], row["patient_id"]))
    return selected


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


def summarize(rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    rows = list(rows)
    by_project: Dict[str, Dict[str, Any]] = {}
    for project_id in sorted({row["study_id"] for row in rows}):
        group = [row for row in rows if row["study_id"] == project_id]
        by_project[project_id] = {
            "n": len(group),
            "sex_counts": {
                sex: sum(row["sex"] == sex for row in group) for sex in sorted({row["sex"] for row in group})
            },
            "age_group_60_counts": {
                age_group: sum(row["age_group_60"] == age_group for row in group)
                for age_group in sorted({row["age_group_60"] for row in group})
            },
        }
    return {"n_rows": len(rows), "by_project": by_project}


def main() -> None:
    args = parse_args()
    projects = args.project_id or DEFAULT_PROJECTS
    candidates: List[Dict[str, str]] = []
    candidate_counts: Dict[str, int] = {}
    for project_id in projects:
        metadata_paths = find_project_metadata(args.root, project_id, args.max_candidates_per_project)
        candidate_counts[project_id] = len(metadata_paths)
        for metadata_path in metadata_paths:
            row = make_row(args.root, metadata_path)
            if row is not None:
                candidates.append(row)

    selected = balanced_select(candidates, args.per_project, args.seed)
    write_csv(args.output_csv, selected)
    summary = {
        "root": str(args.root),
        "projects": projects,
        "per_project": args.per_project,
        "max_candidates_per_project": args.max_candidates_per_project,
        "candidate_metadata_counts": candidate_counts,
        "candidate_summary": summarize(candidates),
        "selected_summary": summarize(selected),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
