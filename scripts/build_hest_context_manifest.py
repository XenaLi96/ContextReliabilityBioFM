#!/usr/bin/env python3
"""Build a ContextShift-Bio metadata manifest from HEST metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

DEFAULT_METADATA_CSV = Path("data/metadata/hest/HEST_v1_3_0.csv")
DEFAULT_OUTPUT_CSV = Path("data/metadata/hest/metadata_manifest.csv")
DEFAULT_RAW_ROOT = Path("data/raw/hest")

MANIFEST_COLUMNS = [
    "sample_id",
    "patient_id",
    "study_id",
    "organ",
    "disease",
    "disease_subtype",
    "age",
    "age_bin",
    "sex",
    "species",
    "platform",
    "assay",
    "site",
    "scanner",
    "stain",
    "tissue_processing",
    "cell_type",
    "spatial_niche",
    "perturbation_identity",
    "perturbation_target",
    "treatment_condition",
    "split",
    "split_strategy",
    "split_group",
    "task",
    "metric",
    "raw_data_root",
    "wsi_path",
    "st_path",
    "patch_path",
    "source_metadata_id",
    "notes",
]

FIELD_ALIASES = {
    "sample_id": ["sample_id", "id", "slide_id", "name"],
    "patient_id": ["patient_id", "donor_id", "subject_id", "case_id", "patient", "donor"],
    "study_id": ["study_id", "dataset_title", "cohort", "study", "dataset", "publication", "study_link", "source"],
    "organ": ["organ", "tissue", "tissue_type"],
    "disease": ["disease", "disease_state", "diagnosis", "cancer_type", "oncotree_code"],
    "disease_subtype": ["disease_subtype", "subtype", "oncotree_code", "tumor_subtype"],
    "age": ["age", "patient_age", "donor_age"],
    "age_bin": ["age_bin", "age_range", "age_group", "age_band"],
    "sex": ["sex", "gender", "donor_sex", "patient_sex"],
    "species": ["species", "organism"],
    "platform": ["platform", "technology", "st_technology", "spatial_technology", "tech"],
    "assay": ["assay", "sequencing_assay", "technology", "st_technology"],
    "site": ["site", "dataset_title", "institution", "center", "cohort", "study", "study_link", "download_page_link1", "source"],
    "scanner": ["scanner", "scanner_model"],
    "stain": ["stain", "staining"],
    "tissue_processing": ["tissue_processing", "preservation_method", "preservation", "sectioning", "ffpe_fresh_frozen"],
}

MISSING = {"", "na", "n/a", "none", "null", "unknown", "nan"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--split-axis",
        choices=("unassigned", "cross_patient", "site_platform"),
        default="unassigned",
    )
    parser.add_argument("--auto-holdout-platform", action="store_true")
    parser.add_argument("--holdout-platform")
    parser.add_argument("--holdout-site")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING


def normalize(value: str) -> str:
    value = (value or "").strip()
    return "NA" if is_missing(value) else value


def first_existing(row: Dict[str, str], names: Sequence[str]) -> str:
    lower_to_key = {key.lower(): key for key in row}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is not None and not is_missing(row.get(key, "")):
            return normalize(row.get(key, ""))
    return "NA"


def stable_unit_bucket(value: str, seed: int) -> float:
    payload = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def choose_auto_holdout(rows: List[Dict[str, str]], field: str) -> str:
    counts = Counter(row[field] for row in rows if row[field] != "NA")
    eligible = [(count, value) for value, count in counts.items() if count >= 3]
    if not eligible:
        return ""
    eligible.sort()
    return eligible[0][1]


def assign_splits(
    rows: List[Dict[str, str]],
    split_axis: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    holdout_platform: str = "",
    holdout_site: str = "",
) -> None:
    if split_axis == "unassigned":
        for row in rows:
            row["split"] = "unassigned"
            row["split_strategy"] = "unassigned"
            row["split_group"] = "NA"
        return

    if split_axis == "site_platform":
        split_group = holdout_platform or holdout_site
        split_field = "platform" if holdout_platform else "site"
        for row in rows:
            if split_group and row.get(split_field) == split_group:
                row["split"] = "ood_test"
            else:
                bucket = stable_unit_bucket(row["patient_id"], seed)
                row["split"] = "val" if bucket < val_fraction else "train"
            row["split_strategy"] = "site_platform"
            row["split_group"] = split_group or "NA"
        return

    for row in rows:
        bucket = stable_unit_bucket(row["patient_id"], seed)
        if bucket < test_fraction:
            split = "test"
        elif bucket < test_fraction + val_fraction:
            split = "val"
        else:
            split = "train"
        row["split"] = split
        row["split_strategy"] = "cross_patient"
        row["split_group"] = "patient_id"


def derive_paths(sample_id: str, raw_root: Path) -> Dict[str, str]:
    return {
        "raw_data_root": str(raw_root),
        "wsi_path": str(raw_root / "wsis" / f"{sample_id}.tif"),
        "st_path": str(raw_root / "st" / f"{sample_id}.h5ad"),
        "patch_path": str(raw_root / "patches" / f"{sample_id}.h5"),
    }


def build_manifest(rows: Iterable[Dict[str, str]], raw_root: Path) -> List[Dict[str, str]]:
    manifest: List[Dict[str, str]] = []
    for source_row in rows:
        row = {column: "NA" for column in MANIFEST_COLUMNS}
        for field, aliases in FIELD_ALIASES.items():
            row[field] = first_existing(source_row, aliases)

        if row["sample_id"] == "NA":
            raise ValueError("HEST metadata row has no sample id/id field")
        if row["patient_id"] == "NA":
            row["patient_id"] = row["sample_id"]
            row["notes"] = "patient_id_missing_used_sample_id"
        if row["study_id"] == "NA":
            row["study_id"] = row["site"] if row["site"] != "NA" else "HEST"
        if row["site"] == "NA":
            row["site"] = row["study_id"]
        if row["age_bin"] == "NA" and row["age"] != "NA":
            row["age_bin"] = row["age"]

        row.update(derive_paths(row["sample_id"], raw_root))
        row["task"] = "histology_to_gene_prediction"
        row["metric"] = (
            "average_gene_pearson;average_gene_spearman;"
            "worst_group_gene_pearson;worst_group_gene_spearman;nonzero_calibration"
        )
        row["source_metadata_id"] = row["sample_id"]
        manifest.append(row)
    return manifest


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    summary = {
        "metadata_csv": str(args.metadata_csv),
        "output_csv": str(args.output_csv),
        "row_count": len(rows),
        "split_axis": args.split_axis,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "platform_counts": dict(Counter(row["platform"] for row in rows)),
        "site_counts": dict(Counter(row["site"] for row in rows)),
        "organ_counts": dict(Counter(row["organ"] for row in rows)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    source_rows = read_csv(args.metadata_csv)
    manifest = build_manifest(source_rows, args.raw_root)

    holdout_platform = args.holdout_platform or ""
    holdout_site = args.holdout_site or ""
    if args.split_axis == "site_platform" and args.auto_holdout_platform and not holdout_platform:
        holdout_platform = choose_auto_holdout(manifest, "platform")

    assign_splits(
        manifest,
        args.split_axis,
        args.seed,
        args.val_fraction,
        args.test_fraction,
        holdout_platform=holdout_platform,
        holdout_site=holdout_site,
    )
    write_csv(args.output_csv, manifest)
    if args.summary_json:
        write_summary(args.summary_json, manifest, args)
    print(json.dumps({"rows": len(manifest), "output_csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
