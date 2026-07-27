#!/usr/bin/env python3
"""Select a small HEST public-replication subset from a context manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

DEFAULT_MANIFEST = Path("data/metadata/hest/metadata_manifest.csv")
DEFAULT_OUTPUT_IDS = Path("data/metadata/hest/selected_hest_ids.txt")
DEFAULT_OUTPUT_MANIFEST = Path("data/metadata/hest/selected_metadata_manifest.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-ids", type=Path, default=DEFAULT_OUTPUT_IDS)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--species", default="Homo sapiens")
    parser.add_argument(
        "--platforms",
        nargs="*",
        default=["Visium", "Spatial Transcriptomics", "Xenium", "Xenium 5k", "Visium HD", "Visium HD 3'"],
    )
    parser.add_argument("--min-per-platform", type=int, default=3)
    parser.add_argument("--max-per-platform", type=int, default=10)
    parser.add_argument("--min-platforms", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_rank(row: Dict[str, str], seed: int) -> str:
    payload = f"{seed}:{row.get('sample_id', '')}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def same_token(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.manifest_csv)
    if not rows:
        raise SystemExit(f"No rows in {args.manifest_csv}")

    allowed_platforms = {platform.strip().lower() for platform in args.platforms}
    filtered = [
        row
        for row in rows
        if same_token(row.get("species", ""), args.species)
        and row.get("platform", "").strip().lower() in allowed_platforms
    ]

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in filtered:
        grouped[row["platform"]].append(row)

    eligible_platforms = {
        platform: sorted(platform_rows, key=lambda row: stable_rank(row, args.seed))
        for platform, platform_rows in grouped.items()
        if len(platform_rows) >= args.min_per_platform
    }
    if len(eligible_platforms) < args.min_platforms:
        counts = {platform: len(platform_rows) for platform, platform_rows in grouped.items()}
        raise SystemExit(
            "Not enough eligible platforms for site/platform replication. "
            f"Need {args.min_platforms}, got {len(eligible_platforms)}. Counts: {counts}"
        )

    selected: List[Dict[str, str]] = []
    for platform in sorted(eligible_platforms):
        selected.extend(eligible_platforms[platform][: args.max_per_platform])
    selected.sort(key=lambda row: (row["platform"], row["sample_id"]))

    args.output_ids.parent.mkdir(parents=True, exist_ok=True)
    with args.output_ids.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(f"{row['sample_id']}\n")

    write_csv(args.output_manifest, selected, list(rows[0].keys()))
    summary = {
        "manifest_csv": str(args.manifest_csv),
        "output_ids": str(args.output_ids),
        "output_manifest": str(args.output_manifest),
        "selected_count": len(selected),
        "selected_by_platform": {
            platform: sum(row["platform"] == platform for row in selected)
            for platform in sorted({row["platform"] for row in selected})
        },
        "species": args.species,
        "platforms": args.platforms,
        "min_per_platform": args.min_per_platform,
        "max_per_platform": args.max_per_platform,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
