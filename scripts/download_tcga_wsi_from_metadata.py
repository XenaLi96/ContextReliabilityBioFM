#!/usr/bin/env python3
"""Download TCGA slide files listed in a metadata CSV."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests


GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TCGA WSI files from a metadata CSV."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--organ",
        default=None,
        help="Optional organ_group filter.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL download manifest.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected_bytes(row: Dict[str, str]) -> int:
    size_gb = float(row.get("slide_size_gb") or 0)
    return int(size_gb * (1024 ** 3))


def filter_rows(rows: Iterable[Dict[str, str]], organ: str | None) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    for row in rows:
        if organ and row.get("organ_group") != organ:
            continue
        if not row.get("slide_file_id") or not row.get("slide_file_name"):
            continue
        selected.append(row)
    return selected


def needs_redownload(target: Path, row: Dict[str, str]) -> bool:
    if not target.exists() or target.stat().st_size == 0:
        return True
    exp_bytes = expected_bytes(row)
    if exp_bytes <= 0:
        return False
    return target.stat().st_size < int(exp_bytes * 0.90)


def download_one(
    session: requests.Session,
    row: Dict[str, str],
    outdir: Path,
    timeout: int,
    retries: int,
) -> Tuple[str, Path, int]:
    organ = row.get("organ_group", "unknown")
    target_dir = outdir / organ
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / row["slide_file_name"]
    file_id = row["slide_file_id"]

    if not needs_redownload(target_path, row):
        return "skipped_existing", target_path, target_path.stat().st_size

    url = f"{GDC_DATA_URL}/{file_id}"
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp_path.replace(target_path)
            return "downloaded", target_path, target_path.stat().st_size
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                time.sleep(min(5 * attempt, 20))

    raise RuntimeError(f"Download failed for {file_id}: {last_error}")


def main() -> None:
    args = parse_args()
    rows = filter_rows(read_rows(args.metadata_csv), args.organ)
    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        raise ValueError("No rows matched the download criteria.")

    manifest_path = args.manifest_jsonl
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "metadata_csv": str(args.metadata_csv),
        "outdir": str(args.outdir),
        "organ": args.organ,
        "limit": args.limit,
        "requested": len(rows),
        "downloaded": 0,
        "skipped_existing": 0,
        "failed": 0,
        "total_bytes": 0,
    }

    session = requests.Session()
    for index, row in enumerate(rows, start=1):
        record = {
            "index": index,
            "organ_group": row.get("organ_group"),
            "project_id": row.get("project_id"),
            "case_id": row.get("case_id"),
            "slide_file_id": row.get("slide_file_id"),
            "slide_file_name": row.get("slide_file_name"),
        }
        try:
            status, target_path, size_bytes = download_one(
                session=session,
                row=row,
                outdir=args.outdir,
                timeout=args.timeout,
                retries=args.retries,
            )
            record["status"] = status
            record["target_path"] = str(target_path)
            record["size_bytes"] = size_bytes
            summary[status] += 1
            summary["total_bytes"] += size_bytes
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            summary["failed"] += 1

        if manifest_path is not None:
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
