#!/usr/bin/env python3
"""Download HEST-1k metadata or a selected sample subset from Hugging Face."""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List

DEFAULT_REPO_ID = "MahmoodLab/hest"
DEFAULT_LOCAL_DIR = Path("data/raw/hest")
DEFAULT_METADATA_DIR = Path("data/metadata/hest")
DEFAULT_METADATA_FILE = "HEST_v1_3_0.csv"
DEFAULT_COMPONENTS = ["metadata", "st", "patches", "thumbnails"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-all", action="store_true")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument(
        "--components",
        nargs="*",
        default=DEFAULT_COMPONENTS,
        help=(
            "HEST top-level components to download for selected ids. "
            "Use `all` for every matching file. Default: metadata st patches thumbnails."
        ),
    )
    parser.add_argument(
        "--unzip-cellvit",
        action="store_true",
        help="Unzip downloaded cellvit_seg zip files after download.",
    )
    return parser.parse_args()


def require_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with "
            "`python3 -m pip install --user huggingface-hub`, accept HEST terms, "
            "then run `huggingface-cli login`."
        ) from exc
    return snapshot_download


def read_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = [sample_id.strip() for sample_id in args.ids if sample_id.strip()]
    if args.ids_file:
        with args.ids_file.open(encoding="utf-8") as handle:
            ids.extend(
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            )
    deduped = sorted(set(ids))
    return deduped


def patterns_for_ids(sample_ids: Iterable[str], metadata_file: str, components: List[str]) -> List[str]:
    patterns = [metadata_file]
    if "all" in {component.lower() for component in components}:
        for sample_id in sample_ids:
            patterns.append(f"*{sample_id}[_.]**")
        return patterns

    for sample_id in sample_ids:
        for component in components:
            component = component.strip().strip("/")
            if component:
                patterns.append(f"{component}/*{sample_id}[_.]**")
    return patterns


def unzip_cellvit(local_dir: Path) -> None:
    seg_dir = local_dir / "cellvit_seg"
    if not seg_dir.exists():
        return
    for path_zip in seg_dir.glob("*.zip"):
        target_dir = seg_dir / path_zip.stem
        if target_dir.exists():
            continue
        with zipfile.ZipFile(path_zip, "r") as zip_ref:
            zip_ref.extractall(seg_dir)


def main() -> None:
    args = parse_args()
    snapshot_download = require_huggingface_hub()

    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if not token_present:
        print(
            "No HF_TOKEN/HUGGINGFACE_HUB_TOKEN environment variable detected. "
            "snapshot_download will still use a cached `huggingface-cli login` token if present.",
            file=sys.stderr,
        )

    if args.metadata_only:
        args.metadata_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=[args.metadata_file],
            local_dir=str(args.metadata_dir),
        )
        print(args.metadata_dir / args.metadata_file)
        return

    sample_ids = read_ids(args)
    if not sample_ids and not args.download_all:
        raise SystemExit(
            "Refusing to download HEST without --ids/--ids-file. "
            "The full dataset is multi-TB; pass --download-all explicitly if intended."
        )

    patterns = "*" if args.download_all else patterns_for_ids(sample_ids, args.metadata_file, args.components)
    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(args.local_dir),
    )
    if args.unzip_cellvit:
        unzip_cellvit(args.local_dir)
    print(args.local_dir)


if __name__ == "__main__":
    main()
