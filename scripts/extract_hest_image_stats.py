#!/usr/bin/env python3
"""Extract fixed image-statistic features from downloaded HEST patch h5 files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

DEFAULT_MANIFEST = Path("data/metadata/hest/selected_metadata_manifest.csv")
DEFAULT_RAW_ROOT = Path("data/raw/hest")
DEFAULT_OUTDIR = Path("data/embeddings/hest/image_stats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def decode_barcodes(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    decoded = []
    for value in flat:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return np.asarray(decoded, dtype=object)


def batch_features(images: np.ndarray) -> np.ndarray:
    x = images.astype(np.float32) / 255.0
    n = x.shape[0]
    flat = x.reshape(n, -1, 3)

    rgb_mean = flat.mean(axis=1)
    rgb_std = flat.std(axis=1)
    rgb_min = flat.min(axis=1)
    rgb_max = flat.max(axis=1)

    gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).reshape(n, -1)
    gray_stats = np.stack(
        [
            gray.mean(axis=1),
            gray.std(axis=1),
            gray.min(axis=1),
            gray.max(axis=1),
        ],
        axis=1,
    )

    maxc = flat.max(axis=2)
    minc = flat.min(axis=2)
    saturation = (maxc - minc) / np.maximum(maxc, 1e-6)
    sat_stats = np.stack(
        [
            saturation.mean(axis=1),
            saturation.std(axis=1),
            saturation.min(axis=1),
            saturation.max(axis=1),
        ],
        axis=1,
    )

    h, w = x.shape[1], x.shape[2]
    center = x[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, :].reshape(n, -1, 3).mean(axis=1)
    top_left = x[:, : h // 2, : w // 2, :].reshape(n, -1, 3).mean(axis=1)
    top_right = x[:, : h // 2, w // 2 :, :].reshape(n, -1, 3).mean(axis=1)
    bottom_left = x[:, h // 2 :, : w // 2, :].reshape(n, -1, 3).mean(axis=1)
    bottom_right = x[:, h // 2 :, w // 2 :, :].reshape(n, -1, 3).mean(axis=1)

    return np.concatenate(
        [
            rgb_mean,
            rgb_std,
            rgb_min,
            rgb_max,
            gray_stats,
            sat_stats,
            center,
            top_left,
            top_right,
            bottom_left,
            bottom_right,
        ],
        axis=1,
    ).astype(np.float32)


def extract_sample(sample_id: str, raw_root: Path, outdir: Path, batch_size: int, skip_existing: bool) -> Dict[str, object]:
    patch_path = raw_root / "patches" / f"{sample_id}.h5"
    if not patch_path.is_file():
        raise FileNotFoundError(patch_path)

    out_path = outdir / f"{sample_id}.npz"
    if skip_existing and out_path.is_file():
        data = np.load(out_path, allow_pickle=True)
        return {
            "sample_id": sample_id,
            "n_patches": int(data["features"].shape[0]),
            "n_features": int(data["features"].shape[1]),
            "output": str(out_path),
            "status": "skipped_existing",
        }

    with h5py.File(patch_path, "r") as handle:
        images = handle["img"]
        coords = handle["coords"][:]
        barcodes = decode_barcodes(handle["barcode"][:])
        features = []
        for start in range(0, images.shape[0], batch_size):
            end = min(start + batch_size, images.shape[0])
            features.append(batch_features(images[start:end]))
    feature_arr = np.vstack(features).astype(np.float32)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=feature_arr, barcodes=barcodes, coords=coords)
    return {
        "sample_id": sample_id,
        "n_patches": int(feature_arr.shape[0]),
        "n_features": int(feature_arr.shape[1]),
        "output": str(out_path),
        "status": "written",
    }


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest_csv)
    summary = {
        "manifest_csv": str(args.manifest_csv),
        "raw_root": str(args.raw_root),
        "outdir": str(args.outdir),
        "samples": [],
    }
    for row in rows:
        sample_id = row["sample_id"]
        info = extract_sample(sample_id, args.raw_root, args.outdir, args.batch_size, args.skip_existing)
        summary["samples"].append(info)
        print(json.dumps(info), flush=True)

    summary["n_samples"] = len(summary["samples"])
    summary["total_patches"] = sum(item["n_patches"] for item in summary["samples"])
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({"n_samples": summary["n_samples"], "total_patches": summary["total_patches"]}, indent=2))


if __name__ == "__main__":
    main()
