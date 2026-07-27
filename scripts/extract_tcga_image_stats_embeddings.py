#!/usr/bin/env python3
"""Extract low-level image-statistic embeddings for TCGA slides.

The output schema matches the pathology FM feature extractors: one compressed
``embedding`` vector per slide under ``features/``. This makes the existing
TCGA context-shift evaluator reusable for color/texture control baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openslide
from PIL import Image

from run_uni_batch_extract import (  # noqa: E402
    build_sample_key,
    can_try_image_fallback,
    group_rows,
    is_wsi_path,
    load_rows,
    pick_patch_coords,
    resolve_input_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-suffix", default="_image_stats_embedding.npz")
    parser.add_argument("--group-by", choices=("slide", "case"), default="slide")
    parser.add_argument("--path-column", default="slide_file_name")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--num-patches", type=int, default=16)
    parser.add_argument("--read-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--missing-policy", choices=("skip", "fail"), default="skip")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    mask = delta > 1e-8
    r_mask = mask & (maxc == r)
    g_mask = mask & (maxc == g)
    b_mask = mask & (maxc == b)
    hue[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    hue = hue / 6.0
    sat = np.zeros_like(maxc)
    nonzero = maxc > 1e-8
    sat[nonzero] = delta[nonzero] / maxc[nonzero]
    val = maxc
    return np.stack([hue, sat, val], axis=-1)


def channel_stats(values: np.ndarray, prefix: str, names: List[str]) -> Tuple[List[float], List[str]]:
    feats: List[float] = []
    feat_names: List[str] = []
    for idx, name in enumerate(names):
        x = values[:, idx] if values.ndim == 2 else values
        stats = [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.percentile(x, 10)),
            float(np.percentile(x, 50)),
            float(np.percentile(x, 90)),
        ]
        feats.extend(stats)
        feat_names.extend(
            [
                f"{prefix}_{name}_mean",
                f"{prefix}_{name}_std",
                f"{prefix}_{name}_p10",
                f"{prefix}_{name}_p50",
                f"{prefix}_{name}_p90",
            ]
        )
    return feats, feat_names


def hist_features(values: np.ndarray, prefix: str, bins: int = 8) -> Tuple[List[float], List[str]]:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=True)
    hist = np.nan_to_num(hist.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return hist.astype(float).tolist(), [f"{prefix}_hist_{i:02d}" for i in range(bins)]


def patch_features(image: Image.Image) -> Tuple[np.ndarray, List[str]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    flat_rgb = rgb.reshape(-1, 3)
    gray = rgb.mean(axis=2)
    hsv = rgb_to_hsv(rgb).reshape(-1, 3)
    tissue_mask = (gray < 0.92) & (np.std(rgb, axis=2) > 0.03)

    gy, gx = np.gradient(gray.astype(np.float32))
    grad_mag = np.sqrt(gx * gx + gy * gy)

    feats: List[float] = []
    names: List[str] = []
    part, part_names = channel_stats(flat_rgb, "rgb", ["r", "g", "b"])
    feats.extend(part)
    names.extend(part_names)
    part, part_names = channel_stats(hsv, "hsv", ["h", "s", "v"])
    feats.extend(part)
    names.extend(part_names)
    part, part_names = channel_stats(gray.reshape(-1), "gray", ["gray"])
    feats.extend(part)
    names.extend(part_names)
    for idx, channel in enumerate(["r", "g", "b"]):
        part, part_names = hist_features(flat_rgb[:, idx], f"rgb_{channel}")
        feats.extend(part)
        names.extend(part_names)
    for idx, channel in enumerate(["h", "s", "v"]):
        part, part_names = hist_features(hsv[:, idx], f"hsv_{channel}")
        feats.extend(part)
        names.extend(part_names)
    part, part_names = hist_features(gray.reshape(-1), "gray")
    feats.extend(part)
    names.extend(part_names)
    feats.extend(
        [
            float(tissue_mask.mean()),
            float(np.mean(grad_mag)),
            float(np.std(grad_mag)),
            float(np.percentile(grad_mag, 90)),
            float(np.mean(grad_mag[tissue_mask])) if tissue_mask.any() else 0.0,
        ]
    )
    names.extend(["tissue_fraction", "edge_mean", "edge_std", "edge_p90", "edge_tissue_mean"])
    return np.asarray(feats, dtype=np.float32), names


def extract_wsi_stats(slide_path: Path, num_patches: int, read_size: int, seed: int) -> Tuple[np.ndarray, Dict[str, object], List[str]]:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        coords = pick_patch_coords(slide=slide, num_patches=num_patches, read_size=read_size, seed=seed)
        patches = [slide.read_region((x, y), 0, (read_size, read_size)).convert("RGB") for x, y in coords]
        slide_dimensions = list(slide.dimensions)
    finally:
        slide.close()

    patch_arrays = []
    feature_names: List[str] = []
    for patch in patches:
        feats, feature_names = patch_features(patch)
        patch_arrays.append(feats)
    patch_matrix = np.stack(patch_arrays, axis=0)
    embedding = np.concatenate(
        [
            patch_matrix.mean(axis=0),
            patch_matrix.std(axis=0),
        ],
        axis=0,
    ).astype(np.float32)
    names = [f"mean_{name}" for name in feature_names] + [f"std_{name}" for name in feature_names]
    meta = {
        "input_kind": "wsi",
        "slide_dimensions": slide_dimensions,
        "num_patches": int(len(coords)),
        "patch_coords": coords,
        "patch_feature_shape": list(patch_matrix.shape),
    }
    return embedding, meta, names


def extract_image_stats(image_path: Path) -> Tuple[np.ndarray, Dict[str, object], List[str]]:
    image = Image.open(image_path).convert("RGB")
    feats, names = patch_features(image)
    embedding = np.concatenate([feats, np.zeros_like(feats)], axis=0).astype(np.float32)
    names = [f"mean_{name}" for name in names] + [f"std_{name}" for name in names]
    return embedding, {"input_kind": "image", "image_size": list(image.size)}, names


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    metadata_csv = args.metadata_csv.resolve()
    root_dir = args.root_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    rows = group_rows(load_rows(metadata_csv), args.group_by, args.case_column)
    if args.limit is not None:
        rows = rows[: args.limit]

    skipped_missing = 0
    if args.missing_policy == "skip":
        filtered = []
        for row in rows:
            resolved_path = resolve_input_path(root_dir, row.get(args.path_column, "").strip())
            if resolved_path.exists():
                filtered.append(row)
            else:
                skipped_missing += 1
        rows = filtered

    manifest_path = output_dir / "manifest.jsonl"
    feature_names_path = output_dir / "feature_names.txt"
    results: List[Dict[str, object]] = []
    feature_names: List[str] = []
    start_time = time.time()

    for index, row in enumerate(rows, start=1):
        sample_key = build_sample_key(row, args.group_by, args.case_column)
        raw_relpath = row.get(args.path_column, "").strip()
        resolved_path = resolve_input_path(root_dir, raw_relpath)
        out_npz = features_dir / f"{sample_key.replace(os.sep, '_')}{args.embedding_suffix}"
        status = "ok"
        error = None
        extra_meta: Dict[str, object] = {}
        try:
            if args.skip_existing and out_npz.exists():
                payload = np.load(out_npz, allow_pickle=True)
                embedding = payload["embedding"].astype(np.float32)
                extra_meta = {"input_kind": "existing", "embedding_shape": list(embedding.shape)}
            else:
                if not resolved_path.exists():
                    raise FileNotFoundError(f"Input not found: {resolved_path}")
                if is_wsi_path(resolved_path):
                    try:
                        embedding, extra_meta, feature_names = extract_wsi_stats(
                            resolved_path,
                            num_patches=max(1, args.num_patches),
                            read_size=args.read_size,
                            seed=args.seed,
                        )
                    except openslide.OpenSlideError:
                        if not can_try_image_fallback(resolved_path):
                            raise
                        embedding, extra_meta, feature_names = extract_image_stats(resolved_path)
                        extra_meta["input_kind"] = "image_fallback"
                else:
                    embedding, extra_meta, feature_names = extract_image_stats(resolved_path)
                np.savez_compressed(
                    out_npz,
                    embedding=embedding.astype(np.float32),
                    sample_key=sample_key,
                    source_path=str(resolved_path),
                    metadata_json=json.dumps(row, ensure_ascii=False),
                )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"

        record = {
            "index": index,
            "sample_key": sample_key,
            "source_path": str(resolved_path),
            "output_npz": str(out_npz) if status == "ok" else None,
            "status": status,
            "error": error,
            "num_patches": args.num_patches,
            "read_size": args.read_size,
            "group_metadata": row,
        }
        record.update(extra_meta)
        results.append(record)
        if index % 25 == 0:
            print(f"[image-stats] processed {index}/{len(rows)}", flush=True)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if feature_names:
        feature_names_path.write_text("\n".join(feature_names) + "\n", encoding="utf-8")

    summary = {
        "metadata_csv": str(metadata_csv),
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "embedding_suffix": args.embedding_suffix,
        "group_by": args.group_by,
        "num_requested": int(len(rows)),
        "num_succeeded": int(sum(1 for row in results if row["status"] == "ok")),
        "num_failed": int(sum(1 for row in results if row["status"] != "ok")),
        "num_skipped_missing": int(skipped_missing),
        "num_patches": int(args.num_patches),
        "read_size": int(args.read_size),
        "runtime_seconds": round(time.time() - start_time, 3),
        "manifest_path": str(manifest_path),
        "feature_names_path": str(feature_names_path) if feature_names else "",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
