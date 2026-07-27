#!/usr/bin/env python3

"""Batch UNI feature extraction from a metadata CSV and a root data directory.

This script avoids the broken `get_encoder` path and loads the local UNI
checkpoint directly through `timm`.

It supports two input modes:
- WSI-like slide files via OpenSlide
- ordinary image files via PIL

For each row/group, it extracts a small number of tissue patches and stores a
single representative embedding per sample in a unified output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import openslide
import timm
import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CHECKPOINT = Path("checkpoints/uni/pytorch_model.bin")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
IMAGE_FALLBACK_MAX_BYTES = 512 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch UNI feature extraction from metadata CSV entries."
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="Metadata CSV containing slide/image entries.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Root directory that contains the slide/image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Unified output directory for embeddings and manifests.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to local UNI checkpoint.",
    )
    parser.add_argument(
        "--group-by",
        choices=("slide", "case"),
        default="slide",
        help="Traverse by slide or deduplicate rows by case_id.",
    )
    parser.add_argument(
        "--path-column",
        default="slide_file_name",
        help="CSV column that stores the relative slide/image path.",
    )
    parser.add_argument(
        "--case-column",
        default="case_id",
        help="CSV column used when group-by=case.",
    )
    parser.add_argument(
        "--sample-id-column",
        default=None,
        help="Optional CSV column to use as the sample identifier.",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=1,
        help="Number of tissue patches sampled per slide before pooling.",
    )
    parser.add_argument(
        "--read-size",
        type=int,
        default=512,
        help="Patch read size from level 0 for WSI inputs.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Model input size after resizing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Patch batch size for a single sample.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Preferred device. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows/groups to process.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=("skip", "fail"),
        default="skip",
        help="How to handle metadata rows whose files are absent under root-dir.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    print("[1/4] build UNI model skeleton", flush=True)
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )

    print(f"[2/4] load checkpoint from {checkpoint_path}", flush=True)
    load_kwargs = {"map_location": "cpu"}
    try:
        state_dict = torch.load(
            checkpoint_path,
            mmap=True,
            weights_only=False,
            **load_kwargs,
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, **load_kwargs)

    print("[3/4] load checkpoint into model", flush=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    print(f"[4/4] model ready on {device}", flush=True)
    return model


def pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def normalize_relpath(value: str) -> Path:
    value = value.strip()
    if not value:
        return Path(value)
    return Path(value)


def resolve_input_path(root_dir: Path, relative_path: str) -> Path:
    candidate = root_dir / normalize_relpath(relative_path)
    if candidate.exists():
        return candidate
    basename = Path(relative_path).name
    candidate = root_dir / basename
    if candidate.exists():
        return candidate
    matches = list(root_dir.rglob(basename))
    if matches:
        return matches[0]
    return root_dir / relative_path


def load_rows(metadata_csv: Path) -> List[Dict[str, str]]:
    with metadata_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in metadata CSV: {metadata_csv}")
    return rows


def group_rows(
    rows: Sequence[Dict[str, str]],
    group_by: str,
    case_column: str,
) -> List[Dict[str, str]]:
    if group_by == "slide":
        return list(rows)

    grouped: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
    for row in rows:
        key = row.get(case_column, "").strip()
        if not key:
            continue
        if key not in grouped:
            grouped[key] = row
    return list(grouped.values())


def is_wsi_path(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix not in IMAGE_SUFFIXES


def get_thumbnail_and_mask(slide: openslide.OpenSlide, max_side: int = 2048):
    width, height = slide.dimensions
    scale = max(width, height) / max_side
    if scale < 1:
        scale = 1
    thumb_size = (max(1, int(width / scale)), max(1, int(height / scale)))
    thumbnail = slide.get_thumbnail(thumb_size).convert("RGB")
    arr = np.asarray(thumbnail)
    gray = arr.mean(axis=2)
    color_std = arr.std(axis=2)
    mask = (gray < 225) & (color_std > 8)
    return thumbnail, mask


def is_tissue_patch(pil_image: Image.Image) -> bool:
    rgb = np.asarray(pil_image.convert("RGB"))
    gray = rgb.mean(axis=2)
    return bool((gray < 235).mean() > 0.2 and rgb.std() > 10)


def pick_patch_coords(
    slide: openslide.OpenSlide,
    num_patches: int,
    read_size: int,
    seed: int,
):
    thumbnail, mask = get_thumbnail_and_mask(slide)
    thumb_w, thumb_h = thumbnail.size
    width, height = slide.dimensions
    scale_x = width / thumb_w
    scale_y = height / thumb_h

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("No tissue region detected from thumbnail mask.")

    rng = np.random.default_rng(seed)
    candidates = rng.permutation(len(xs))
    coords = []
    half = read_size // 2

    for idx in candidates:
        x = int(xs[idx] * scale_x)
        y = int(ys[idx] * scale_y)
        x = min(max(0, x - half), max(0, width - read_size))
        y = min(max(0, y - half), max(0, height - read_size))
        patch = slide.read_region((x, y), 0, (read_size, read_size)).convert("RGB")
        if not is_tissue_patch(patch):
            continue
        coords.append((x, y))
        if len(coords) >= num_patches:
            break

    if not coords:
        raise RuntimeError("Failed to sample any tissue patches from the slide.")
    return coords


def extract_wsi_embedding(
    model: torch.nn.Module,
    device: torch.device,
    slide_path: Path,
    num_patches: int,
    read_size: int,
    image_size: int,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        coords = pick_patch_coords(
            slide=slide,
            num_patches=num_patches,
            read_size=read_size,
            seed=seed,
        )
        patches = [
            slide.read_region((x, y), 0, (read_size, read_size)).convert("RGB")
            for x, y in coords
        ]
        slide_dimensions = list(slide.dimensions)
    finally:
        slide.close()

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    batch_tensors = torch.stack([transform(patch) for patch in patches], dim=0).to(
        device
    )

    patch_embeddings: List[np.ndarray] = []
    with torch.inference_mode():
        for start_idx in range(0, len(batch_tensors), batch_size):
            batch = batch_tensors[start_idx : start_idx + batch_size]
            emb = model(batch).detach().cpu().numpy().astype(np.float32)
            patch_embeddings.append(emb)
    patch_embeddings_arr = np.concatenate(patch_embeddings, axis=0)
    sample_embedding = patch_embeddings_arr.mean(axis=0, keepdims=False).astype(
        np.float32
    )

    meta = {
        "input_kind": "wsi",
        "slide_dimensions": slide_dimensions,
        "num_patches": len(coords),
        "patch_coords": coords,
        "patch_embeddings_shape": list(patch_embeddings_arr.shape),
    }
    return sample_embedding, meta


def extract_image_embedding(
    model: torch.nn.Module,
    device: torch.device,
    image_path: Path,
    image_size: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    image = Image.open(image_path).convert("RGB")
    tensor = pil_to_tensor(image, image_size=image_size).unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = model(tensor).detach().cpu().numpy().astype(np.float32)[0]
    return emb, {"input_kind": "image", "image_size": list(image.size)}


def can_try_image_fallback(path: Path) -> bool:
    try:
        return path.stat().st_size <= IMAGE_FALLBACK_MAX_BYTES
    except FileNotFoundError:
        return False


def build_sample_key(row: Dict[str, str], group_by: str, case_column: str) -> str:
    if group_by == "case":
        value = row.get(case_column, "").strip()
        if value:
            return value
    value = row.get("slide_file_name", "").strip()
    if value:
        return Path(value).stem
    value = row.get("slide_file_id", "").strip()
    if value:
        return value
    return f"sample_{abs(hash(tuple(sorted(row.items()))))}"


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    metadata_csv = args.metadata_csv.resolve()
    root_dir = args.root_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root dir not found: {root_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    model = load_model(checkpoint_path, device)

    rows = load_rows(metadata_csv)
    rows = group_rows(rows, args.group_by, args.case_column)
    if args.limit is not None:
        rows = rows[: args.limit]

    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    results: List[Dict[str, object]] = []
    start_time = time.time()
    skipped_missing = 0

    if args.missing_policy == "skip":
        filtered_rows: List[Dict[str, str]] = []
        for row in rows:
            raw_relpath = row.get(args.path_column, "").strip()
            resolved_path = resolve_input_path(root_dir, raw_relpath)
            if resolved_path.exists():
                filtered_rows.append(row)
            else:
                skipped_missing += 1
        rows = filtered_rows

    for index, row in enumerate(rows, start=1):
        sample_key = build_sample_key(row, args.group_by, args.case_column)
        raw_relpath = row.get(args.path_column, "").strip()
        resolved_path = resolve_input_path(root_dir, raw_relpath)
        sample_prefix = sample_key.replace(os.sep, "_")
        out_npz = features_dir / f"{sample_prefix}_uni_embedding.npz"
        status = "ok"
        error = None
        embedding = None
        extra_meta: Dict[str, object] = {}

        try:
            if not resolved_path.exists():
                raise FileNotFoundError(f"Input not found: {resolved_path}")

            if is_wsi_path(resolved_path):
                try:
                    embedding, extra_meta = extract_wsi_embedding(
                        model=model,
                        device=device,
                        slide_path=resolved_path,
                        num_patches=max(1, args.num_patches),
                        read_size=args.read_size,
                        image_size=args.image_size,
                        batch_size=max(1, args.batch_size),
                        seed=args.seed,
                    )
                except openslide.OpenSlideError as exc:
                    if not can_try_image_fallback(resolved_path):
                        raise
                    embedding, image_meta = extract_image_embedding(
                        model=model,
                        device=device,
                        image_path=resolved_path,
                        image_size=args.image_size,
                    )
                    extra_meta = {
                        "input_kind": "image_fallback",
                        "wsi_fallback_error": f"{type(exc).__name__}: {exc}",
                    }
                    extra_meta.update(image_meta)
            else:
                embedding, extra_meta = extract_image_embedding(
                    model=model,
                    device=device,
                    image_path=resolved_path,
                    image_size=args.image_size,
                )

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
            "group_by": args.group_by,
            "source_path": str(resolved_path),
            "output_npz": str(out_npz) if status == "ok" else None,
            "status": status,
            "error": error,
            "checkpoint_path": str(checkpoint_path),
            "device": str(device),
            "input_image_size": args.image_size,
            "num_patches": args.num_patches,
            "read_size": args.read_size,
            "group_metadata": row,
        }
        record.update(extra_meta)
        results.append(record)

    with manifest_path.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "metadata_csv": str(metadata_csv),
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "group_by": args.group_by,
        "num_requested": len(rows),
        "num_succeeded": sum(1 for r in results if r["status"] == "ok"),
        "num_failed": sum(1 for r in results if r["status"] != "ok"),
        "num_skipped_missing": skipped_missing,
        "device": str(device),
        "input_image_size": args.image_size,
        "num_patches": args.num_patches,
        "read_size": args.read_size,
        "runtime_seconds": round(time.time() - start_time, 3),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "manifest_path": str(manifest_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
