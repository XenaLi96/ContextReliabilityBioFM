#!/usr/bin/env python3
"""Batch CONCH feature extraction from a TCGA-style metadata CSV."""

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
import torch
from PIL import Image

from run_uni_batch_extract import (  # noqa: E402
    build_sample_key,
    can_try_image_fallback,
    group_rows,
    is_wsi_path,
    load_rows,
    pick_patch_coords,
    resolve_device,
    resolve_input_path,
)


DEFAULT_CHECKPOINT = Path("checkpoints/conch/pytorch_model.bin")
DEFAULT_CONCH_REPO = Path("external/CONCH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch CONCH feature extraction from metadata CSV entries."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--conch-repo", type=Path, default=DEFAULT_CONCH_REPO)
    parser.add_argument("--group-by", choices=("slide", "case"), default="slide")
    parser.add_argument("--path-column", default="slide_file_name")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--num-patches", type=int, default=16)
    parser.add_argument("--read-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--missing-policy", choices=("skip", "fail"), default="skip")
    return parser.parse_args()


def load_model(checkpoint_path: Path, conch_repo: Path, device: torch.device):
    sys.path.insert(0, str(conch_repo))
    from conch.open_clip_custom import create_model_from_pretrained

    print("[1/3] load CONCH model", flush=True)
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=str(checkpoint_path),
    )
    print("[2/3] move CONCH model", flush=True)
    model.eval().to(device)
    print(f"[3/3] model ready on {device}", flush=True)
    return model, preprocess


def embed_patches(
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    patches: List[Image.Image],
    batch_size: int,
) -> np.ndarray:
    tensors = torch.stack([preprocess(patch.convert("RGB")) for patch in patches], dim=0)
    tensors = tensors.to(device)
    outputs: List[np.ndarray] = []
    with torch.inference_mode():
        for start_idx in range(0, len(tensors), batch_size):
            batch = tensors[start_idx : start_idx + batch_size]
            emb = model.encode_image(
                batch,
                proj_contrast=False,
                normalize=False,
            )
            outputs.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def extract_wsi_embedding(
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    slide_path: Path,
    num_patches: int,
    read_size: int,
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

    patch_embeddings = embed_patches(
        model=model,
        preprocess=preprocess,
        device=device,
        patches=patches,
        batch_size=batch_size,
    )
    sample_embedding = patch_embeddings.mean(axis=0).astype(np.float32)
    meta = {
        "input_kind": "wsi",
        "slide_dimensions": slide_dimensions,
        "num_patches": len(coords),
        "patch_coords": coords,
        "patch_embeddings_shape": list(patch_embeddings.shape),
    }
    return sample_embedding, meta


def extract_image_embedding(
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    image_path: Path,
) -> Tuple[np.ndarray, Dict[str, object]]:
    image = Image.open(image_path).convert("RGB")
    embedding = embed_patches(
        model=model,
        preprocess=preprocess,
        device=device,
        patches=[image],
        batch_size=1,
    )[0]
    return embedding, {"input_kind": "image", "image_size": list(image.size)}


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    metadata_csv = args.metadata_csv.resolve()
    root_dir = args.root_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    conch_repo = args.conch_repo.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root dir not found: {root_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not conch_repo.is_dir():
        raise NotADirectoryError(f"CONCH repo not found: {conch_repo}")

    device = resolve_device(args.device)
    model, preprocess = load_model(checkpoint_path, conch_repo, device)

    rows = load_rows(metadata_csv)
    rows = group_rows(rows, args.group_by, args.case_column)
    if args.limit is not None:
        rows = rows[: args.limit]

    skipped_missing = 0
    if args.missing_policy == "skip":
        filtered_rows = []
        for row in rows:
            raw_relpath = row.get(args.path_column, "").strip()
            resolved_path = resolve_input_path(root_dir, raw_relpath)
            if resolved_path.exists():
                filtered_rows.append(row)
            else:
                skipped_missing += 1
        rows = filtered_rows

    results: List[Dict[str, object]] = []
    start_time = time.time()
    for index, row in enumerate(rows, start=1):
        sample_key = build_sample_key(row, args.group_by, args.case_column)
        raw_relpath = row.get(args.path_column, "").strip()
        resolved_path = resolve_input_path(root_dir, raw_relpath)
        sample_prefix = sample_key.replace(os.sep, "_")
        out_npz = features_dir / f"{sample_prefix}_conch_embedding.npz"
        status = "ok"
        error = None
        extra_meta: Dict[str, object] = {}

        try:
            if not resolved_path.exists():
                raise FileNotFoundError(f"Input not found: {resolved_path}")
            if is_wsi_path(resolved_path):
                try:
                    embedding, extra_meta = extract_wsi_embedding(
                        model=model,
                        preprocess=preprocess,
                        device=device,
                        slide_path=resolved_path,
                        num_patches=max(1, args.num_patches),
                        read_size=args.read_size,
                        batch_size=max(1, args.batch_size),
                        seed=args.seed,
                    )
                except openslide.OpenSlideError as exc:
                    if not can_try_image_fallback(resolved_path):
                        raise
                    embedding, image_meta = extract_image_embedding(
                        model=model,
                        preprocess=preprocess,
                        device=device,
                        image_path=resolved_path,
                    )
                    extra_meta = {
                        "input_kind": "image_fallback",
                        "wsi_fallback_error": f"{type(exc).__name__}: {exc}",
                    }
                    extra_meta.update(image_meta)
            else:
                embedding, extra_meta = extract_image_embedding(
                    model=model,
                    preprocess=preprocess,
                    device=device,
                    image_path=resolved_path,
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
            "num_patches": args.num_patches,
            "read_size": args.read_size,
            "group_metadata": row,
        }
        record.update(extra_meta)
        results.append(record)

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "metadata_csv": str(metadata_csv),
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "features_dir": str(features_dir),
        "checkpoint_path": str(checkpoint_path),
        "conch_repo": str(conch_repo),
        "num_input_rows": len(rows),
        "num_succeeded": sum(record["status"] == "ok" for record in results),
        "num_failed": sum(record["status"] == "failed" for record in results),
        "skipped_missing": skipped_missing,
        "num_patches": args.num_patches,
        "read_size": args.read_size,
        "batch_size": args.batch_size,
        "runtime_seconds": round(time.time() - start_time, 3),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
