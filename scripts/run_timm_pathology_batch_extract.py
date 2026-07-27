#!/usr/bin/env python3
"""Batch feature extraction for local timm pathology FMs.

The output schema matches the existing UNI/CONCH extractors: each sample gets a
compressed npz with an ``embedding`` array, so the downstream TCGA audit scripts
can reuse the same loader by changing only the suffix.
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
import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.layers import SwiGLUPacked
from torchvision import transforms

from run_uni_batch_extract import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_sample_key,
    can_try_image_fallback,
    group_rows,
    is_wsi_path,
    load_rows,
    pick_patch_coords,
    resolve_device,
    resolve_input_path,
)


MODEL_DEFAULTS = {
    "virchow2": {
        "checkpoint": Path("checkpoints/virchow2/pytorch_model.bin"),
        "suffix": "_virchow2_embedding.npz",
        "pooling": "cls_patch_mean",
    },
    "hoptimus0": {
        "checkpoint": Path("checkpoints/hoptimus0/pytorch_model.bin"),
        "suffix": "_hoptimus0_embedding.npz",
        "pooling": "cls",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-kind", choices=sorted(MODEL_DEFAULTS), default="virchow2")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--embedding-suffix", default=None)
    parser.add_argument("--pooling", choices=("cls", "cls_patch_mean"), default=None)
    parser.add_argument("--group-by", choices=("slide", "case"), default="slide")
    parser.add_argument("--path-column", default="slide_file_name")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--num-patches", type=int, default=16)
    parser.add_argument("--read-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--missing-policy", choices=("skip", "fail"), default="skip")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_virchow2() -> torch.nn.Module:
    return timm.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        img_size=224,
        num_classes=0,
        init_values=1e-5,
        mlp_ratio=6832 / 1280,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        reg_tokens=4,
        no_embed_class=False,
    )


def build_hoptimus0() -> torch.nn.Module:
    return timm.create_model(
        "vit_giant_patch14_reg4_dinov2",
        pretrained=False,
        img_size=224,
        num_classes=0,
    )


def load_model(model_kind: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    print(f"[1/4] build {model_kind} skeleton", flush=True)
    if model_kind == "virchow2":
        model = build_virchow2()
    elif model_kind == "hoptimus0":
        model = build_hoptimus0()
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    print(f"[2/4] load checkpoint from {checkpoint_path}", flush=True)
    state_dict = load_state_dict(checkpoint_path)
    print("[3/4] load checkpoint into model", flush=True)
    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    print(f"[4/4] model ready on {device}", flush=True)
    return model


def make_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def pool_tokens(model: torch.nn.Module, batch: torch.Tensor, pooling: str) -> torch.Tensor:
    tokens = model.forward_features(batch)
    if isinstance(tokens, dict):
        tokens = tokens.get("x_norm_patchtokens", tokens.get("x", tokens.get("tokens")))
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]
    if tokens.ndim == 2:
        return tokens
    cls_token = tokens[:, 0]
    if pooling == "cls":
        return cls_token
    num_prefix = int(getattr(model, "num_prefix_tokens", 1))
    patch_tokens = tokens[:, num_prefix:]
    patch_mean = patch_tokens.mean(dim=1)
    return torch.cat([cls_token, patch_mean], dim=1)


def embed_patches(
    model: torch.nn.Module,
    device: torch.device,
    patches: List[Image.Image],
    image_size: int,
    batch_size: int,
    pooling: str,
) -> np.ndarray:
    transform = make_transform(image_size)
    tensors = torch.stack([transform(patch.convert("RGB")) for patch in patches], dim=0).to(device)
    outputs: List[np.ndarray] = []
    with torch.inference_mode():
        for start_idx in range(0, len(tensors), batch_size):
            batch = tensors[start_idx : start_idx + batch_size]
            emb = pool_tokens(model, batch, pooling=pooling)
            outputs.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def extract_wsi_embedding(
    model: torch.nn.Module,
    device: torch.device,
    slide_path: Path,
    num_patches: int,
    read_size: int,
    image_size: int,
    batch_size: int,
    seed: int,
    pooling: str,
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
        device=device,
        patches=patches,
        image_size=image_size,
        batch_size=batch_size,
        pooling=pooling,
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
    device: torch.device,
    image_path: Path,
    image_size: int,
    batch_size: int,
    pooling: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    image = Image.open(image_path).convert("RGB")
    embedding = embed_patches(
        model=model,
        device=device,
        patches=[image],
        image_size=image_size,
        batch_size=batch_size,
        pooling=pooling,
    )[0]
    return embedding, {"input_kind": "image", "image_size": list(image.size)}


def main() -> None:
    args = parse_args()
    defaults = MODEL_DEFAULTS[args.model_kind]
    checkpoint_path = (args.checkpoint_path or defaults["checkpoint"]).resolve()
    embedding_suffix = args.embedding_suffix or str(defaults["suffix"])
    pooling = args.pooling or str(defaults["pooling"])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    metadata_csv = args.metadata_csv.resolve()
    root_dir = args.root_dir.resolve()
    output_dir = args.output_dir.resolve()
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
    model = load_model(args.model_kind, checkpoint_path, device)

    rows = group_rows(load_rows(metadata_csv), args.group_by, args.case_column)
    if args.limit is not None:
        rows = rows[: args.limit]

    skipped_missing = 0
    if args.missing_policy == "skip":
        filtered_rows = []
        for row in rows:
            resolved_path = resolve_input_path(root_dir, row.get(args.path_column, "").strip())
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
        out_npz = features_dir / f"{sample_prefix}{embedding_suffix}"
        status = "ok"
        error = None
        extra_meta: Dict[str, object] = {}

        try:
            if args.skip_existing and out_npz.exists():
                extra_meta = {"input_kind": "existing", "skipped_existing": True}
                raise StopIteration
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
                        pooling=pooling,
                    )
                except openslide.OpenSlideError as exc:
                    if not can_try_image_fallback(resolved_path):
                        raise
                    embedding, image_meta = extract_image_embedding(
                        model=model,
                        device=device,
                        image_path=resolved_path,
                        image_size=args.image_size,
                        batch_size=max(1, args.batch_size),
                        pooling=pooling,
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
                    batch_size=max(1, args.batch_size),
                    pooling=pooling,
                )

            np.savez_compressed(
                out_npz,
                embedding=embedding.astype(np.float32),
                sample_key=sample_key,
                source_path=str(resolved_path),
                metadata_json=json.dumps(row, ensure_ascii=False),
                model_kind=args.model_kind,
                pooling=pooling,
            )
        except StopIteration:
            pass
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
            "model_kind": args.model_kind,
            "checkpoint_path": str(checkpoint_path),
            "embedding_suffix": embedding_suffix,
            "pooling": pooling,
            "device": str(device),
            "input_image_size": args.image_size,
            "num_patches": args.num_patches,
            "read_size": args.read_size,
            "group_metadata": row,
        }
        record.update(extra_meta)
        results.append(record)
        print(f"[{index}/{len(rows)}] {sample_key} {status}", flush=True)

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    ok = sum(1 for record in results if record["status"] == "ok")
    summary = {
        "metadata_csv": str(metadata_csv),
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "features_dir": str(features_dir),
        "model_kind": args.model_kind,
        "checkpoint_path": str(checkpoint_path),
        "embedding_suffix": embedding_suffix,
        "pooling": pooling,
        "n_rows": len(results),
        "n_ok": ok,
        "n_failed": len(results) - ok,
        "skipped_missing": skipped_missing,
        "elapsed_sec": time.time() - start_time,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
