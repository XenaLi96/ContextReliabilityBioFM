#!/usr/bin/env python3
"""Extract frozen UNI/CONCH patch embeddings from HEST patch h5 files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
from PIL import Image


DEFAULT_MANIFEST = Path("data/metadata/hest/selected_metadata_manifest.csv")
DEFAULT_RAW_ROOT = Path("data/raw/hest")
DEFAULT_UNI_CHECKPOINT = Path("checkpoints/uni/pytorch_model.bin")
DEFAULT_CONCH_CHECKPOINT = Path("checkpoints/conch/pytorch_model.bin")
DEFAULT_CONCH_REPO = Path("external/CONCH")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model", choices=("uni", "conch"), required=True)
    parser.add_argument("--uni-checkpoint", type=Path, default=DEFAULT_UNI_CHECKPOINT)
    parser.add_argument("--conch-checkpoint", type=Path, default=DEFAULT_CONCH_CHECKPOINT)
    parser.add_argument("--conch-repo", type=Path, default=DEFAULT_CONCH_REPO)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260606)
    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def decode_barcodes(values: np.ndarray) -> np.ndarray:
    decoded = []
    for value in values.reshape(-1):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return np.asarray(decoded, dtype=object)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def normalize_uni_batch(images: np.ndarray, image_size: int) -> torch.Tensor:
    if images.shape[1] != image_size or images.shape[2] != image_size:
        pil_images = [Image.fromarray(img.astype(np.uint8)).resize((image_size, image_size)) for img in images]
        images = np.stack([np.asarray(img, dtype=np.uint8) for img in pil_images], axis=0)
    x = images.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    tensor = torch.from_numpy(x)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def load_uni_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    import timm

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
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    print("[3/4] load checkpoint into model", flush=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    print(f"[4/4] UNI ready on {device}", flush=True)
    return model


def load_conch_model(checkpoint_path: Path, conch_repo: Path, device: torch.device):
    sys.path.insert(0, str(conch_repo))
    from conch.open_clip_custom import create_model_from_pretrained

    print("[1/3] load CONCH model", flush=True)
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=str(checkpoint_path),
    )
    model.eval().to(device)
    print(f"[2/3] CONCH ready on {device}", flush=True)
    return model, preprocess


def iter_batches(dataset, batch_size: int) -> Iterable[Tuple[int, int, np.ndarray]]:
    for start in range(0, dataset.shape[0], batch_size):
        end = min(start + batch_size, dataset.shape[0])
        yield start, end, dataset[start:end]


def embed_uni_sample(
    model: torch.nn.Module,
    images,
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    with torch.inference_mode():
        for _, _, image_batch in iter_batches(images, batch_size):
            tensor = normalize_uni_batch(np.asarray(image_batch), image_size=image_size).to(device)
            emb = model(tensor).detach().cpu().numpy().astype(np.float32)
            chunks.append(emb)
    return np.vstack(chunks).astype(np.float32)


def embed_conch_sample(
    model: torch.nn.Module,
    preprocess,
    images,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    with torch.inference_mode():
        for _, _, image_batch in iter_batches(images, batch_size):
            tensors = torch.stack(
                [preprocess(Image.fromarray(img.astype(np.uint8)).convert("RGB")) for img in np.asarray(image_batch)],
                dim=0,
            ).to(device)
            emb = model.encode_image(tensors, proj_contrast=False, normalize=False)
            chunks.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks).astype(np.float32)


def extract_sample(
    sample_id: str,
    raw_root: Path,
    outdir: Path,
    args: argparse.Namespace,
    model_obj,
    device: torch.device,
) -> Dict[str, object]:
    patch_path = raw_root / "patches" / f"{sample_id}.h5"
    if not patch_path.is_file():
        raise FileNotFoundError(patch_path)

    out_path = outdir / f"{sample_id}.npz"
    if args.skip_existing and out_path.is_file():
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
        if args.model == "uni":
            features = embed_uni_sample(
                model=model_obj,
                images=images,
                batch_size=max(1, args.batch_size),
                image_size=args.image_size,
                device=device,
            )
        else:
            model, preprocess = model_obj
            features = embed_conch_sample(
                model=model,
                preprocess=preprocess,
                images=images,
                batch_size=max(1, args.batch_size),
                device=device,
            )

    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features.astype(np.float32),
        barcodes=barcodes,
        coords=coords,
        sample_id=sample_id,
        model=args.model,
    )
    return {
        "sample_id": sample_id,
        "n_patches": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "output": str(out_path),
        "status": "written",
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    if not args.manifest_csv.is_file():
        raise FileNotFoundError(args.manifest_csv)
    if not args.raw_root.is_dir():
        raise NotADirectoryError(args.raw_root)

    device = resolve_device(args.device)
    if args.model == "uni":
        if not args.uni_checkpoint.is_file():
            raise FileNotFoundError(args.uni_checkpoint)
        model_obj = load_uni_model(args.uni_checkpoint, device)
    else:
        if not args.conch_checkpoint.is_file():
            raise FileNotFoundError(args.conch_checkpoint)
        if not args.conch_repo.is_dir():
            raise NotADirectoryError(args.conch_repo)
        model_obj = load_conch_model(args.conch_checkpoint, args.conch_repo, device)

    rows = read_manifest(args.manifest_csv)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.outdir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    sample_rows = []
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        try:
            info = extract_sample(sample_id, args.raw_root, args.outdir, args, model_obj, device)
        except Exception as exc:  # noqa: BLE001
            info = {
                "sample_id": sample_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        info["index"] = index
        sample_rows.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    summary = {
        "manifest_csv": str(args.manifest_csv),
        "raw_root": str(args.raw_root),
        "outdir": str(args.outdir),
        "model": args.model,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": args.batch_size,
        "n_samples": len(sample_rows),
        "num_succeeded": sum(row["status"] in {"written", "skipped_existing"} for row in sample_rows),
        "num_failed": sum(row["status"] == "failed" for row in sample_rows),
        "total_patches": sum(int(row.get("n_patches", 0)) for row in sample_rows),
        "runtime_seconds": round(time.time() - start_time, 3),
        "samples": sample_rows,
    }
    with (args.outdir / "feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({k: summary[k] for k in ["model", "num_succeeded", "num_failed", "total_patches", "runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
