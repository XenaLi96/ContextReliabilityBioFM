#!/usr/bin/env python3
"""Extract frozen scGPT cell embeddings for selected CELLxGENE cells.

The official scGPT ``embed_data`` helper densifies the full expression matrix.
For CELLxGENE tissue-scale h5ad files, this script keeps the selected
expression matrix sparse and tokenizes each inference batch on demand.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm import tqdm


DEFAULT_SCGPT_SOURCE_DIR = Path(
    os.environ.get("SCGPT_SOURCE_DIR", "external/scGPT")
)
DEFAULT_SCGPT_VENDOR_DIR = Path(
    os.environ.get("SCGPT_VENDOR_DIR", "external/scgpt_vendor")
)


class SimpleVocab:
    """Minimal token vocabulary interface required by scGPT inference."""

    def __init__(self, tokens: List[str], default_token: str | None = None):
        self._itos = list(tokens)
        self._stoi = {token: idx for idx, token in enumerate(self._itos)}
        self._default_index = self._stoi[default_token] if default_token in self._stoi else None

    @classmethod
    def from_json(cls, path: Path, default_token: str | None = "<pad>") -> "SimpleVocab":
        with path.open("r", encoding="utf-8") as handle:
            token_to_idx = json.load(handle)
        size = max(int(idx) for idx in token_to_idx.values()) + 1
        tokens: List[str | None] = [None] * size
        for token, idx in token_to_idx.items():
            tokens[int(idx)] = token
        missing = [idx for idx, token in enumerate(tokens) if token is None]
        if missing:
            raise ValueError(f"{path} has missing vocabulary indices: {missing[:5]}")
        return cls([str(token) for token in tokens], default_token=default_token)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __len__(self) -> int:
        return len(self._itos)

    def __getitem__(self, item: str | int) -> int | str:
        if isinstance(item, int):
            return self._itos[item]
        if item in self._stoi:
            return self._stoi[item]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(f"Token {item!r} is not in the vocabulary.")

    def __call__(self, tokens: List[str]) -> List[int]:
        return [int(self[token]) for token in tokens]

    def append_token(self, token: str) -> int:
        if token in self._stoi:
            return self._stoi[token]
        idx = len(self._itos)
        self._itos.append(token)
        self._stoi[token] = idx
        return idx

    def set_default_index(self, index: int | None) -> None:
        self._default_index = index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--selected-cells-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gene-col", default="feature_name")
    parser.add_argument("--scgpt-source-dir", type=Path, default=DEFAULT_SCGPT_SOURCE_DIR)
    parser.add_argument("--vendor-dir", type=Path, default=DEFAULT_SCGPT_VENDOR_DIR)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--read-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast during inference.")
    parser.add_argument("--no-l2-normalize", action="store_true")
    return parser.parse_args()


def append_import_path(path: Path | None) -> None:
    if path is None:
        return
    if path.exists():
        text = str(path)
        if text not in sys.path:
            sys.path.append(text)


def ensure_scgpt_shim(source_dir: Path) -> str:
    """Register a minimal scgpt package so imports avoid scGPT's heavy __init__."""
    package_dir = source_dir / "scgpt"
    if not package_dir.exists():
        raise FileNotFoundError(package_dir)

    version = "unknown"
    init_file = package_dir / "__init__.py"
    if init_file.exists():
        for line in init_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("__version__"):
                version = line.split("=", 1)[1].strip().strip("\"'")
                break

    logger = logging.getLogger("scGPT")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    root_pkg = types.ModuleType("scgpt")
    root_pkg.__path__ = [str(package_dir)]
    root_pkg.__version__ = version
    root_pkg.logger = logger
    sys.modules["scgpt"] = root_pkg

    for subpackage in ["model", "utils", "tokenizer"]:
        module = types.ModuleType(f"scgpt.{subpackage}")
        module.__path__ = [str(package_dir / subpackage)]
        sys.modules[f"scgpt.{subpackage}"] = module

    return version


def setup_scgpt_imports(args: argparse.Namespace):
    append_import_path(args.vendor_dir)
    scgpt_version = ensure_scgpt_shim(args.scgpt_source_dir)
    from scgpt.data_collator import DataCollator
    from scgpt.model.model import TransformerModel
    from scgpt.utils.util import load_pretrained

    return DataCollator, TransformerModel, load_pretrained, scgpt_version


def load_selected(path: Path, max_cells: int) -> pd.DataFrame:
    selected = pd.read_csv(path)
    if "cell_index" not in selected.columns:
        raise ValueError(f"{path} must contain a cell_index column")
    if "label" not in selected.columns and "cell_type" in selected.columns:
        selected = selected.rename(columns={"cell_type": "label"})
    if max_cells and max_cells > 0:
        selected = selected.head(max_cells).copy()
    selected["cell_index"] = pd.to_numeric(selected["cell_index"], errors="raise").astype(int)
    return selected.reset_index(drop=True)


def load_model_and_vocab(args: argparse.Namespace, device: torch.device):
    DataCollator, TransformerModel, load_pretrained, scgpt_version = setup_scgpt_imports(args)
    vocab_file = args.model_dir / "vocab.json"
    config_file = args.model_dir / "args.json"
    model_file = args.model_dir / "best_model.pt"
    for path in [vocab_file, config_file, model_file]:
        if not path.exists():
            raise FileNotFoundError(path)

    with config_file.open("r", encoding="utf-8") as handle:
        model_configs = json.load(handle)

    pad_token = model_configs.get("pad_token", "<pad>")
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab = SimpleVocab.from_json(vocab_file, default_token=pad_token)
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[pad_token])

    input_emb_style = model_configs.get("input_emb_style", "continuous")
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=int(model_configs["embsize"]),
        nhead=int(model_configs["nheads"]),
        d_hid=int(model_configs["d_hid"]),
        nlayers=int(model_configs["nlayers"]),
        nlayers_cls=int(model_configs.get("n_layers_cls", 3)),
        n_cls=1,
        vocab=vocab,
        dropout=float(model_configs.get("dropout", 0.2)),
        pad_token=pad_token,
        pad_value=int(model_configs.get("pad_value", -2)),
        do_mvc=bool(model_configs.get("MVC", True)),
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        input_emb_style=input_emb_style,
        n_input_bins=int(model_configs.get("n_bins", 51)) if input_emb_style == "category" else None,
        use_fast_transformer=False,
        pre_norm=bool(model_configs.get("pre_norm", False)),
    )

    print(f"[scgpt] loading checkpoint {model_file}", flush=True)
    state = torch.load(model_file, map_location="cpu")
    if isinstance(state, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict) and state and all(str(key).startswith("module.") for key in state.keys()):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    load_pretrained(model, state, verbose=False)
    model.to(device)
    model.eval()
    disable_torch112_transformer_fastpath(model)

    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[pad_token],
        pad_value=int(model_configs.get("pad_value", -2)),
        do_mlm=False,
        do_binning=True,
        max_length=args.max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    return model, vocab, collator, model_configs, scgpt_version


def disable_torch112_transformer_fastpath(model) -> None:
    """Avoid PyTorch 1.12 CUDA TransformerEncoder fused mask bug.

    In torch 1.12, eval-mode TransformerEncoderLayer can route to
    ``torch._transformer_encoder_layer_fwd``. On CUDA that path treats
    ``src_key_padding_mask`` as a transformer mask and rejects the shape. Setting
    encoder layers to train-mode disables the fused path; dropout probabilities
    are set to zero so inference remains deterministic.
    """
    encoder = getattr(model, "transformer_encoder", None)
    layers = getattr(encoder, "layers", [])
    for layer in layers:
        layer.train()
        for dropout_name in ["dropout", "dropout1", "dropout2"]:
            dropout = getattr(layer, dropout_name, None)
            if dropout is not None and hasattr(dropout, "p"):
                dropout.p = 0.0
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "dropout"):
            self_attn.dropout = 0.0


def gene_mapping(
    adata: ad.AnnData,
    vocab,
    gene_col: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if gene_col == "index":
        gene_names = pd.Index(adata.var.index).astype(str).to_numpy()
    else:
        if gene_col not in adata.var.columns:
            raise ValueError(f"{gene_col} not found in adata.var")
        gene_names = adata.var[gene_col].astype(str).to_numpy()

    in_vocab = np.asarray([name in vocab for name in gene_names], dtype=bool)
    vocab_ids = np.full(len(gene_names), -1, dtype=np.int64)
    vocab_ids[in_vocab] = np.asarray(vocab(gene_names[in_vocab].tolist()), dtype=np.int64)

    all_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    for idx, (_, row) in enumerate(adata.var.iterrows()):
        base = {
            "feature_matrix_index": int(idx),
            "feature_id": row.get("feature_id", ""),
            "feature_name": row.get("feature_name", gene_names[idx]),
            "in_vocab": bool(in_vocab[idx]),
            "vocab_id": int(vocab_ids[idx]),
        }
        all_rows.append(base)
        if in_vocab[idx]:
            feature_rows.append({**base, "rank": len(feature_rows) + 1})
    return in_vocab, vocab_ids[in_vocab], pd.DataFrame(all_rows), pd.DataFrame(feature_rows)


def load_expression_subset(
    adata: ad.AnnData,
    row_indices: np.ndarray,
    gene_mask: np.ndarray,
) -> sparse.csr_matrix:
    order = np.argsort(row_indices)
    sorted_rows = row_indices[order]
    x_sorted = sparse.csr_matrix(adata.X[sorted_rows, :])
    x_sorted = x_sorted[:, gene_mask].tocsr().astype(np.float32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return x_sorted[inverse, :].tocsr()


def make_examples(
    x_batch: sparse.csr_matrix,
    gene_ids: np.ndarray,
    cls_id: int,
    pad_value: int,
) -> Tuple[List[Dict[str, torch.Tensor]], np.ndarray]:
    examples: List[Dict[str, torch.Tensor]] = []
    lengths = np.zeros(x_batch.shape[0], dtype=np.int32)
    for local_idx in range(x_batch.shape[0]):
        row = x_batch.getrow(local_idx)
        indices = row.indices
        values = row.data.astype(np.float32, copy=False)
        genes = np.empty(len(indices) + 1, dtype=np.int64)
        expr = np.empty(len(indices) + 1, dtype=np.float32)
        genes[0] = cls_id
        expr[0] = float(pad_value)
        if len(indices):
            genes[1:] = gene_ids[indices]
            expr[1:] = values
        lengths[local_idx] = len(genes)
        examples.append(
            {
                "id": torch.tensor(local_idx, dtype=torch.long),
                "genes": torch.from_numpy(genes),
                "expressions": torch.from_numpy(expr),
            }
        )
    return examples, lengths


def encode_embeddings(
    x: sparse.csr_matrix,
    gene_ids: np.ndarray,
    model,
    vocab,
    collator,
    model_configs: Dict[str, object],
    batch_size: int,
    max_length: int,
    device: torch.device,
    use_amp: bool,
    l2_normalize: bool,
    desc: str = "Embedding cells",
) -> Tuple[np.ndarray, np.ndarray]:
    n_cells = x.shape[0]
    emb_dim = int(model_configs["embsize"])
    embeddings = np.zeros((n_cells, emb_dim), dtype=np.float32)
    token_lengths = np.zeros(n_cells, dtype=np.int32)
    cls_id = int(vocab["<cls>"])
    pad_token_id = int(vocab[model_configs.get("pad_token", "<pad>")])
    pad_value = int(model_configs.get("pad_value", -2))

    with torch.no_grad():
        for start in tqdm(range(0, n_cells, batch_size), desc=desc):
            end = min(start + batch_size, n_cells)
            examples, lengths = make_examples(x[start:end], gene_ids, cls_id, pad_value)
            token_lengths[start:end] = lengths
            data_dict = collator(examples)
            input_gene_ids = data_dict["gene"].to(device)
            input_expr = data_dict["expr"].to(device)
            src_key_padding_mask = input_gene_ids.eq(pad_token_id)
            if device.type == "cuda" and use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    encoded = model._encode(
                        input_gene_ids,
                        input_expr,
                        src_key_padding_mask=src_key_padding_mask,
                    )
            else:
                encoded = model._encode(
                    input_gene_ids,
                    input_expr,
                    src_key_padding_mask=src_key_padding_mask,
                )
            batch_emb = encoded[:, 0, :].detach().cpu().numpy().astype(np.float32)
            embeddings[start:end] = batch_emb

    if l2_normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = (embeddings / norms).astype(np.float32)

    return embeddings, token_lengths


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = load_selected(args.selected_cells_csv, args.max_cells)
    row_indices = selected["cell_index"].to_numpy(dtype=int)
    adata = ad.read_h5ad(args.h5ad, backed="r")
    model, vocab, collator, model_configs, scgpt_version = load_model_and_vocab(args, device)

    gene_mask, gene_ids, gene_vocab_df, feature_df = gene_mapping(adata, vocab, args.gene_col)
    if int(gene_mask.sum()) == 0:
        raise ValueError("No h5ad genes matched the scGPT vocabulary.")
    print(
        f"[scgpt] matched {int(gene_mask.sum())}/{adata.n_vars} genes; "
        f"n_cells={len(selected)} device={device}",
        flush=True,
    )

    embedding_chunks: List[np.ndarray] = []
    token_length_chunks: List[np.ndarray] = []
    t0 = time.time()
    n_chunks = int(np.ceil(len(row_indices) / args.read_chunk_size))
    for chunk_i, read_start in enumerate(range(0, len(row_indices), args.read_chunk_size), start=1):
        read_end = min(read_start + args.read_chunk_size, len(row_indices))
        print(
            f"[scgpt] chunk {chunk_i}/{n_chunks}: cells {read_start}:{read_end}",
            flush=True,
        )
        x_chunk = load_expression_subset(adata, row_indices[read_start:read_end], gene_mask)
        emb_chunk, len_chunk = encode_embeddings(
            x=x_chunk,
            gene_ids=gene_ids,
            model=model,
            vocab=vocab,
            collator=collator,
            model_configs=model_configs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            use_amp=args.amp,
            l2_normalize=not args.no_l2_normalize,
            desc=f"Embedding chunk {chunk_i}/{n_chunks}",
        )
        embedding_chunks.append(emb_chunk)
        token_length_chunks.append(len_chunk)
        done = read_end
        elapsed = max(time.time() - t0, 1e-6)
        rate = done / elapsed
        remaining = (len(row_indices) - done) / rate if rate > 0 else float("nan")
        print(
            f"[scgpt] completed {done}/{len(row_indices)} cells; "
            f"{rate:.2f} cells/s; eta {remaining / 60:.1f} min",
            flush=True,
        )

    embeddings = np.concatenate(embedding_chunks, axis=0)
    token_lengths = np.concatenate(token_length_chunks, axis=0)

    selected.to_csv(args.output_dir / "metadata.csv", index=False)
    gene_vocab_df.to_csv(args.output_dir / "gene_vocab_match.csv", index=False)
    feature_df.to_csv(args.output_dir / "feature_genes.csv", index=False)
    token_df = pd.DataFrame(
        {
            "cell_index": selected["cell_index"].to_numpy(dtype=int),
            "n_nonzero_vocab_genes": token_lengths - 1,
            "sequence_length_before_truncation": token_lengths,
            "truncated_by_max_length": token_lengths > args.max_length,
        }
    )
    token_df.to_csv(args.output_dir / "token_lengths.csv", index=False)
    np.savez_compressed(args.output_dir / "scgpt_embeddings.npz", embeddings=embeddings)

    quantiles = np.quantile(token_lengths, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).tolist()
    summary = {
        "model_name": "scgpt_continual",
        "h5ad": str(args.h5ad),
        "selected_cells_csv": str(args.selected_cells_csv),
        "model_dir": str(args.model_dir),
        "gene_col": args.gene_col,
        "n_cells": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "n_h5ad_genes": int(adata.n_vars),
        "n_vocab_matched_genes": int(gene_mask.sum()),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "read_chunk_size": int(args.read_chunk_size),
        "device": str(device),
        "amp": bool(args.amp),
        "l2_normalized": not args.no_l2_normalize,
        "transformer_fastpath_disabled": True,
        "token_length_quantiles": quantiles,
        "n_truncated_cells": int(np.sum(token_lengths > args.max_length)),
        "seed": int(args.seed),
        "torch_version": torch.__version__,
        "scgpt_version": scgpt_version,
        "model_config_subset": {
            key: model_configs.get(key)
            for key in [
                "input_style",
                "input_emb_style",
                "n_bins",
                "max_seq_len",
                "nlayers",
                "nheads",
                "embsize",
                "d_hid",
                "dropout",
                "pad_value",
                "pad_token",
            ]
        },
        "outputs": {
            "metadata": str(args.output_dir / "metadata.csv"),
            "embeddings": str(args.output_dir / "scgpt_embeddings.npz"),
            "feature_genes": str(args.output_dir / "feature_genes.csv"),
            "gene_vocab_match": str(args.output_dir / "gene_vocab_match.csv"),
            "token_lengths": str(args.output_dir / "token_lengths.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
