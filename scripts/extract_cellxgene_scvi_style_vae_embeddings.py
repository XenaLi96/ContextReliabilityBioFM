#!/usr/bin/env python3
"""Train a lightweight scVI-style VAE and extract CELLxGENE latent embeddings.

This is a local baseline, not the official scvi-tools implementation. It uses a
VAE with a negative-binomial reconstruction objective on HVG counts and exports
the encoder mean as a latent representation compatible with the existing
CELLxGENE embedding audit scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--selected-cells-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-genes-csv", type=Path, default=None)
    parser.add_argument("--n-top-genes", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_expression_subset(adata: ad.AnnData, row_indices: np.ndarray) -> sparse.csr_matrix:
    row_indices = np.asarray(row_indices, dtype=int)
    order = np.argsort(row_indices)
    sorted_rows = row_indices[order]
    x_sorted = sparse.csr_matrix(adata.X[sorted_rows, :])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return x_sorted[inverse, :].tocsr()


def select_variable_genes(x: sparse.csr_matrix, n_top_genes: int) -> np.ndarray:
    means = np.asarray(x.mean(axis=0)).ravel()
    second = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    variances = second - means * means
    expressed = np.asarray((x > 0).sum(axis=0)).ravel()
    variances[expressed < 5] = -np.inf
    n_top = min(int(n_top_genes), int(np.isfinite(variances).sum()))
    if n_top <= 0:
        raise ValueError("No genes passed the minimum expression filter.")
    top = np.argpartition(variances, -n_top)[-n_top:]
    return top[np.argsort(variances[top])[::-1]]


def load_feature_indices(path: Path | None, x_all: sparse.csr_matrix, n_top_genes: int) -> np.ndarray:
    if path is None:
        return select_variable_genes(x_all, n_top_genes)
    df = pd.read_csv(path)
    if "feature_matrix_index" not in df.columns:
        raise ValueError(f"{path} has no feature_matrix_index column")
    indices = pd.to_numeric(df["feature_matrix_index"], errors="raise").astype(int).to_numpy()
    if len(indices) == 0:
        raise ValueError(f"{path} contains no feature rows")
    return indices[:n_top_genes]


def feature_rows(adata: ad.AnnData, feature_idx: np.ndarray) -> List[Dict[str, object]]:
    var = adata.var.iloc[feature_idx].copy()
    rows: List[Dict[str, object]] = []
    for rank, (matrix_idx, (_, row)) in enumerate(zip(feature_idx, var.iterrows()), start=1):
        rows.append(
            {
                "rank": rank,
                "feature_matrix_index": int(matrix_idx),
                "feature_id": row.get("feature_id", ""),
                "feature_name": row.get("feature_name", ""),
            }
        )
    return rows


class ScviStyleVAE(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.z_mean(h), self.z_logvar(h).clamp(-8.0, 8.0)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(0.5 * logvar)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        gene_logits = self.decoder(z)
        return gene_logits, mean, logvar


def nb_negative_log_likelihood(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0.0)
    mu = mu.clamp_min(1e-5)
    theta = theta.clamp(1e-4, 1e4)
    log_likelihood = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + x * (torch.log(mu) - torch.log(theta + mu))
    )
    return -log_likelihood.sum(dim=1).mean()


def train_vae(
    x_input: np.ndarray,
    x_counts: np.ndarray,
    library_size: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[ScviStyleVAE, List[Dict[str, float]]]:
    model = ScviStyleVAE(x_input.shape[1], args.hidden_dim, args.latent_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = TensorDataset(
        torch.tensor(x_input, dtype=torch.float32),
        torch.tensor(x_counts, dtype=torch.float32),
        torch.tensor(library_size, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_seen = 0
        kl_weight = args.beta * min(1.0, epoch / max(1, args.kl_warmup_epochs))
        for xb, counts, lib in loader:
            xb = xb.to(args.device)
            counts = counts.to(args.device)
            lib = lib.to(args.device).view(-1, 1).clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            gene_logits, mean, logvar = model(xb)
            proportions = torch.softmax(gene_logits, dim=1)
            mu = proportions * lib
            theta = torch.exp(model.log_theta).view(1, -1)
            recon = nb_negative_log_likelihood(counts, mu, theta)
            kl = -0.5 * torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
            loss = recon + kl_weight * kl
            loss.backward()
            optimizer.step()

            batch_n = int(xb.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_recon += float(recon.item()) * batch_n
            total_kl += float(kl.item()) * batch_n
            n_seen += batch_n

        row = {
            "epoch": float(epoch),
            "loss": total_loss / n_seen,
            "reconstruction_nll": total_recon / n_seen,
            "kl": total_kl / n_seen,
            "kl_weight": float(kl_weight),
        }
        history.append(row)
        print(
            f"[vae] epoch={epoch} loss={row['loss']:.4f} recon={row['reconstruction_nll']:.4f} kl={row['kl']:.4f}",
            flush=True,
        )
    return model, history


def encode_latent(
    model: ScviStyleVAE,
    x_input: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    latents: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x_input), batch_size):
            xb = torch.tensor(x_input[start : start + batch_size], dtype=torch.float32, device=device)
            mean, _ = model.encode(xb)
            latents.append(mean.cpu().numpy())
    return np.concatenate(latents, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected_cells_csv)
    if args.max_cells and args.max_cells > 0:
        selected = selected.head(args.max_cells).copy()
    if "cell_index" not in selected.columns:
        raise ValueError("selected cells CSV must contain cell_index")

    adata = ad.read_h5ad(args.h5ad, backed="r")
    row_indices = pd.to_numeric(selected["cell_index"], errors="raise").astype(int).to_numpy()
    x_all = load_expression_subset(adata, row_indices)
    feature_idx = load_feature_indices(args.feature_genes_csv, x_all, args.n_top_genes)
    x_hvg = x_all[:, feature_idx].tocsr().astype(np.float32)
    x_counts = np.asarray(x_hvg.toarray(), dtype=np.float32)
    x_counts = np.nan_to_num(x_counts, nan=0.0, posinf=0.0, neginf=0.0)
    x_counts[x_counts < 0] = 0.0
    library_size = x_counts.sum(axis=1).astype(np.float32)
    library_size[library_size <= 0] = 1.0
    x_log_norm = np.log1p((x_counts / library_size[:, None]) * 10000.0).astype(np.float32)
    scaler = StandardScaler()
    x_input = scaler.fit_transform(x_log_norm).astype(np.float32)

    model, history = train_vae(x_input, x_counts, library_size, args)
    embeddings = encode_latent(model, x_input, args.batch_size * 4, args.device)

    selected.to_csv(args.output_dir / "metadata.csv", index=False)
    pd.DataFrame(feature_rows(adata, feature_idx)).to_csv(args.output_dir / "feature_genes.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "train_history.csv", index=False)
    np.savez_compressed(args.output_dir / "scvi_style_vae_embeddings.npz", embeddings=embeddings)
    torch.save(model.state_dict(), args.output_dir / "model_state.pt")

    nonzero_values = x_counts[x_counts > 0]
    integer_like = bool(np.allclose(nonzero_values[: min(len(nonzero_values), 10000)], np.round(nonzero_values[: min(len(nonzero_values), 10000)]))) if len(nonzero_values) else False
    summary = {
        "model_name": "scvi_style_vae",
        "note": "Local scVI-style VAE baseline; not official scvi-tools/scArches.",
        "h5ad": str(args.h5ad),
        "selected_cells_csv": str(args.selected_cells_csv),
        "feature_genes_csv": str(args.feature_genes_csv) if args.feature_genes_csv else None,
        "n_cells": int(embeddings.shape[0]),
        "n_input_genes": int(len(feature_idx)),
        "latent_dim": int(embeddings.shape[1]),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "input_nonzero_integer_like_sample": integer_like,
        "final_loss": history[-1] if history else {},
        "outputs": {
            "metadata": str(args.output_dir / "metadata.csv"),
            "feature_genes": str(args.output_dir / "feature_genes.csv"),
            "train_history": str(args.output_dir / "train_history.csv"),
            "embeddings": str(args.output_dir / "scvi_style_vae_embeddings.npz"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
