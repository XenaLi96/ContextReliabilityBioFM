#!/usr/bin/env python3
"""Run factorized z^y/z^c representation diagnostics for CELLxGENE embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_cellxgene_embedding_audit import clean_string, infer_fold_count, read_embedding, write_csv  # noqa: E402
from train_support_calibrated_interventions import (  # noqa: E402
    conditional_coral_loss,
    conditional_mean_alignment_loss,
    grad_reverse,
    label_context_group_dro,
    support_eligible_mask,
    supported_tensors,
)


METHODS = ["factorized_erm", "factorized_sabca"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--embedding-key", default="embeddings")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Geneformer")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--context-field", required=True)
    parser.add_argument("--methods", nargs="*", default=["factorized_erm", "factorized_sabca"])
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-context", type=float, default=0.5)
    parser.add_argument("--lambda-adv", type=float, default=0.25)
    parser.add_argument("--lambda-label-context-dro", type=float, default=0.5)
    parser.add_argument("--lambda-cond-mmd", type=float, default=0.1)
    parser.add_argument("--lambda-cond-coral", type=float, default=0.02)
    parser.add_argument("--lambda-orth", type=float, default=0.02)
    parser.add_argument("--sabca-min-group-size", type=int, default=20)
    parser.add_argument("--sabca-max-sample-weight", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def normalize_metadata(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    out = df.copy()
    if label_column != "label":
        out = out.rename(columns={label_column: "label"})
    for column in out.columns:
        out[column] = out[column].astype(object).map(clean_string).astype(object)
    out["cell_index"] = pd.to_numeric(df["cell_index"], errors="raise").astype(int)
    return out


def inverse_frequency_weights(keys: Sequence[object]) -> np.ndarray:
    key_arr = np.asarray(keys, dtype=str)
    counts = pd.Series(key_arr).value_counts().to_dict()
    weights = np.asarray([1.0 / float(counts[key]) for key in key_arr], dtype=np.float32)
    return weights / float(weights.mean())


def clip_and_renormalize(weights: np.ndarray, max_weight: float) -> np.ndarray:
    if max_weight > 0:
        weights = np.minimum(weights, float(max_weight))
    return weights / float(weights.mean())


def label_context_weights(y_train_int: np.ndarray, c_train_int: np.ndarray, max_weight: float) -> np.ndarray:
    keys = [f"{int(y)}::{int(c)}" for y, c in zip(y_train_int, c_train_int)]
    return clip_and_renormalize(inverse_frequency_weights(keys), max_weight)


class Branch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorizedHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int, n_classes: int, n_contexts: int, dropout: float):
        super().__init__()
        self.task_branch = Branch(input_dim, hidden_dim, proj_dim, dropout)
        self.context_branch = Branch(input_dim, hidden_dim, proj_dim, dropout)
        self.task_classifier = nn.Linear(proj_dim, n_classes)
        self.task_context_classifier = nn.Linear(proj_dim, max(n_contexts, 1))
        self.context_classifier = nn.Linear(proj_dim, max(n_contexts, 1))

    def forward(self, x: torch.Tensor, grl_strength: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_y = self.task_branch(x)
        z_c = self.context_branch(x)
        task_logits = self.task_classifier(z_y)
        task_context_logits = self.task_context_classifier(grad_reverse(z_y, grl_strength) if grl_strength else z_y.detach())
        context_logits = self.context_classifier(z_c)
        return task_logits, task_context_logits, context_logits, z_y, z_c


def orthogonality_loss(z_y: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
    if z_y.shape[0] < 2:
        return z_y.new_tensor(0.0)
    y_centered = z_y - z_y.mean(dim=0, keepdim=True)
    c_centered = z_c - z_c.mean(dim=0, keepdim=True)
    y_norm = y_centered / y_centered.norm(dim=0, keepdim=True).clamp_min(1e-6)
    c_norm = c_centered / c_centered.norm(dim=0, keepdim=True).clamp_min(1e-6)
    corr = y_norm.T @ c_norm
    return corr.pow(2).mean()


def fit_context_probe(x_train: np.ndarray, c_train: np.ndarray, x_test: np.ndarray, c_test: np.ndarray, seed: int) -> float:
    if len(np.unique(c_train.astype(str))) < 2 or len(np.unique(c_test.astype(str))) < 2:
        return float("nan")
    clf = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-3,
        random_state=seed,
    )
    clf.fit(x_train, c_train.astype(str))
    pred = clf.predict(x_test)
    return float(balanced_accuracy_score(c_test.astype(str), pred))


def supported_test_mask(
    y_train: Sequence[object],
    c_train: Sequence[object],
    y_test: Sequence[object],
    c_test: Sequence[object],
    min_group_size: int,
) -> np.ndarray:
    train = pd.DataFrame({"y": np.asarray(y_train, dtype=str), "c": np.asarray(c_train, dtype=str)})
    counts = train.groupby(["y", "c"]).size()
    supported = {key for key, n in counts.items() if int(n) >= min_group_size}
    return np.asarray([(str(y), str(c)) in supported for y, c in zip(y_test, c_test)], dtype=bool)


def train_factorized(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_train: np.ndarray,
    x_test: np.ndarray,
    method: str,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train.astype(str))
    context_encoder = LabelEncoder()
    c_train_int = context_encoder.fit_transform(c_train.astype(str))
    n_classes = int(len(label_encoder.classes_))
    n_contexts = int(max(1, len(context_encoder.classes_)))
    support_np = support_eligible_mask(y_train_int, c_train_int, args.sabca_min_group_size)
    if method == "factorized_sabca":
        sample_weights = label_context_weights(y_train_int, c_train_int, args.sabca_max_sample_weight)
    else:
        sample_weights = np.ones(len(y_train_int), dtype=np.float32)

    model = FactorizedHead(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        n_classes=n_classes,
        n_contexts=n_contexts,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(reduction="none")
    ctx_ce = nn.CrossEntropyLoss()

    order = rng.permutation(len(x_train))
    dataset = TensorDataset(
        torch.tensor(x_train[order], dtype=torch.float32),
        torch.tensor(y_train_int[order], dtype=torch.long),
        torch.tensor(c_train_int[order], dtype=torch.long),
        torch.tensor(sample_weights[order], dtype=torch.float32),
        torch.tensor(support_np[order], dtype=torch.bool),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    is_sabca = method == "factorized_sabca"
    model.train()
    for _ in range(args.epochs):
        for xb, yb, cb, wb, sb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            cb = cb.to(args.device)
            wb = wb.to(args.device)
            sb = sb.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            adv_strength = args.lambda_adv if is_sabca and n_contexts > 1 else 0.0
            task_logits, task_context_logits, context_logits, z_y, z_c = model(xb, grl_strength=adv_strength)
            sample_losses = ce(task_logits, yb)
            loss = (sample_losses * wb).sum() / wb.sum().clamp_min(1e-8)
            loss = loss + args.lambda_context * ctx_ce(context_logits, cb)

            supported = supported_tensors(sb, sample_losses, yb, cb, task_logits, z_y, z_c)
            if supported is not None:
                reg_losses, reg_y, reg_c, reg_logits, reg_z_y, reg_z_c = supported
            else:
                reg_losses, reg_y, reg_c, reg_logits, reg_z_y, reg_z_c = sample_losses, yb, cb, task_logits, z_y, z_c

            if is_sabca:
                if n_contexts > 1:
                    if supported is not None and int(sb.sum().item()) >= 2:
                        loss = loss + args.lambda_adv * ctx_ce(task_context_logits[sb], cb[sb])
                    else:
                        loss = loss + args.lambda_adv * ctx_ce(task_context_logits, cb)
                loss = loss + args.lambda_label_context_dro * label_context_group_dro(reg_losses, reg_y, reg_c)
                loss = loss + args.lambda_cond_mmd * conditional_mean_alignment_loss(reg_z_y, reg_y, reg_c)
                loss = loss + args.lambda_cond_coral * conditional_coral_loss(reg_z_y, reg_y, reg_c)
                loss = loss + args.lambda_orth * orthogonality_loss(reg_z_y, reg_z_c)
            loss.backward()
            optimizer.step()

    def encode(x_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits_list: List[np.ndarray] = []
        zy_list: List[np.ndarray] = []
        zc_list: List[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(x_arr), args.batch_size * 4):
                xb = torch.tensor(x_arr[start : start + args.batch_size * 4], dtype=torch.float32, device=args.device)
                logits, _, _, z_y, z_c = model(xb)
                logits_list.append(logits.cpu().numpy())
                zy_list.append(z_y.cpu().numpy())
                zc_list.append(z_c.cpu().numpy())
        return np.concatenate(logits_list, axis=0), np.concatenate(zy_list, axis=0), np.concatenate(zc_list, axis=0)

    train_logits, z_y_train, z_c_train = encode(x_train)
    test_logits, z_y_test, z_c_test = encode(x_test)
    pred = label_encoder.inverse_transform(test_logits.argmax(axis=1))
    return pred, z_y_train, z_y_test, z_c_train, z_c_test


def collect_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    unknown = [method for method in args.methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {METHODS}")
    metadata = normalize_metadata(pd.read_csv(args.metadata_csv), args.label_column)
    if args.context_field not in metadata.columns:
        raise ValueError(f"Missing context field: {args.context_field}")
    x = read_embedding(args.embedding_file, args.embedding_key).astype(np.float32)
    if x.shape[0] != len(metadata):
        raise ValueError(f"Embedding row count {x.shape[0]} != metadata rows {len(metadata)}")
    y = metadata["label"].astype(str).to_numpy()
    c = metadata[args.context_field].astype(str).to_numpy()
    groups = metadata["donor_id"].astype(str).to_numpy()
    n_folds = infer_fold_count(metadata, "label", args.n_folds)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx]).astype(np.float32)
        x_test = scaler.transform(x[test_idx]).astype(np.float32)
        support_mask = supported_test_mask(y[train_idx], c[train_idx], y[test_idx], c[test_idx], args.sabca_min_group_size)

        frozen_probe = fit_context_probe(x_train, c[train_idx], x_test, c[test_idx], args.seed + fold)
        frozen_supported_probe = float("nan")
        if int(support_mask.sum()) > 0 and len(np.unique(c[test_idx][support_mask])) >= 2:
            frozen_supported_probe = fit_context_probe(
                x_train,
                c[train_idx],
                x_test[support_mask],
                c[test_idx][support_mask],
                args.seed + fold + 1000,
            )
        rows.append(
            {
                "fold": fold,
                "model_name": args.model_name,
                "context_field": args.context_field,
                "method": "frozen_h",
                "representation": "frozen_h",
                "task_ba": float("nan"),
                "supported_task_ba": float("nan"),
                "context_probe_ba": frozen_probe,
                "supported_context_probe_ba": frozen_supported_probe,
                "delta_context_probe_vs_frozen": float("nan"),
                "delta_supported_context_probe_vs_frozen": float("nan"),
                "supported_fraction": float(support_mask.mean()),
                "n_test": int(len(test_idx)),
                "n_supported_test": int(support_mask.sum()),
            }
        )

        for method in args.methods:
            print(f"[factorized-repdiag] fold={fold} method={method}", flush=True)
            pred, z_y_train, z_y_test, z_c_train, z_c_test = train_factorized(
                x_train,
                y[train_idx],
                c[train_idx],
                x_test,
                method,
                args,
                seed=args.seed + fold * 100 + METHODS.index(method),
            )
            task_ba = float(balanced_accuracy_score(y[test_idx].astype(str), pred.astype(str)))
            supported_task_ba = float("nan")
            if int(support_mask.sum()) > 0:
                supported_task_ba = float(
                    balanced_accuracy_score(y[test_idx][support_mask].astype(str), pred[support_mask].astype(str))
                )
            for representation, train_repr, test_repr in [
                ("z_y", z_y_train, z_y_test),
                ("z_c", z_c_train, z_c_test),
            ]:
                probe = fit_context_probe(train_repr, c[train_idx], test_repr, c[test_idx], args.seed + fold * 37)
                supported_probe = float("nan")
                if int(support_mask.sum()) > 0 and len(np.unique(c[test_idx][support_mask])) >= 2:
                    supported_probe = fit_context_probe(
                        train_repr,
                        c[train_idx],
                        test_repr[support_mask],
                        c[test_idx][support_mask],
                        args.seed + fold * 37 + 1000,
                    )
                rows.append(
                    {
                        "fold": fold,
                        "model_name": args.model_name,
                        "context_field": args.context_field,
                        "method": method,
                        "representation": representation,
                        "task_ba": task_ba if representation == "z_y" else float("nan"),
                        "supported_task_ba": supported_task_ba if representation == "z_y" else float("nan"),
                        "context_probe_ba": probe,
                        "supported_context_probe_ba": supported_probe,
                        "delta_context_probe_vs_frozen": probe - frozen_probe,
                        "delta_supported_context_probe_vs_frozen": supported_probe - frozen_supported_probe,
                        "supported_fraction": float(support_mask.mean()),
                        "n_test": int(len(test_idx)),
                        "n_supported_test": int(support_mask.sum()),
                    }
                )
            for local_i, idx in enumerate(test_idx):
                pred_rows.append(
                    {
                        "fold": fold,
                        "method": method,
                        "cell_index": int(metadata.iloc[idx]["cell_index"]),
                        "donor_id": metadata.iloc[idx]["donor_id"],
                        "true_label": str(y[idx]),
                        "pred_label": str(pred[local_i]),
                        "context_field": args.context_field,
                        "context_value": str(c[idx]),
                        "supported_label_context": bool(support_mask[local_i]),
                    }
                )
    return rows, pred_rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    df = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    for (method, representation), sub in df.groupby(["method", "representation"], sort=False):
        out: Dict[str, object] = {"method": method, "representation": representation, "n_folds": int(len(sub))}
        for metric in [
            "task_ba",
            "supported_task_ba",
            "context_probe_ba",
            "supported_context_probe_ba",
            "delta_context_probe_vs_frozen",
            "delta_supported_context_probe_vs_frozen",
            "supported_fraction",
        ]:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else float("nan")
        summary_rows.append(out)
    return summary_rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, pred_rows = collect_rows(args)
    summary_rows = summarize(rows)
    write_csv(args.output_dir / "fold_factorized_representation_diagnostics.csv", rows)
    write_csv(args.output_dir / "prediction_diagnostics.csv", pred_rows)
    write_csv(args.output_dir / "summary_factorized_representation_diagnostics.csv", summary_rows)
    summary = {
        "model_name": args.model_name,
        "context_field": args.context_field,
        "methods": args.methods,
        "n_rows": len(rows),
        "summary": summary_rows,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
