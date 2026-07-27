#!/usr/bin/env python3
"""Run post-adaptation representation diagnostics for CELLxGENE embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    METHODS,
    SABCA_FAMILY,
    ProjectorHead,
    conditional_mean_alignment_loss,
    consistency_pref_loss,
    context_group_loss_regularizer,
    group_loss_variance,
    label_context_group_dro,
    method_weights,
    supervised_sample_weights,
    supcon_cross_context_loss,
    support_eligible_mask,
    supported_tensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--embedding-key", default="embeddings")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Geneformer")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--context-field", required=True)
    parser.add_argument("--methods", nargs="*", default=["erm_mlp", "label_context_reweight", "sabca"])
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-fair-var", type=float, default=0.5)
    parser.add_argument("--lambda-adv", type=float, default=0.25)
    parser.add_argument("--lambda-supcon", type=float, default=0.15)
    parser.add_argument("--lambda-consistency", type=float, default=0.15)
    parser.add_argument("--lambda-group-dro", type=float, default=1.0)
    parser.add_argument("--lambda-max-gap", type=float, default=0.5)
    parser.add_argument("--lambda-cond-mmd", type=float, default=0.2)
    parser.add_argument("--sabca-min-group-size", type=int, default=20)
    parser.add_argument("--sabca-max-sample-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.2)
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


def fit_context_probe(
    x_train: np.ndarray,
    c_train: np.ndarray,
    x_test: np.ndarray,
    c_test: np.ndarray,
    seed: int,
) -> Tuple[float, np.ndarray]:
    if len(np.unique(c_train.astype(str))) < 2 or len(np.unique(c_test.astype(str))) < 2:
        return float("nan"), np.asarray([], dtype=str)
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
    return float(balanced_accuracy_score(c_test.astype(str), pred)), pred


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


def train_projector(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_train: np.ndarray,
    x_test: np.ndarray,
    method: str,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder, LabelEncoder]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train.astype(str))
    context_encoder = LabelEncoder()
    c_train_int = context_encoder.fit_transform(c_train.astype(str))
    n_classes = int(len(label_encoder.classes_))
    n_contexts = int(max(1, len(context_encoder.classes_)))
    if method in SABCA_FAMILY and method != "sabca_no_support_gate":
        supported_np = support_eligible_mask(y_train_int, c_train_int, args.sabca_min_group_size)
    else:
        supported_np = np.ones(len(y_train_int), dtype=bool)

    model = ProjectorHead(
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
    weights = method_weights(method, args)
    sample_weights = supervised_sample_weights(method, y_train_int, c_train_int, args)

    order = rng.permutation(len(x_train))
    dataset = TensorDataset(
        torch.tensor(x_train[order], dtype=torch.float32),
        torch.tensor(y_train_int[order], dtype=torch.long),
        torch.tensor(c_train_int[order], dtype=torch.long),
        torch.tensor(sample_weights[order], dtype=torch.float32),
        torch.tensor(supported_np[order], dtype=torch.bool),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model.train()
    for _ in range(args.epochs):
        for xb, yb, cb, wb, sb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            cb = cb.to(args.device)
            wb = wb.to(args.device)
            sb = sb.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, context_logits, z = model(xb, grl_strength=weights["adv"])
            sample_losses = ce(logits, yb)
            loss = (sample_losses * wb).sum() / wb.sum().clamp_min(1e-8)
            supported = supported_tensors(sb, sample_losses, yb, cb, logits, z)
            if method in SABCA_FAMILY and method != "sabca_no_support_gate" and supported is not None:
                reg_losses, reg_y, reg_c, reg_logits, reg_z = supported
            else:
                reg_losses, reg_y, reg_c, reg_logits, reg_z = sample_losses, yb, cb, logits, z
            if weights["fair"]:
                loss = loss + weights["fair"] * group_loss_variance(reg_losses, reg_y, reg_c)
            if weights["adv"] and n_contexts > 1:
                if method in SABCA_FAMILY and method != "sabca_no_support_gate" and supported is not None:
                    loss = loss + weights["adv"] * ctx_ce(context_logits[sb], cb[sb])
                else:
                    loss = loss + weights["adv"] * ctx_ce(context_logits, cb)
            if weights["supcon"]:
                loss = loss + weights["supcon"] * supcon_cross_context_loss(reg_z, reg_y, reg_c, args.temperature)
            if weights["consistency"]:
                loss = loss + weights["consistency"] * consistency_pref_loss(reg_logits, reg_y, reg_c)
            if weights["group_dro"]:
                loss = loss + weights["group_dro"] * context_group_loss_regularizer(reg_losses, reg_c, "max")
            if weights["label_context_dro"]:
                loss = loss + weights["label_context_dro"] * label_context_group_dro(reg_losses, reg_y, reg_c)
            if weights["max_gap"]:
                loss = loss + weights["max_gap"] * context_group_loss_regularizer(reg_losses, reg_c, "gap")
            if weights["cond_mmd"]:
                loss = loss + weights["cond_mmd"] * conditional_mean_alignment_loss(reg_z, reg_y, reg_c)
            loss.backward()
            optimizer.step()

    def encode(x_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logits_list: List[np.ndarray] = []
        z_list: List[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(x_arr), args.batch_size * 4):
                xb = torch.tensor(x_arr[start : start + args.batch_size * 4], dtype=torch.float32, device=args.device)
                logits, _, z = model(xb)
                logits_list.append(logits.cpu().numpy())
                z_list.append(z.cpu().numpy())
        return np.concatenate(logits_list, axis=0), np.concatenate(z_list, axis=0)

    train_logits, z_train = encode(x_train)
    test_logits, z_test = encode(x_test)
    pred = label_encoder.inverse_transform(test_logits.argmax(axis=1))
    return pred, z_train, z_test, label_encoder, context_encoder


def collect_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    unknown = [method for method in args.methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    metadata = normalize_metadata(pd.read_csv(args.metadata_csv), args.label_column)
    if args.context_field not in metadata.columns:
        raise ValueError(f"Missing context field {args.context_field}")
    x = read_embedding(args.embedding_file, args.embedding_key).astype(np.float32)
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
        supported_mask = supported_test_mask(
            y[train_idx],
            c[train_idx],
            y[test_idx],
            c[test_idx],
            args.sabca_min_group_size,
        )
        frozen_probe, _ = fit_context_probe(x_train, c[train_idx], x_test, c[test_idx], args.seed + fold)
        frozen_supported_probe = float("nan")
        if supported_mask.sum() > 0 and len(np.unique(c[test_idx][supported_mask])) >= 2:
            frozen_supported_probe, _ = fit_context_probe(
                x_train,
                c[train_idx],
                x_test[supported_mask],
                c[test_idx][supported_mask],
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
                "context_probe_ba": frozen_probe,
                "supported_context_probe_ba": frozen_supported_probe,
                "supported_fraction": float(supported_mask.mean()),
                "n_test": int(len(test_idx)),
                "n_supported_test": int(supported_mask.sum()),
            }
        )
        for method in args.methods:
            print(f"[repdiag] fold={fold} method={method}", flush=True)
            pred, z_train, z_test, _, _ = train_projector(
                x_train,
                y[train_idx],
                c[train_idx],
                x_test,
                method,
                args,
                seed=args.seed + fold * 100 + METHODS.index(method),
            )
            task_ba = float(balanced_accuracy_score(y[test_idx].astype(str), pred.astype(str)))
            context_probe, _ = fit_context_probe(z_train, c[train_idx], z_test, c[test_idx], args.seed + fold * 17)
            supported_context_probe = float("nan")
            supported_task_ba = float("nan")
            if supported_mask.sum() > 0 and len(np.unique(c[test_idx][supported_mask])) >= 2:
                supported_context_probe, _ = fit_context_probe(
                    z_train,
                    c[train_idx],
                    z_test[supported_mask],
                    c[test_idx][supported_mask],
                    args.seed + fold * 17 + 1000,
                )
                supported_task_ba = float(
                    balanced_accuracy_score(y[test_idx][supported_mask].astype(str), pred[supported_mask].astype(str))
                )
            rows.append(
                {
                    "fold": fold,
                    "model_name": args.model_name,
                    "context_field": args.context_field,
                    "method": method,
                    "representation": "adapter_z",
                    "task_ba": task_ba,
                    "supported_task_ba": supported_task_ba,
                    "context_probe_ba": context_probe,
                    "supported_context_probe_ba": supported_context_probe,
                    "context_probe_delta_vs_frozen": context_probe - frozen_probe,
                    "supported_context_probe_delta_vs_frozen": supported_context_probe - frozen_supported_probe,
                    "supported_fraction": float(supported_mask.mean()),
                    "n_test": int(len(test_idx)),
                    "n_supported_test": int(supported_mask.sum()),
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
                        "supported_label_context": bool(supported_mask[local_i]),
                    }
                )
    return rows, pred_rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    df = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    for (method, representation), sub in df.groupby(["method", "representation"], sort=False):
        out: Dict[str, object] = {
            "method": method,
            "representation": representation,
            "n_folds": int(len(sub)),
        }
        for metric in [
            "task_ba",
            "supported_task_ba",
            "context_probe_ba",
            "supported_context_probe_ba",
            "context_probe_delta_vs_frozen",
            "supported_context_probe_delta_vs_frozen",
            "supported_fraction",
        ]:
            if metric in sub.columns:
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
    write_csv(args.output_dir / "fold_representation_diagnostics.csv", rows)
    write_csv(args.output_dir / "prediction_diagnostics.csv", pred_rows)
    write_csv(args.output_dir / "summary_representation_diagnostics.csv", summary_rows)
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
