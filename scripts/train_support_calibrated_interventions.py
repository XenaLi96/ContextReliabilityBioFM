#!/usr/bin/env python3
"""Train support-calibrated interventions on frozen CELLxGENE embeddings.

The four manuscript-facing methods are:

- erm_mlp: supervised MLP-head baseline.
- label_context_reweight: LC-Reweight.
- sca_lite: SCA-Align, which adds support-gated conditional alignment.
- group_dro: external worst-group comparator.

The remaining identifiers reproduce sensitivity analyses and appendix
diagnostics. They are not additional proposed methods:

- fair_var: FairLoRA-style per label-context loss variance regularization.
- fair_var: FairLoRA-style per label-context loss variance regularization.
- adv_context: domain-adversarial/context-adversarial projector via GRL.
- supcon_cross_context: supervised contrastive loss using same-label,
  different-context positives.
- consistency_pref: preference-like logit consistency across same-label,
  different-context pairs.
- hybrid: a conservative combination of the above.
- max_gap: differentiable max-min context loss gap regularization.
- cond_mmd: class-conditional context mean alignment in projector space.
- cond_coral: class-conditional context covariance alignment in projector space.
- irm: invariant-risk-minimization penalty across context groups.
- fishr: Fishr-style gradient-variance matching across context groups.
- harmony_style: deterministic embedding-level context-centering correction
  followed by the same ERM MLP head.
- context_reweight: inverse-frequency context reweighting for the supervised
  loss.
- label_context_reweight: inverse-frequency label-context reweighting for the
  supervised loss.
- lc_reweight_pow05/lc_reweight_pow075/lc_reweight_pow085/
  lc_reweight_pow09/lc_reweight_pow095/lc_reweight_pow125: conservative
  power-scaled label-context inverse-frequency reweighting.
- lc_reweight_clip2/lc_reweight_clip3: clipped label-context inverse-frequency
  reweighting.
- lc_label_balanced and lc_label_balanced_pow05/pow075: per-label context
  balancing with weights proportional to (N_y / n_{y,c})^alpha.
- lc_donor_reweight/lc_donor_pow075: donor-count label-context inverse-
  frequency reweighting.
- stdr_pow085/stdr_pow09/stdr_pow095: label-conditional support-tempered
  context-distribution reweighting. These preserve each label's total training
  weight while interpolating between empirical and uniform contexts.
- sabca: support-aware bio-context adapter objective. It combines
  label-context balanced ERM with worst label-context risk, context-adversarial
  suppression, class-conditional alignment, and cross-context consistency only
  on label-context regions with enough support.
- sabca_no_support_gate: SABCA without the support-eligibility gate.
- sabca_no_context_alignment: SABCA without adversarial/alignment/consistency
  losses.
- sabca_no_label_context_balancing: SABCA without label-context balanced
  sample weights or worst label-context risk.
- scea: support-calibrated episodic adapter. It combines label-context
  balanced ERM, context episodic/CVaR risk, worst label-context risk,
  support-gated conditional alignment, cross-context supervised contrastive
  learning, and Fishr-style context risk stabilization.
- scea_no_episode: SCEA without context episodic/CVaR risk.
- scea_no_support_gate: SCEA without the support-eligibility gate.
- scea_no_alignment: SCEA without conditional alignment/contrastive/stability
  terms.
- scea_no_cvar: SCEA without context episodic/CVaR or worst label-context risk.
- sca_lite/sca_mmd/sca_coral/sca_supcon: lightweight support-calibrated
  module variants that keep label-context balanced ERM and test individual
  alignment/contrastive components without episodic/CVaR risk.
- sca_soft_dro/sca_soft_cvar: conservative robust-risk variants with weaker
  label-context DRO or weaker CVaR than full SCEA.
- sca_multi_lite/sca_multi_soft_dro: multi-context variants that apply the
  support-gated regularizers over auxiliary protected/clinical context fields
  such as age, sex, and disease in addition to the primary benchmark context.
- reweight_plus: conservative validation-gated Reweight backbone. It trains a
  small set of reweight-based support-calibrated adapters on an inner training
  split, adopts an adapter only when it beats pure label-context reweighting on
  held-in validation worst-context score, and otherwise falls back to Reweight.
- sacro_risk: support-adaptive context risk optimization prototype. It keeps
  the frozen-embedding head simple and replaces fixed inverse-frequency
  reweighting with donor-aware soft support, dynamic label-context risk
  multipliers, and label-conditional local CVaR.

It evaluates patient-level donor CV and leave-one-context-out splits.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_cellxgene_embedding_audit import (  # noqa: E402
    DEFAULT_CONTEXT_FIELDS,
    clean_string,
    infer_fold_count,
    label_context_audit,
    metric_row,
    read_embedding,
    summarize_method_predictions,
    summarize_metric_gaps,
    write_csv,
)


METHODS = [
    "erm_mlp",
    "fair_var",
    "adv_context",
    "supcon_cross_context",
    "consistency_pref",
    "hybrid",
    "group_dro",
    "label_context_dro",
    "max_gap",
    "cond_mmd",
    "cond_coral",
    "irm",
    "fishr",
    "harmony_style",
    "context_reweight",
    "label_context_reweight",
    "lc_reweight_pow05",
    "lc_reweight_pow075",
    "lc_reweight_pow085",
    "lc_reweight_pow09",
    "lc_reweight_pow095",
    "lc_reweight_pow125",
    "lc_reweight_clip2",
    "lc_reweight_clip3",
    "lc_label_balanced",
    "lc_label_balanced_pow05",
    "lc_label_balanced_pow075",
    "lc_donor_reweight",
    "lc_donor_pow075",
    "stdr_pow085",
    "stdr_pow09",
    "stdr_pow095",
    "sabca",
    "sabca_no_support_gate",
    "sabca_no_context_alignment",
    "sabca_no_label_context_balancing",
    "scea",
    "scea_no_episode",
    "scea_no_support_gate",
    "scea_no_alignment",
    "scea_no_cvar",
    "sca_lite",
    "sca_mmd",
    "sca_coral",
    "sca_supcon",
    "sca_soft_dro",
    "sca_soft_cvar",
    "sca_multi_lite",
    "sca_multi_soft_dro",
    "reweight_plus",
    "sca_align_tuned",
    "group_dro_tuned",
    "sacro_risk",
]

SABCA_FAMILY = {
    "sabca",
    "sabca_no_support_gate",
    "sabca_no_context_alignment",
    "sabca_no_label_context_balancing",
}

SCEA_FAMILY = {
    "scea",
    "scea_no_episode",
    "scea_no_support_gate",
    "scea_no_alignment",
    "scea_no_cvar",
}

SCA_SEARCH_FAMILY = {
    "sca_lite",
    "sca_mmd",
    "sca_coral",
    "sca_supcon",
    "sca_soft_dro",
    "sca_soft_cvar",
    "sca_multi_lite",
    "sca_multi_soft_dro",
}

MULTI_CONTEXT_METHODS = {
    "sca_multi_lite",
    "sca_multi_soft_dro",
}

REWEIGHT_PLUS_CANDIDATES = [
    "label_context_reweight",
    "sca_lite",
    "sca_mmd",
    "sca_soft_cvar",
]

SUPPORT_AWARE_METHODS = SABCA_FAMILY | SCEA_FAMILY | SCA_SEARCH_FAMILY
NO_SUPPORT_GATE_METHODS = {"sabca_no_support_gate", "scea_no_support_gate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--embedding-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="embedding")
    parser.add_argument("--embedding-key", default="embeddings")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--context-field", required=True)
    parser.add_argument("--context-fields", nargs="*", default=[*DEFAULT_CONTEXT_FIELDS, "disease"])
    parser.add_argument("--aux-regularizer-context-fields", nargs="*", default=[])
    parser.add_argument("--leave-one-context-fields", nargs="*", default=[])
    parser.add_argument("--methods", nargs="*", default=["erm_mlp", "fair_var", "adv_context", "supcon_cross_context", "consistency_pref", "hybrid"])
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
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
    parser.add_argument("--lambda-cond-coral", type=float, default=0.05)
    parser.add_argument("--lambda-irm", type=float, default=0.5)
    parser.add_argument("--lambda-fishr", type=float, default=0.1)
    parser.add_argument("--lambda-scea-episode", type=float, default=0.75)
    parser.add_argument("--scea-cvar-fraction", type=float, default=0.3)
    parser.add_argument("--sabca-min-group-size", type=int, default=20)
    parser.add_argument(
        "--support-min-donors",
        type=int,
        default=0,
        help="Minimum distinct donors per label-context cell for support-gated adaptation; 0 reproduces sample-only legacy runs.",
    )
    parser.add_argument("--sabca-max-sample-weight", type=float, default=5.0)
    parser.add_argument("--sacro-alpha", type=float, default=1.0)
    parser.add_argument("--sacro-beta", type=float, default=0.75)
    parser.add_argument("--sacro-kappa", type=float, default=10.0)
    parser.add_argument("--sacro-confidence-floor", type=float, default=0.25)
    parser.add_argument("--sacro-risk-ema", type=float, default=0.2)
    parser.add_argument("--sacro-risk-clamp", type=float, default=1.5)
    parser.add_argument("--sacro-min-sample-weight", type=float, default=0.05)
    parser.add_argument("--sacro-max-sample-weight", type=float, default=5.0)
    parser.add_argument("--lambda-sacro-cvar", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--min-holdout-cells", type=int, default=200)
    parser.add_argument("--min-holdout-labels", type=int, default=2)
    parser.add_argument("--write-training-diagnostics", action="store_true")
    parser.add_argument("--diagnostic-epoch-interval", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalize_metadata(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    out = df.copy()
    if label_column != "label":
        out = out.rename(columns={label_column: "label"})
    required = ["cell_index", "donor_id", "label"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    for column in out.columns:
        out[column] = out[column].astype(object).map(clean_string).astype(object)
    out["cell_index"] = pd.to_numeric(df["cell_index"], errors="raise").astype(int)
    return out


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.strength * grad_output, None


def grad_reverse(x: torch.Tensor, strength: float) -> torch.Tensor:
    return GradientReverseFn.apply(x, strength)


class ProjectorHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int, n_classes: int, n_contexts: int, dropout: float):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(proj_dim, n_classes)
        self.context_classifier = nn.Linear(proj_dim, max(n_contexts, 1))

    def forward(self, x: torch.Tensor, grl_strength: float = 0.0):
        z = self.projector(x)
        logits = self.classifier(z)
        context_logits = self.context_classifier(grad_reverse(z, grl_strength) if grl_strength else z.detach())
        return logits, context_logits, z


def group_loss_variance(losses: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    group_ids = y * (int(c.max().item()) + 1) + c
    unique = torch.unique(group_ids)
    if unique.numel() < 2:
        return losses.new_tensor(0.0)
    group_losses = torch.stack([losses[group_ids == group_id].mean() for group_id in unique])
    return group_losses.var(unbiased=False)


def grouped_losses(losses: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    unique = torch.unique(group_ids)
    if unique.numel() < 2:
        return losses.new_zeros((0,))
    return torch.stack([losses[group_ids == group_id].mean() for group_id in unique])


def context_group_loss_regularizer(losses: torch.Tensor, c: torch.Tensor, mode: str) -> torch.Tensor:
    group_losses = grouped_losses(losses, c)
    if group_losses.numel() < 2:
        return losses.new_tensor(0.0)
    if mode == "max":
        return group_losses.max()
    if mode == "gap":
        return group_losses.max() - group_losses.min()
    raise ValueError(f"Unknown group regularizer mode: {mode}")


def context_cvar_loss(losses: torch.Tensor, c: torch.Tensor, fraction: float) -> torch.Tensor:
    """Average the highest-loss context groups in the current batch."""
    group_losses = grouped_losses(losses, c)
    if group_losses.numel() < 2:
        return losses.new_tensor(0.0)
    tail_fraction = min(max(float(fraction), 1e-6), 1.0)
    top_k = max(1, int(math.ceil(tail_fraction * int(group_losses.numel()))))
    return torch.topk(group_losses, k=top_k, largest=True).values.mean()


def label_context_group_dro(losses: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    group_ids = y * (int(c.max().item()) + 1) + c
    group_losses = grouped_losses(losses, group_ids)
    if group_losses.numel() < 2:
        return losses.new_tensor(0.0)
    return group_losses.max()


def sacro_label_conditional_cvar_loss(
    losses: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    confidence_by_group: torch.Tensor,
    n_contexts: int,
    fraction: float,
) -> torch.Tensor:
    """Local robust risk over contexts within each label, calibrated by q_{y,c}."""
    label_losses: List[torch.Tensor] = []
    tail_fraction = min(max(float(fraction), 1e-6), 1.0)
    group_ids = y * int(n_contexts) + c
    for label in torch.unique(y):
        label_mask = y == label
        contexts = torch.unique(c[label_mask])
        group_risks: List[torch.Tensor] = []
        for context in contexts:
            group = int(label.item()) * int(n_contexts) + int(context.item())
            mask = group_ids == group
            if int(mask.sum().item()) < 2:
                continue
            confidence = confidence_by_group[group].clamp_min(0.0)
            if float(confidence.item()) <= 0.0:
                continue
            group_risks.append(confidence * losses[mask].mean())
        if len(group_risks) < 2:
            continue
        stacked = torch.stack(group_risks)
        top_k = max(1, int(math.ceil(tail_fraction * int(stacked.numel()))))
        label_losses.append(torch.topk(stacked, k=top_k, largest=True).values.mean())
    if not label_losses:
        return losses.new_tensor(0.0)
    return torch.stack(label_losses).mean()


def conditional_mean_alignment_loss(z: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for label in torch.unique(y):
        label_mask = y == label
        contexts = torch.unique(c[label_mask])
        if contexts.numel() < 2:
            continue
        context_means = []
        for context in contexts:
            mask = label_mask & (c == context)
            if int(mask.sum().item()) >= 2:
                context_means.append(z[mask].mean(dim=0))
        if len(context_means) < 2:
            continue
        means = torch.stack(context_means, dim=0)
        label_mean = means.mean(dim=0, keepdim=True)
        losses.append(((means - label_mean) ** 2).sum(dim=1).mean())
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()


def covariance_matrix(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_zeros((z.shape[1], z.shape[1]))
    centered = z - z.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(z.shape[0] - 1)


def conditional_coral_loss(z: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for label in torch.unique(y):
        label_mask = y == label
        contexts = torch.unique(c[label_mask])
        if contexts.numel() < 2:
            continue
        covariances = []
        for context in contexts:
            mask = label_mask & (c == context)
            if int(mask.sum().item()) >= 3:
                covariances.append(covariance_matrix(z[mask]))
        if len(covariances) < 2:
            continue
        stacked = torch.stack(covariances, dim=0)
        target = stacked.mean(dim=0, keepdim=True).detach()
        losses.append(((stacked - target) ** 2).mean())
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()


def irm_penalty(logits: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    penalties: List[torch.Tensor] = []
    for context in torch.unique(c):
        mask = c == context
        if int(mask.sum().item()) < 2:
            continue
        scale = torch.ones((), device=logits.device, requires_grad=True)
        loss = nn.functional.cross_entropy(logits[mask] * scale, y[mask])
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        penalties.append(grad.pow(2))
    if not penalties:
        return logits.new_tensor(0.0)
    return torch.stack(penalties).mean()


def fishr_logit_penalty(logits: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    if logits.shape[0] < 3:
        return logits.new_tensor(0.0)
    probs = nn.functional.softmax(logits, dim=1)
    one_hot = nn.functional.one_hot(y, num_classes=logits.shape[1]).to(dtype=logits.dtype)
    logit_grads = probs - one_hot
    if logit_grads.shape[0] < 2:
        return logits.new_tensor(0.0)
    target_var = logit_grads.var(dim=0, unbiased=False).detach()
    penalties: List[torch.Tensor] = []
    for context in torch.unique(c):
        mask = c == context
        if int(mask.sum().item()) < 3:
            continue
        group_var = logit_grads[mask].var(dim=0, unbiased=False)
        penalties.append(((group_var - target_var) ** 2).mean())
    if not penalties:
        return logits.new_tensor(0.0)
    return torch.stack(penalties).mean()


def supcon_cross_context_loss(z: torch.Tensor, y: torch.Tensor, c: torch.Tensor, temperature: float) -> torch.Tensor:
    if z.shape[0] < 3:
        return z.new_tensor(0.0)
    z = nn.functional.normalize(z, dim=1)
    logits = z @ z.T / temperature
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    positives = (y[:, None] == y[None, :]) & (c[:, None] != c[None, :]) & ~eye
    valid = positives.sum(dim=1) > 0
    if not torch.any(valid):
        return z.new_tensor(0.0)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~eye).to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    pos_log_prob = (log_prob * positives.to(logits.dtype)).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    return -pos_log_prob[valid].mean()


def consistency_pref_loss(class_logits: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    probs = nn.functional.softmax(class_logits, dim=1)
    losses: List[torch.Tensor] = []
    for label in torch.unique(y):
        idx = torch.where(y == label)[0]
        if idx.numel() < 2:
            continue
        contexts = c[idx]
        pairs: List[Tuple[int, int]] = []
        for i in idx.tolist():
            candidates = idx[contexts != c[i]]
            if candidates.numel() > 0:
                pairs.append((i, int(candidates[0].item())))
        if not pairs:
            continue
        left = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=class_logits.device)
        right = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=class_logits.device)
        losses.append(nn.functional.mse_loss(probs[left], probs[right]))
    if not losses:
        return class_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def support_eligible_mask(
    y_int: np.ndarray,
    c_int: np.ndarray,
    min_group_size: int,
    donor_ids: Sequence[object] | None = None,
    min_donors: int = 0,
) -> np.ndarray:
    """Return cells whose label is supported in at least two contexts.

    A label-context cell is supported only when it satisfies both the sample
    threshold and, when requested, the independent-donor threshold.  Keeping
    ``min_donors=0`` preserves the historical sample-only protocol for old
    artifacts; formal support-calibrated runs set it explicitly.
    """
    y_arr = np.asarray(y_int, dtype=int)
    c_arr = np.asarray(c_int, dtype=int)
    donors = (
        np.asarray(donor_ids, dtype=str)
        if donor_ids is not None
        else np.asarray([f"row_{index}" for index in range(len(y_arr))], dtype=str)
    )
    if len(donors) != len(y_arr):
        raise ValueError(f"Donor row count {len(donors)} != support rows {len(y_arr)}")
    counts: Dict[Tuple[int, int], int] = {}
    donor_sets: Dict[Tuple[int, int], set[str]] = {}
    for y_value, c_value, donor in zip(y_arr.tolist(), c_arr.tolist(), donors.tolist()):
        key = (int(y_value), int(c_value))
        counts[key] = counts.get(key, 0) + 1
        donor_sets.setdefault(key, set()).add(str(donor))

    supported_contexts_by_label: Dict[int, set[int]] = {}
    for (label, context), count in counts.items():
        if count >= min_group_size and len(donor_sets[(label, context)]) >= int(min_donors):
            supported_contexts_by_label.setdefault(label, set()).add(context)

    eligible = np.zeros(len(y_arr), dtype=bool)
    for index, (label, context) in enumerate(zip(y_arr.tolist(), c_arr.tolist())):
        contexts = supported_contexts_by_label.get(int(label), set())
        eligible[index] = len(contexts) >= 2 and int(context) in contexts
    return eligible


def supported_tensors(
    supported: torch.Tensor,
    *tensors: torch.Tensor,
) -> Tuple[torch.Tensor, ...] | None:
    mask = supported.to(dtype=torch.bool)
    if int(mask.sum().item()) < 2:
        return None
    return tuple(tensor[mask] for tensor in tensors)


def support_coverage_tables(
    metadata: pd.DataFrame,
    label_column: str,
    context_fields: Sequence[str],
    min_group_size: int,
    min_donors: int = 0,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    count_rows: List[Dict[str, object]] = []
    label_values = metadata[label_column].astype(str)
    total_labels = int(label_values.nunique())

    for field in sorted(set(context_fields)):
        if field not in metadata.columns:
            summary_rows.append(
                {
                    "context_field": field,
                    "status": "missing_field",
                    "min_group_size": int(min_group_size),
                    "min_donors": int(min_donors),
                }
            )
            continue
        context_values = metadata[field].astype(str)
        y_encoder = LabelEncoder()
        c_encoder = LabelEncoder()
        y_int = y_encoder.fit_transform(label_values.to_numpy())
        c_int = c_encoder.fit_transform(context_values.to_numpy())
        donor_values = metadata["donor_id"].astype(str).to_numpy()
        eligible_mask = support_eligible_mask(
            y_int,
            c_int,
            min_group_size,
            donor_ids=donor_values,
            min_donors=min_donors,
        )

        counts_df = (
            pd.DataFrame(
                {
                    "label": label_values.to_numpy(),
                    "context": context_values.to_numpy(),
                    "donor_id": donor_values,
                }
            )
            .groupby(["label", "context"], sort=True)
            .agg(n_cells=("donor_id", "size"), n_donors=("donor_id", "nunique"))
            .reset_index()
        )
        supported_contexts_by_label: Dict[str, set[str]] = {}
        for _, row in counts_df.iterrows():
            if int(row["n_cells"]) >= min_group_size and int(row["n_donors"]) >= int(min_donors):
                supported_contexts_by_label.setdefault(str(row["label"]), set()).add(str(row["context"]))
        counts_df["supported_sample_count"] = counts_df["n_cells"].astype(int) >= min_group_size
        counts_df["supported_donor_count"] = counts_df["n_donors"].astype(int) >= int(min_donors)
        counts_df["supported_label_context"] = (
            counts_df["supported_sample_count"] & counts_df["supported_donor_count"]
        )
        counts_df["support_eligible_label"] = counts_df["label"].astype(str).map(
            lambda label: len(supported_contexts_by_label.get(label, set())) >= 2
        )
        counts_df["support_eligible_cell"] = counts_df["supported_label_context"] & counts_df["support_eligible_label"]

        for _, row in counts_df.iterrows():
            count_rows.append(
                {
                    "context_field": field,
                    "label": str(row["label"]),
                    "context_value": str(row["context"]),
                    "n_cells": int(row["n_cells"]),
                    "n_donors": int(row["n_donors"]),
                    "supported_sample_count": bool(row["supported_sample_count"]),
                    "supported_donor_count": bool(row["supported_donor_count"]),
                    "supported_label_context": bool(row["supported_label_context"]),
                    "support_eligible_label": bool(row["support_eligible_label"]),
                    "support_eligible_cell": bool(row["support_eligible_cell"]),
                    "min_group_size": int(min_group_size),
                    "min_donors": int(min_donors),
                }
            )

        summary_rows.append(
            {
                "context_field": field,
                "status": "ok",
                "min_group_size": int(min_group_size),
                "min_donors": int(min_donors),
                "n_cells": int(len(metadata)),
                "eligible_cells": int(eligible_mask.sum()),
                "eligible_fraction": float(np.mean(eligible_mask)),
                "total_labels": total_labels,
                "eligible_labels": int(label_values[eligible_mask].nunique()) if bool(eligible_mask.any()) else 0,
                "total_contexts": int(context_values.nunique()),
                "eligible_contexts": int(context_values[eligible_mask].nunique()) if bool(eligible_mask.any()) else 0,
                "total_label_context_pairs": int(len(counts_df)),
                "eligible_label_context_pairs": int(counts_df["support_eligible_cell"].sum()),
            }
        )
    return summary_rows, count_rows


def method_weights(method: str, args: argparse.Namespace) -> Dict[str, float]:
    base = {
        "fair": 0.0,
        "adv": 0.0,
        "supcon": 0.0,
        "consistency": 0.0,
        "group_dro": 0.0,
        "label_context_dro": 0.0,
        "max_gap": 0.0,
        "cond_mmd": 0.0,
        "cond_coral": 0.0,
        "irm": 0.0,
        "fishr": 0.0,
        "context_cvar": 0.0,
        "multi_context": 0.0,
    }
    if method == "erm_mlp":
        return base
    if method == "fair_var":
        return {**base, "fair": args.lambda_fair_var}
    if method == "adv_context":
        return {**base, "adv": args.lambda_adv}
    if method == "supcon_cross_context":
        return {**base, "supcon": args.lambda_supcon}
    if method == "consistency_pref":
        return {**base, "consistency": args.lambda_consistency}
    if method == "hybrid":
        return {**base,
            "fair": 0.5 * args.lambda_fair_var,
            "adv": 0.5 * args.lambda_adv,
            "supcon": 0.5 * args.lambda_supcon,
            "consistency": 0.5 * args.lambda_consistency,
        }
    if method == "group_dro":
        return {**base, "group_dro": args.lambda_group_dro}
    if method == "label_context_dro":
        return {**base, "label_context_dro": args.lambda_group_dro}
    if method == "max_gap":
        return {**base, "max_gap": args.lambda_max_gap}
    if method == "cond_mmd":
        return {**base, "cond_mmd": args.lambda_cond_mmd}
    if method == "cond_coral":
        return {**base, "cond_coral": args.lambda_cond_coral}
    if method == "irm":
        return {**base, "irm": args.lambda_irm}
    if method == "fishr":
        return {**base, "fishr": args.lambda_fishr}
    if method in {
        "context_reweight",
        "label_context_reweight",
        "lc_reweight_pow05",
        "lc_reweight_pow075",
        "lc_reweight_pow085",
        "lc_reweight_pow09",
        "lc_reweight_pow095",
        "lc_reweight_pow125",
        "lc_reweight_clip2",
        "lc_reweight_clip3",
        "lc_label_balanced",
        "lc_label_balanced_pow05",
        "lc_label_balanced_pow075",
        "lc_donor_reweight",
        "lc_donor_pow075",
        "stdr_pow085",
        "stdr_pow09",
        "stdr_pow095",
        "harmony_style",
    }:
        return base
    if method == "sacro_risk":
        return base
    if method in {"sabca", "sabca_no_support_gate"}:
        return {**base,
            "fair": 0.25 * args.lambda_fair_var,
            "adv": args.lambda_adv,
            "supcon": args.lambda_supcon,
            "consistency": args.lambda_consistency,
            "label_context_dro": 0.5 * args.lambda_group_dro,
            "cond_mmd": 0.5 * args.lambda_cond_mmd,
        }
    if method == "sabca_no_context_alignment":
        return {**base,
            "fair": 0.25 * args.lambda_fair_var,
            "label_context_dro": 0.5 * args.lambda_group_dro,
        }
    if method == "sabca_no_label_context_balancing":
        return {**base,
            "fair": 0.25 * args.lambda_fair_var,
            "adv": args.lambda_adv,
            "supcon": args.lambda_supcon,
            "consistency": args.lambda_consistency,
            "cond_mmd": 0.5 * args.lambda_cond_mmd,
        }
    if method in {"scea", "scea_no_support_gate"}:
        return {**base,
            "fair": 0.25 * args.lambda_fair_var,
            "supcon": args.lambda_supcon,
            "consistency": 0.5 * args.lambda_consistency,
            "label_context_dro": 0.5 * args.lambda_group_dro,
            "cond_mmd": 0.5 * args.lambda_cond_mmd,
            "cond_coral": args.lambda_cond_coral,
            "fishr": args.lambda_fishr,
            "context_cvar": args.lambda_scea_episode,
        }
    if method == "scea_no_episode":
        return {**base,
            "fair": 0.25 * args.lambda_fair_var,
            "supcon": args.lambda_supcon,
            "consistency": 0.5 * args.lambda_consistency,
            "label_context_dro": 0.5 * args.lambda_group_dro,
            "cond_mmd": 0.5 * args.lambda_cond_mmd,
            "cond_coral": args.lambda_cond_coral,
            "fishr": args.lambda_fishr,
        }
    if method == "scea_no_alignment":
        return {**base,
            "fair": 0.25 * args.lambda_fair_var,
            "label_context_dro": 0.5 * args.lambda_group_dro,
            "context_cvar": args.lambda_scea_episode,
        }
    if method == "scea_no_cvar":
        return {**base,
            "fair": 0.25 * args.lambda_fair_var,
            "supcon": args.lambda_supcon,
            "consistency": 0.5 * args.lambda_consistency,
            "cond_mmd": 0.5 * args.lambda_cond_mmd,
            "cond_coral": args.lambda_cond_coral,
            "fishr": args.lambda_fishr,
        }
    if method == "sca_lite":
        return {**base,
            "fair": 0.15 * args.lambda_fair_var,
            "supcon": 0.75 * args.lambda_supcon,
            "consistency": 0.25 * args.lambda_consistency,
            "cond_mmd": 0.5 * args.lambda_cond_mmd,
            "cond_coral": 0.5 * args.lambda_cond_coral,
        }
    if method == "sca_mmd":
        return {**base,
            "fair": 0.15 * args.lambda_fair_var,
            "cond_mmd": args.lambda_cond_mmd,
        }
    if method == "sca_coral":
        return {**base,
            "fair": 0.15 * args.lambda_fair_var,
            "cond_coral": args.lambda_cond_coral,
        }
    if method == "sca_supcon":
        return {**base,
            "fair": 0.15 * args.lambda_fair_var,
            "supcon": args.lambda_supcon,
            "consistency": 0.5 * args.lambda_consistency,
        }
    if method == "sca_soft_dro":
        return {**base,
            "fair": 0.15 * args.lambda_fair_var,
            "supcon": 0.75 * args.lambda_supcon,
            "consistency": 0.25 * args.lambda_consistency,
            "label_context_dro": 0.25 * args.lambda_group_dro,
            "cond_mmd": 0.5 * args.lambda_cond_mmd,
            "cond_coral": 0.5 * args.lambda_cond_coral,
        }
    if method == "sca_soft_cvar":
        return {**base,
            "fair": 0.15 * args.lambda_fair_var,
            "supcon": 0.75 * args.lambda_supcon,
            "consistency": 0.25 * args.lambda_consistency,
            "label_context_dro": 0.25 * args.lambda_group_dro,
            "cond_mmd": 0.5 * args.lambda_cond_mmd,
            "cond_coral": 0.5 * args.lambda_cond_coral,
            "context_cvar": 0.25 * args.lambda_scea_episode,
        }
    if method == "sca_multi_lite":
        return {**base,
            "fair": 0.15 * args.lambda_fair_var,
            "supcon": 0.75 * args.lambda_supcon,
            "consistency": 0.25 * args.lambda_consistency,
            "cond_mmd": 0.5 * args.lambda_cond_mmd,
            "cond_coral": 0.5 * args.lambda_cond_coral,
            "multi_context": 1.0,
        }
    if method == "sca_multi_soft_dro":
        return {**base,
            "fair": 0.15 * args.lambda_fair_var,
            "supcon": 0.75 * args.lambda_supcon,
            "consistency": 0.25 * args.lambda_consistency,
            "label_context_dro": 0.25 * args.lambda_group_dro,
            "cond_mmd": 0.5 * args.lambda_cond_mmd,
            "cond_coral": 0.5 * args.lambda_cond_coral,
            "multi_context": 1.0,
        }
    raise ValueError(f"Unknown method: {method}")


def inverse_frequency_weights(keys: Sequence[object]) -> np.ndarray:
    key_arr = np.asarray(keys, dtype=str)
    counts = pd.Series(key_arr).value_counts().to_dict()
    weights = np.asarray([1.0 / float(counts[key]) for key in key_arr], dtype=np.float32)
    return weights / float(weights.mean())


def power_inverse_frequency_weights(keys: Sequence[object], power: float) -> np.ndarray:
    key_arr = np.asarray(keys, dtype=str)
    counts = pd.Series(key_arr).value_counts().to_dict()
    weights = np.asarray([1.0 / float(counts[key]) for key in key_arr], dtype=np.float32)
    weights = np.power(weights, float(power)).astype(np.float32)
    return weights / float(weights.mean())


def clip_and_renormalize(weights: np.ndarray, max_weight: float) -> np.ndarray:
    if max_weight > 0:
        weights = np.minimum(weights, float(max_weight))
    return weights / float(weights.mean())


def label_balanced_context_weights(y_int: np.ndarray, c_int: np.ndarray, power: float = 1.0) -> np.ndarray:
    y_arr = np.asarray(y_int, dtype=int)
    c_arr = np.asarray(c_int, dtype=int)
    n_contexts = int(np.max(c_arr)) + 1
    group_ids = y_arr * n_contexts + c_arr
    label_counts = np.bincount(y_arr).astype(np.float32)
    group_counts = np.bincount(group_ids, minlength=(int(np.max(y_arr)) + 1) * n_contexts).astype(np.float32)
    weights = label_counts[y_arr] / np.maximum(group_counts[group_ids], 1.0)
    weights = np.power(weights, float(power)).astype(np.float32)
    return (weights / float(weights.mean())).astype(np.float32)


def stdr_tempered_context_weights(y_int: np.ndarray, c_int: np.ndarray, alpha: float) -> np.ndarray:
    """Label-conditional tempered reweighting from empirical to uniform contexts.

    For each label y, q_alpha(c|y) is proportional to p(c|y)^(1-alpha), and
    the sample importance weight q_alpha(c|y) / p(c|y) preserves the total
    weight of label y.
    """
    y_arr = np.asarray(y_int, dtype=int)
    c_arr = np.asarray(c_int, dtype=int)
    weights = np.ones(len(y_arr), dtype=np.float32)
    alpha_value = float(alpha)
    for label in np.unique(y_arr):
        label_mask = y_arr == label
        label_contexts, label_counts = np.unique(c_arr[label_mask], return_counts=True)
        n_label = float(label_counts.sum())
        if n_label <= 0:
            continue
        p_context = label_counts.astype(np.float64) / n_label
        normalizer = float(np.power(p_context, 1.0 - alpha_value).sum())
        context_weights = np.power(p_context, -alpha_value) / max(normalizer, 1e-12)
        mapping = {int(context): float(weight) for context, weight in zip(label_contexts, context_weights)}
        weights[label_mask] = np.asarray([mapping[int(context)] for context in c_arr[label_mask]], dtype=np.float32)
    return weights.astype(np.float32)


def donor_inverse_frequency_weights(
    y_int: np.ndarray,
    c_int: np.ndarray,
    donor_train: Sequence[object] | None,
    power: float = 1.0,
) -> np.ndarray:
    if donor_train is None:
        keys = [f"{int(y)}::{int(c)}" for y, c in zip(y_int, c_int)]
        return power_inverse_frequency_weights(keys, power)
    rows = pd.DataFrame(
        {
            "group": [f"{int(y)}::{int(c)}" for y, c in zip(y_int, c_int)],
            "donor": np.asarray(donor_train, dtype=str),
        }
    )
    donor_counts = rows.groupby("group", sort=False)["donor"].nunique().to_dict()
    weights = np.asarray([1.0 / max(float(donor_counts[group]), 1.0) for group in rows["group"]], dtype=np.float32)
    weights = np.power(weights, float(power)).astype(np.float32)
    return weights / float(weights.mean())


def sacro_group_confidence(
    y_train_int: np.ndarray,
    c_train_int: np.ndarray,
    donor_train: Sequence[object] | None,
    n_classes: int,
    n_contexts: int,
    kappa: float,
) -> np.ndarray:
    """Continuous support confidence q_{y,c} from cell and donor support."""
    group_count = np.zeros(n_classes * n_contexts, dtype=np.float32)
    donor_sets: List[set[str]] = [set() for _ in range(n_classes * n_contexts)]
    donors = (
        np.asarray(donor_train, dtype=str)
        if donor_train is not None
        else np.asarray([f"cell_{idx}" for idx in range(len(y_train_int))], dtype=str)
    )
    for y_value, c_value, donor in zip(y_train_int.tolist(), c_train_int.tolist(), donors.tolist()):
        group = int(y_value) * n_contexts + int(c_value)
        group_count[group] += 1.0
        donor_sets[group].add(str(donor))
    donor_count = np.asarray([float(len(values)) for values in donor_sets], dtype=np.float32)
    kappa_value = max(float(kappa), 1e-6)
    cell_conf = group_count / (group_count + kappa_value)
    donor_conf = donor_count / (donor_count + kappa_value)
    return (cell_conf * donor_conf).astype(np.float32)


def sacro_floor_confidence(confidence: np.ndarray, floor: float) -> np.ndarray:
    floor_value = min(max(float(floor), 0.0), 1.0)
    return (floor_value + (1.0 - floor_value) * np.asarray(confidence, dtype=np.float32)).astype(np.float32)


def sacro_static_sample_weights(
    y_train_int: np.ndarray,
    c_train_int: np.ndarray,
    donor_train: Sequence[object] | None,
    n_classes: int,
    n_contexts: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """Frequency balance calibrated by soft support confidence."""
    y_arr = np.asarray(y_train_int, dtype=int)
    c_arr = np.asarray(c_train_int, dtype=int)
    group_ids = y_arr * n_contexts + c_arr
    label_counts = np.bincount(y_arr, minlength=n_classes).astype(np.float32)
    group_counts = np.bincount(group_ids, minlength=n_classes * n_contexts).astype(np.float32)
    confidence = sacro_group_confidence(y_arr, c_arr, donor_train, n_classes, n_contexts, args.sacro_kappa)
    confidence = sacro_floor_confidence(confidence, args.sacro_confidence_floor)
    alpha = float(args.sacro_alpha)
    balance = np.power(label_counts[y_arr] / np.maximum(group_counts[group_ids], 1.0), alpha)
    weights = balance * confidence[group_ids]
    weights = np.clip(weights, float(args.sacro_min_sample_weight), float(args.sacro_max_sample_weight))
    return (weights / float(weights.mean())).astype(np.float32)


def supervised_sample_weights(
    method: str,
    y_train_int: np.ndarray,
    c_train_int: np.ndarray,
    args: argparse.Namespace,
    donor_train: Sequence[object] | None = None,
) -> np.ndarray:
    if method == "context_reweight":
        return inverse_frequency_weights(c_train_int)
    keys = [f"{int(y)}::{int(c)}" for y, c in zip(y_train_int, c_train_int)]
    if method == "lc_reweight_pow05":
        return power_inverse_frequency_weights(keys, 0.5)
    if method == "lc_reweight_pow075":
        return power_inverse_frequency_weights(keys, 0.75)
    if method == "lc_reweight_pow085":
        return power_inverse_frequency_weights(keys, 0.85)
    if method == "lc_reweight_pow09":
        return power_inverse_frequency_weights(keys, 0.9)
    if method == "lc_reweight_pow095":
        return power_inverse_frequency_weights(keys, 0.95)
    if method == "lc_reweight_pow125":
        return power_inverse_frequency_weights(keys, 1.25)
    if method == "lc_reweight_clip2":
        return clip_and_renormalize(inverse_frequency_weights(keys), 2.0)
    if method == "lc_reweight_clip3":
        return clip_and_renormalize(inverse_frequency_weights(keys), 3.0)
    if method == "lc_label_balanced":
        return label_balanced_context_weights(y_train_int, c_train_int)
    if method == "lc_label_balanced_pow05":
        return label_balanced_context_weights(y_train_int, c_train_int, 0.5)
    if method == "lc_label_balanced_pow075":
        return label_balanced_context_weights(y_train_int, c_train_int, 0.75)
    if method == "lc_donor_reweight":
        return donor_inverse_frequency_weights(y_train_int, c_train_int, donor_train)
    if method == "lc_donor_pow075":
        return donor_inverse_frequency_weights(y_train_int, c_train_int, donor_train, 0.75)
    if method == "stdr_pow085":
        return stdr_tempered_context_weights(y_train_int, c_train_int, 0.85)
    if method == "stdr_pow09":
        return stdr_tempered_context_weights(y_train_int, c_train_int, 0.9)
    if method == "stdr_pow095":
        return stdr_tempered_context_weights(y_train_int, c_train_int, 0.95)
    if method == "sacro_risk":
        n_classes = int(np.max(y_train_int)) + 1
        n_contexts = int(np.max(c_train_int)) + 1
        return sacro_static_sample_weights(y_train_int, c_train_int, donor_train, n_classes, n_contexts, args)
    if (
        method in {"label_context_reweight", "sabca", "sabca_no_support_gate", "sabca_no_context_alignment"}
        or method in SCEA_FAMILY
        or method in SCA_SEARCH_FAMILY
    ):
        weights = inverse_frequency_weights(keys)
        if method in SUPPORT_AWARE_METHODS:
            return clip_and_renormalize(weights, args.sabca_max_sample_weight)
        return weights
    return np.ones(len(y_train_int), dtype=np.float32)


def effective_sample_size(weights: np.ndarray) -> float:
    weight_arr = np.asarray(weights, dtype=np.float64)
    denom = float(np.square(weight_arr).sum())
    if denom <= 0:
        return 0.0
    return float(np.square(weight_arr.sum()) / denom)


def min_group_ess_fraction(values: np.ndarray, weights: np.ndarray) -> float:
    fractions: List[float] = []
    value_arr = np.asarray(values)
    weight_arr = np.asarray(weights, dtype=np.float64)
    for value in np.unique(value_arr):
        mask = value_arr == value
        n_group = int(mask.sum())
        if n_group > 0:
            fractions.append(effective_sample_size(weight_arr[mask]) / float(n_group))
    return float(min(fractions)) if fractions else float("nan")


def donor_label_ess_fraction(
    labels: np.ndarray,
    weights: np.ndarray,
    donor_train: Sequence[object] | None,
) -> float:
    if donor_train is None:
        return float("nan")
    rows = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=str),
            "donor": np.asarray(donor_train, dtype=str),
            "weight": np.asarray(weights, dtype=np.float64),
        }
    )
    fractions: List[float] = []
    for _, label_rows in rows.groupby("label", sort=False):
        donor_totals = label_rows.groupby("donor", sort=False)["weight"].sum().to_numpy(dtype=np.float64)
        n_donors = int(len(donor_totals))
        if n_donors > 0:
            fractions.append(effective_sample_size(donor_totals) / float(n_donors))
    return float(min(fractions)) if fractions else float("nan")


def weight_diagnostic_row(
    method: str,
    y_train_int: np.ndarray,
    c_train_int: np.ndarray,
    weights: np.ndarray,
    donor_train: Sequence[object] | None,
    prefix: Mapping[str, object],
) -> Dict[str, object]:
    weight_arr = np.asarray(weights, dtype=np.float64)
    label_df = pd.DataFrame({"label": y_train_int, "weight": weight_arr})
    label_totals = label_df.groupby("label")["weight"].sum()
    label_counts = label_df.groupby("label")["weight"].size().astype(float)
    label_weight_to_count = label_totals / label_counts
    label_total_mean = float(label_totals.mean()) if len(label_totals) else float("nan")
    label_total_cv = (
        float(label_totals.std(ddof=0) / label_total_mean)
        if len(label_totals) and abs(label_total_mean) > 1e-12
        else float("nan")
    )
    label_ratio_mean = float(label_weight_to_count.mean()) if len(label_weight_to_count) else float("nan")
    label_ratio_cv = (
        float(label_weight_to_count.std(ddof=0) / label_ratio_mean)
        if len(label_weight_to_count) and abs(label_ratio_mean) > 1e-12
        else float("nan")
    )
    positive = weight_arr[weight_arr > 0]
    min_positive = float(positive.min()) if len(positive) else float("nan")
    return {
        **dict(prefix),
        "method": method,
        "n_train_cells": int(len(weight_arr)),
        "n_labels": int(len(np.unique(y_train_int))),
        "n_contexts": int(len(np.unique(c_train_int))),
        "n_label_context_groups": int(len(np.unique(np.asarray(y_train_int) * (int(np.max(c_train_int)) + 1) + c_train_int))),
        "weight_mean": float(weight_arr.mean()),
        "weight_sum": float(weight_arr.sum()),
        "weight_std": float(weight_arr.std(ddof=0)),
        "weight_min": float(weight_arr.min()) if len(weight_arr) else float("nan"),
        "weight_max": float(weight_arr.max()) if len(weight_arr) else float("nan"),
        "weight_p95": float(np.percentile(weight_arr, 95)) if len(weight_arr) else float("nan"),
        "weight_p99": float(np.percentile(weight_arr, 99)) if len(weight_arr) else float("nan"),
        "weight_max_min_ratio": float(weight_arr.max() / min_positive) if len(weight_arr) and min_positive > 0 else float("nan"),
        "ess": effective_sample_size(weight_arr),
        "ess_fraction": effective_sample_size(weight_arr) / float(len(weight_arr)) if len(weight_arr) else float("nan"),
        "min_label_ess_fraction": min_group_ess_fraction(y_train_int, weight_arr),
        "min_context_ess_fraction": min_group_ess_fraction(c_train_int, weight_arr),
        "min_donor_label_ess_fraction": donor_label_ess_fraction(y_train_int, weight_arr, donor_train),
        "label_total_weight_cv": label_total_cv,
        "label_total_weight_min": float(label_totals.min()) if len(label_totals) else float("nan"),
        "label_total_weight_max": float(label_totals.max()) if len(label_totals) else float("nan"),
        "label_weight_to_count_ratio_cv": label_ratio_cv,
        "label_weight_to_count_ratio_min": float(label_weight_to_count.min()) if len(label_weight_to_count) else float("nan"),
        "label_weight_to_count_ratio_max": float(label_weight_to_count.max()) if len(label_weight_to_count) else float("nan"),
    }


def predict_model_labels(
    model: nn.Module,
    x_eval: np.ndarray,
    batch_size: int,
    device: str,
    label_encoder: LabelEncoder,
) -> np.ndarray:
    preds: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x_eval), batch_size * 4):
            xb = torch.tensor(x_eval[start : start + batch_size * 4], dtype=torch.float32, device=device)
            logits, _, _ = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    pred_int = np.concatenate(preds, axis=0)
    return label_encoder.inverse_transform(pred_int)


def eval_diagnostic_metrics(
    y_eval: np.ndarray,
    pred_eval: np.ndarray,
    context_eval: np.ndarray | None,
) -> Dict[str, object]:
    overall = metric_row(y_eval, pred_eval, {})
    out: Dict[str, object] = {
        "eval_accuracy": overall["accuracy"],
        "eval_balanced_accuracy": overall["balanced_accuracy"],
        "eval_macro_f1": overall["macro_f1"],
    }
    if context_eval is None or len(context_eval) == 0:
        return out
    context_scores: List[float] = []
    for value in np.unique(context_eval.astype(str)):
        mask = context_eval.astype(str) == value
        if int(mask.sum()) > 0:
            context_scores.append(float(metric_row(y_eval[mask], pred_eval[mask], {})["balanced_accuracy"]))
    if context_scores:
        out["eval_worst_context_balanced_accuracy"] = float(min(context_scores))
        out["eval_best_context_balanced_accuracy"] = float(max(context_scores))
        out["eval_context_gap"] = float(max(context_scores) - min(context_scores))
    return out


def context_center_embeddings(
    x_train: np.ndarray,
    context_train: np.ndarray,
    x_test: np.ndarray,
    context_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Harmony-style deterministic context centering in standardized embedding space.

    The correction estimates per-context mean offsets on the training split only.
    Test contexts unseen in training are left unchanged, which is the conservative
    behavior for leave-one-context evaluation.
    """
    c_train = np.asarray(context_train, dtype=str)
    c_test = np.asarray(context_test, dtype=str)
    global_mean = x_train.mean(axis=0, keepdims=True)
    offsets: Dict[str, np.ndarray] = {}
    for context in sorted(set(c_train.tolist())):
        mask = c_train == context
        if int(mask.sum()) < 2:
            continue
        offsets[context] = x_train[mask].mean(axis=0, keepdims=True) - global_mean
    x_train_corr = x_train.copy()
    for context, offset in offsets.items():
        x_train_corr[c_train == context] = x_train_corr[c_train == context] - offset
    x_test_corr = x_test.copy()
    for context, offset in offsets.items():
        x_test_corr[c_test == context] = x_test_corr[c_test == context] - offset
    return x_train_corr.astype(np.float32), x_test_corr.astype(np.float32)


def encode_aux_contexts(aux_context_train: np.ndarray | None) -> np.ndarray:
    if aux_context_train is None or aux_context_train.size == 0:
        n_rows = 0 if aux_context_train is None else int(aux_context_train.shape[0])
        return np.zeros((n_rows, 0), dtype=np.int64)
    aux_arr = np.asarray(aux_context_train, dtype=str)
    if aux_arr.ndim == 1:
        aux_arr = aux_arr[:, None]
    encoded = np.zeros(aux_arr.shape, dtype=np.int64)
    for col in range(aux_arr.shape[1]):
        encoded[:, col] = LabelEncoder().fit_transform(aux_arr[:, col])
    return encoded


def aux_support_masks(
    y_train_int: np.ndarray,
    aux_train_int: np.ndarray,
    min_group_size: int,
    donor_train: Sequence[object] | None = None,
    min_donors: int = 0,
) -> np.ndarray:
    if aux_train_int.size == 0:
        return np.ones(aux_train_int.shape, dtype=bool)
    masks = np.ones(aux_train_int.shape, dtype=bool)
    for col in range(aux_train_int.shape[1]):
        masks[:, col] = support_eligible_mask(
            y_train_int,
            aux_train_int[:, col],
            min_group_size,
            donor_ids=donor_train,
            min_donors=min_donors,
        )
    return masks


def weighted_context_regularizer(
    weights: Mapping[str, float],
    reg_losses: torch.Tensor,
    reg_y: torch.Tensor,
    reg_c: torch.Tensor,
    reg_logits: torch.Tensor,
    reg_z: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    total = reg_losses.new_tensor(0.0)
    if weights["fair"]:
        total = total + weights["fair"] * group_loss_variance(reg_losses, reg_y, reg_c)
    if weights["supcon"]:
        total = total + weights["supcon"] * supcon_cross_context_loss(reg_z, reg_y, reg_c, args.temperature)
    if weights["consistency"]:
        total = total + weights["consistency"] * consistency_pref_loss(reg_logits, reg_y, reg_c)
    if weights["group_dro"]:
        total = total + weights["group_dro"] * context_group_loss_regularizer(reg_losses, reg_c, "max")
    if weights["label_context_dro"]:
        total = total + weights["label_context_dro"] * label_context_group_dro(reg_losses, reg_y, reg_c)
    if weights["max_gap"]:
        total = total + weights["max_gap"] * context_group_loss_regularizer(reg_losses, reg_c, "gap")
    if weights["context_cvar"]:
        total = total + weights["context_cvar"] * context_cvar_loss(reg_losses, reg_c, args.scea_cvar_fraction)
    if weights["cond_mmd"]:
        total = total + weights["cond_mmd"] * conditional_mean_alignment_loss(reg_z, reg_y, reg_c)
    if weights["cond_coral"]:
        total = total + weights["cond_coral"] * conditional_coral_loss(reg_z, reg_y, reg_c)
    if weights["irm"]:
        total = total + weights["irm"] * irm_penalty(reg_logits, reg_y, reg_c)
    if weights["fishr"]:
        total = total + weights["fishr"] * fishr_logit_penalty(reg_logits, reg_y, reg_c)
    return total


def regularizer_view(
    support_mask: torch.Tensor,
    sample_losses: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    logits: torch.Tensor,
    z: torch.Tensor,
    support_gated: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    supported = supported_tensors(support_mask, sample_losses, y, c, logits, z)
    if support_gated and supported is not None:
        reg_losses, reg_y, reg_c, reg_logits, reg_z = supported
        return reg_losses, reg_y, reg_c, reg_logits, reg_z
    return sample_losses, y, c, logits, z


def make_inner_validation_split(
    y_train: np.ndarray,
    context_train: np.ndarray,
    seed: int,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Build an inner validation split using only the outer training fold."""
    n = len(y_train)
    if n < 20:
        return None
    y_arr = y_train.astype(str)
    c_arr = context_train.astype(str)
    stratify = np.asarray([f"{label}::{context}" for label, context in zip(y_arr, c_arr)], dtype=object)
    counts = pd.Series(stratify).value_counts()
    if counts.empty or int(counts.min()) < 2 or len(counts) < 2:
        stratify = y_arr
        counts = pd.Series(stratify).value_counts()
    rng = np.random.default_rng(seed)
    if not counts.empty and int(counts.min()) >= 2 and len(counts) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        inner_train, inner_val = next(splitter.split(np.zeros(n), stratify))
    else:
        order = rng.permutation(n)
        val_n = max(2, int(round(test_size * n)))
        inner_val = np.sort(order[:val_n])
        inner_train = np.sort(order[val_n:])
    if len(inner_train) < 10 or len(inner_val) < 2:
        return None
    if len(np.unique(y_arr[inner_train])) < 2 or len(np.unique(y_arr[inner_val])) < 2:
        return None
    return inner_train.astype(int), inner_val.astype(int)


def validation_context_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    context_values: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return robust validation score, worst-bin BA, gap, and overall BA."""
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    context_values = context_values.astype(str)
    def safe_balanced_accuracy(local_true: np.ndarray, local_pred: np.ndarray) -> float:
        recalls: List[float] = []
        for label in sorted(set(local_true.tolist())):
            mask = local_true == label
            if int(mask.sum()) == 0:
                continue
            recalls.append(float(np.mean(local_pred[mask] == label)))
        if not recalls:
            return 0.0
        return float(np.mean(recalls))

    overall = safe_balanced_accuracy(y_true, y_pred)
    context_scores: List[float] = []
    for context in sorted(set(context_values.tolist())):
        mask = context_values == context
        if int(mask.sum()) < 2:
            continue
        if len(np.unique(y_true[mask])) < 2:
            continue
        context_scores.append(safe_balanced_accuracy(y_true[mask], y_pred[mask]))
    if not context_scores:
        worst = overall
        gap = 0.0
    else:
        worst = float(min(context_scores))
        gap = float(max(context_scores) - worst)
    score = worst - 0.10 * gap + 0.05 * overall
    return score, worst, gap, overall


def train_predict_reweight_plus(
    x_train: np.ndarray,
    y_train: np.ndarray,
    context_train: np.ndarray,
    x_test: np.ndarray,
    context_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    aux_context_train: np.ndarray | None = None,
    donor_train: np.ndarray | None = None,
) -> np.ndarray:
    split = make_inner_validation_split(y_train, context_train, seed)
    if split is None:
        print("[reweight_plus] no stable inner validation split; falling back to label_context_reweight", flush=True)
        return train_predict(
            x_train,
            y_train,
            context_train,
            x_test,
            context_test,
            "label_context_reweight",
            args,
            seed=seed + 7919,
            aux_context_train=aux_context_train,
            donor_train=donor_train,
        )

    inner_train, inner_val = split
    aux_inner_train = aux_context_train[inner_train] if aux_context_train is not None else None
    donor_inner_train = donor_train[inner_train] if donor_train is not None else None
    scores: Dict[str, Tuple[float, float, float, float]] = {}
    for offset, candidate in enumerate(REWEIGHT_PLUS_CANDIDATES):
        pred_val = train_predict(
            x_train[inner_train],
            y_train[inner_train],
            context_train[inner_train],
            x_train[inner_val],
            context_train[inner_val],
            candidate,
            args,
            seed=seed + 101 * (offset + 1),
            aux_context_train=aux_inner_train,
            donor_train=donor_inner_train,
        )
        scores[candidate] = validation_context_score(y_train[inner_val], pred_val, context_train[inner_val])

    reweight_score = scores["label_context_reweight"][0]
    best_method = max(scores, key=lambda name: scores[name][0])
    # Keep the method anchored to Reweight unless an adapter clears a small,
    # held-in robust-score margin. This avoids replacing strong Reweight rows
    # with noisier modules.
    if best_method != "label_context_reweight" and scores[best_method][0] < reweight_score + 0.002:
        best_method = "label_context_reweight"
    score_text = "; ".join(
        f"{name}:score={values[0]:.4f},worst={values[1]:.4f},gap={values[2]:.4f}"
        for name, values in sorted(scores.items())
    )
    print(f"[reweight_plus] selected={best_method}; {score_text}", flush=True)
    return train_predict(
        x_train,
        y_train,
        context_train,
        x_test,
        context_test,
        best_method,
        args,
        seed=seed + 1543,
        aux_context_train=aux_context_train,
        donor_train=donor_train,
    )


def train_predict_validation_tuned(
    x_train: np.ndarray,
    y_train: np.ndarray,
    context_train: np.ndarray,
    x_test: np.ndarray,
    context_test: np.ndarray,
    method: str,
    args: argparse.Namespace,
    seed: int,
    aux_context_train: np.ndarray | None = None,
    donor_train: np.ndarray | None = None,
) -> np.ndarray:
    """Select one of three strengths using only an inner training split."""
    split = make_inner_validation_split(y_train, context_train, seed)
    base_method = "sca_lite" if method == "sca_align_tuned" else "group_dro"
    strengths = (0.5, 1.0, 2.0)
    if split is None:
        selected_strength = 1.0
        scores: Dict[float, Tuple[float, float, float, float]] = {}
    else:
        inner_train, inner_val = split
        scores = {}
        for offset, strength in enumerate(strengths):
            candidate_args = copy.copy(args)
            if method == "sca_align_tuned":
                for field in (
                    "lambda_fair_var",
                    "lambda_supcon",
                    "lambda_consistency",
                    "lambda_cond_mmd",
                    "lambda_cond_coral",
                ):
                    setattr(candidate_args, field, float(getattr(args, field)) * strength)
            else:
                candidate_args.lambda_group_dro = float(args.lambda_group_dro) * strength
            pred_val = train_predict(
                x_train[inner_train],
                y_train[inner_train],
                context_train[inner_train],
                x_train[inner_val],
                context_train[inner_val],
                base_method,
                candidate_args,
                seed=seed + 101 * (offset + 1),
                aux_context_train=(
                    aux_context_train[inner_train]
                    if aux_context_train is not None
                    else None
                ),
                donor_train=(
                    donor_train[inner_train] if donor_train is not None else None
                ),
            )
            scores[strength] = validation_context_score(
                y_train[inner_val],
                pred_val,
                context_train[inner_val],
            )
        selected_strength = max(scores, key=lambda value: scores[value][0])
    score_text = "; ".join(
        f"{strength:g}:score={values[0]:.4f},worst={values[1]:.4f},gap={values[2]:.4f}"
        for strength, values in sorted(scores.items())
    )
    print(
        f"[{method}] selected_strength={selected_strength:g}; {score_text}",
        flush=True,
    )
    selected_args = copy.copy(args)
    if method == "sca_align_tuned":
        for field in (
            "lambda_fair_var",
            "lambda_supcon",
            "lambda_consistency",
            "lambda_cond_mmd",
            "lambda_cond_coral",
        ):
            setattr(
                selected_args,
                field,
                float(getattr(args, field)) * selected_strength,
            )
    else:
        selected_args.lambda_group_dro = (
            float(args.lambda_group_dro) * selected_strength
        )
    return train_predict(
        x_train,
        y_train,
        context_train,
        x_test,
        context_test,
        base_method,
        selected_args,
        seed=seed + 1543,
        aux_context_train=aux_context_train,
        donor_train=donor_train,
    )


def train_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    context_train: np.ndarray,
    x_test: np.ndarray,
    context_test: np.ndarray,
    method: str,
    args: argparse.Namespace,
    seed: int,
    aux_context_train: np.ndarray | None = None,
    donor_train: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    diagnostic_prefix: Mapping[str, object] | None = None,
    weight_diagnostic_rows: List[Dict[str, object]] | None = None,
    epoch_diagnostic_rows: List[Dict[str, object]] | None = None,
) -> np.ndarray:
    if method in {"sca_align_tuned", "group_dro_tuned"}:
        return train_predict_validation_tuned(
            x_train,
            y_train,
            context_train,
            x_test,
            context_test,
            method,
            args,
            seed=seed,
            aux_context_train=aux_context_train,
            donor_train=donor_train,
        )
    if method == "reweight_plus":
        return train_predict_reweight_plus(
            x_train,
            y_train,
            context_train,
            x_test,
            context_test,
            args,
            seed=seed,
            aux_context_train=aux_context_train,
            donor_train=donor_train,
        )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    effective_method = "erm_mlp" if method == "harmony_style" else method
    if method == "harmony_style":
        x_train, x_test = context_center_embeddings(x_train, context_train, x_test, context_test)

    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train.astype(str))
    context_encoder = LabelEncoder()
    c_train_int = context_encoder.fit_transform(context_train.astype(str))
    aux_train_int = encode_aux_contexts(aux_context_train)
    if aux_train_int.shape[0] == 0 and len(y_train_int) > 0:
        aux_train_int = np.zeros((len(y_train_int), 0), dtype=np.int64)
    if aux_train_int.shape[0] != len(y_train_int):
        raise ValueError(f"Aux context row count {aux_train_int.shape[0]} != training rows {len(y_train_int)}")
    n_classes = int(len(label_encoder.classes_))
    n_contexts = int(max(1, len(context_encoder.classes_)))
    sacro_confidence_np = sacro_group_confidence(
        y_train_int,
        c_train_int,
        donor_train,
        n_classes,
        n_contexts,
        args.sacro_kappa,
    )
    sacro_confidence_np = sacro_floor_confidence(sacro_confidence_np, args.sacro_confidence_floor)
    support_gated_method = (
        effective_method in SUPPORT_AWARE_METHODS
        and effective_method not in NO_SUPPORT_GATE_METHODS
    )
    if support_gated_method:
        supported_np = support_eligible_mask(
            y_train_int,
            c_train_int,
            args.sabca_min_group_size,
            donor_ids=donor_train,
            min_donors=args.support_min_donors,
        )
    else:
        supported_np = np.ones(len(y_train_int), dtype=bool)
    aux_supported_np = aux_support_masks(
        y_train_int,
        aux_train_int,
        args.sabca_min_group_size,
        donor_train=donor_train,
        min_donors=args.support_min_donors,
    )

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
    weights = method_weights(effective_method, args)
    sample_weight_np = supervised_sample_weights(
        effective_method,
        y_train_int,
        c_train_int,
        args,
        donor_train=donor_train,
    )
    if args.write_training_diagnostics and weight_diagnostic_rows is not None:
        weight_diagnostic_rows.append(
            weight_diagnostic_row(
                method,
                y_train_int,
                c_train_int,
                sample_weight_np,
                donor_train,
                diagnostic_prefix or {},
            )
        )

    order = rng.permutation(len(x_train))
    dataset = TensorDataset(
        torch.tensor(x_train[order], dtype=torch.float32),
        torch.tensor(y_train_int[order], dtype=torch.long),
        torch.tensor(c_train_int[order], dtype=torch.long),
        torch.tensor(aux_train_int[order], dtype=torch.long),
        torch.tensor(sample_weight_np[order], dtype=torch.float32),
        torch.tensor(supported_np[order], dtype=torch.bool),
        torch.tensor(aux_supported_np[order], dtype=torch.bool),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    sacro_confidence = torch.tensor(sacro_confidence_np, dtype=torch.float32, device=args.device)
    sacro_group_risk_ema = torch.zeros(n_classes * n_contexts, dtype=torch.float32, device=args.device)
    model.train()
    for epoch in range(1, int(args.epochs) + 1):
        epoch_objective = 0.0
        epoch_weighted_ce = 0.0
        epoch_unweighted_ce = 0.0
        epoch_grad_norms: List[float] = []
        epoch_cells = 0
        for xb, yb, cb, aux_cb, wb, sb, aux_sb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            cb = cb.to(args.device)
            aux_cb = aux_cb.to(args.device)
            wb = wb.to(args.device)
            sb = sb.to(args.device)
            aux_sb = aux_sb.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, context_logits, z = model(xb, grl_strength=weights["adv"])
            sample_losses = ce(logits, yb)
            if effective_method == "sacro_risk":
                group_ids = yb * n_contexts + cb
                risk_matrix = sacro_group_risk_ema.view(n_classes, n_contexts)
                confidence_matrix = sacro_confidence.view(n_classes, n_contexts)
                active = (confidence_matrix > 0).to(risk_matrix.dtype)
                label_mean_risk = (risk_matrix * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
                centered_risk = (sacro_group_risk_ema[group_ids] - label_mean_risk[yb]).clamp(
                    min=-float(args.sacro_risk_clamp),
                    max=float(args.sacro_risk_clamp),
                )
                dynamic_w = torch.exp(float(args.sacro_beta) * centered_risk)
                wb = wb * dynamic_w
                wb = wb / wb.mean().clamp_min(1e-8)
            loss = (sample_losses * wb).sum() / wb.sum().clamp_min(1e-8)
            if effective_method == "sacro_risk" and float(args.lambda_sacro_cvar) > 0:
                loss = loss + float(args.lambda_sacro_cvar) * sacro_label_conditional_cvar_loss(
                    sample_losses,
                    yb,
                    cb,
                    sacro_confidence,
                    n_contexts,
                    args.scea_cvar_fraction,
                )
            if weights["adv"] and n_contexts > 1:
                supported = supported_tensors(sb, sample_losses, yb, cb, logits, z)
                if support_gated_method and supported is not None:
                    loss = loss + weights["adv"] * ctx_ce(context_logits[sb], cb[sb])
                else:
                    loss = loss + weights["adv"] * ctx_ce(context_logits, cb)
            context_views = [
                regularizer_view(sb, sample_losses, yb, cb, logits, z, support_gated_method),
            ]
            if weights["multi_context"] and aux_cb.shape[1] > 0:
                for aux_index in range(aux_cb.shape[1]):
                    context_views.append(
                        regularizer_view(
                            aux_sb[:, aux_index],
                            sample_losses,
                            yb,
                            aux_cb[:, aux_index],
                            logits,
                            z,
                            support_gated_method,
                        )
                    )
            regularizer_loss = sample_losses.new_tensor(0.0)
            for reg_losses, reg_y, reg_c, reg_logits, reg_z in context_views:
                regularizer_loss = regularizer_loss + weighted_context_regularizer(
                    weights,
                    reg_losses,
                    reg_y,
                    reg_c,
                    reg_logits,
                    reg_z,
                    args,
                )
            loss = loss + regularizer_loss / max(1, len(context_views))
            loss.backward()
            grad_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_sq += float(param.grad.detach().norm(2).item() ** 2)
            epoch_grad_norms.append(float(math.sqrt(grad_sq)))
            optimizer.step()
            batch_n = int(yb.shape[0])
            epoch_cells += batch_n
            epoch_objective += float(loss.detach().cpu().item()) * batch_n
            epoch_weighted_ce += float(((sample_losses * wb).sum() / wb.sum().clamp_min(1e-8)).detach().cpu().item()) * batch_n
            epoch_unweighted_ce += float(sample_losses.mean().detach().cpu().item()) * batch_n
            if effective_method == "sacro_risk":
                with torch.no_grad():
                    group_ids = yb * n_contexts + cb
                    for group in torch.unique(group_ids):
                        mask = group_ids == group
                        group_loss = sample_losses[mask].mean().detach()
                        sacro_group_risk_ema[group] = (
                            (1.0 - float(args.sacro_risk_ema)) * sacro_group_risk_ema[group]
                            + float(args.sacro_risk_ema) * group_loss
                        )
        if (
            args.write_training_diagnostics
            and epoch_diagnostic_rows is not None
            and int(args.diagnostic_epoch_interval) > 0
            and (epoch % int(args.diagnostic_epoch_interval) == 0 or epoch == int(args.epochs))
        ):
            model.eval()
            row: Dict[str, object] = {
                **dict(diagnostic_prefix or {}),
                "method": method,
                "epoch": int(epoch),
                "train_objective": float(epoch_objective / max(epoch_cells, 1)),
                "train_weighted_ce": float(epoch_weighted_ce / max(epoch_cells, 1)),
                "train_unweighted_ce": float(epoch_unweighted_ce / max(epoch_cells, 1)),
                "grad_norm_mean": float(np.mean(epoch_grad_norms)) if epoch_grad_norms else float("nan"),
                "grad_norm_max": float(np.max(epoch_grad_norms)) if epoch_grad_norms else float("nan"),
            }
            if y_eval is not None:
                pred_eval = predict_model_labels(model, x_test, args.batch_size, args.device, label_encoder)
                row.update(eval_diagnostic_metrics(np.asarray(y_eval, dtype=str), pred_eval, context_test))
            epoch_diagnostic_rows.append(row)
            model.train()

    model.eval()
    return predict_model_labels(model, x_test, args.batch_size, args.device, label_encoder)


def append_prediction_rows(
    rows: List[Dict[str, object]],
    metadata: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str,
    prefix: Mapping[str, object],
    context_fields: Sequence[str],
) -> None:
    test_meta = metadata.iloc[indices]
    for local_i, (_, meta_row) in enumerate(test_meta.iterrows()):
        row = {
            **dict(prefix),
            "method": method,
            "cell_index": int(meta_row["cell_index"]),
            "donor_id": meta_row["donor_id"],
            "true_label": str(y_true[local_i]),
            "pred_label": str(y_pred[local_i]),
        }
        for field in sorted(set(context_fields) | set(DEFAULT_CONTEXT_FIELDS) | {"disease"}):
            if field in test_meta.columns:
                row[field] = meta_row[field]
        rows.append(row)


def context_value_matrix(metadata: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    valid_fields = [field for field in fields if field in metadata.columns]
    if not valid_fields:
        return np.zeros((len(metadata), 0), dtype=object)
    return metadata[valid_fields].astype(str).to_numpy(dtype=object)


def run_patient_cv(
    x: np.ndarray,
    metadata: pd.DataFrame,
    y: np.ndarray,
    context_values: np.ndarray,
    aux_context_values: np.ndarray,
    methods: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    groups = metadata["donor_id"].astype(str).to_numpy()
    n_folds = infer_fold_count(metadata, "label", args.n_folds)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    weight_diagnostic_rows: List[Dict[str, object]] = []
    epoch_diagnostic_rows: List[Dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx]).astype(np.float32)
        x_test = scaler.transform(x[test_idx]).astype(np.float32)
        for method in methods:
            print(f"[mitigation] patient_cv fold={fold} method={method}", flush=True)
            pred = train_predict(
                x_train,
                y[train_idx],
                context_values[train_idx],
                x_test,
                context_values[test_idx],
                method,
                args,
                seed=args.seed + fold * 100 + METHODS.index(method),
                aux_context_train=aux_context_values[train_idx],
                donor_train=metadata["donor_id"].astype(str).to_numpy()[train_idx],
                y_eval=y[test_idx],
                diagnostic_prefix={
                    "fold": fold,
                    "split_type": "patient_level_cv",
                    "context_field": args.context_field,
                    "context_value": "fold_test",
                    "seed": int(args.seed),
                    "train_seed": int(args.seed + fold * 100 + METHODS.index(method)),
                },
                weight_diagnostic_rows=weight_diagnostic_rows,
                epoch_diagnostic_rows=epoch_diagnostic_rows,
            )
            fold_rows.append(
                metric_row(
                    y[test_idx],
                    pred,
                    {
                        "method": method,
                        "fold": fold,
                        "split_type": "patient_level_cv",
                        "n_train_cells": int(len(train_idx)),
                        "n_test_cells": int(len(test_idx)),
                    },
                )
            )
            append_prediction_rows(
                pred_rows,
                metadata,
                test_idx,
                y[test_idx],
                pred,
                method,
                {"fold": fold, "split_type": "patient_level_cv"},
                args.context_fields,
            )
    subgroup_rows, gap_rows = summarize_method_predictions(pred_rows, args.context_fields, "patient_level_cv")
    return fold_rows, subgroup_rows, gap_rows, pred_rows, weight_diagnostic_rows, epoch_diagnostic_rows


def run_leave_one_context(
    x: np.ndarray,
    metadata: pd.DataFrame,
    y: np.ndarray,
    context_values: np.ndarray,
    aux_context_values: np.ndarray,
    methods: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    fields = args.leave_one_context_fields or [args.context_field]
    metric_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    weight_diagnostic_rows: List[Dict[str, object]] = []
    epoch_diagnostic_rows: List[Dict[str, object]] = []

    for field in fields:
        if field not in metadata.columns:
            skipped_rows.append({"context_field": field, "context_value": "ALL", "reason": "missing_field"})
            continue
        for context_value in sorted(metadata[field].dropna().astype(str).unique()):
            test_mask = metadata[field].astype(str).to_numpy() == context_value
            test_idx = np.flatnonzero(test_mask)
            train_idx = np.flatnonzero(~test_mask)
            prefix = {"split_type": "leave_one_context", "context_field": field, "context_value": context_value}
            if len(test_idx) < args.min_holdout_cells:
                skipped_rows.append({**prefix, "reason": "too_few_holdout_cells", "n_test_cells": int(len(test_idx))})
                continue
            if len(np.unique(y[test_idx])) < args.min_holdout_labels:
                skipped_rows.append(
                    {
                        **prefix,
                        "reason": "too_few_holdout_labels",
                        "n_test_cells": int(len(test_idx)),
                        "n_test_labels": int(len(np.unique(y[test_idx]))),
                    }
                )
                continue
            if len(np.unique(y[train_idx])) < 2:
                skipped_rows.append({**prefix, "reason": "too_few_train_labels", "n_train_cells": int(len(train_idx))})
                continue
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx]).astype(np.float32)
            x_test = scaler.transform(x[test_idx]).astype(np.float32)
            for method in methods:
                print(f"[mitigation] leave_one field={field} value={context_value} method={method}", flush=True)
                train_seed = int(args.seed + len(metric_rows) * 17 + METHODS.index(method))
                pred = train_predict(
                    x_train,
                    y[train_idx],
                    context_values[train_idx],
                    x_test,
                    context_values[test_idx],
                    method,
                    args,
                    seed=train_seed,
                    aux_context_train=aux_context_values[train_idx],
                    donor_train=metadata["donor_id"].astype(str).to_numpy()[train_idx],
                    y_eval=y[test_idx],
                    diagnostic_prefix={
                        **prefix,
                        "seed": int(args.seed),
                        "train_seed": train_seed,
                    },
                    weight_diagnostic_rows=weight_diagnostic_rows,
                    epoch_diagnostic_rows=epoch_diagnostic_rows,
                )
                metric_rows.append(
                    metric_row(
                        y[test_idx],
                        pred,
                        {
                            **prefix,
                            "method": method,
                            "n_train_cells": int(len(train_idx)),
                            "n_test_cells": int(len(test_idx)),
                        },
                    )
                )
                append_prediction_rows(pred_rows, metadata, test_idx, y[test_idx], pred, method, prefix, args.context_fields)

    gap_rows: List[Dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in metric_rows}):
        method_rows = [row for row in metric_rows if row["method"] == method]
        method_gaps = summarize_metric_gaps(method_rows)
        for row in method_gaps:
            row["method"] = method
        gap_rows.extend(method_gaps)
    return metric_rows, gap_rows, pred_rows, skipped_rows, weight_diagnostic_rows, epoch_diagnostic_rows


def main() -> None:
    args = parse_args()
    unknown = [method for method in args.methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {METHODS}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = normalize_metadata(pd.read_csv(args.metadata_csv), args.label_column)
    if args.context_field not in metadata.columns:
        raise ValueError(f"Missing context field: {args.context_field}")
    x = read_embedding(args.embedding_file, args.embedding_key).astype(np.float32)
    if x.shape[0] != len(metadata):
        raise ValueError(f"Embedding row count {x.shape[0]} != metadata rows {len(metadata)}")
    y = metadata["label"].astype(str).to_numpy()
    context_values = metadata[args.context_field].astype(str).to_numpy()
    aux_regularizer_context_fields = [
        field for field in args.aux_regularizer_context_fields if field != args.context_field
    ]
    aux_context_values = context_value_matrix(metadata, aux_regularizer_context_fields)

    (
        fold_rows,
        subgroup_rows,
        gap_rows,
        pred_rows,
        patient_weight_diagnostics,
        patient_epoch_diagnostics,
    ) = run_patient_cv(
        x,
        metadata,
        y,
        context_values,
        aux_context_values,
        args.methods,
        args,
    )
    (
        leave_metrics,
        leave_gaps,
        leave_preds,
        leave_skipped,
        leave_weight_diagnostics,
        leave_epoch_diagnostics,
    ) = run_leave_one_context(
        x,
        metadata,
        y,
        context_values,
        aux_context_values,
        args.methods,
        args,
    )
    association_rows, context_count_rows = label_context_audit(metadata, "label", args.context_fields)
    support_summary_rows, support_count_rows = support_coverage_tables(
        metadata,
        "label",
        sorted(set(args.context_fields) | {args.context_field}),
        args.sabca_min_group_size,
        args.support_min_donors,
    )

    write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "predictions.csv", pred_rows)
    write_csv(args.output_dir / "subgroup_metrics.csv", subgroup_rows)
    write_csv(args.output_dir / "subgroup_gaps.csv", gap_rows)
    write_csv(args.output_dir / "leave_one_context_metrics.csv", leave_metrics)
    write_csv(args.output_dir / "leave_one_context_gaps.csv", leave_gaps)
    write_csv(args.output_dir / "leave_one_context_predictions.csv", leave_preds)
    write_csv(args.output_dir / "leave_one_context_skipped.csv", leave_skipped)
    write_csv(args.output_dir / "label_context_association.csv", association_rows)
    write_csv(args.output_dir / "label_context_counts.csv", context_count_rows)
    write_csv(args.output_dir / "support_coverage.csv", support_summary_rows)
    write_csv(args.output_dir / "support_label_context_counts.csv", support_count_rows)
    write_csv(args.output_dir / "training_weight_diagnostics.csv", patient_weight_diagnostics + leave_weight_diagnostics)
    write_csv(args.output_dir / "training_epoch_diagnostics.csv", patient_epoch_diagnostics + leave_epoch_diagnostics)

    summary = {
        "model_name": args.model_name,
        "embedding_file": str(args.embedding_file),
        "metadata_csv": str(args.metadata_csv),
        "embedding_shape": [int(x.shape[0]), int(x.shape[1])],
        "context_field": args.context_field,
        "aux_regularizer_context_fields": aux_regularizer_context_fields,
        "methods": args.methods,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "support_min_samples": int(args.sabca_min_group_size),
        "support_min_donors": int(args.support_min_donors),
        "patient_level_gaps": gap_rows,
        "leave_one_context_gaps": leave_gaps,
        "leave_one_context_skipped": leave_skipped,
        "support_coverage": support_summary_rows,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
