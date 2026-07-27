#!/usr/bin/env python3
"""Build paper-ready tables from CELLxGENE external-baseline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


METHOD_LABELS = {
    "erm_mlp": "ERM",
    "label_context_reweight": "Reweight",
    "adv_context": "DANN",
    "group_dro": "GroupDRO",
    "cond_mmd": "MMD",
    "cond_coral": "CORAL",
    "irm": "IRM",
    "fishr": "Fishr",
    "harmony_style": "Harmony-style",
    "sabca": "SABCA",
    "scea": "SCEA",
    "scea_no_episode": "no episode",
    "scea_no_support_gate": "no support gate",
    "scea_no_alignment": "no alignment",
    "scea_no_cvar": "no CVaR",
    "sca_lite": "SCA-lite",
    "sca_mmd": "MMD only",
    "sca_coral": "CORAL only",
    "sca_supcon": "SupCon only",
    "sca_soft_dro": "soft DRO",
    "sca_soft_cvar": "soft CVaR",
    "sca_multi_lite": "multi SCA",
    "sca_multi_soft_dro": "multi soft DRO",
    "reweight_plus": "Reweight++",
    "lc_reweight_pow085": "STDR-0.85",
    "lc_reweight_pow09": "STDR-0.90",
    "lc_reweight_pow095": "STDR-0.95",
    "stdr_pow085": "STDR-0.85",
    "stdr_pow09": "STDR-0.90",
    "stdr_pow095": "STDR-0.95",
}

DEFAULT_METHOD_ORDER = [
    "erm_mlp",
    "label_context_reweight",
    "adv_context",
    "group_dro",
    "cond_mmd",
    "cond_coral",
    "irm",
    "fishr",
    "harmony_style",
    "sabca",
]

SCEA_METHOD_ORDER = [
    *DEFAULT_METHOD_ORDER,
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
]

TASKS = [
    ("geneformer_v1", "assay", "Geneformer", "assay"),
    ("geneformer_v1", "dataset_id", "Geneformer", "dataset"),
    ("scgpt_continual", "assay", "scGPT", "assay"),
    ("scgpt_continual", "dataset_id", "scGPT", "dataset"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/cellxgene_external_baselines"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_tables"))
    parser.add_argument("--method-order", nargs="*", default=DEFAULT_METHOD_ORDER)
    return parser.parse_args()


def load_task(input_root: Path, model: str, context: str, model_label: str, context_label: str) -> pd.DataFrame:
    path = input_root / model / context / "summary" / "aggregate_gaps.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["summary_metric"].astype(str).isin(["gap", "worst_value"])].copy()
    if df.empty:
        return df
    pivot = df.pivot_table(
        index=["split_type", "method", "n_seeds"],
        columns="summary_metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    pivot["model"] = model
    pivot["context"] = context
    pivot["model_label"] = model_label
    pivot["context_label"] = context_label
    return pivot


def format_cell(gap: float, worst: float) -> str:
    return f"{gap:.3f} / {worst:.3f}"


def build_rows(df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if df.empty:
        return rows
    for _, row in df.iterrows():
        method = str(row["method"])
        rows.append(
            {
                "model": row["model"],
                "context": row["context"],
                "model_label": row["model_label"],
                "context_label": row["context_label"],
                "split_type": row["split_type"],
                "split_label": "patient-CV" if row["split_type"] == "patient_level_cv" else "leave-one",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_seeds": int(row["n_seeds"]),
                "gap": float(row["gap"]),
                "worst_ba": float(row["worst_value"]),
                "cell": format_cell(float(row["gap"]), float(row["worst_value"])),
            }
        )
    return rows


def latex_rows(rows: List[Dict[str, object]], method_order: List[str]) -> str:
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    out_lines: List[str] = []
    for model, context, split_type in [
        ("geneformer_v1", "assay", "patient_level_cv"),
        ("geneformer_v1", "assay", "leave_one_context"),
        ("geneformer_v1", "dataset_id", "patient_level_cv"),
        ("geneformer_v1", "dataset_id", "leave_one_context"),
        ("scgpt_continual", "assay", "patient_level_cv"),
        ("scgpt_continual", "assay", "leave_one_context"),
        ("scgpt_continual", "dataset_id", "patient_level_cv"),
        ("scgpt_continual", "dataset_id", "leave_one_context"),
    ]:
        subset = df[(df["model"] == model) & (df["context"] == context) & (df["split_type"] == split_type)]
        if subset.empty:
            continue
        label = f"{subset['model_label'].iloc[0]} {subset['context_label'].iloc[0]} {subset['split_label'].iloc[0]}"
        cell_by_method = {str(row["method"]): str(row["cell"]) for _, row in subset.iterrows()}
        values = [cell_by_method.get(method, "--") for method in method_order]
        out_lines.append(label + " & " + " & ".join(values) + r" \\")
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_task(args.input_root, *task) for task in TASKS]
    df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    rows = build_rows(df)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_dir / "external_baseline_adaptation_table.csv", index=False)
    tex = latex_rows(rows, args.method_order)
    (args.output_dir / "external_baseline_adaptation_rows.tex").write_text(tex, encoding="utf-8")
    summary = {
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "n_rows": int(len(rows)),
        "methods": METHOD_LABELS,
        "method_order": args.method_order,
        "tex_rows": tex,
    }
    (args.output_dir / "external_baseline_adaptation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
