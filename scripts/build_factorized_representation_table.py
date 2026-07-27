#!/usr/bin/env python3
"""Build paper-ready rows for factorized representation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


TASK_LABELS = {
    "geneformer_bone_marrow_assay": ("Geneformer", "assay"),
    "geneformer_bone_marrow_dataset_id": ("Geneformer", "dataset"),
    "scgpt_bone_marrow_assay": ("scGPT", "assay"),
    "scgpt_bone_marrow_dataset_id": ("scGPT", "dataset"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/cellxgene_factorized_representation_diagnostics"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_tables"))
    return parser.parse_args()


def collect(input_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted(input_root.glob("*/summary_factorized_representation_diagnostics.csv")):
        task_key = summary_path.parent.name
        model_label, context_label = TASK_LABELS.get(task_key, (task_key, ""))
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            method = str(row["method"])
            representation = str(row["representation"])
            if method == "frozen_h":
                display = "frozen $h$"
            elif representation == "z_y":
                display = f"{method} $z^y$"
            elif representation == "z_c":
                display = f"{method} $z^c$"
            else:
                display = f"{method} {representation}"
            rows.append(
                {
                    "task_key": task_key,
                    "model": model_label,
                    "context": context_label,
                    "method": method,
                    "representation": representation,
                    "display": display,
                    "task_ba": row.get("task_ba_mean"),
                    "supported_task_ba": row.get("supported_task_ba_mean"),
                    "context_probe_ba": row.get("context_probe_ba_mean"),
                    "supported_context_probe_ba": row.get("supported_context_probe_ba_mean"),
                    "delta_supported_context_probe_vs_frozen": row.get("delta_supported_context_probe_vs_frozen_mean"),
                    "n_folds": int(row.get("n_folds", 0)),
                }
            )
    return rows


def fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "--"
    if pd.isna(val):
        return "--"
    return f"{val:.3f}"


def latex_rows(rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    order = {"frozen_h": 0, "factorized_erm": 1, "factorized_sabca": 2}
    rep_order = {"frozen_h": 0, "z_y": 1, "z_c": 2}
    for row in sorted(rows, key=lambda r: (str(r["task_key"]), order.get(str(r["method"]), 99), rep_order.get(str(r["representation"]), 99))):
        task = f"{row['model']} {row['context']}"
        lines.append(
            task
            + " & "
            + str(row["display"])
            + " & "
            + fmt(row["task_ba"])
            + " & "
            + fmt(row["context_probe_ba"])
            + " & "
            + fmt(row["supported_context_probe_ba"])
            + " & "
            + fmt(row["delta_supported_context_probe_vs_frozen"])
            + r" \\"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.input_root)
    pd.DataFrame(rows).to_csv(args.output_dir / "factorized_representation_diagnostic_table.csv", index=False)
    tex = latex_rows(rows)
    (args.output_dir / "factorized_representation_diagnostic_rows.tex").write_text(tex, encoding="utf-8")
    summary = {"input_root": str(args.input_root), "output_dir": str(args.output_dir), "n_rows": len(rows), "tex_rows": tex}
    (args.output_dir / "factorized_representation_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
