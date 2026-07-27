#!/usr/bin/env python3
"""Build paper-ready residualized control summary tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/residualized_embedding_controls"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_tables"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(val):
        return "--"
    return f"{val:.3f}"


def task_label(path: Path, summary: Dict[str, object]) -> str:
    mode = str(summary.get("mode", ""))
    label = str(summary.get("label_column", "label"))
    model = str(summary.get("model_name", path.parent.name))
    if mode == "cellxgene":
        context = path.parent.name
        return f"CELLxGENE bone marrow {model} {context}"
    if label == "TP53_status":
        task = "LUAD TP53"
    elif label == "KRAS_status":
        task = "LUAD KRAS"
    elif label == "IDH_status":
        task = "LGG IDH"
    else:
        task = label
    return f"{task} {model}"


def collect(input_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted(input_root.glob("**/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = summary_path.parent
        table = read_csv(run_dir / "residualized_control_summary.csv")
        if table.empty:
            continue
        for _, row in table.iterrows():
            out = row.to_dict()
            out.update(
                {
                    "run_dir": str(run_dir),
                    "mode": summary.get("mode"),
                    "model_name": summary.get("model_name"),
                    "label_column": summary.get("label_column"),
                    "task_label": task_label(run_dir, summary),
                    "embedding_dim": summary.get("embedding_shape", [None, None])[1],
                    "artifact_dim": summary.get("artifact_shape", [None, None])[1],
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def delta_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if df.empty:
        return pd.DataFrame()
    key_cols = ["task_label", "context_field"]
    for key, group in df.groupby(key_cols, dropna=False):
        base = group[group["control_type"].astype(str) == "none"]
        artifact = group[group["control_type"].astype(str) == "artifact"]
        label = group[group["control_type"].astype(str) == "label"]
        artifact_label = group[group["control_type"].astype(str) == "artifact_label"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        art_row = artifact.iloc[0] if not artifact.empty else pd.Series(dtype=object)
        label_row = label.iloc[0] if not label.empty else pd.Series(dtype=object)
        both_row = artifact_label.iloc[0] if not artifact_label.empty else pd.Series(dtype=object)
        rows.append(
            {
                "task_label": key[0],
                "context_field": key[1],
                "base_probe_ba": base_row.get("context_probe_ba", np.nan),
                "artifact_resid_probe_ba": art_row.get("context_probe_ba", np.nan),
                "label_resid_probe_ba": label_row.get("context_probe_ba", np.nan),
                "artifact_label_resid_probe_ba": both_row.get("context_probe_ba", np.nan),
                "base_patient_cv_gap": base_row.get("patient_cv_ba_gap", np.nan),
                "artifact_resid_patient_cv_gap": art_row.get("patient_cv_ba_gap", np.nan),
                "base_leave_one_gap": base_row.get("leave_one_ba_gap", np.nan),
                "artifact_resid_leave_one_gap": art_row.get("leave_one_ba_gap", np.nan),
                "run_dir": base_row.get("run_dir", ""),
            }
        )
    return pd.DataFrame(rows)


def write_tex(delta: pd.DataFrame, path: Path) -> str:
    lines: List[str] = []
    if delta.empty:
        path.write_text("", encoding="utf-8")
        return ""
    priority = delta.copy()
    priority["sort_gap"] = pd.to_numeric(priority["base_leave_one_gap"], errors="coerce").fillna(
        pd.to_numeric(priority["base_patient_cv_gap"], errors="coerce")
    )
    priority = priority.sort_values("sort_gap", ascending=False).head(12)
    for _, row in priority.iterrows():
        lines.append(
            f"{row['task_label']} & {row['context_field']} & "
            f"{fmt(row['base_probe_ba'])} $\\rightarrow$ {fmt(row['artifact_resid_probe_ba'])} & "
            f"{fmt(row['label_resid_probe_ba'])} & "
            f"{fmt(row['base_leave_one_gap'])} $\\rightarrow$ {fmt(row['artifact_resid_leave_one_gap'])} \\\\"
        )
    text = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = collect(args.input_root)
    delta = delta_table(raw)
    raw_path = args.output_dir / "residualized_control_summary.csv"
    delta_path = args.output_dir / "residualized_control_delta_table.csv"
    tex_path = args.output_dir / "residualized_control_rows.tex"
    summary_path = args.output_dir / "residualized_control_summary.json"
    raw.to_csv(raw_path, index=False)
    delta.to_csv(delta_path, index=False)
    tex_rows = write_tex(delta, tex_path)
    payload = {
        "input_root": str(args.input_root),
        "n_raw_rows": int(len(raw)),
        "n_delta_rows": int(len(delta)),
        "raw_path": str(raw_path),
        "delta_path": str(delta_path),
        "tex_rows_path": str(tex_path),
        "tex_rows": tex_rows,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
