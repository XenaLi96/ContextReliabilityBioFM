#!/usr/bin/env python3
"""Build the four-method main benchmark table from formal aggregate artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


METHODS = ["erm_mlp", "label_context_reweight", "sca_lite", "group_dro"]
MODEL_LABELS = {
    "geneformer_v1": "Geneformer",
    "scgpt_continual": "scGPT",
    "scvi_style_vae": "scVI-style",
}
CONTEXT_LABELS = {"assay": "assay", "dataset_id": "dataset"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/cellxgene_support_calibrated_formal"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/paper_tables/support_calibrated_main_benchmark.csv"))
    parser.add_argument("--output-tex", type=Path, default=Path("tables/main_intervention_rows.tex"))
    parser.add_argument(
        "--cross-tissue-output-csv",
        type=Path,
        default=Path("data/paper_tables/cross_tissue_intervention_benchmark.csv"),
    )
    parser.add_argument(
        "--cross-tissue-output-tex",
        type=Path,
        default=Path("tables/cross_tissue_intervention_rows.tex"),
    )
    return parser.parse_args()


def path_metadata(path: Path, root: Path) -> Tuple[str, str, str]:
    relative = path.relative_to(root).parts
    if len(relative) < 5:
        raise ValueError(f"Unexpected aggregate path: {path}")
    return relative[0], relative[1], relative[2]


def load(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(root.glob("**/summary/aggregate_gaps.csv")):
        task, model, context = path_metadata(path, root)
        frame = pd.read_csv(path)
        frame = frame.loc[frame["method"].isin(METHODS) & frame["summary_metric"].isin(["worst_value", "gap"])]
        pivot = frame.pivot_table(index=["split_type", "method"], columns="summary_metric", values="mean", aggfunc="first").reset_index()
        for row in pivot.to_dict(orient="records"):
            rows.append({"task": task, "model": model, "context_field": context, **row})
    if not rows:
        raise FileNotFoundError(f"No formal aggregate tables under {root}")
    return pd.DataFrame.from_records(rows)


def row_label(row: pd.Series) -> str:
    task = str(row["task"])
    model = MODEL_LABELS.get(str(row["model"]), str(row["model"]))
    context = CONTEXT_LABELS.get(str(row["context_field"]), str(row["context_field"]))
    if task.startswith("bone_marrow"):
        return f"{model} / {context}"
    tissue = task.replace("_assay", "").replace("_dataset", "").replace("_", " ").title()
    return f"{tissue} / {model} / {context}"


def format_cell(worst: float, gap: float, bold: bool) -> str:
    cell = f"{worst:.3f} / {gap:.3f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def task_records(frame: pd.DataFrame, split: str, extra: bool) -> List[Dict[str, object]]:
    subset = frame.loc[frame["split_type"] == split].copy()
    is_extra = ~subset["task"].astype(str).str.startswith("bone_marrow")
    subset = subset.loc[is_extra if extra else ~is_extra]
    records: List[Dict[str, object]] = []
    for keys, group in subset.groupby(["task", "model", "context_field"], sort=True):
        if set(group["method"]) != set(METHODS):
            continue
        by_method = group.set_index("method")
        best = group.sort_values(["worst_value", "gap"], ascending=[False, True]).iloc[0]["method"]
        row = {"task": keys[0], "model": keys[1], "context_field": keys[2], "split_type": split}
        row["task_label"] = row_label(pd.Series(row))
        for method in METHODS:
            row[f"{method}_worst"] = float(by_method.loc[method, "worst_value"])
            row[f"{method}_gap"] = float(by_method.loc[method, "gap"])
            row[f"{method}_cell"] = format_cell(
                row[f"{method}_worst"], row[f"{method}_gap"], method == best
            )
        records.append(row)
    preferred = {("geneformer_v1", "assay"): 0, ("geneformer_v1", "dataset_id"): 1,
                 ("scgpt_continual", "assay"): 2, ("scgpt_continual", "dataset_id"): 3}
    if extra:
        model_order = {"geneformer_v1": 0, "scgpt_continual": 1, "scvi_style_vae": 2}
        records.sort(
            key=lambda row: (
                str(row["task"]),
                model_order.get(str(row["model"]), 9),
            )
        )
    else:
        records.sort(key=lambda row: preferred.get((row["model"], row["context_field"]), 10))
    return records


def write_tex(
    blocks: List[Tuple[str, List[Dict[str, object]]]], path: Path, macro_name: str
) -> None:
    lines: List[str] = []
    for title, rows in blocks:
        if not rows:
            continue
        if lines:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{title}}}}} \\")
        for row in rows:
            cells = [row["task_label"]] + [row[f"{method}_cell"] for method in METHODS]
            lines.append(" & ".join(cells) + r" \\")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "%\n".join(lines)
    path.write_text(f"\\newcommand{{\\{macro_name}}}{{%\n" + body + "%\n}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = load(args.input_root)
    main_blocks = [
        ("Observed-context patient-CV", task_records(frame, "patient_level_cv", extra=False)),
        ("Held-out-context leave-one", task_records(frame, "leave_one_context", extra=False)),
    ]
    cross_blocks = [
        ("Support-gated cross-tissue leave-one", task_records(frame, "leave_one_context", extra=True)),
    ]
    main_records = [row for _, rows in main_blocks for row in rows]
    cross_records = [row for _, rows in cross_blocks for row in rows]
    output = pd.DataFrame.from_records(main_records)
    cross_output = pd.DataFrame.from_records(cross_records)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    cross_output.to_csv(args.cross_tissue_output_csv, index=False)
    write_tex(main_blocks, args.output_tex, "maininterventionrows")
    write_tex(cross_blocks, args.cross_tissue_output_tex, "crosstissueinterventionrows")
    print(f"wrote {args.output_csv} ({len(output)} rows)")
    print(f"wrote {args.output_tex}")
    print(f"wrote {args.cross_tissue_output_csv} ({len(cross_output)} rows)")
    print(f"wrote {args.cross_tissue_output_tex}")


if __name__ == "__main__":
    main()
