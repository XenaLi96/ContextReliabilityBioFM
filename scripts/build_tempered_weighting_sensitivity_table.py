#!/usr/bin/env python3
"""Build the appendix alpha/ESS/weight-ratio table without method acronyms."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


METHOD_ALPHA = {
    "stdr_pow085": 0.85,
    "stdr_pow09": 0.90,
    "stdr_pow095": 0.95,
    "label_context_reweight": 1.00,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/cellxgene_stdr_formal"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/paper_tables/tempered_weighting_sensitivity.csv"))
    parser.add_argument("--output-tex", type=Path, default=Path("tables/tempered_weighting_rows.tex"))
    return parser.parse_args()


def display_model(value: str) -> str:
    return "Geneformer" if "geneformer" in value.lower() else "scGPT"


def display_context(value: str) -> str:
    return "dataset" if value == "dataset_id" else value


def display_split(value: str) -> str:
    return "patient-CV" if value == "patient_level_cv" else "leave-one"


def load_performance(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/*/summary/aggregate_gaps.csv")):
        model, context = path.relative_to(root).parts[:2]
        frame = pd.read_csv(path)
        frame = frame.loc[frame["method"].isin(METHOD_ALPHA) & (frame["summary_metric"] == "worst_value")]
        for row in frame.to_dict(orient="records"):
            rows.append({
                "model": model,
                "context_field": context,
                "split_type": row["split_type"],
                "method": row["method"],
                "alpha": METHOD_ALPHA[str(row["method"])],
                "worst_ba": float(row["mean"]),
            })
    if not rows:
        raise FileNotFoundError(f"No aggregate gap tables found under {root}")
    return pd.DataFrame.from_records(rows)


def load_diagnostics(root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(root.glob("*/*/seed_*/*/training_weight_diagnostics.csv")):
        model, context = path.relative_to(root).parts[:2]
        frame = pd.read_csv(path)
        frame = frame.loc[frame["method"].isin(METHOD_ALPHA)].copy()
        frame["model"] = model
        frame["context_field"] = context
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No training-weight diagnostics found under {root}")
    data = pd.concat(frames, ignore_index=True)
    return data.groupby(["model", "context_field", "split_type", "method"], sort=True).agg(
        ess_fraction_median=("ess_fraction", "median"),
        weight_max_min_ratio_median=("weight_max_min_ratio", "median"),
    ).reset_index()


def format_ratio(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value >= 1000:
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / (10 ** exponent)
        return f"${mantissa:.1f}\\times 10^{{{exponent}}}$"
    return f"{value:.1f}"


def write_tex(frame: pd.DataFrame, output: Path) -> None:
    lines: List[str] = []
    task_keys = ["model", "context_field", "split_type"]
    for task, group in frame.groupby(task_keys, sort=True):
        model, context, split = task
        by_alpha = {float(row.alpha): row for row in group.itertuples(index=False)}
        cells = [f"{display_model(model)} {display_context(context)} {display_split(split)}"]
        for alpha in [0.85, 0.90, 0.95, 1.00]:
            row = by_alpha.get(alpha)
            if row is None:
                cells.append("--")
            else:
                cells.append(
                    f"{row.worst_ba:.3f} / {row.ess_fraction_median:.3f} / "
                    f"{format_ratio(float(row.weight_max_min_ratio_median))}"
                )
        lines.append(" & ".join(cells) + r" \\")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    performance = load_performance(args.input_root)
    diagnostics = load_diagnostics(args.input_root)
    result = performance.merge(
        diagnostics,
        on=["model", "context_field", "split_type", "method"],
        how="left",
        validate="one_to_one",
    ).sort_values(["model", "context_field", "split_type", "alpha"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    write_tex(result, args.output_tex)
    print(f"wrote {args.output_csv} ({len(result)} rows)")
    print(f"wrote {args.output_tex}")


if __name__ == "__main__":
    main()
