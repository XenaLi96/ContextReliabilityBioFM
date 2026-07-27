#!/usr/bin/env python3
"""Aggregate multi-seed semi-synthetic support-identifiability runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "erm_mlp": "ERM",
    "label_context_reweight": "LC-Reweight",
    "sca_lite": "SCA-Align",
    "adv_context": "DANN",
    "sabca": "support-gated adapter prototype",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_tables"))
    parser.add_argument("--glob", default="seed_*/support_curve_results.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260630)
    return parser.parse_args()


def finite_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def percentile_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(val):
        return "--"
    return f"{val:.{digits}f}"


def load_rows(input_root: Path, pattern: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(input_root.glob(pattern)):
        frame = pd.read_csv(path)
        seed = path.parent.name[len("seed_") :] if path.parent.name.startswith("seed_") else path.parent.name
        frame["seed"] = seed
        frame["source_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No support result files matched {input_root / pattern}")
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["target_support_fraction", "method"]
    for (target, method), sub in df.groupby(group_cols, dropna=False):
        worst = pd.to_numeric(sub["worst_context_ba"], errors="coerce").to_numpy(dtype=float)
        gap = pd.to_numeric(sub["gap"], errors="coerce").to_numpy(dtype=float)
        overall = pd.to_numeric(sub["overall_ba"], errors="coerce").to_numpy(dtype=float)
        coverage = pd.to_numeric(sub["actual_support_coverage_ge20"], errors="coerce").to_numpy(dtype=float)
        worst_lo, worst_hi = percentile_ci(worst, rng, n_boot)
        gap_lo, gap_hi = percentile_ci(gap, rng, n_boot)
        rows.append(
            {
                "target_support_fraction": float(target),
                "actual_support_coverage_mean": float(np.nanmean(coverage)),
                "method": method,
                "method_label": METHOD_LABELS.get(str(method), str(method)),
                "n_seeds": int(sub["seed"].nunique()),
                "overall_ba_mean": finite_mean(sub["overall_ba"]),
                "worst_context_ba_mean": float(np.nanmean(worst)),
                "worst_context_ba_ci_low": worst_lo,
                "worst_context_ba_ci_high": worst_hi,
                "gap_mean": float(np.nanmean(gap)),
                "gap_ci_low": gap_lo,
                "gap_ci_high": gap_hi,
                "train_cells_mean": finite_mean(sub["train_cells"]),
                "test_cells_mean": finite_mean(sub["test_cells"]),
                "supported_pairs_ge20_mean": finite_mean(sub["supported_pairs_ge20"]),
                "total_pairs_mean": finite_mean(sub["total_pairs"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_support_fraction", "method_label"])


def best_by_level(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for _, sub in summary.groupby("target_support_fraction", dropna=False):
        rows.append(sub.loc[sub["worst_context_ba_mean"].astype(float).idxmax()])
    return pd.DataFrame(rows).sort_values("target_support_fraction")


def write_tex(summary: pd.DataFrame, output_path: Path) -> str:
    method_order = ["ERM", "LC-Reweight", "SCA-Align", "DANN", "support-gated adapter prototype"]
    lines: List[str] = []
    for target, sub in summary.groupby("target_support_fraction", dropna=False):
        by_method = {str(row.method_label): row for row in sub.itertuples(index=False)}
        cells = [fmt(target)]
        for label in method_order:
            row = by_method.get(label)
            if row is None:
                cells.append("--")
            else:
                cells.append(f"{fmt(row.worst_context_ba_mean)} [{fmt(row.worst_context_ba_ci_low)}, {fmt(row.worst_context_ba_ci_high)}] / {fmt(row.gap_mean)}")
        lines.append(" & ".join(cells) + r" \\")
    text = "\n".join(lines) + ("\n" if lines else "")
    output_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = load_rows(args.input_root, args.glob)
    summary = aggregate(df, rng, args.bootstrap_samples)
    best = best_by_level(summary)

    summary_path = args.output_dir / "support_identifiability_multiseed_table.csv"
    best_path = args.output_dir / "support_identifiability_multiseed_best_by_level.csv"
    rows_path = args.output_dir / "support_identifiability_multiseed_rows.tex"
    json_path = args.output_dir / "support_identifiability_multiseed_summary.json"
    summary.to_csv(summary_path, index=False)
    best.to_csv(best_path, index=False)
    tex_rows = write_tex(summary, rows_path)
    payload = {
        "input_root": str(args.input_root),
        "n_raw_rows": int(len(df)),
        "n_summary_rows": int(len(summary)),
        "n_seed_dirs": int(df["seed"].nunique()),
        "summary_path": str(summary_path),
        "best_by_level_path": str(best_path),
        "tex_rows_path": str(rows_path),
        "tex_rows": tex_rows,
        "best_by_level": best.to_dict("records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
