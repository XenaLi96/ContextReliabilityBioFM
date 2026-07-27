#!/usr/bin/env python3
"""Plot the semi-synthetic support-identifiability curve."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    method_order = ["erm_mlp", "label_context_reweight", "adv_context", "sabca"]
    labels = {
        "erm_mlp": "ERM",
        "label_context_reweight": "Reweight",
        "adv_context": "DANN",
        "sabca": "SCRA-full",
    }
    colors = {
        "erm_mlp": "#555555",
        "label_context_reweight": "#0072B2",
        "adv_context": "#D55E00",
        "sabca": "#009E73",
    }
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for method in method_order:
        sub = df[df["method"] == method].sort_values("actual_support_coverage_ge20")
        if sub.empty:
            continue
        ax.plot(
            sub["actual_support_coverage_ge20"],
            sub["worst_context_ba"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=colors[method],
            label=labels[method],
        )
    ax.set_xlabel("Observed label-context support coverage")
    ax.set_ylabel("Worst-context balanced accuracy")
    ax.set_ylim(0.25, 0.82)
    ax.set_xlim(0.18, 0.60)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output)
    if args.output.suffix.lower() != ".png":
        fig.savefig(args.output.with_suffix(".png"), dpi=300)


if __name__ == "__main__":
    main()
