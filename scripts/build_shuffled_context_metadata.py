#!/usr/bin/env python3
"""Create a metadata CSV with one context column shuffled for negative controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--context-field", required=True)
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--within-label", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.context_field not in df.columns:
        raise ValueError(f"Missing context field: {args.context_field}")
    label_column = args.label_column
    if args.within_label and label_column not in df.columns:
        if "label" in df.columns:
            label_column = "label"
        else:
            raise ValueError(f"Missing label column for within-label shuffle: {args.label_column}")

    rng = np.random.default_rng(args.seed)
    out = df.copy()
    if args.within_label:
        shuffled = out[args.context_field].copy()
        for _, index in out.groupby(label_column, dropna=False).groups.items():
            values = out.loc[index, args.context_field].to_numpy(copy=True)
            rng.shuffle(values)
            shuffled.loc[index] = values
        out[args.context_field] = shuffled
    else:
        values = out[args.context_field].to_numpy(copy=True)
        rng.shuffle(values)
        out[args.context_field] = values

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
