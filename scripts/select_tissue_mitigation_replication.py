#!/usr/bin/env python3
"""Select extra-tissue mitigation replications by a fixed support protocol.

The selection is deterministic and independent of mitigation outcomes:

1. use the completed scGPT ten-tissue ERM audit;
2. require at least 80% of cells to lie in support-eligible label-context
   cells, where support means >=20 cells and >=5 distinct donors;
3. require an ERM leave-one-context balanced-accuracy gap >=0.25;
4. rank eligible tissues by the ERM leave-one gap, then support coverage,
   then tissue name;
5. choose one assay and one dataset task.  If the same tissue tops both
   rankings, retain it for the context with the larger leave-one gap and use
   the next-ranked distinct tissue for the other context.

The script writes the complete eligibility table as well as the final selected
tasks so that selection can be audited before model training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


CONTEXT_ALIASES = {"assay": "assay", "dataset_id": "dataset_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("data/paper_tables/scgpt_ten_tissue_context_bins.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tissue_mitigation_replication_selection"),
    )
    parser.add_argument("--min-support-coverage", type=float, default=0.8)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-donors", type=int, default=5)
    parser.add_argument("--min-leave-one-gap", type=float, default=0.25)
    parser.add_argument("--exclude-tissues", nargs="*", default=["bone_marrow"])
    return parser.parse_args()


def support_profile(
    metadata: pd.DataFrame,
    label_column: str,
    context_column: str,
    min_samples: int,
    min_donors: int,
) -> Dict[str, object]:
    required = {label_column, context_column, "donor_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata missing columns {missing}")

    rows = metadata[[label_column, context_column, "donor_id"]].astype(str).copy()
    counts = (
        rows.groupby([label_column, context_column], sort=True)
        .agg(n_samples=("donor_id", "size"), n_donors=("donor_id", "nunique"))
        .reset_index()
    )
    counts["supported_cell"] = (
        (counts["n_samples"] >= int(min_samples))
        & (counts["n_donors"] >= int(min_donors))
    )
    supported_contexts = (
        counts.loc[counts["supported_cell"]]
        .groupby(label_column, sort=False)[context_column]
        .nunique()
    )
    counts["supported_label"] = counts[label_column].map(supported_contexts).fillna(0).astype(int) >= 2
    counts["support_eligible"] = counts["supported_cell"] & counts["supported_label"]

    eligible_pairs = counts.loc[counts["support_eligible"], [label_column, context_column]].copy()
    eligible_pairs["eligible"] = True
    rows = rows.merge(eligible_pairs, on=[label_column, context_column], how="left")
    eligible = rows["eligible"].fillna(False).astype(bool)

    return {
        "n_cells": int(len(rows)),
        "n_donors": int(rows["donor_id"].nunique()),
        "n_labels": int(rows[label_column].nunique()),
        "n_contexts": int(rows[context_column].nunique()),
        "support_eligible_cells": int(eligible.sum()),
        "support_coverage": float(eligible.mean()) if len(eligible) else 0.0,
        "supported_pairs": int(counts["support_eligible"].sum()),
        "total_pairs": int(len(counts)),
        "min_supported_pair_donors": (
            int(counts.loc[counts["support_eligible"], "n_donors"].min())
            if bool(counts["support_eligible"].any())
            else 0
        ),
    }


def load_summary_path(source_dir: str) -> Path:
    return Path(source_dir) / "summary.json"


def build_eligibility_table(args: argparse.Namespace) -> pd.DataFrame:
    audit = pd.read_csv(args.audit_csv)
    audit = audit.loc[audit["context"].isin(CONTEXT_ALIASES)].copy()
    audit["tissue_key"] = audit["dataset"].astype(str).str.split(":", n=1).str[-1]
    audit = audit.loc[~audit["tissue_key"].isin(set(args.exclude_tissues))].copy()
    records: List[Dict[str, object]] = []
    for row in audit.to_dict(orient="records"):
        summary_path = load_summary_path(str(row["source"]))
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        metadata_path = Path(summary["metadata_csv"])
        metadata = pd.read_csv(metadata_path)
        context = CONTEXT_ALIASES[str(row["context"])]
        profile = support_profile(
            metadata,
            label_column="label",
            context_column=context,
            min_samples=args.min_samples,
            min_donors=args.min_donors,
        )
        record = {
            "tissue": str(row["dataset"]).split(":", 1)[-1],
            "context": context,
            "model": str(row["embedding"]),
            "metadata_csv": str(metadata_path),
            "embedding_file": str(summary["embedding_file"]),
            "erm_leave_one_gap": float(row["leave_one_gap"]),
            "erm_leave_one_worst_ba": float(row["leave_one_worst_ba"]),
            **profile,
        }
        record["eligible"] = bool(
            record["support_coverage"] >= args.min_support_coverage
            and record["min_supported_pair_donors"] >= args.min_donors
            and record["erm_leave_one_gap"] >= args.min_leave_one_gap
        )
        records.append(record)
    result = pd.DataFrame.from_records(records)
    return result.sort_values(
        ["context", "eligible", "erm_leave_one_gap", "support_coverage", "tissue"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)


def ranked_candidates(table: pd.DataFrame, context: str) -> List[Dict[str, object]]:
    rows = table.loc[(table["context"] == context) & table["eligible"]].copy()
    rows = rows.sort_values(
        ["erm_leave_one_gap", "support_coverage", "tissue"],
        ascending=[False, False, True],
    )
    return rows.to_dict(orient="records")


def choose_distinct_tasks(table: pd.DataFrame) -> List[Dict[str, object]]:
    assay = ranked_candidates(table, "assay")
    dataset = ranked_candidates(table, "dataset_id")
    if not assay or not dataset:
        raise RuntimeError("No eligible assay or dataset replication task under the fixed thresholds")
    assay_pick = assay[0]
    dataset_pick = dataset[0]
    if assay_pick["tissue"] != dataset_pick["tissue"]:
        return [assay_pick, dataset_pick]

    if float(assay_pick["erm_leave_one_gap"]) >= float(dataset_pick["erm_leave_one_gap"]):
        dataset_pick = next((row for row in dataset if row["tissue"] != assay_pick["tissue"]), None)
    else:
        assay_pick = next((row for row in assay if row["tissue"] != dataset_pick["tissue"]), None)
    if assay_pick is None or dataset_pick is None:
        raise RuntimeError("Fixed distinct-tissue rule cannot be satisfied under the eligibility thresholds")
    return [assay_pick, dataset_pick]


def main() -> None:
    args = parse_args()
    table = build_eligibility_table(args)
    selected = pd.DataFrame.from_records(choose_distinct_tasks(table))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "eligibility_table.csv", index=False)
    all_eligible = table.loc[table["eligible"]].copy()
    all_eligible.to_csv(args.output_dir / "all_eligible_tasks.csv", index=False)
    selected.to_csv(args.output_dir / "selected_tasks.csv", index=False)
    protocol = {
        "audit_csv": str(args.audit_csv),
        "min_support_coverage": float(args.min_support_coverage),
        "min_samples_per_label_context": int(args.min_samples),
        "min_donors_per_label_context": int(args.min_donors),
        "min_erm_leave_one_gap": float(args.min_leave_one_gap),
        "excluded_reference_tissues": list(args.exclude_tissues),
        "selection_rule": "rank by ERM leave-one gap, support coverage, tissue; enforce distinct tissues deterministically",
        "all_eligible": all_eligible.to_dict(orient="records"),
        "selected": selected.to_dict(orient="records"),
    }
    with (args.output_dir / "selection_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
    print(selected[["tissue", "context", "support_coverage", "erm_leave_one_gap"]].to_string(index=False))


if __name__ == "__main__":
    main()
