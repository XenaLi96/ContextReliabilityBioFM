#!/usr/bin/env python3
"""Build focused patient-context hotspot tables for manuscript review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd


PATIENT_CONTEXTS = {"age_group", "sex", "disease"}
AGE_DEVELOPMENT_BINS = {"prenatal", "child", "unknown", "adult_unspecified"}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-table-dir", type=Path, default=Path("data/paper_tables"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/patient_context_focus_tables"))
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def clean_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def finite(value: object) -> bool:
    return math.isfinite(clean_float(value))


def severity_flag(gap: float, worst_ba: float) -> str:
    if finite(gap) and (gap >= 0.25 or (finite(worst_ba) and worst_ba < 0.65)):
        return "high"
    if finite(gap) and (gap >= 0.10 or (finite(worst_ba) and worst_ba < 0.75)):
        return "medium"
    return "low"


def support_flag(support: float) -> str:
    if not finite(support):
        return "unknown_support"
    if support < 0.50:
        return "support_limited"
    if support < 0.75:
        return "partial_support"
    return "well_supported"


def biology_note(row: Mapping[str, object]) -> str:
    context = str(row.get("context", ""))
    dataset = str(row.get("dataset", "")).lower()
    worst = str(row.get("worst_context_value", "")).lower()
    best = str(row.get("best_context_value", "")).lower()
    values = f"{worst} {best} {str(row.get('bin_counts', '')).lower()}"
    if context == "age_group":
        if worst in AGE_DEVELOPMENT_BINS or best in AGE_DEVELOPMENT_BINS or any(v in values for v in AGE_DEVELOPMENT_BINS):
            return "developmental_or_missing_age_shift; interpret separately from adult aging"
        return "adult_age_shift; may reflect aging biology and donor composition"
    if context == "sex":
        return "sex_biology_and_sampling; check immune/hormonal biology, support, and tissue composition"
    if context == "disease":
        if "bone_marrow" in dataset:
            return "hematologic disease biology; disease bins can encode malignant/immune cell-state and composition changes"
        if "lymph_node" in dataset:
            return "tumor-draining/metastatic lymph-node biology; normal-vs-cancer shifts are biologically meaningful"
        if "esophagus" in dataset:
            return "Barrett/metaplasia biology; do not treat disease signal as removable nuisance"
        if "pancreas" in dataset:
            return "diabetes/islet-immune biology; disease context may change endocrine and immune states"
        if "placenta" in dataset:
            return "infection/inflammatory placenta biology; support is often limited"
        if "stomach" in dataset:
            return "gastritis/metaplasia biology; inflammation and epithelial-state shifts expected"
        return "disease biology; preserve disease signal while reporting worst-bin robustness"
    return "patient context"


def manuscript_use(row: Mapping[str, object]) -> str:
    context = str(row.get("context", ""))
    support = support_flag(clean_float(row.get("support_coverage_ge20")))
    note = biology_note(row)
    if context == "disease":
        return f"primary patient-context robustness endpoint; {support}; disease signal should be preserved"
    if context == "age_group" and "developmental_or_missing" in note:
        return f"boundary/support endpoint; {support}; separate prenatal/unknown from adult-only analysis"
    if context == "age_group":
        return f"adult clinical-context endpoint; {support}; check aging biology and support"
    if context == "sex":
        return f"patient-context fairness/biology endpoint; {support}; report because support is often complete"
    return f"patient-context endpoint; {support}"


def build_single_cell_hotspots(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df[df["context"].astype(str).isin(PATIENT_CONTEXTS)].copy()
    if work.empty:
        return pd.DataFrame()
    work["gap"] = pd.to_numeric(work["patient_cv_best_minus_worst_gap"], errors="coerce")
    work["average_minus_worst_gap"] = pd.to_numeric(work["patient_cv_average_minus_worst_gap"], errors="coerce")
    work["worst_ba"] = pd.to_numeric(work["worst_bin_ba"], errors="coerce")
    work["support_coverage_ge20"] = pd.to_numeric(work["support_coverage_ge20"], errors="coerce")
    work["severity_flag"] = [
        severity_flag(gap, worst)
        for gap, worst in zip(work["gap"], work["worst_ba"])
    ]
    work["support_flag"] = [support_flag(value) for value in work["support_coverage_ge20"]]
    work["biology_interpretation"] = [biology_note(row) for row in work.to_dict("records")]
    work["manuscript_use"] = [manuscript_use(row) for row in work.to_dict("records")]
    work["_severity_rank"] = work["severity_flag"].map(SEVERITY_ORDER).fillna(99)
    preferred = [
        "context",
        "dataset",
        "tissue_label",
        "model_label",
        "average_ba",
        "worst_bin_ba",
        "gap",
        "average_minus_worst_gap",
        "best_context_value",
        "worst_context_value",
        "n_cells",
        "n_donors",
        "n_context_values",
        "support_coverage_ge20",
        "support_flag",
        "label_context_cramers_v",
        "severity_flag",
        "biology_interpretation",
        "manuscript_use",
        "bin_counts",
        "source_file",
    ]
    cols = [c for c in preferred if c in work.columns]
    return work.sort_values(["_severity_rank", "gap"], ascending=[True, False])[cols].head(top_k)


def build_single_cell_hotspots_by_context(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    all_rows = build_single_cell_hotspots(df, top_k=10**9)
    if all_rows.empty:
        return all_rows
    return (
        all_rows.sort_values(["context", "gap"], ascending=[True, False])
        .groupby("context", group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def build_context_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df[df["context"].astype(str).isin(PATIENT_CONTEXTS)].copy()
    if work.empty:
        return pd.DataFrame()
    work["gap"] = pd.to_numeric(work["patient_cv_best_minus_worst_gap"], errors="coerce")
    work["average_minus_worst_gap"] = pd.to_numeric(work["patient_cv_average_minus_worst_gap"], errors="coerce")
    work["support_coverage_ge20"] = pd.to_numeric(work["support_coverage_ge20"], errors="coerce")
    rows = []
    for context, sub in work.groupby("context"):
        max_row = sub.loc[sub["gap"].idxmax()]
        rows.append(
            {
                "context": context,
                "n_rows": int(len(sub)),
                "gap_mean": float(sub["gap"].mean()),
                "gap_median": float(sub["gap"].median()),
                "gap_p90": float(sub["gap"].quantile(0.90)),
                "gap_max": float(sub["gap"].max()),
                "average_minus_worst_gap_mean": float(sub["average_minus_worst_gap"].mean()),
                "support_coverage_mean": float(sub["support_coverage_ge20"].mean()),
                "support_coverage_median": float(sub["support_coverage_ge20"].median()),
                "max_gap_dataset": str(max_row.get("dataset", "")),
                "max_gap_model": str(max_row.get("model_label", "")),
                "max_gap_worst_bin": str(max_row.get("worst_context_value", "")),
                "max_gap_best_bin": str(max_row.get("best_context_value", "")),
                "max_gap_biology_interpretation": biology_note(max_row.to_dict()),
            }
        )
    return pd.DataFrame(rows).sort_values("gap_max", ascending=False)


def build_adult_age_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "context" not in df.columns:
        return pd.DataFrame()
    work = df[df["context"].astype(str) == "age_adult_only"].copy()
    if work.empty:
        return pd.DataFrame()
    work["gap"] = pd.to_numeric(work["gap"], errors="coerce")
    work["worst_ba"] = pd.to_numeric(work["worst_ba"], errors="coerce")
    work["support_coverage_ge20"] = pd.to_numeric(work["support_coverage_ge20"], errors="coerce")
    work["severity_flag"] = [
        severity_flag(gap, worst)
        for gap, worst in zip(work["gap"], work["worst_ba"])
    ]
    work["support_flag"] = [support_flag(value) for value in work["support_coverage_ge20"]]
    work["biology_interpretation"] = "adult_age_shift; use as prenatal/unknown sensitivity control"
    preferred = [
        "dataset",
        "embedding",
        "overall_ba",
        "worst_ba",
        "gap",
        "gap_ci_low",
        "gap_ci_high",
        "best_context_value",
        "worst_context_value",
        "n_cells",
        "n_donors",
        "support_coverage_ge20",
        "support_flag",
        "label_context_cramers_v",
        "severity_flag",
        "biology_interpretation",
        "bin_counts",
        "source_file",
    ]
    return work.sort_values("gap", ascending=False)[[c for c in preferred if c in work.columns]]


def build_tcga_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["gap"] = pd.to_numeric(work["gap"], errors="coerce")
    work["worst_ba"] = pd.to_numeric(work["worst_ba"], errors="coerce")
    work["severity_flag"] = [
        severity_flag(gap, worst)
        for gap, worst in zip(work["gap"], work["worst_ba"])
    ]
    work["manuscript_use"] = "pathology patient-context screen; small-n bins need CI-aware wording"
    preferred = [
        "dataset",
        "embedding",
        "task",
        "context",
        "overall_ba",
        "worst_ba",
        "gap",
        "gap_ci_low",
        "gap_ci_high",
        "directional_gap",
        "n_predictions",
        "severity_flag",
        "manuscript_use",
        "source_file",
    ]
    return work.sort_values("gap", ascending=False)[[c for c in preferred if c in work.columns]]


def build_residualized_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df[df["context_field"].astype(str).isin(PATIENT_CONTEXTS)].copy()
    if work.empty:
        return pd.DataFrame()
    numeric = [
        "base_probe_ba",
        "artifact_resid_probe_ba",
        "label_resid_probe_ba",
        "artifact_label_resid_probe_ba",
        "base_patient_cv_gap",
        "artifact_resid_patient_cv_gap",
        "base_leave_one_gap",
        "artifact_resid_leave_one_gap",
    ]
    for col in numeric:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    rows = []
    for context, sub in work.groupby("context_field"):
        rows.append(
            {
                "context": context,
                "n_rows": int(len(sub)),
                "base_probe_ba_mean": float(sub["base_probe_ba"].mean()),
                "artifact_resid_probe_ba_mean": float(sub["artifact_resid_probe_ba"].mean()),
                "label_resid_probe_ba_mean": float(sub["label_resid_probe_ba"].mean()),
                "base_patient_cv_gap_mean": float(sub["base_patient_cv_gap"].mean()),
                "artifact_resid_patient_cv_gap_mean": float(sub["artifact_resid_patient_cv_gap"].mean()),
                "base_leave_one_gap_mean": float(sub["base_leave_one_gap"].mean()),
                "artifact_resid_leave_one_gap_mean": float(sub["artifact_resid_leave_one_gap"].mean()),
                "interpretation": (
                    "artifact residualization is a control, not proof that patient context is nuisance; "
                    "compare probe reduction with downstream worst-bin gap"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def summary_json(tables: Mapping[str, pd.DataFrame]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    context_summary = tables.get("context_summary", pd.DataFrame())
    if not context_summary.empty:
        out["single_cell_context_summary"] = context_summary.to_dict("records")
    hotspots = tables.get("single_cell_hotspots", pd.DataFrame())
    if not hotspots.empty:
        out["top_single_cell_hotspots"] = hotspots.head(5).to_dict("records")
    by_context = tables.get("single_cell_hotspots_by_context", pd.DataFrame())
    if not by_context.empty:
        out["top_single_cell_hotspots_by_context"] = {
            str(context): sub.head(3).to_dict("records")
            for context, sub in by_context.groupby("context", sort=True)
        }
    adult = tables.get("adult_age_hotspots", pd.DataFrame())
    if not adult.empty:
        out["adult_age_top_gap"] = adult.head(1).to_dict("records")[0]
    tcga = tables.get("tcga_hotspots", pd.DataFrame())
    if not tcga.empty:
        out["tcga_top_gap"] = tcga.head(1).to_dict("records")[0]
    residual = tables.get("residualized_context_summary", pd.DataFrame())
    if not residual.empty:
        out["residualized_context_summary"] = residual.to_dict("records")
    return out


def clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    paper = args.paper_table_dir
    single_cell = read_csv(paper / "patient_context_tissue_model_bins.csv")
    demographic = read_csv(paper / "single_cell_demographic_context_bins.csv")
    tcga = read_csv(paper / "tcga_demographic_context_bins.csv")
    residualized = read_csv(paper / "residualized_control_delta_table.csv")

    tables = {
        "single_cell_hotspots": build_single_cell_hotspots(single_cell, args.top_k),
        "single_cell_hotspots_by_context": build_single_cell_hotspots_by_context(single_cell, args.top_k),
        "context_summary": build_context_summary(single_cell),
        "adult_age_hotspots": build_adult_age_table(demographic),
        "tcga_hotspots": build_tcga_table(tcga),
        "residualized_context_summary": build_residualized_table(residualized),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        write_csv(args.output_dir / f"{name}.csv", table)
    summary = clean_json(summary_json(tables))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
