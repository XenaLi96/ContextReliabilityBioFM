#!/usr/bin/env python3
"""Build paper-ready context-bin performance tables for main2.tex."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


CONTEXT_DIR_TO_FIELD = {
    "assay": "assay",
    "dataset": "dataset_id",
    "dataset_id": "dataset_id",
}

MODEL_LABELS = {
    "geneformer_v1": "Geneformer",
    "scgpt_continual": "scGPT",
    "scvi_style_vae": "scVI-style",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_tables"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def strip_prefix_suffix(text: str, prefix: str, suffix: str) -> str:
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith(suffix):
        text = text[: -len(suffix)]
    return text


def clean_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def fmt_num(value: object, digits: int = 3) -> str:
    value = clean_float(value)
    if math.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def fmt_range(values: Iterable[float], digits: int = 3) -> str:
    vals = [clean_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return "--"
    return f"{min(vals):.{digits}f}--{max(vals):.{digits}f}"


def fmt_ci(value: float, ci_low: Optional[float], ci_high: Optional[float]) -> str:
    if ci_low is None or ci_high is None or math.isnan(ci_low) or math.isnan(ci_high):
        return fmt_num(value)
    return f"{fmt_num(value)} [{fmt_num(ci_low)}, {fmt_num(ci_high)}]"


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def audit_dir(tissue: str, model_key: str, context_dir: str) -> Path:
    return Path("data") / f"cellxgene_{tissue}_embedding_audit" / f"{model_key}_{context_dir}_erm"


def load_cellxgene_row(
    tissue: str,
    data_task: str,
    model_key: str,
    context_dir: str,
    split_label: str = "patient-CV",
) -> Dict[str, object]:
    context_field = CONTEXT_DIR_TO_FIELD[context_dir]
    path = audit_dir(tissue, model_key, context_dir)
    metrics = read_csv(path / "subgroup_metrics.csv")
    gaps = read_csv(path / "subgroup_gaps.csv")
    leave_gaps = read_csv(path / "leave_one_context_gaps.csv")
    summary = {}
    if (path / "summary.json").exists():
        with open(path / "summary.json") as handle:
            summary = json.load(handle)

    overall = metrics[
        (metrics["split_type"].astype(str) == "patient_level_cv")
        & (metrics["context_field"].astype(str) == "overall")
    ]
    if overall.empty:
        raise ValueError(f"Missing overall patient-CV metrics in {path}")
    overall_row = overall.iloc[0]

    patient_gap = gaps[
        (gaps["split_type"].astype(str) == "patient_level_cv")
        & (gaps["context_field"].astype(str) == context_field)
        & (gaps["metric"].astype(str) == "balanced_accuracy")
    ]
    if patient_gap.empty:
        raise ValueError(f"Missing patient-CV gap for {context_field} in {path}")
    patient_row = patient_gap.iloc[0]

    leave_row = None
    if not leave_gaps.empty:
        leave = leave_gaps[
            (leave_gaps["context_field"].astype(str) == context_field)
            & (leave_gaps["metric"].astype(str) == "balanced_accuracy")
        ]
        if not leave.empty:
            leave_row = leave.iloc[0]

    return {
        "domain": "single_cell",
        "data_task": data_task,
        "dataset": f"CELLxGENE:{tissue}",
        "embedding": MODEL_LABELS.get(model_key, model_key),
        "context": context_field,
        "split": split_label,
        "n": int(clean_float(overall_row.get("n"))),
        "average_ba": clean_float(overall_row.get("balanced_accuracy")),
        "average_ba_ci_low": float("nan"),
        "average_ba_ci_high": float("nan"),
        "worst_bin_ba": clean_float(patient_row.get("worst_value")),
        "worst_bin_ba_ci_low": float("nan"),
        "worst_bin_ba_ci_high": float("nan"),
        "gap": clean_float(patient_row.get("gap")),
        "gap_ci_low": float("nan"),
        "gap_ci_high": float("nan"),
        "best_bin": str(patient_row.get("best_context_value")),
        "worst_bin": str(patient_row.get("worst_context_value")),
        "leave_one_gap": clean_float(leave_row.get("gap")) if leave_row is not None else float("nan"),
        "leave_one_worst_ba": clean_float(leave_row.get("worst_value")) if leave_row is not None else float("nan"),
        "source": str(path),
        "note": f"{summary.get('embedding_shape', ['?'])[0]} cells",
    }


def collect_bone_marrow_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_key in ["geneformer_v1", "scgpt_continual", "scvi_style_vae"]:
        for context_dir in ["assay", "dataset"]:
            rows.append(
                load_cellxgene_row(
                    tissue="bone_marrow",
                    data_task="Bone marrow cell type",
                    model_key=model_key,
                    context_dir=context_dir,
                )
            )
    return rows


def collect_ten_tissue_rows() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    detail_rows: List[Dict[str, object]] = []
    for path in sorted(Path("data").glob("cellxgene_*_embedding_audit/scgpt_continual_*_erm")):
        tissue = strip_prefix_suffix(path.parent.name, "cellxgene_", "_embedding_audit")
        run_name = path.name
        if run_name.endswith("_assay_erm"):
            context_dir = "assay"
        elif run_name.endswith("_dataset_erm"):
            context_dir = "dataset"
        else:
            continue
        detail_rows.append(
            load_cellxgene_row(
                tissue=tissue,
                data_task=f"{tissue.replace('_', ' ')} cell type",
                model_key="scgpt_continual",
                context_dir=context_dir,
            )
        )

    detail = pd.DataFrame(detail_rows)
    summary_rows: List[Dict[str, object]] = []
    for context, group in detail.groupby("context", sort=True):
        max_patient = group.loc[group["gap"].astype(float).idxmax()]
        max_leave = group.loc[group["leave_one_gap"].astype(float).idxmax()]
        summary_rows.append(
            {
                "domain": "single_cell",
                "data_task": "Ten-tissue cell type",
                "dataset": "CELLxGENE:10 tissues",
                "embedding": "scGPT",
                "context": context,
                "split": "patient-CV",
                "n": int(group[["dataset", "n"]].drop_duplicates()["n"].sum()),
                "average_ba": float("nan"),
                "average_ba_range": fmt_range(group["average_ba"]),
                "average_ba_ci_low": float("nan"),
                "average_ba_ci_high": float("nan"),
                "worst_bin_ba": float("nan"),
                "worst_bin_ba_range": fmt_range(group["worst_bin_ba"]),
                "worst_bin_ba_ci_low": float("nan"),
                "worst_bin_ba_ci_high": float("nan"),
                "gap": clean_float(max_patient["gap"]),
                "gap_ci_low": float("nan"),
                "gap_ci_high": float("nan"),
                "best_bin": "",
                "worst_bin": "",
                "leave_one_gap": clean_float(max_leave["leave_one_gap"]),
                "leave_one_worst_ba": clean_float(max_leave["leave_one_worst_ba"]),
                "source": "data/cellxgene_*_embedding_audit/scgpt_continual_*_erm",
                "note": (
                    f"patient gap max {fmt_num(max_patient['gap'])} "
                    f"({str(max_patient['dataset']).split(':')[-1]}); "
                    f"leave-one max {fmt_num(max_leave['leave_one_gap'])} "
                    f"({str(max_leave['dataset']).split(':')[-1]})"
                ),
            }
        )
    return detail_rows, summary_rows


def tcga_ci_lookup() -> Dict[Tuple[str, str, str, str], Tuple[float, float, float]]:
    ci = read_csv(Path("data/tcga_image_context_shift_bootstrap_ci/leave_one_context_bootstrap_ci.csv"))
    lookup: Dict[Tuple[str, str, str, str], Tuple[float, float, float]] = {}
    if ci.empty:
        return lookup
    sub = ci[
        (ci["method"].astype(str) == "erm")
        & (ci["metric"].astype(str) == "balanced_accuracy")
    ]
    for _, row in sub.iterrows():
        key = (
            str(row["task"]),
            str(row["model"]),
            str(row["context_field"]),
            str(row["quantity"]),
        )
        lookup[key] = (
            clean_float(row["observed"]),
            clean_float(row["ci_low"]),
            clean_float(row["ci_high"]),
        )
    return lookup


def load_tcga_rows() -> List[Dict[str, object]]:
    summary = read_csv(Path("data/tcga_image_context_shift_summary/combined_leave_one_context_summary.csv"))
    if summary.empty:
        return []
    lookup = tcga_ci_lookup()
    rows: List[Dict[str, object]] = []
    sub = summary[
        (summary["method"].astype(str) == "erm")
        & (summary["metric"].astype(str) == "balanced_accuracy")
        & (summary["context_field"].astype(str).isin(["site", "primary_diagnosis"]))
    ].copy()
    for _, row in sub.iterrows():
        task = str(row["task"])
        model = str(row["model"])
        context = str(row["context_field"])
        overall = lookup.get((task, model, context, "overall"), (clean_float(row["balanced_accuracy"]), float("nan"), float("nan")))
        worst = lookup.get((task, model, context, "worst_value"), (clean_float(row["worst_context_metric"]), float("nan"), float("nan")))
        gap = lookup.get((task, model, context, "best_minus_worst"), (clean_float(row["best_minus_worst"]), float("nan"), float("nan")))
        rows.append(
            {
                "domain": "pathology",
                "data_task": task.replace("_status", "").replace("_", " "),
                "dataset": "TCGA",
                "embedding": model,
                "context": context,
                "split": "leave-one-context",
                "n": int(clean_float(row["n"])),
                "average_ba": overall[0],
                "average_ba_ci_low": overall[1],
                "average_ba_ci_high": overall[2],
                "worst_bin_ba": worst[0],
                "worst_bin_ba_ci_low": worst[1],
                "worst_bin_ba_ci_high": worst[2],
                "gap": gap[0],
                "gap_ci_low": gap[1],
                "gap_ci_high": gap[2],
                "best_bin": str(row["best_context_value"]),
                "worst_bin": str(row["worst_context_value"]),
                "leave_one_gap": gap[0],
                "leave_one_worst_ba": worst[0],
                "source": str(row["result_dir"]),
                "note": f"{int(clean_float(row['n_holdout_contexts']))} held-out contexts",
            }
        )
    return rows


def focus_main_rows(
    bone_rows: List[Dict[str, object]],
    ten_summary_rows: List[Dict[str, object]],
    tcga_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rows.extend(
        row
        for row in bone_rows
        if row["embedding"] in {"Geneformer", "scGPT", "scVI-style"}
        and row["context"] in {"assay", "dataset_id"}
    )
    rows.extend(sorted(ten_summary_rows, key=lambda r: str(r["context"])))
    for task, model, context in [("KRAS", "CONCH", "site"), ("IDH", "UNI", "site")]:
        match = [
            row
            for row in tcga_rows
            if row["data_task"] == task and row["embedding"] == model and row["context"] == context
        ]
        rows.extend(match)
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def latex_rows(rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    for row in rows:
        avg = row.get("average_ba_range") or fmt_ci(
            clean_float(row["average_ba"]),
            clean_float(row.get("average_ba_ci_low")),
            clean_float(row.get("average_ba_ci_high")),
        )
        worst = row.get("worst_bin_ba_range") or fmt_ci(
            clean_float(row["worst_bin_ba"]),
            clean_float(row.get("worst_bin_ba_ci_low")),
            clean_float(row.get("worst_bin_ba_ci_high")),
        )
        gap = fmt_ci(
            clean_float(row["gap"]),
            clean_float(row.get("gap_ci_low")),
            clean_float(row.get("gap_ci_high")),
        )
        leave_gap = clean_float(row.get("leave_one_gap"))
        if not math.isnan(leave_gap):
            gap = f"{gap}; leave-one {fmt_num(leave_gap)}"
        note = row.get("note", "")
        if row["data_task"] == "Ten-tissue cell type":
            gap = str(note)
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row["data_task"]),
                latex_escape(row["embedding"]),
                latex_escape(row["context"]),
                latex_escape(row["split"]),
                avg,
                worst,
                latex_escape(gap),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bone_rows = collect_bone_marrow_rows()
    ten_detail_rows, ten_summary_rows = collect_ten_tissue_rows()
    tcga_rows = load_tcga_rows()
    main_rows = focus_main_rows(bone_rows, ten_summary_rows, tcga_rows)

    write_csv(args.output_dir / "bone_marrow_context_bins.csv", bone_rows)
    write_csv(args.output_dir / "scgpt_ten_tissue_context_bins.csv", ten_detail_rows)
    write_csv(args.output_dir / "tcga_pathology_context_bins.csv", tcga_rows)
    write_csv(args.output_dir / "main_context_bin_bias_table.csv", main_rows)

    (args.output_dir / "main_context_bin_bias_rows.tex").write_text(latex_rows(main_rows))
    summary = {
        "n_bone_marrow_rows": len(bone_rows),
        "n_ten_tissue_detail_rows": len(ten_detail_rows),
        "n_tcga_rows": len(tcga_rows),
        "n_main_rows": len(main_rows),
        "ten_tissue_total_cells": int(pd.DataFrame(ten_detail_rows)[["dataset", "n"]].drop_duplicates()["n"].sum()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
