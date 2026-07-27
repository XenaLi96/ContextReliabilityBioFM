#!/usr/bin/env python3
"""Draw a bubble atlas preview for the context-bias manuscript figure.

The figure is intentionally generated from existing project tables. Missing
model/dataset combinations are left blank rather than imputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "data" / "paper_tables"
SCA = ROOT / "data" / "sca_search_tables"
OUT = ROOT / "figures"


MODEL_ORDER = [
    "Geneformer",
    "scGPT",
    "scVI-style",
    "HVG-PCA",
    "QC-count",
    "UNI",
    "CONCH",
    "Virchow2",
    "H-optimus0",
    "ImageStats",
]

SINGLE_CELL_COLUMNS = {"Geneformer", "scGPT", "scVI-style", "HVG-PCA", "QC-count"}
PATHOLOGY_COLUMNS = {"UNI", "CONCH", "Virchow2", "H-optimus0", "ImageStats"}
SINGLE_CELL_PANEL_MODELS = ["Geneformer", "scGPT", "scVI-style", "HVG-PCA", "QC-count"]
PATHOLOGY_PANEL_MODELS = ["UNI", "CONCH", "Virchow2", "H-optimus0", "ImageStats"]

MODEL_LABELS = {
    "Geneformer": "Geneformer",
    "scGPT": "scGPT",
    "scVI-style": "scVI\n(scArches)",
    "HVG-PCA": "HVG-PCA\n(control)",
    "QC-count": "QC/count\n(control)",
    "UNI": "UNI\n(vision)",
    "CONCH": "CONCH\n(histology)",
    "Virchow2": "Virchow2",
    "H-optimus0": "H-optimus0\n(histology)",
    "ImageStats": "ImageStats\n(control)",
}

DOMAIN_COLORS = {
    "SINGLE-CELL": "#6F8F7B",
    "PATHOLOGY": "#6E8397",
    "GENOMICS / CLINICAL": "#907D9A",
}

METHOD_COLORS = {
    "Baseline": "#70777D",
    "LC-Reweight": "#D09152",
    "SCA-Align": "#5F8A78",
}

METHOD_MARKERS = {
    "Baseline": "o",
    "LC-Reweight": "s",
    "SCA-Align": "^",
}


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".svg"))


def add_card(ax: plt.Axes) -> None:
    card = FancyBboxPatch(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor="#C9D1D9",
        facecolor="white",
        zorder=-10,
        clip_on=False,
    )
    ax.add_patch(card)


def panel_badge(ax: plt.Axes, label: str, x: float = 0.02, y: float = 0.96) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="circle,pad=0.20", facecolor="#087A3D", edgecolor="none"),
        zorder=10,
    )


def gap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "red_to_blue_worst_ba",
        [
            (0.00, "#B65F5A"),
            (0.24, "#D09A7A"),
            (0.50, "#D8C99A"),
            (0.74, "#94AFC0"),
            (1.00, "#587FA2"),
        ],
    )


def atlas_rows() -> list[dict[str, str]]:
    return [
        {"domain": "SINGLE-CELL", "label": "Pancreas assay", "dataset_match": "CELLxGENE:pancreas", "context": "assay"},
        {"domain": "SINGLE-CELL", "label": "Esophagus assay", "dataset_match": "CELLxGENE:esophagus", "context": "assay"},
        {"domain": "SINGLE-CELL", "label": "Stomach assay", "dataset_match": "CELLxGENE:stomach", "context": "assay"},
        {"domain": "SINGLE-CELL", "label": "Lymph node dataset", "dataset_match": "CELLxGENE:lymph_node", "context": "dataset_id"},
        {"domain": "SINGLE-CELL", "label": "Ovary assay", "dataset_match": "CELLxGENE:ovary", "context": "assay"},
        {"domain": "SINGLE-CELL", "label": "Bone marrow assay", "dataset_match": "CELLxGENE:bone_marrow", "context": "assay"},
        {"domain": "PATHOLOGY", "label": "TCGA IDH site", "dataset_match": "TCGA", "task_match": "IDH", "context": "site"},
        {"domain": "PATHOLOGY", "label": "TCGA IDH diagnosis", "dataset_match": "TCGA", "task_match": "IDH", "context": "primary_diagnosis"},
        {"domain": "PATHOLOGY", "label": "TCGA KRAS site", "dataset_match": "TCGA", "task_match": "KRAS", "context": "site"},
        {"domain": "PATHOLOGY", "label": "TCGA KRAS diagnosis", "dataset_match": "TCGA", "task_match": "KRAS", "context": "primary_diagnosis"},
        {"domain": "PATHOLOGY", "label": "TCGA TP53 site", "dataset_match": "TCGA", "task_match": "TP53", "context": "site"},
        {"domain": "PATHOLOGY", "label": "TCGA TP53 diagnosis", "dataset_match": "TCGA", "task_match": "TP53", "context": "primary_diagnosis"},
        {"domain": "GENOMICS / CLINICAL", "label": "Bone marrow disease", "dataset_match": "CELLxGENE:bone_marrow", "context": "disease"},
        {"domain": "GENOMICS / CLINICAL", "label": "Lymph node disease", "dataset_match": "CELLxGENE:lymph_node", "context": "disease"},
        {"domain": "GENOMICS / CLINICAL", "label": "Pancreas disease", "dataset_match": "CELLxGENE:pancreas", "context": "disease"},
        {"domain": "GENOMICS / CLINICAL", "label": "Esophagus age", "dataset_match": "CELLxGENE:esophagus", "context": "age_group"},
        {"domain": "GENOMICS / CLINICAL", "label": "Placenta age", "dataset_match": "CELLxGENE:placenta", "context": "age_group"},
        {"domain": "GENOMICS / CLINICAL", "label": "Bladder sex", "dataset_match": "CELLxGENE:bladder_organ", "context": "sex"},
    ]


def normalize_model(raw: str) -> str:
    text = str(raw)
    if text in {"Geneformer", "geneformer_v1"}:
        return "Geneformer"
    if text in {"scGPT", "scgpt_continual"}:
        return "scGPT"
    if text.startswith("scVI"):
        return "scVI-style"
    if text.startswith("H-optimus"):
        return "H-optimus0"
    if "hvg_pca" in text.lower():
        return "HVG-PCA"
    if "qc_count" in text.lower() or text == "QC-count":
        return "QC-count"
    if text == "ImageStats":
        return "ImageStats"
    return text


def series_from(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def applicable_models(domain: str) -> set[str]:
    if domain == "PATHOLOGY":
        return PATHOLOGY_COLUMNS
    return SINGLE_CELL_COLUMNS


def task_from_pathology_dataset(value: str) -> str:
    text = str(value).replace("TCGA:", "")
    for suffix in ["_status", " status"]:
        text = text.replace(suffix, "")
    return text.upper()


def strip_prefix_suffix(text: str, prefix: str, suffix: str) -> str:
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith(suffix):
        text = text[: -len(suffix)]
    return text


def single_cell_control_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted((ROOT / "data").glob("cellxgene_*_embedding_audit/*/summary.json")):
        run_dir = summary_path.parent
        run_name = run_dir.name
        if "smoke" in run_name:
            continue
        model = normalize_model(run_name)
        if model not in {"HVG-PCA", "QC-count"}:
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        dataset = strip_prefix_suffix(run_dir.parent.name, "cellxgene_", "_embedding_audit")
        for row in summary.get("patient_level_gaps", []):
            if str(row.get("method")) != "erm" or str(row.get("metric")) != "balanced_accuracy":
                continue
            gap = float(row.get("gap", np.nan))
            worst = float(row.get("worst_value", np.nan))
            if not np.isfinite(gap) or not np.isfinite(worst):
                continue
            rows.append(
                {
                    "domain": "single_cell",
                    "data_task": f"{dataset.replace('_', ' ')} cell type",
                    "dataset": f"CELLxGENE:{dataset}",
                    "embedding": model,
                    "context": str(row.get("context_field", "")),
                    "split": "patient-CV",
                    "average_ba": worst + gap,
                    "worst_bin_ba": worst,
                    "gap": gap,
                    "source": str(run_dir),
                }
            )
    support_table = ROOT / "data" / "reviewer_control_tables" / "single_cell_artifact_control_summary.csv"
    if support_table.exists() and rows:
        support = pd.read_csv(support_table)
        support["model_norm"] = support["representation"].astype(str).map(normalize_model)
        support_lookup = {
            (str(r["dataset"]), str(r["model_norm"]), str(r["context"])): r.get("support_coverage", np.nan)
            for _, r in support.iterrows()
        }
        for row in rows:
            row["support_coverage_ge20"] = support_lookup.get((str(row["dataset"]), str(row["embedding"]), str(row["context"])), np.nan)
    return rows


def pathology_control_rows() -> list[dict[str, object]]:
    path = ROOT / "data" / "reviewer_control_tables" / "pathology_artifact_control_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    out: list[dict[str, object]] = []
    for _, row in df[df["representation"].astype(str) == "ImageStats"].iterrows():
        gap = float(row.get("leave_one_ba_gap", np.nan))
        worst = float(row.get("worst_context_ba", np.nan))
        if not np.isfinite(gap) or not np.isfinite(worst):
            continue
        out.append(
            {
                "domain": "pathology",
                "data_task": task_from_pathology_dataset(str(row.get("dataset", ""))),
                "dataset": "TCGA",
                "embedding": "ImageStats",
                "context": str(row.get("context", "")),
                "split": "leave-one-context",
                "average_ba": worst + gap,
                "worst_bin_ba": worst,
                "gap": gap,
                "source": str(row.get("source_dir", "")),
            }
        )
    return out


def load_bias_points() -> dict[tuple[int, str], dict[str, float | str]]:
    frames = []
    for path in [
        PAPER / "main_context_bin_bias_table.csv",
        PAPER / "scgpt_ten_tissue_context_bins.csv",
        PAPER / "cellxgene_tissue_model_context_bins.csv",
        PAPER / "tcga_pathology_context_bins.csv",
        PAPER / "patient_context_tissue_model_bins.csv",
    ]:
        if path.exists():
            frames.append(pd.read_csv(path))
    control_rows = single_cell_control_rows() + pathology_control_rows()
    if control_rows:
        frames.append(pd.DataFrame(control_rows))
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["embedding_norm_src"] = series_from(df, "embedding").fillna(series_from(df, "model_label"))
    df["gap_plot_src"] = pd.to_numeric(series_from(df, "gap"), errors="coerce")
    patient_gap = pd.to_numeric(series_from(df, "patient_cv_average_minus_worst_gap"), errors="coerce")
    avg_minus_worst = pd.to_numeric(series_from(df, "average_ba"), errors="coerce") - pd.to_numeric(series_from(df, "worst_bin_ba"), errors="coerce")
    df["gap_plot_src"] = df["gap_plot_src"].fillna(patient_gap).fillna(avg_minus_worst)
    rows = atlas_rows()
    points: dict[tuple[int, str], dict[str, float | str]] = {}
    for row_i, spec in enumerate(rows):
        sub = df.copy()
        sub["dataset_str"] = series_from(sub, "dataset").astype(str)
        sub["task_str"] = series_from(sub, "data_task").astype(str)
        sub["model_norm"] = sub["embedding_norm_src"].astype(str).map(normalize_model)
        sub["context_str"] = series_from(sub, "context").astype(str)
        sub = sub[
            sub["dataset_str"].str.contains(spec["dataset_match"], regex=False, na=False)
            & (sub["context_str"] == spec["context"])
        ]
        task_match = spec.get("task_match")
        if task_match:
            sub = sub[sub["task_str"].str.contains(task_match, regex=False, na=False)]
        for model in MODEL_ORDER:
            model_sub = sub[sub["model_norm"] == model]
            if model_sub.empty:
                continue
            best = model_sub.sort_values("gap_plot_src", ascending=False).iloc[0]
            points[(row_i, model)] = {
                "gap": float(best.get("gap_plot_src", np.nan)),
                "avg": float(best.get("average_ba", np.nan)),
                "worst": float(best.get("worst_bin_ba", np.nan)),
                "support": float(best.get("support_coverage_ge20", best.get("support_coverage", np.nan))),
                "domain": spec["domain"],
            }
    return points


def load_adaptation_improvements() -> dict[tuple[str, str], float]:
    table = SCA / "external_baseline_adaptation_table.csv"
    if not table.exists():
        return {}
    df = pd.read_csv(table)
    keep = df[df["method"].isin(["erm_mlp", "sca_lite"])].copy()
    pivot = keep.pivot_table(index=["model_label", "context_label", "split_label"], columns="method", values="gap", aggfunc="first")
    improvements: dict[tuple[str, str], float] = {}
    for (model, context, _split), row in pivot.iterrows():
        if "erm_mlp" not in row or "sca_lite" not in row:
            continue
        value = float(row["erm_mlp"] - row["sca_lite"])
        key = (normalize_model(str(model)), str(context))
        improvements[key] = max(improvements.get(key, -np.inf), value)
    return improvements


def plot_bubble_atlas(ax: plt.Axes, *, show_card: bool = True) -> None:
    if show_card:
        add_card(ax)
    rows = atlas_rows()
    points = load_bias_points()
    cmap = gap_cmap()
    norm_worst = Normalize(0.0, 1.0)

    row_step = 1.13
    top_indices = [i for i, spec in enumerate(rows) if spec["domain"] != "PATHOLOGY"]
    bottom_indices = [i for i, spec in enumerate(rows) if spec["domain"] == "PATHOLOGY"]
    block_gap = 3.05
    bottom_y = {row_i: (len(bottom_indices) - 1 - j) * row_step for j, row_i in enumerate(bottom_indices)}
    top_start = len(bottom_indices) * row_step + block_gap
    top_y = {row_i: top_start + (len(top_indices) - 1 - j) * row_step for j, row_i in enumerate(top_indices)}
    y_by_row = {**bottom_y, **top_y}

    model_x = np.arange(5) * 1.10 + 4.05
    label_x = 3.20
    panel_right = model_x[-1] + 0.85
    top_max = max(top_y.values())
    bottom_max = max(bottom_y.values())

    def draw_model_header(models: list[str], y_top: float) -> None:
        for x, model in zip(model_x, models):
            ax.text(x, y_top + 1.48, MODEL_LABELS[model], ha="center", va="bottom", fontsize=5.7, fontweight="bold")
            ax.text(x, y_top + 1.02, model_icon(model), ha="center", va="center", fontsize=10.2, color=model_color(model), fontweight="bold")

    draw_model_header(SINGLE_CELL_PANEL_MODELS, top_max)
    draw_model_header(PATHOLOGY_PANEL_MODELS, bottom_max)
    ax.plot([0.72, panel_right - 0.05], [top_start - 1.50, top_start - 1.50], color="#E5DED4", linewidth=0.8)

    domain_ranges: dict[str, list[int]] = {}
    for i, spec in enumerate(rows):
        y = y_by_row[i]
        domain_ranges.setdefault(spec["domain"], []).append(y)
        ax.text(label_x, y, spec["label"], ha="right", va="center", fontsize=6.5, color=DOMAIN_COLORS[spec["domain"]], fontweight="bold" if i in {1, 6, 12} else None)
        models = PATHOLOGY_PANEL_MODELS if spec["domain"] == "PATHOLOGY" else SINGLE_CELL_PANEL_MODELS
        for x, model in zip(model_x, models):
            value = points.get((i, model))
            if not value:
                continue
            gap = float(value["gap"])
            worst = float(value["worst"])
            support = value.get("support", np.nan)
            if not np.isfinite(gap) or not np.isfinite(worst):
                continue
            size = 15 + 210 * min(max(gap, 0.0), 0.5) / 0.5
            face = cmap(norm_worst(max(min(worst, 1.0), 0.0)))
            ax.scatter(x, y, s=size, c=[face], edgecolors="#F8F4EC", linewidths=0.75, alpha=0.96, zorder=3)
            if np.isfinite(support) and support < 0.50:
                ax.scatter(
                    [x],
                    [y],
                    s=[size * 1.35],
                    facecolors="none",
                    edgecolors="#8A8177",
                    linewidths=0.70,
                    linestyles=(0, (2, 2)),
                    zorder=4,
                )

    for domain, ys in domain_ranges.items():
        y0, y1 = min(ys) - 0.40, max(ys) + 0.40
        color = DOMAIN_COLORS[domain]
        x = 0.78
        label_x = {
            "SINGLE-CELL": 0.16,
            "GENOMICS / CLINICAL": 0.45,
            "PATHOLOGY": 0.24,
        }[domain]
        ax.plot([x, x], [y0, y1], color=color, linewidth=1.1)
        ax.plot([x, x + 0.10], [y0, y0], color=color, linewidth=1.1)
        ax.plot([x, x + 0.10], [y1, y1], color=color, linewidth=1.1)
        domain_label = "CLINICAL" if domain == "GENOMICS / CLINICAL" else domain
        ax.text(label_x, (y0 + y1) / 2, domain_label, rotation=90, ha="center", va="center", fontsize=8, fontweight="bold", color=color)

    ax.text(
        0.5,
        1.035,
        "Same-modality bias bubble-dot atlas",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )
    ax.set_xlim(0.0, panel_right)
    ax.set_ylim(-1.0, top_max + 3.36)
    ax.set_axis_off()


def model_icon(model: str) -> str:
    return {
        "Geneformer": "Ω",
        "scGPT": "◌",
        "scVI-style": "V",
        "HVG-PCA": "P",
        "QC-count": "Q",
        "UNI": "✣",
        "CONCH": "◎",
        "Virchow2": "V",
        "H-optimus0": "H",
        "ImageStats": "I",
    }[model]


def model_color(model: str) -> str:
    return {
        "Geneformer": "#617C68",
        "scGPT": "#68889A",
        "scVI-style": "#6D86A8",
        "HVG-PCA": "#9A8F77",
        "QC-count": "#8E8075",
        "UNI": "#5D7186",
        "CONCH": "#88739A",
        "Virchow2": "#6E668D",
        "H-optimus0": "#9A695C",
        "ImageStats": "#8C877E",
    }[model]


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def load_adaptation_rows() -> pd.DataFrame:
    """Load the formal five-seed bone-marrow comparison used in the manuscript."""
    method_labels = {
        "erm_mlp": "Baseline",
        "label_context_reweight": "LC-Reweight",
        "sca_lite": "SCA-Align",
    }
    model_labels = {
        "geneformer_v1": "Geneformer",
        "scgpt_continual": "scGPT",
    }
    context_labels = {
        "assay": "assay context",
        "dataset_id": "dataset-source context",
    }
    split_labels = {
        "patient_level_cv": "Observed-context patient-CV",
        "leave_one_context": "Unseen-context leave-one",
    }
    rows: list[dict[str, object]] = []
    paths = sorted(
        (ROOT / "data" / "cellxgene_support_calibrated_formal").glob(
            "bone_marrow_*/*/*/summary/per_seed_gaps.csv"
        )
    )
    for path_i, path in enumerate(paths):
        df = pd.read_csv(path)
        model_key = path.parents[2].name
        context_key = path.parents[1].name
        for split_key, regime in split_labels.items():
            split_df = df[df["split_type"].astype(str) == split_key]
            for method_i, (method_key, method_label) in enumerate(method_labels.items()):
                values = pd.to_numeric(
                    split_df.loc[split_df["method"].astype(str) == method_key, "gap"],
                    errors="coerce",
                ).dropna().to_numpy()
                mean, ci_low, ci_high = bootstrap_mean_ci(
                    values,
                    seed=20260725 + 100 * path_i + 10 * method_i + (split_key == "leave_one_context"),
                )
                rows.append(
                    {
                        "regime": regime,
                        "model": model_labels.get(model_key, model_key),
                        "context": context_labels.get(context_key, context_key),
                        "method": method_label,
                        "gap_mean": mean,
                        "gap_ci_low": ci_low,
                        "gap_ci_high": ci_high,
                        "n_seeds": len(values),
                    }
                )
    return pd.DataFrame(rows)


def plot_mitigation(ax: plt.Axes, *, show_card: bool = True) -> None:
    if show_card:
        add_card(ax)
    ax.text(
        0.5,
        1.055,
        "Bone-marrow context gap after bounded intervention",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )
    data = load_adaptation_rows()
    regime_order = ["Observed-context patient-CV", "Unseen-context leave-one"]
    task_order = [
        ("Geneformer", "assay context"),
        ("Geneformer", "dataset-source context"),
        ("scGPT", "assay context"),
        ("scGPT", "dataset-source context"),
    ]
    method_order = ["Baseline", "LC-Reweight", "SCA-Align"]
    method_offsets = {"Baseline": 0.18, "LC-Reweight": 0.0, "SCA-Align": -0.18}
    group_bases = {
        "Observed-context patient-CV": 7.65,
        "Unseen-context leave-one": 2.65,
    }
    group_faces = {
        "Observed-context patient-CV": "#F4F7F5",
        "Unseen-context leave-one": "#F4F6F9",
    }

    for regime in regime_order:
        base = group_bases[regime]
        ax.axhspan(base - 3.42, base + 0.52, color=group_faces[regime], zorder=-5)
        ax.text(
            -0.175,
            base + 0.72,
            regime,
            ha="left",
            va="bottom",
            fontsize=6.2,
            fontweight="bold",
            color="#334B5C",
        )
        for task_i, (model, context) in enumerate(task_order):
            yi = base - task_i
            task = data[
                (data["regime"] == regime)
                & (data["model"] == model)
                & (data["context"] == context)
            ]
            ax.text(
                -0.175,
                yi,
                f"{model} · {context}",
                ha="left",
                va="center",
                fontsize=5.8,
                color="#30363B",
            )
            for method in method_order:
                row = task[task["method"] == method]
                if row.empty:
                    continue
                row = row.iloc[0]
                y_method = yi + method_offsets[method]
                mean = float(row["gap_mean"])
                low = float(row["gap_ci_low"])
                high = float(row["gap_ci_high"])
                ax.errorbar(
                    mean,
                    y_method,
                    xerr=np.array([[mean - low], [high - mean]]),
                    fmt=METHOD_MARKERS[method],
                    markersize=3.9,
                    markerfacecolor=METHOD_COLORS[method],
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    ecolor=METHOD_COLORS[method],
                    elinewidth=1.0,
                    capsize=1.8,
                    capthick=0.8,
                    zorder=3,
                )

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.18, 0.56)
    ax.set_ylim(-1.05, 8.95)
    ax.set_xlabel("Context gap (lower is better)", fontsize=7)
    ax.set_xticks(np.arange(0, 0.56, 0.10))
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.6)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            linestyle="",
            color=METHOD_COLORS[method],
            label=method,
            markersize=5,
        )
        for method in method_order
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.01),
        ncol=3,
        fontsize=5.8,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def failure_scatter_data() -> pd.DataFrame:
    df = pd.read_csv(PAPER / "main_context_bin_bias_table.csv")
    more = []
    for p in [PAPER / "scgpt_ten_tissue_context_bins.csv", PAPER / "tcga_pathology_context_bins.csv", PAPER / "patient_context_tissue_model_bins.csv"]:
        if p.exists():
            more.append(pd.read_csv(p))
    if more:
        df = pd.concat([df, *more], ignore_index=True, sort=False)
    df["gap_plot"] = pd.to_numeric(df.get("average_ba", np.nan), errors="coerce") - pd.to_numeric(df.get("worst_bin_ba", np.nan), errors="coerce")
    df = df[np.isfinite(df["gap_plot"]) & np.isfinite(df["average_ba"]) & np.isfinite(df["worst_bin_ba"])].copy()
    df["embedding_plot"] = series_from(df, "embedding").fillna(series_from(df, "model_label")).astype(str)
    df["domain_plot"] = df.get("domain", "").astype(str).replace(
        {"single_cell": "Single-cell technical", "pathology": "Pathology"}
    ).fillna("Clinical context")
    clinical_contexts = {"age_group", "sex", "disease"}
    df.loc[df["context"].astype(str).isin(clinical_contexts), "domain_plot"] = "Clinical context"
    dedup_keys = ["dataset", "data_task", "embedding_plot", "context", "split"]
    df = df.sort_values("gap_plot", ascending=False).drop_duplicates(dedup_keys)
    return df.reset_index(drop=True)


def plot_scatter_and_support(
    ax_scatter: plt.Axes,
    ax_support: plt.Axes,
    *,
    show_card: bool = True,
) -> None:
    if show_card:
        add_card(ax_scatter)
        add_card(ax_support)
    df = failure_scatter_data()
    colors = {
        "Single-cell technical": "#72A18B",
        "Pathology": "#6D89A2",
        "Clinical context": "#927E9E",
    }
    markers = {
        "Single-cell technical": "o",
        "Pathology": "s",
        "Clinical context": "D",
    }
    ax_scatter.axhspan(0.0, 0.5, xmin=(0.65 - 0.48) / (1.01 - 0.48), color="#FAF1EE", alpha=0.65, zorder=-4)
    for domain in ["Single-cell technical", "Pathology", "Clinical context"]:
        sub = df[df["domain_plot"] == domain]
        if sub.empty:
            continue
        ax_scatter.scatter(
            sub["average_ba"],
            sub["worst_bin_ba"],
            s=17,
            color=colors.get(domain, "#999999"),
            marker=markers.get(domain, "o"),
            alpha=0.76,
            edgecolor="#F8F4EC",
            linewidth=0.45,
            label=domain if domain else "Other",
        )
    ax_scatter.plot([0.48, 1.0], [0.48, 1.0], color="#9A9A9A", linestyle=(0, (3, 2)), linewidth=0.8)
    headline_specs = [
        ("CELLxGENE:esophagus", "scGPT", "dataset_id", "Esophagus · scGPT · dataset"),
        ("CELLxGENE:pancreas", "scGPT", "assay", "Pancreas · scGPT · assay"),
        ("TCGA", "H-optimus0", "site", "TCGA KRAS · H-optimus0 · site"),
    ]
    annotation_offsets = [(14, 7), (18, 10), (12, 8)]
    for (dataset, embedding, context, label), offset in zip(headline_specs, annotation_offsets):
        sub = df[
            (df["dataset"].astype(str) == dataset)
            & (df["embedding_plot"].astype(str) == embedding)
            & (df["context"].astype(str) == context)
        ]
        if dataset == "TCGA":
            sub = sub[sub["data_task"].astype(str) == "KRAS"]
        if sub.empty:
            continue
        row = sub.sort_values("worst_bin_ba").iloc[0]
        ax_scatter.annotate(
            f"{label}\n{row['average_ba']:.2f} → {row['worst_bin_ba']:.3f}",
            xy=(float(row["average_ba"]), float(row["worst_bin_ba"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.0,
            color="#39434A",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.78),
            arrowprops=dict(arrowstyle="-", color="#8B949A", linewidth=0.55),
        )
    ax_scatter.set_xlim(0.48, 1.01)
    ax_scatter.set_ylim(-0.02, 1.02)
    ax_scatter.set_xlabel("Average balanced accuracy", fontsize=7)
    ax_scatter.set_ylabel("Worst-bin balanced accuracy", fontsize=7)
    ax_scatter.set_title("Average performance hides context collapse", fontsize=8, fontweight="bold", pad=7)
    ax_scatter.legend(loc="upper left", fontsize=5.0, markerscale=0.75, handletextpad=0.25)
    ax_scatter.grid(color="#E8E8E8", linewidth=0.6)
    severe_n = int(((df["average_ba"] >= 0.65) & (df["worst_bin_ba"] < 0.50)).sum())
    ax_scatter.text(
        0.985,
        0.965,
        f"All {len(df)} evaluations\n{severe_n} high-average failures",
        transform=ax_scatter.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color="#6A6F73",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#D8DEE2", alpha=0.88),
    )

    support = pd.read_csv(PAPER / "support_identifiability_multiseed_table.csv")
    support = support[support["method"].isin(["label_context_reweight"])].sort_values("target_support_fraction")
    x = support["actual_support_coverage_mean"].to_numpy(dtype=float)
    ax_support.axvspan(0.20, 0.30, color="#F7EDE8", alpha=0.75, zorder=-4)
    series_specs = [
        (
            "Worst-context BA",
            support["worst_context_ba_mean"].to_numpy(dtype=float),
            support["worst_context_ba_ci_low"].to_numpy(dtype=float),
            support["worst_context_ba_ci_high"].to_numpy(dtype=float),
            "#567B9A",
            "o",
        ),
        (
            "Context gap",
            support["gap_mean"].to_numpy(dtype=float),
            support["gap_ci_low"].to_numpy(dtype=float),
            support["gap_ci_high"].to_numpy(dtype=float),
            "#B96C5C",
            "s",
        ),
    ]
    for label, mean, low, high, color, marker in series_specs:
        ax_support.errorbar(
            x,
            mean,
            yerr=np.vstack([mean - low, high - mean]),
            color=color,
            marker=marker,
            markersize=3.4,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.15,
            elinewidth=0.9,
            capsize=1.8,
            label=label,
            zorder=3,
        )
    ax_support.set_xlim(0.20, 0.53)
    ax_support.set_ylim(0, 1.02)
    ax_support.set_xlabel("Observed label–context support coverage", fontsize=7)
    ax_support.set_ylabel("Balanced-accuracy quantity", fontsize=7)
    ax_support.set_title("Support sensitivity (LC-Reweight)", fontsize=8, fontweight="bold", pad=7)
    ax_support.grid(color="#E8E8E8", linewidth=0.6)
    ax_support.legend(loc="center right", fontsize=5.0, handletextpad=0.35)
    ax_support.text(
        0.215,
        0.08,
        "lower-support\nregion",
        ha="left",
        va="bottom",
        fontsize=5.0,
        color="#8B5B50",
    )
    train_cells = " · ".join(f"{v / 1000:.1f}k" for v in support["train_cells_mean"])
    ax_support.text(
        0.5,
        0.985,
        f"Mean train cells: {train_cells}",
        transform=ax_support.transAxes,
        ha="center",
        va="top",
        fontsize=5.0,
        color="#687178",
    )
    ax_support.text(
        0.5,
        0.025,
        "5 seeds; bars = 95% CI · coverage and train n co-vary",
        transform=ax_support.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.0,
        color="#687178",
    )


def plot_legend(ax: plt.Axes) -> None:
    add_card(ax)
    ax.set_axis_off()
    ax.text(0.02, 0.86, "How to read the bubble glyphs (Panel A)", fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    xs = [0.08, 0.29, 0.48, 0.66, 0.83]
    titles = [
        "1. Larger bubble",
        "2. Warmer fill color",
        "3. Thicker ring",
        "4. Green arc",
        "5. Dashed ring",
    ]
    desc = [
        "larger context gap\n(average - worst-bin)",
        "lower worst-bin\nbalanced accuracy",
        "more support\ncoverage",
        "larger post-adaptation\ngap reduction",
        "low support\nsetting",
    ]
    cmap = gap_cmap()
    for x, title, text in zip(xs, titles, desc):
        ax.text(x, 0.70, title, ha="center", fontsize=7, fontweight="bold", transform=ax.transAxes)
        ax.text(x, 0.21, text, ha="center", va="center", fontsize=6.2, transform=ax.transAxes)
    ax.scatter([xs[0] - 0.02, xs[0] + 0.05], [0.46, 0.46], s=[390, 80], color="#C8C2B8", edgecolor="#F8F4EC", linewidth=0.7, transform=ax.transAxes, clip_on=False)
    for i, c in enumerate([0.08, 0.26, 0.48, 0.70, 0.92]):
        ax.scatter(xs[1] - 0.055 + i * 0.028, 0.46, s=100, color=cmap(c), edgecolor="#F8F4EC", linewidth=0.35, transform=ax.transAxes, clip_on=False)
    ax.scatter(xs[2] - 0.025, 0.46, s=190, color="white", edgecolor="#DAD2C6", linewidth=0.5, transform=ax.transAxes, clip_on=False)
    ax.scatter(xs[2] + 0.025, 0.46, s=190, color="white", edgecolor="#DAD2C6", linewidth=2.0, transform=ax.transAxes, clip_on=False)
    ax.scatter(xs[3], 0.46, s=210, color="white", edgecolor="#DAD2C6", linewidth=0.8, transform=ax.transAxes, clip_on=False)
    ax.add_patch(Arc((xs[3] + 0.006, 0.465), 0.060, 0.18, theta1=305, theta2=70, color="#6F927E", linewidth=1.5, transform=ax.transAxes))
    ax.scatter(xs[4], 0.46, s=210, color="white", edgecolor="#8A8177", linewidth=0.9, linestyle=(0, (2, 2)), transform=ax.transAxes, clip_on=False)
    for x in [0.19, 0.38, 0.57, 0.75]:
        ax.plot([x, x], [0.14, 0.70], color="#DDDDDD", linewidth=0.7, transform=ax.transAxes)


def plot_takeaway(ax: plt.Axes) -> None:
    add_card(ax)
    ax.set_axis_off()
    ax.text(0.18, 0.52, "◎", fontsize=34, color="#6F927E", transform=ax.transAxes, ha="center", va="center")
    ax.text(0.33, 0.77, "Takeaway", fontsize=9, fontweight="bold", color="#52675B", transform=ax.transAxes)
    ax.text(
        0.33,
        0.30,
        "Average performance can hide\nworst-bin collapse; support-\ncovered shifts are more fixable.",
        fontsize=7.7,
        color="#52675B",
        transform=ax.transAxes,
        linespacing=1.35,
    )


def plot_bubble_encoding_legend(ax: plt.Axes) -> None:
    add_card(ax)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for x in [0.365, 0.740]:
        ax.plot([x, x], [0.18, 0.82], color="#E3E8EB", linewidth=0.7)

    title_y, glyph_y, label_y = 0.72, 0.49, 0.25

    ax.text(0.055, title_y, "Size = context gap", ha="left", va="center", fontsize=7.2, fontweight="bold", color="#3F474D")
    for x, size in zip([0.220, 0.270, 0.330], [16, 40, 76]):
        ax.scatter(x, glyph_y, s=size, color="#A9B3B8", edgecolor="white", linewidth=0.5, zorder=3)
    ax.text(0.215, label_y, "small", ha="center", va="center", fontsize=6.0, color="#5B6268")
    ax.text(0.330, label_y, "large", ha="center", va="center", fontsize=6.0, color="#5B6268")

    ax.text(0.405, title_y, "Color = worst-bin BA", ha="left", va="center", fontsize=7.2, fontweight="bold", color="#3F474D")
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, extent=(0.560, 0.720, 0.40, 0.58), cmap=gap_cmap(), aspect="auto", origin="lower", zorder=2)
    ax.add_patch(Rectangle((0.560, 0.40), 0.160, 0.18, transform=ax.transData, fill=False, edgecolor="#CDD3D8", linewidth=0.5))
    ax.text(0.560, label_y, "low/worse", ha="left", va="center", fontsize=6.0, color="#8A5A51")
    ax.text(0.720, label_y, "high/better", ha="right", va="center", fontsize=6.0, color="#4E6E88")

    ax.text(0.785, title_y, "Low support", ha="left", va="center", fontsize=7.2, fontweight="bold", color="#3F474D")
    ax.scatter(
        [0.925],
        [glyph_y],
        s=[78],
        facecolors="none",
        edgecolors="#8A8177",
        linewidths=1.0,
        linestyles=(0, (2, 2)),
        zorder=3,
    )
    ax.text(0.925, label_y, "dashed ring", ha="center", va="center", fontsize=6.0, color="#5B6268")


def main() -> None:
    setup()
    # Calibrated to the final ACM double-column width.  Keeping the physical
    # canvas close to its LaTeX display size prevents unreadable down-scaling.
    fig = plt.figure(figsize=(7.25, 7.18))
    gs = fig.add_gridspec(
        4,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[2.55, 0.32, 1.55, 1.80],
        left=0.055,
        right=0.985,
        top=0.970,
        bottom=0.055,
        wspace=0.27,
        hspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_b = fig.add_subplot(gs[2, :])
    ax_c1 = fig.add_subplot(gs[3, 0])
    ax_c2 = fig.add_subplot(gs[3, 1])

    plot_bubble_atlas(ax_a, show_card=False)
    for collection in ax_a.collections:
        sizes = collection.get_sizes()
        if len(sizes):
            collection.set_sizes(np.asarray(sizes) * 0.30)
    for text in ax_a.texts:
        if text.get_text() == "Same-modality bias bubble-dot atlas":
            text.set_text("Context-failure atlas across biomedical embeddings")
            text.set_fontsize(9.0)
            text.set_y(1.012)
    ax_a.text(
        -0.018,
        1.012,
        "a",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        clip_on=False,
    )
    plot_bubble_encoding_legend(ax_legend)
    plot_mitigation(ax_b, show_card=False)
    ax_b.set_xlabel("")
    ax_b.text(
        -0.018,
        1.055,
        "b",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        clip_on=False,
    )
    plot_scatter_and_support(ax_c1, ax_c2, show_card=False)
    for label, axis in [("c", ax_c1), ("d", ax_c2)]:
        axis.text(
            -0.055,
            1.055,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            clip_on=False,
        )

    stem = OUT / "figure2_context_reliability_results"
    save_all(fig, stem)
    fig.savefig(stem.with_name(stem.name + "_preview").with_suffix(".png"), dpi=180)
    print(stem.with_name(stem.name + "_preview").with_suffix(".png"))


if __name__ == "__main__":
    main()
