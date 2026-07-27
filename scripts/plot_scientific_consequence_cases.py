#!/usr/bin/env python3
"""Create combined and single-column figures for the scientific-consequence cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import PercentFormatter


MODEL_ORDER = ["Sequencing", "CONCH", "H-optimus0", "UNI", "Virchow2"]
MODEL_COLORS = {
    "Sequencing": "#26343D",
    "CONCH": "#3978A8",
    "H-optimus0": "#3B8C86",
    "UNI": "#C9872C",
    "Virchow2": "#A7544E",
}
MODEL_MARKERS = {
    "Sequencing": "o",
    "CONCH": "s",
    "H-optimus0": "D",
    "UNI": "^",
    "Virchow2": "v",
}
METHOD_ORDER = ["Source annotation", "ERM", "LC-Reweight", "SCA-Align", "GroupDRO"]
LINEAGE_COLORS = {
    "Acinar": "#2F7D68",
    "Ductal": "#B7658A",
    "Other": "#D9DEE1",
}
INK = "#25333B"
MUTED = "#68747B"
GRID = "#DCE2E5"
ALERT = "#A7544E"
PANEL_FACE = "#F6F7F7"
ACINAR_LABELS = {"acinar cell", "pancreatic acinar cell"}
DUCTAL_LABEL = "pancreatic ductal cell"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("data/scientific_consequence_cases"),
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path(
            "data/cellxgene_support_calibrated_formal/pancreas_assay/"
            "scgpt_continual/assay"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "data/scientific_consequence_cases/review/"
            "figure_scientific_consequences_review"
        ),
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path(
            "data/scientific_consequence_cases/review/"
            "figure_scientific_consequences_source.csv"
        ),
    )
    parser.add_argument(
        "--tp53-output-prefix",
        type=Path,
        default=Path(
            "data/scientific_consequence_cases/review/"
            "figure_scientific_consequence_tp53_single_column"
        ),
    )
    parser.add_argument(
        "--pancreas-output-prefix",
        type=Path,
        default=Path(
            "data/scientific_consequence_cases/review/"
            "figure_scientific_consequence_pancreas_single_column"
        ),
    )
    return parser.parse_args()


def tint(hex_color: str, amount: float = 0.70) -> tuple[float, float, float]:
    rgb = np.asarray(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * amount)


def rounded_percentages(values: list[float]) -> list[float]:
    """Round to one decimal place while preserving an exact displayed sum of 100.0%."""
    scaled = np.asarray(values, dtype=float) * 1000.0
    units = np.floor(scaled + 1e-12).astype(int)
    remainder = int(1000 - units.sum())
    if remainder > 0:
        order = np.argsort(-(scaled - units), kind="stable")
        units[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(scaled - units, kind="stable")
        units[order[: -remainder]] -= 1
    return (units / 10.0).tolist()


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
    )


def case_tag(ax: plt.Axes, text: str, color: str, x: float = 0.0, y: float = 1.105) -> None:
    ax.text(
        x,
        y,
        text.upper(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.08",
            "facecolor": tint(color, 0.88),
            "edgecolor": "none",
        },
    )


def load_tp53(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = pd.read_csv(case_dir / "pathology_tp53_site_pair.csv")
    boot = pd.read_csv(case_dir / "pathology_tp53_site_pair_bootstrap.csv")
    sequencing = {
        "model": "Sequencing",
        "site_a": str(pair.iloc[0]["site_a"]),
        "site_b": str(pair.iloc[0]["site_b"]),
        "prevalence_a": float(pair.iloc[0]["true_prevalence_a"]),
        "prevalence_b": float(pair.iloc[0]["true_prevalence_b"]),
        "contrast": float(pair.iloc[0]["true_difference_a_minus_b"]),
        "contrast_ci_low": float(boot.iloc[0]["true_difference_ci_low"]),
        "contrast_ci_high": float(boot.iloc[0]["true_difference_ci_high"]),
        "distortion": 0.0,
        "distortion_ci_low": np.nan,
        "distortion_ci_high": np.nan,
        "direction_reversal": False,
    }
    records = [sequencing]
    for row in pair.to_dict(orient="records"):
        ci = boot.loc[boot["model"].eq(row["model"])].iloc[0]
        records.append(
            {
                "model": row["model"],
                "site_a": str(row["site_a"]),
                "site_b": str(row["site_b"]),
                "prevalence_a": float(row["predicted_prevalence_a"]),
                "prevalence_b": float(row["predicted_prevalence_b"]),
                "contrast": float(row["predicted_difference_a_minus_b"]),
                "contrast_ci_low": float(ci["predicted_difference_ci_low"]),
                "contrast_ci_high": float(ci["predicted_difference_ci_high"]),
                "distortion": float(row["predicted_difference_a_minus_b"])
                - float(row["true_difference_a_minus_b"]),
                "distortion_ci_low": float(ci["prediction_minus_truth_ci_low"]),
                "distortion_ci_high": float(ci["prediction_minus_truth_ci_high"]),
                "direction_reversal": bool(row["direction_reversal"]),
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
    frame = frame.sort_values("model").reset_index(drop=True)
    support = pd.DataFrame(
        [
            {"site": "50", "tp53_positive": 18, "tp53_negative": 13, "n_patients": 31},
            {"site": "86", "tp53_positive": 17, "tp53_negative": 17, "n_patients": 34},
        ]
    )
    return frame, support


def load_pancreas(case_dir: Path, prediction_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    abundance = pd.read_csv(case_dir / "pancreas_abundance_summary.csv")
    cell_error = pd.read_csv(case_dir / "pancreas_cell_error_summary.csv")
    support = pd.read_csv(case_dir / "pancreas_acinar_support.csv")

    true_acinar = float(abundance["true_acinar_lineage_abundance"].iloc[0])
    true_ductal = float(
        pd.read_csv(case_dir / "pancreas_abundance_per_seed.csv")["true_ductal_abundance"].iloc[0]
    )
    rows = [
        {
            "method_label": "Source annotation",
            "acinar": true_acinar,
            "ductal": true_ductal,
            "other": 1.0 - true_acinar - true_ductal,
            "n_seeds": 1,
        }
    ]
    for method in METHOD_ORDER[1:]:
        row = abundance.loc[abundance["method_label"].eq(method)].iloc[0]
        acinar = float(row["predicted_acinar_lineage_abundance_mean"])
        ductal = float(row["predicted_ductal_abundance_mean"])
        rows.append(
            {
                "method_label": method,
                "acinar": acinar,
                "ductal": ductal,
                "other": 1.0 - acinar - ductal,
                "n_seeds": int(row["n_seeds"]),
            }
        )
    composition = pd.DataFrame.from_records(rows)

    prediction_files = sorted(prediction_root.glob("seed_*/*/leave_one_context_predictions.csv"))
    donor_rows = []
    for path in prediction_files:
        seed = path.parts[-3].replace("seed_", "")
        pred = pd.read_csv(path)
        pred = pred.loc[
            pred["method"].eq("erm_mlp")
            & pred["context_value"].eq("microwell-seq")
            & pred["true_label"].isin(ACINAR_LABELS)
        ].copy()
        for donor, group in pred.groupby("donor_id", sort=True):
            acinar = float(group["pred_label"].isin(ACINAR_LABELS).mean())
            ductal = float(group["pred_label"].eq(DUCTAL_LABEL).mean())
            donor_rows.append(
                {
                    "seed": seed,
                    "donor": donor,
                    "n_true_acinar_cells": int(len(group)),
                    "acinar": acinar,
                    "ductal": ductal,
                    "other": 1.0 - acinar - ductal,
                }
            )
    donors = pd.DataFrame.from_records(donor_rows)
    donors = (
        donors.groupby("donor", sort=True)
        .agg(
            n_true_acinar_cells=("n_true_acinar_cells", "first"),
            acinar=("acinar", "mean"),
            ductal=("ductal", "mean"),
            other=("other", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    target_support = support.loc[
        support["context_value"].eq("microwell-seq")
        & support["label"].eq("acinar cell")
    ].iloc[0]
    flow = cell_error.loc[cell_error["method_label"].eq("ERM")].iloc[0]
    boundary = pd.DataFrame(
        [
            {
                "acinar": float(flow["acinar_lineage_recall_mean"]),
                "ductal": float(flow["acinar_to_ductal_rate_mean"]),
                "other": 1.0
                - float(flow["acinar_lineage_recall_mean"])
                - float(flow["acinar_to_ductal_rate_mean"]),
                "n_true_acinar_cells": int(flow["n_true_acinar_cells"]),
                "n_acinar_positive_donors": int(target_support["n_donors"]),
                "donor_threshold": int(target_support["min_donors"]),
            }
        ]
    )
    donors["n_acinar_positive_donors"] = int(boundary.iloc[0]["n_acinar_positive_donors"])
    donors["donor_threshold"] = int(boundary.iloc[0]["donor_threshold"])
    return composition, donors, boundary


def plot_prevalence(ax: plt.Axes, tp53: pd.DataFrame, support: pd.DataFrame) -> None:
    y = np.arange(len(MODEL_ORDER))[::-1]
    height = 0.29
    for ypos, model in zip(y, MODEL_ORDER):
        row = tp53.loc[tp53["model"].astype(str).eq(model)].iloc[0]
        color = MODEL_COLORS[model]
        ax.barh(ypos + 0.17, row["prevalence_a"], height=height, color=color, zorder=2)
        ax.barh(
            ypos - 0.17,
            row["prevalence_b"],
            height=height,
            color=tint(color, 0.68),
            edgecolor=color,
            linewidth=0.45,
            zorder=2,
        )
        if model == "Sequencing":
            ax.text(
                0.018,
                ypos + 0.17,
                "site 50",
                va="center",
                ha="left",
                fontsize=5.0,
                color="white",
                fontweight="bold",
                zorder=3,
            )
            ax.text(
                0.018,
                ypos - 0.17,
                "site 86",
                va="center",
                ha="left",
                fontsize=5.0,
                color=INK,
                fontweight="bold",
                zorder=3,
            )
        ax.text(
            min(float(row["prevalence_a"]) + 0.018, 0.96),
            ypos + 0.17,
            f"{row['prevalence_a']:.0%}",
            va="center",
            ha="left",
            fontsize=5.3,
            color=INK,
        )
        ax.text(
            min(float(row["prevalence_b"]) + 0.018, 0.96),
            ypos - 0.17,
            f"{row['prevalence_b']:.0%}",
            va="center",
            ha="left",
            fontsize=5.3,
            color=INK,
        )
    ax.axhspan(3.47, 4.52, color=PANEL_FACE, zorder=-2)
    ax.set_yticks(y)
    ax.set_yticklabels(MODEL_ORDER, fontsize=6.2)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("TP53-positive prevalence", fontsize=6.4)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=5.8)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_title(
        "Image-derived prevalence reverses the site contrast",
        loc="left",
        fontsize=7.6,
        fontweight="bold",
        pad=7,
    )
    panel_label(ax, "a", x=-0.16, y=1.235)
    case_tag(ax, "Label–site support met", MODEL_COLORS["CONCH"], x=0.0, y=1.215)


def forest_interval(
    ax: plt.Axes,
    y: float,
    low: float,
    high: float,
    point: float,
    color: str,
    marker: str,
) -> None:
    ax.plot([low, high], [y, y], color=color, linewidth=1.45, solid_capstyle="round", zorder=2)
    ax.plot([low, low], [y - 0.08, y + 0.08], color=color, linewidth=0.75, zorder=2)
    ax.plot([high, high], [y - 0.08, y + 0.08], color=color, linewidth=0.75, zorder=2)
    ax.scatter(
        [point],
        [y],
        s=27,
        marker=marker,
        color=color,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )


def plot_contrasts(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    tp53: pd.DataFrame,
    show_tag: bool = True,
) -> None:
    y = np.arange(len(MODEL_ORDER))[::-1]
    for ypos, model in zip(y, MODEL_ORDER):
        row = tp53.loc[tp53["model"].astype(str).eq(model)].iloc[0]
        color = MODEL_COLORS[model]
        forest_interval(
            ax_left,
            ypos,
            100.0 * float(row["contrast_ci_low"]),
            100.0 * float(row["contrast_ci_high"]),
            100.0 * float(row["contrast"]),
            color,
            MODEL_MARKERS[model],
        )
        if model != "Sequencing":
            forest_interval(
                ax_right,
                ypos,
                100.0 * float(row["distortion_ci_low"]),
                100.0 * float(row["distortion_ci_high"]),
                100.0 * float(row["distortion"]),
                color,
                MODEL_MARKERS[model],
            )
    for ax in (ax_left, ax_right):
        ax.axvline(0.0, color="#77848A", linestyle=(0, (2.0, 2.0)), linewidth=0.8, zorder=1)
        ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=5.5)
        ax.set_axisbelow(True)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(MODEL_ORDER, fontsize=5.8)
    ax_right.set_yticks(y)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_left.set_xlim(-90, 42)
    ax_left.set_xticks([-80, -40, 0, 40])
    ax_right.set_xlim(-102, 8)
    ax_right.set_xticks([-100, -50, 0])
    ax_left.set_xlabel("Site 50 − site 86 (percentage points)", fontsize=5.7)
    ax_right.set_xlabel(
        "Predicted contrast −\nsequencing contrast\n(percentage points)",
        fontsize=5.5,
    )
    ax_left.set_title("Contrast (95% patient bootstrap CI)", loc="left", fontsize=6.6, fontweight="bold", pad=8)
    ax_right.set_title("Model-induced distortion", loc="left", fontsize=6.6, fontweight="bold", pad=8)
    panel_label(ax_left, "b", x=-0.30, y=1.25 if show_tag else 1.12)
    if show_tag:
        case_tag(ax_left, "Label–site support met", MODEL_COLORS["CONCH"], x=0.0, y=1.215)
    ax_right.text(
        0.98,
        0.98,
        "4/4 reverse\nthe observed point estimate",
        transform=ax_right.transAxes,
        ha="right",
        va="top",
        fontsize=5.4,
        color=ALERT,
        fontweight="bold",
    )
    ax_right.text(
        0.98,
        0.10,
        "All distortion CIs exclude 0",
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.1,
        color=MUTED,
    )


def plot_composition(
    ax: plt.Axes,
    composition: pd.DataFrame,
    panel: str = "c",
    compact: bool = False,
) -> None:
    y = np.arange(len(METHOD_ORDER))[::-1]
    for ypos, method in zip(y, METHOD_ORDER):
        row = composition.loc[composition["method_label"].eq(method)].iloc[0]
        columns = ["acinar", "ductal", "other"]
        displayed = rounded_percentages([float(row[column]) for column in columns])
        left = 0.0
        for (lineage, column), display_value in zip(
            [("Acinar", "acinar"), ("Ductal", "ductal"), ("Other", "other")],
            displayed,
        ):
            value = float(row[column])
            ax.barh(
                ypos,
                value,
                left=left,
                height=0.62,
                color=LINEAGE_COLORS[lineage],
                edgecolor="white",
                linewidth=0.65,
                zorder=2,
            )
            if value >= 0.055:
                text_color = "white" if lineage != "Other" else INK
                ax.text(
                    left + value / 2,
                    ypos,
                    f"{display_value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=5.3,
                    color=text_color,
                    fontweight="bold" if method in {"Source annotation", "ERM"} else "normal",
                )
            left += value
    ax.axhspan(3.48, 4.52, color=PANEL_FACE, zorder=-2)
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_ORDER, fontsize=6.0)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("" if compact else "Reconstructed lineage composition", fontsize=6.4)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=5.8)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_title(
        "Source composition is reversed",
        loc="left",
        fontsize=7.6,
        fontweight="bold",
        pad=7,
    )
    panel_label(ax, panel, x=-0.16, y=1.215)
    case_tag(ax, "Donor support not met", ALERT, x=0.0, y=1.195)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=LINEAGE_COLORS[name], edgecolor="none", label=name.lower())
        for name in ["Acinar", "Ductal", "Other"]
    ]
    ax.legend(
        handles=handles,
        loc="upper right" if compact else "lower center",
        bbox_to_anchor=(1.0, 1.075) if compact else (0.50, -0.26),
        ncol=3,
        frameon=False,
        fontsize=5.5,
        handlelength=1.3,
        columnspacing=1.0,
    )
    if not compact:
        ax.text(
            0.0,
            -0.39,
            f"Panel {panel}: sampling-corrected donor composition; capped target sample, n=754 cells.\n"
            "Other = all non-acinar, non-ductal lineages; source ductal annotation = 0.0%.",
            transform=ax.transAxes,
            fontsize=5.15,
            color=MUTED,
            ha="left",
            va="top",
        )


def ribbon_patch(
    x0: float,
    x1: float,
    y0_low: float,
    y0_high: float,
    y1_low: float,
    y1_high: float,
    color: str,
) -> PathPatch:
    bend = (x1 - x0) * 0.46
    vertices = [
        (x0, y0_low),
        (x0 + bend, y0_low),
        (x1 - bend, y1_low),
        (x1, y1_low),
        (x1, y1_high),
        (x1 - bend, y1_high),
        (x0 + bend, y0_high),
        (x0, y0_high),
        (x0, y0_low),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none", alpha=0.78)


def plot_flow_and_support(
    ax: plt.Axes,
    donors: pd.DataFrame,
    boundary: pd.DataFrame,
    panel: str = "d",
    show_tag: bool = True,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, panel, x=-0.12, y=1.285 if show_tag else 1.15)
    if show_tag:
        case_tag(ax, "Donor support not met", ALERT, x=0.0, y=1.265)
    ax.set_title(
        "Failure is clear; remodeling claim unsupported",
        loc="left",
        fontsize=7.6,
        fontweight="bold",
        pad=7,
    )

    row = boundary.iloc[0]
    proportions = {
        "Acinar": float(row["acinar"]),
        "Ductal": float(row["ductal"]),
        "Other": float(row["other"]),
    }
    x0, x1 = 0.17, 0.69
    y_bottom, y_top = 0.48, 0.91
    height = y_top - y_bottom
    current_left = y_top
    right_positions = {}
    current_right = y_top
    for lineage in ["Acinar", "Ductal", "Other"]:
        band_height = height * proportions[lineage]
        left_high = current_left
        left_low = current_left - band_height
        right_high = current_right
        right_low = current_right - band_height
        ax.add_patch(
            ribbon_patch(
                x0,
                x1,
                left_low,
                left_high,
                right_low,
                right_high,
                LINEAGE_COLORS[lineage],
            )
        )
        right_positions[lineage] = (right_low, right_high)
        current_left = left_low
        current_right = right_low
    ax.add_patch(Rectangle((0.04, y_bottom), 0.13, height, facecolor="#EEF1F2", edgecolor=INK, linewidth=0.8))
    ax.text(0.105, (y_bottom + y_top) / 2, "True\nacinar", ha="center", va="center", fontsize=6.2, fontweight="bold", color=INK)
    for lineage in ["Acinar", "Ductal", "Other"]:
        low, high = right_positions[lineage]
        ax.add_patch(
            Rectangle(
                (x1, low),
                0.12,
                high - low,
                facecolor=LINEAGE_COLORS[lineage],
                edgecolor="white",
                linewidth=0.5,
            )
        )
        center = (low + high) / 2
        text_color = "white" if lineage != "Other" else INK
        ax.text(
            x1 + 0.06,
            center,
            f"{lineage}\n{proportions[lineage]:.1%}",
            ha="center",
            va="center",
            fontsize=5.4,
            color=text_color,
            fontweight="bold" if lineage == "Ductal" else "normal",
        )
    ax.text(
        0.04,
        0.94,
        f"Panel {panel}: capped evaluation sample · 402 annotated acinar cells within 754 target cells",
        fontsize=5.25,
        color=MUTED,
        ha="left",
        va="bottom",
    )

    ax.text(0.04, 0.405, "Acinar-positive donors · mean of five seeds", fontsize=5.5, color=MUTED, ha="left", va="bottom")
    donor_order = [value for value in ["Donor9", "Donor11", "Donor44"] if value in set(donors["donor"])]
    donor_y = np.linspace(0.32, 0.16, len(donor_order))
    bar_x, bar_width, bar_height = 0.19, 0.37, 0.050
    for ypos, donor in zip(donor_y, donor_order):
        donor_row = donors.loc[donors["donor"].eq(donor)].iloc[0]
        ax.text(0.04, ypos, donor, fontsize=5.4, color=INK, ha="left", va="center")
        left = bar_x
        for lineage in ["Acinar", "Ductal", "Other"]:
            value = float(donor_row[lineage.lower()])
            ax.add_patch(
                Rectangle(
                    (left, ypos - bar_height / 2),
                    bar_width * value,
                    bar_height,
                    facecolor=LINEAGE_COLORS[lineage],
                    edgecolor="white",
                    linewidth=0.3,
                )
            )
            left += bar_width * value
        ax.text(
            bar_x + bar_width + 0.015,
            ypos,
            f"n={int(donor_row['n_true_acinar_cells'])}",
            fontsize=4.9,
            color=MUTED,
            ha="left",
            va="center",
        )

    support_donors = int(row["n_acinar_positive_donors"])
    threshold = int(row["donor_threshold"])
    card_x, card_y, card_w, card_h = 0.68, 0.065, 0.31, 0.325
    ax.add_patch(
        Rectangle(
            (card_x, card_y),
            card_w,
            card_h,
            facecolor=tint(ALERT, 0.91),
            edgecolor=tint(ALERT, 0.55),
            linewidth=0.65,
        )
    )
    icon_y = card_y + 0.262
    icon_x = np.linspace(card_x + 0.045, card_x + card_w - 0.045, threshold)
    for index, xpos in enumerate(icon_x):
        filled = index < support_donors
        ax.add_patch(
            Circle(
                (xpos, icon_y),
                0.017,
                facecolor=ALERT if filled else "white",
                edgecolor=ALERT,
                linewidth=0.8,
            )
        )
    ax.text(
        card_x + card_w / 2,
        card_y + 0.207,
        f"{support_donors}  <  {threshold}",
        ha="center",
        va="center",
        fontsize=7.2,
        color=ALERT,
        fontweight="bold",
    )
    ax.text(
        card_x + card_w / 2,
        card_y + 0.158,
        "acinar-positive donors\nobserved / required",
        ha="center",
        va="center",
        fontsize=4.65,
        color=ALERT,
    )
    ax.plot(
        [card_x + 0.025, card_x + card_w - 0.025],
        [card_y + 0.103, card_y + 0.103],
        color=tint(ALERT, 0.55),
        linewidth=0.55,
    )
    ax.text(
        card_x + card_w / 2,
        card_y + 0.052,
        "Withhold biological\nremodeling claim",
        ha="center",
        va="center",
        fontsize=4.55,
        color=INK,
    )


def export_source_data(
    tp53: pd.DataFrame,
    tp53_support: pd.DataFrame,
    composition: pd.DataFrame,
    donors: pd.DataFrame,
    boundary: pd.DataFrame,
    path: Path,
) -> None:
    blocks = []
    block = tp53.copy()
    block.insert(0, "panel", "a_b")
    blocks.append(block)
    block = tp53_support.copy()
    block.insert(0, "panel", "a_support")
    blocks.append(block)
    block = composition.copy()
    block.insert(0, "panel", "c")
    blocks.append(block)
    block = donors.copy()
    block.insert(0, "panel", "d_donor")
    blocks.append(block)
    block = boundary.copy()
    block.insert(0, "panel", "d_flow_support")
    blocks.append(block)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(blocks, ignore_index=True, sort=False).to_csv(path, index=False)


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def compact_panel_header(
    ax: plt.Axes,
    panel: str,
    title: str,
    *,
    tag: str | None = None,
    tag_color: str = INK,
    label_x: float = -0.30,
) -> None:
    ax.text(
        label_x,
        1.17,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
    )
    ax.set_title(title, loc="left", fontsize=5.7, fontweight="bold", pad=5.5)
    if tag is not None:
        ax.text(
            0.0,
            1.19,
            tag.upper(),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=3.8,
            fontweight="bold",
            color=tag_color,
            bbox={
                "boxstyle": "round,pad=0.16,rounding_size=0.05",
                "facecolor": tint(tag_color, 0.89),
                "edgecolor": "none",
            },
        )


def plot_tp53_prevalence_compact(
    ax: plt.Axes,
    tp53: pd.DataFrame,
) -> None:
    """Paired site-prevalence display sized for half of one ACM column."""
    y = np.arange(len(MODEL_ORDER))[::-1]
    for ypos, model in zip(y, MODEL_ORDER):
        row = tp53.loc[tp53["model"].astype(str).eq(model)].iloc[0]
        color = MODEL_COLORS[model]
        site_a = float(row["prevalence_a"])
        site_b = float(row["prevalence_b"])
        ax.plot(
            [site_a, site_b],
            [ypos, ypos],
            color=tint(color, 0.28),
            linewidth=1.15,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            site_a,
            ypos,
            s=17,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            site_b,
            ypos,
            s=17,
            marker="s",
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            1.135,
            ypos,
            f"{100.0 * float(row['contrast']):+.0f}",
            ha="right",
            va="center",
            fontsize=4.3,
            color=ALERT if bool(row["direction_reversal"]) else INK,
            fontweight="bold",
        )

    ax.set_xlim(0.0, 1.16)
    ax.set_ylim(-0.65, len(MODEL_ORDER) - 0.35)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="x", labelsize=4.6, length=2.0, pad=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(MODEL_ORDER, fontsize=4.8)
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_xlabel("TP53-positive prevalence", fontsize=4.8, labelpad=1.8)
    compact_panel_header(
        ax,
        "a",
        "Image prevalence reverses contrast",
        tag="Label–site support met",
        tag_color=MODEL_COLORS["CONCH"],
        label_x=-0.37,
    )
    ax.text(
        1.135,
        len(MODEL_ORDER) - 0.23,
        r"$\Delta$ pp",
        ha="right",
        va="bottom",
        fontsize=4.0,
        color=MUTED,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=INK,
            markeredgecolor="white",
            markersize=3.2,
            label="site 50",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=INK,
            markersize=3.0,
            label="site 86",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(-0.01, -0.34),
        ncol=2,
        frameon=False,
        fontsize=4.2,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.0,
    )


def plot_tp53_distortion_compact(
    ax: plt.Axes,
    tp53: pd.DataFrame,
) -> None:
    """Patient-bootstrap distortion forest sized for half of one ACM column."""
    models = MODEL_ORDER[1:]
    y = np.arange(len(models))[::-1]
    for ypos, model in zip(y, models):
        row = tp53.loc[tp53["model"].astype(str).eq(model)].iloc[0]
        forest_interval(
            ax,
            ypos,
            100.0 * float(row["distortion_ci_low"]),
            100.0 * float(row["distortion_ci_high"]),
            100.0 * float(row["distortion"]),
            MODEL_COLORS[model],
            MODEL_MARKERS[model],
        )

    ax.axvline(0.0, color="#77848A", linestyle=(0, (2.0, 2.0)), linewidth=0.75, zorder=1)
    ax.set_xlim(-105, 7)
    ax.set_ylim(-0.75, len(models) - 0.35)
    ax.set_xticks([-100, -50, 0])
    ax.tick_params(axis="x", labelsize=4.6, length=2.0, pad=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=4.8)
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_xlabel("Predicted − sequencing contrast (pp)", fontsize=4.6, labelpad=1.8)
    compact_panel_header(
        ax,
        "b",
        "Model-induced distortion (95% CI)",
        label_x=-0.41,
    )
    ax.text(
        0.98,
        0.02,
        "4/4 CIs exclude 0",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.2,
        color=ALERT,
        fontweight="bold",
    )


def plot_pancreas_composition_compact(
    ax: plt.Axes,
    composition: pd.DataFrame,
) -> None:
    """Sampling-corrected composition sized for half of one ACM column."""
    y = np.arange(len(METHOD_ORDER))[::-1]
    for ypos, method in zip(y, METHOD_ORDER):
        row = composition.loc[composition["method_label"].eq(method)].iloc[0]
        values = [float(row["acinar"]), float(row["ductal"]), float(row["other"])]
        displayed = rounded_percentages(values)
        left = 0.0
        for lineage, value, display_value in zip(
            ["Acinar", "Ductal", "Other"],
            values,
            displayed,
        ):
            ax.barh(
                ypos,
                value,
                left=left,
                height=0.58,
                color=LINEAGE_COLORS[lineage],
                edgecolor="white",
                linewidth=0.45,
                zorder=2,
            )
            if value >= 0.17:
                ax.text(
                    left + value / 2,
                    ypos,
                    f"{display_value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=4.1,
                    color="white" if lineage != "Other" else INK,
                    fontweight="bold" if method in {"Source annotation", "ERM"} else "normal",
                )
            left += value
        if values[0] < 0.17:
            ax.text(
                values[0] + 0.012,
                ypos,
                f"{displayed[0]:.1f}",
                ha="left",
                va="center",
                fontsize=4.0,
                color=LINEAGE_COLORS["Acinar"],
                fontweight="bold",
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.65, len(METHOD_ORDER) - 0.35)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="x", labelsize=4.6, length=2.0, pad=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(
        ["Source", "ERM", "LC-Reweight", "SCA-Align", "GroupDRO"],
        fontsize=4.6,
    )
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_xlabel("Sampling-corrected composition", fontsize=4.7, labelpad=1.8)
    compact_panel_header(
        ax,
        "a",
        "Source composition is reversed",
        tag="Donor support not met",
        tag_color=ALERT,
        label_x=-0.40,
    )
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=LINEAGE_COLORS[name], edgecolor="none", label=name.lower())
        for name in ["Acinar", "Ductal", "Other"]
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(-0.01, -0.34),
        ncol=3,
        frameon=False,
        fontsize=4.0,
        handlelength=1.0,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.0,
    )


def plot_pancreas_boundary_compact(
    ax: plt.Axes,
    boundary: pd.DataFrame,
) -> None:
    """Prediction-flow and donor-support boundary for half of one ACM column."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    compact_panel_header(
        ax,
        "b",
        "Failure clear; claim unsupported",
        label_x=-0.14,
    )
    row = boundary.iloc[0]
    values = [
        float(row["acinar"]),
        float(row["ductal"]),
        float(row["other"]),
    ]
    displayed = [100.0 * value for value in values]

    ax.text(
        0.02,
        0.91,
        "402 annotated acinar cells\n(capped within 754 target cells)",
        ha="left",
        va="top",
        fontsize=4.05,
        color=MUTED,
    )
    left = 0.02
    bar_y, bar_h, bar_w = 0.55, 0.16, 0.96
    for lineage, value, display_value in zip(
        ["Acinar", "Ductal", "Other"],
        values,
        displayed,
    ):
        width = bar_w * value
        ax.add_patch(
            Rectangle(
                (left, bar_y),
                width,
                bar_h,
                facecolor=LINEAGE_COLORS[lineage],
                edgecolor="white",
                linewidth=0.45,
            )
        )
        if value >= 0.12:
            ax.text(
                left + width / 2,
                bar_y + bar_h / 2,
                f"{lineage}\n{display_value:.1f}%",
                ha="center",
                va="center",
                fontsize=4.0,
                color="white" if lineage != "Other" else INK,
                fontweight="bold" if lineage == "Ductal" else "normal",
            )
        left += width
    ax.text(
        0.02,
        0.49,
        "ERM predictions from true acinar",
        ha="left",
        va="top",
        fontsize=4.2,
        color=MUTED,
    )

    support_donors = int(row["n_acinar_positive_donors"])
    threshold = int(row["donor_threshold"])
    card_y, card_h = 0.08, 0.32
    ax.add_patch(
        Rectangle(
            (0.02, card_y),
            0.96,
            card_h,
            facecolor=tint(ALERT, 0.93),
            edgecolor=tint(ALERT, 0.55),
            linewidth=0.6,
        )
    )
    icon_x = np.linspace(0.08, 0.39, threshold)
    icon_y = 0.30
    for index, xpos in enumerate(icon_x):
        filled = index < support_donors
        ax.add_patch(
            Circle(
                (xpos, icon_y),
                0.027,
                facecolor=ALERT if filled else "white",
                edgecolor=ALERT,
                linewidth=0.75,
            )
        )
    ax.text(
        0.235,
        0.17,
        f"{support_donors} < {threshold} donors",
        ha="center",
        va="center",
        fontsize=4.5,
        color=ALERT,
        fontweight="bold",
    )
    ax.plot([0.49, 0.49], [card_y + 0.04, card_y + card_h - 0.04], color=tint(ALERT, 0.58), linewidth=0.55)
    ax.text(
        0.735,
        0.27,
        "Report prediction\nfailure",
        ha="center",
        va="center",
        fontsize=4.2,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.735,
        0.16,
        "Withhold remodeling\nclaim",
        ha="center",
        va="center",
        fontsize=4.0,
        color=ALERT,
    )


def plot_tp53_single_column(
    tp53: pd.DataFrame,
    support: pd.DataFrame,
    output_prefix: Path,
) -> None:
    """Render the TCGA-LUAD case as a two-panel, single-column figure."""
    fig = plt.figure(figsize=(3.42, 1.78), constrained_layout=False, facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 0.95],
        wspace=0.64,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    plot_tp53_prevalence_compact(ax_a, tp53)
    plot_tp53_distortion_compact(ax_b, tp53)

    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.24, top=0.78)
    save_figure(fig, output_prefix)


def plot_pancreas_single_column(
    composition: pd.DataFrame,
    donors: pd.DataFrame,
    boundary: pd.DataFrame,
    output_prefix: Path,
) -> None:
    """Render the pancreas case as a two-panel, single-column figure."""
    fig = plt.figure(figsize=(3.42, 1.78), constrained_layout=False, facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 0.95],
        wspace=0.56,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    plot_pancreas_composition_compact(ax_a, composition)
    plot_pancreas_boundary_compact(ax_b, boundary)

    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.24, top=0.78)
    save_figure(fig, output_prefix)


def main() -> None:
    args = parse_args()
    tp53, tp53_support = load_tp53(args.case_dir)
    composition, donors, boundary = load_pancreas(args.case_dir, args.prediction_root)
    export_source_data(tp53, tp53_support, composition, donors, boundary, args.source_csv)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.20, 6.35), constrained_layout=False, facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.98, 1.16],
        height_ratios=[1.0, 1.04],
        wspace=0.38,
        hspace=0.53,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    b_grid = outer[0, 1].subgridspec(1, 2, width_ratios=[1.18, 0.96], wspace=0.35)
    ax_b1 = fig.add_subplot(b_grid[0, 0])
    ax_b2 = fig.add_subplot(b_grid[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    plot_prevalence(ax_a, tp53, tp53_support)
    plot_contrasts(ax_b1, ax_b2, tp53)
    plot_composition(ax_c, composition)
    plot_flow_and_support(ax_d, donors, boundary)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.08, top=0.925)
    save_figure(fig, args.output_prefix)
    plot_tp53_single_column(tp53, tp53_support, args.tp53_output_prefix)
    plot_pancreas_single_column(
        composition,
        donors,
        boundary,
        args.pancreas_output_prefix,
    )


if __name__ == "__main__":
    main()
