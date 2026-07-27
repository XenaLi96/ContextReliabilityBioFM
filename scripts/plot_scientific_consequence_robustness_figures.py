#!/usr/bin/env python3
"""Plot the TP53 robustness and pancreas ambiguity manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plot_scientific_consequence_cases import (
    LINEAGE_COLORS,
    MODEL_COLORS,
    MODEL_MARKERS,
    MODEL_ORDER,
    load_pancreas,
    load_tp53,
)


ROOT = Path(__file__).resolve().parents[1]
INK = "#26343D"
MUTED = "#68747B"
GRID = "#DCE2E5"
ALERT = "#A7544E"
GREEN = "#2F7D68"
AMBER = "#B57A28"


def label_panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
    )
    ax.set_title(title, loc="left", fontsize=6.6, fontweight="bold", pad=6)


def style_quant_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.45, zorder=0)
    ax.tick_params(labelsize=5.3, length=2.3, pad=1.5)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
    ):
        fig.savefig(prefix.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_tp53_case(ax: plt.Axes, case: pd.DataFrame) -> None:
    order = MODEL_ORDER
    y = np.arange(len(order))[::-1]
    for position, model in zip(y, order):
        row = case.loc[case["model"].astype(str).eq(model)].iloc[0]
        color = MODEL_COLORS[model]
        estimate = 100.0 * float(row["contrast"])
        low = 100.0 * float(row["contrast_ci_low"])
        high = 100.0 * float(row["contrast_ci_high"])
        if np.isfinite(low) and np.isfinite(high):
            ax.plot([low, high], [position, position], color=color, lw=1.0, zorder=2)
        ax.scatter(
            estimate,
            position,
            s=22,
            marker=MODEL_MARKERS[model],
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.text(
            estimate + (3 if estimate >= 0 else -3),
            position,
            f"{estimate:+.1f}",
            ha="left" if estimate >= 0 else "right",
            va="center",
            fontsize=4.8,
            color=color,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25},
        )
    ax.axvline(0, color="#7B858A", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=5.3)
    ax.set_xlim(-85, 25)
    ax.set_xlabel("Site 50 − Site 86 contrast (pp)", fontsize=5.5)
    style_quant_axis(ax)
    label_panel(ax, "a", "Example: all four image models reverse direction")


def plot_tp53_all_pairs(ax: plt.Axes, pairs: pd.DataFrame) -> None:
    x = 100.0 * pairs["true_difference"].to_numpy(float)
    y = 100.0 * pairs["predicted_difference"].to_numpy(float)
    limit = max(70.0, float(np.nanmax(np.abs(np.r_[x, y]))) * 1.08)
    ax.fill_between(
        [-limit, 0],
        0,
        limit,
        color="#F8E8E5",
        alpha=0.9,
        zorder=0,
    )
    ax.fill_between(
        [0, limit],
        -limit,
        0,
        color="#F8E8E5",
        alpha=0.9,
        zorder=0,
    )
    for model in MODEL_ORDER[1:]:
        subset = pairs.loc[pairs["model"].eq(model)]
        ax.scatter(
            100.0 * subset["true_difference"],
            100.0 * subset["predicted_difference"],
            s=17,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            label=model,
            zorder=3,
        )
    ax.axhline(0, color="#707B80", lw=0.7)
    ax.axvline(0, color="#707B80", lw=0.7)
    ax.plot([-limit, limit], [-limit, limit], color="#9CA4A8", lw=0.55, ls=(0, (3, 2)))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Sequencing contrast (pp)", fontsize=5.5)
    ax.set_ylabel("Image-model contrast (pp)", fontsize=5.5)
    style_quant_axis(ax)
    label_panel(ax, "b", "All 60 supported model–site-pair comparisons")
    ax.text(
        0.98,
        0.04,
        "Shaded quadrants = reversal",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.6,
        color=ALERT,
    )
    ax.legend(
        loc="upper left",
        fontsize=4.2,
        frameon=False,
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.7,
    )


def plot_tp53_sensitivity(
    ax: plt.Axes,
    summary: pd.DataFrame,
) -> None:
    selected_probability = summary.loc[
        summary["calibration"].eq("selected")
        & summary["estimator"].eq("probability")
    ].iloc[0]
    hard = summary.loc[
        summary["calibration"].eq("selected")
        & summary["estimator"].eq("hard")
    ].sort_values("threshold")
    probability = summary.loc[
        summary["estimator"].eq("probability")
        & summary["calibration"].isin(["raw", "platt", "isotonic"])
    ].copy()
    labels = ["Mean\nscore"] + [f"{value:.2f}" for value in hard["threshold"]]
    values = [100.0 * float(selected_probability["sign_reversal_rate"])] + (
        100.0 * hard["sign_reversal_rate"].to_numpy(float)
    ).tolist()
    x = np.arange(len(values))
    ax.plot(x[1:], values[1:], color="#3978A8", marker="o", ms=3.0, lw=1.0)
    ax.scatter(
        [x[0]],
        [values[0]],
        color=GREEN,
        marker="D",
        s=20,
        zorder=3,
    )
    calibration_x = np.arange(len(values) + 1, len(values) + 4)
    calibration_order = ["raw", "platt", "isotonic"]
    calibration_values = []
    for calibration in calibration_order:
        row = probability.loc[probability["calibration"].eq(calibration)].iloc[0]
        calibration_values.append(100.0 * float(row["sign_reversal_rate"]))
    ax.scatter(
        calibration_x,
        calibration_values,
        color=AMBER,
        marker="s",
        s=19,
        zorder=3,
    )
    ax.axvline(len(values) - 0.5, color="#AAB1B5", lw=0.6, ls="--")
    ax.set_xticks(
        np.r_[x, calibration_x],
        labels
        + ["Raw\nscore", "Platt", "Isotonic"],
        rotation=45,
        ha="right",
        fontsize=4.4,
    )
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Direction reversals (%)", fontsize=5.5)
    style_quant_axis(ax)
    label_panel(ax, "c", "Reversals persist across thresholds and calibration")
    ax.text(
        0.44,
        0.97,
        "Hard-prediction threshold",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=4.4,
        color="#3978A8",
    )


def make_tp53_figure() -> None:
    case, _ = load_tp53(ROOT / "data/scientific_consequence_cases")
    pairs = pd.read_csv(
        ROOT / "data/pathology_prevalence_robustness/site_pair_robustness.csv"
    )
    pairs = pairs.loc[
        pairs["task"].eq("TP53_status")
        & pairs["calibration"].eq("selected")
        & pairs["estimator"].eq("probability")
    ].copy()
    summary = pd.read_csv(
        ROOT
        / "data/pathology_prevalence_robustness/site_pair_robustness_summary.csv"
    )
    summary = summary.loc[summary["task"].eq("TP53_status")].copy()
    fig = plt.figure(figsize=(3.42, 4.65), facecolor="white")
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.9, 1.3, 1.0],
        left=0.20,
        right=0.97,
        bottom=0.09,
        top=0.97,
        hspace=0.58,
    )
    plot_tp53_case(fig.add_subplot(grid[0]), case)
    plot_tp53_all_pairs(fig.add_subplot(grid[1]), pairs)
    plot_tp53_sensitivity(fig.add_subplot(grid[2]), summary)
    source = ROOT / "data/figure_source"
    source.mkdir(parents=True, exist_ok=True)
    case.to_csv(source / "figure5_tp53_case.csv", index=False)
    pairs.to_csv(source / "figure5_tp53_all_pairs.csv", index=False)
    summary.to_csv(source / "figure5_tp53_sensitivity.csv", index=False)
    save(
        fig,
        ROOT / "figures/figure5_tp53_scientific_consequence",
    )


def plot_composition(ax: plt.Axes, composition: pd.DataFrame) -> None:
    methods = ["Source annotation", "ERM", "LC-Reweight", "SCA-Align"]
    labels = ["Annotation", "Baseline", "LC-Reweight", "SCA-Align"]
    y = np.arange(len(methods))[::-1]
    for position, method in zip(y, methods):
        row = composition.loc[composition["method_label"].eq(method)].iloc[0]
        left = 0.0
        for lineage, column in (
            ("Acinar", "acinar"),
            ("Ductal", "ductal"),
            ("Other", "other"),
        ):
            value = float(row[column])
            ax.barh(
                position,
                100.0 * value,
                left=100.0 * left,
                height=0.58,
                color=LINEAGE_COLORS[lineage],
                edgecolor="white",
                linewidth=0.45,
                zorder=2,
            )
            if value >= 0.15:
                ax.text(
                    100.0 * (left + value / 2),
                    position,
                    f"{100 * value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color="white" if lineage != "Other" else INK,
                    fontweight="bold",
                )
            left += value
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.0)
    ax.set_xlim(0, 100)
    ax.set_xlabel(
        "Sampling-corrected composition (%)",
        fontsize=5.5,
        labelpad=4.2,
    )
    style_quant_axis(ax)
    label_panel(ax, "a", "Reconstruction shifts lineage composition")
    handles = [
        Patch(
            facecolor=LINEAGE_COLORS[name],
            edgecolor="none",
            label=name,
        )
        for name in ("Acinar", "Ductal", "Other")
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=3,
        fontsize=4.4,
        frameon=False,
        handlelength=1.25,
        handleheight=0.75,
        handletextpad=0.45,
        columnspacing=1.25,
        borderaxespad=0,
    )


def plot_marker_donors(ax: plt.Axes, marker: pd.DataFrame) -> None:
    selected = marker.loc[
        marker["prediction_group"].isin(
            ["acinar_retained", "acinar_miscalled_ductal"]
        )
    ].copy()
    donor = (
        selected.groupby(["donor_id", "prediction_group"], as_index=False)
        .agg(
            score=("acinar_minus_ductal", "mean"),
            low=("acinar_minus_ductal", "min"),
            high=("acinar_minus_ductal", "max"),
            n_cells=("n_cells", "mean"),
        )
    )
    order = ["acinar_retained", "acinar_miscalled_ductal"]
    x_lookup = dict(zip(order, [0, 1]))
    colors = {"Donor9": "#3978A8", "Donor11": "#A7544E"}
    for donor_id, group in donor.groupby("donor_id"):
        group = group.set_index("prediction_group").reindex(order).dropna()
        xs = np.asarray([x_lookup[value] for value in group.index], dtype=float)
        ys = group["score"].to_numpy(float)
        color = colors.get(str(donor_id), "#68747B")
        if len(group) == 2:
            ax.plot(xs, ys, color=color, alpha=0.55, lw=0.8)
        ax.errorbar(
            xs,
            ys,
            yerr=np.vstack(
                [
                    ys - group["low"].to_numpy(float),
                    group["high"].to_numpy(float) - ys,
                ]
            ),
            fmt="o",
            color=color,
            ms=3.8,
            lw=0.8,
            capsize=2,
            label=str(donor_id),
        )
    ax.axhline(0, color="#7B858A", lw=0.7, ls="--")
    ax.set_xticks([0, 1], ["Retained\nas acinar", "Assigned\nto ductal"], fontsize=5.0)
    ax.set_ylabel("Acinar − ductal marker score", fontsize=5.3)
    ax.set_xlim(-0.35, 1.35)
    style_quant_axis(ax)
    label_panel(ax, "b", "Ductal assignments remain marker-ambiguous")
    ax.legend(
        title="Donor mean\n(range over seeds)",
        title_fontsize=4.1,
        fontsize=4.2,
        frameon=False,
        loc="lower left",
    )
    ax.text(
        0.98,
        0.04,
        "Median donor–seed score:\n−0.219 vs −0.576",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.3,
        color=MUTED,
    )


def make_pancreas_figure() -> None:
    composition, _, _ = load_pancreas(
        ROOT / "data/scientific_consequence_cases",
        ROOT
        / "data/cellxgene_support_calibrated_formal/pancreas_assay/"
        "scgpt_continual/assay",
    )
    marker = pd.read_csv(ROOT / "data/pancreas_marker_sanity/donor_marker_scores.csv")
    fig = plt.figure(figsize=(3.42, 3.66), facecolor="white")
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.02, 1.0],
        left=0.22,
        right=0.97,
        bottom=0.08,
        top=0.96,
        hspace=0.72,
    )
    plot_composition(fig.add_subplot(grid[0]), composition)
    plot_marker_donors(fig.add_subplot(grid[1]), marker)
    source = ROOT / "data/figure_source"
    composition.to_csv(source / "figure6_pancreas_composition.csv", index=False)
    marker.to_csv(source / "figure6_pancreas_marker_donors.csv", index=False)
    save(
        fig,
        ROOT / "figures/figure6_pancreas_scientific_consequence",
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )
    make_tp53_figure()
    make_pancreas_figure()
    print("wrote Figure 5 and Figure 6 robustness figures")


if __name__ == "__main__":
    main()
