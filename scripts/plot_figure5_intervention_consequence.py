#!/usr/bin/env python3
"""Plot the three-step biological-consequence intervention case for Figure 5."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/donor_abundance_consequence/differential_abundance_direction.csv"
OUTPUT = ROOT / "figures/figure5_intervention_consequence"
SOURCE_OUTPUT = ROOT / "data/figure_source/figure5_intervention_consequence.csv"

INK = "#26343D"
MUTED = "#68747B"
GRID = "#DCE2E5"
ALERT = "#B44D4D"
BASELINE = "#7B858A"
LC = "#D08B3E"
SCA = "#2F7D68"
TRUE = "#2D5F8B"

METHOD_ORDER = ["ERM", "LC-Reweight", "SCA-Align"]
METHOD_LABELS = {
    "ERM": "Baseline",
    "LC-Reweight": "LC-Reweight",
    "SCA-Align": "SCA-Align",
}
METHOD_COLORS = {"ERM": BASELINE, "LC-Reweight": LC, "SCA-Align": SCA}


def select_case() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE)
    case = frame.loc[
        frame["task"].eq("bone_marrow_dataset")
        & frame["model"].eq("geneformer_v1")
        & frame["context_field"].eq("dataset_id")
        & frame["split_type"].eq("leave_one_context")
        & frame["label"].eq("neutrophil")
        & frame["comparison"].eq("B-cell non-Hodgkin lymphoma - normal")
        & frame["method_label"].isin(METHOD_ORDER)
    ].copy()
    if case["seed"].nunique() != 5:
        raise RuntimeError("Figure 5 case must contain five evaluation seeds.")
    truth = case["true_delta"].unique()
    if len(truth) != 1:
        raise RuntimeError("Figure 5 case has inconsistent annotated contrasts.")
    return case


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)
    ax.tick_params(labelsize=5.2, length=2.2, pad=1.5)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    ax.set_title(title, loc="left", fontsize=6.5, fontweight="bold", pad=6)


def panel_a(ax: plt.Axes, case: pd.DataFrame) -> None:
    baseline = case.loc[case["method_label"].eq("ERM")].sort_values("seed")
    truth = 100.0 * float(baseline["true_delta"].iloc[0])
    values = 100.0 * baseline["predicted_delta"].to_numpy(float)

    ax.axvspan(-20, 0, color="#EAF1F6", zorder=0)
    ax.axvspan(0, 15, color="#F8E8E5", zorder=0)
    ax.axvline(0, color="#929A9E", lw=0.7, zorder=1)
    ax.axvline(truth, color=TRUE, lw=1.2, zorder=2)
    jitter = np.linspace(-0.18, 0.18, len(values))
    ax.scatter(
        values,
        np.ones_like(values) + jitter,
        s=22,
        color=ALERT,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    mean = float(values.mean())
    ax.scatter(
        [mean],
        [1],
        s=45,
        marker="D",
        color=INK,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    ax.text(
        truth,
        1.30,
        f"Annotated\n{truth:+.1f} pp",
        ha="center",
        va="bottom",
        fontsize=5.0,
        color=TRUE,
        fontweight="bold",
    )
    ax.text(
        mean,
        0.80,
        f"Baseline mean\n{mean:+.1f} pp",
        ha="center",
        va="top",
        fontsize=5.0,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.97,
        0.92,
        "5/5 seeds\nreverse direction",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color=ALERT,
        fontweight="bold",
    )
    ax.set_xlim(-20, 15)
    ax.set_ylim(0.52, 1.55)
    ax.set_yticks([])
    ax.set_xlabel(
        "Neutrophil abundance: B-cell lymphoma − normal (percentage points)",
        fontsize=5.3,
    )
    style_axis(ax)
    panel_label(ax, "a", "Dataset shift reverses a disease-associated comparison")


def panel_b(ax: plt.Axes, case: pd.DataFrame) -> None:
    truth = 100.0 * float(case["true_delta"].iloc[0])
    y = np.arange(len(METHOD_ORDER))[::-1]
    ax.axvspan(-20, 0, color="#EAF1F6", zorder=0)
    ax.axvspan(0, 15, color="#F8E8E5", zorder=0)
    ax.axvline(0, color="#929A9E", lw=0.7, zorder=1)
    ax.axvline(truth, color=TRUE, lw=1.0, ls=(0, (3, 2)), zorder=2)

    for position, method in zip(y, METHOD_ORDER):
        subset = case.loc[case["method_label"].eq(method)].sort_values("seed")
        values = 100.0 * subset["predicted_delta"].to_numpy(float)
        jitter = np.linspace(-0.10, 0.10, len(values))
        ax.scatter(
            values,
            position + jitter,
            s=18,
            color=METHOD_COLORS[method],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        mean = float(values.mean())
        ax.scatter(
            [mean],
            [position],
            marker="D",
            s=38,
            color=METHOD_COLORS[method],
            edgecolor=INK,
            linewidth=0.45,
            zorder=4,
        )
        ax.text(
            mean + (0.8 if mean >= 0 else -0.8),
            position + 0.22,
            f"{mean:+.1f}",
            ha="left" if mean >= 0 else "right",
            va="center",
            fontsize=4.9,
            color=METHOD_COLORS[method],
            fontweight="bold",
        )

    ax.text(
        truth,
        2.48,
        f"Annotation {truth:+.1f}",
        ha="center",
        va="bottom",
        fontsize=4.8,
        color=TRUE,
    )
    ax.set_xlim(-20, 15)
    ax.set_ylim(-0.45, 2.68)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=5.2)
    ax.set_xlabel("Estimated cohort contrast (percentage points)", fontsize=5.3)
    style_axis(ax)
    panel_label(ax, "b", "SCA-Align restores the average direction")


def panel_c(ax: plt.Axes, case: pd.DataFrame) -> None:
    truth = float(case["true_delta"].iloc[0])
    pivot = case.pivot(index="seed", columns="method_label", values="predicted_delta")
    baseline = 100.0 * (pivot["ERM"] - truth).abs()
    sca = 100.0 * (pivot["SCA-Align"] - truth).abs()
    for seed in pivot.index:
        ax.plot(
            [0, 1],
            [baseline.loc[seed], sca.loc[seed]],
            color="#AEB7BB",
            lw=0.75,
            zorder=1,
        )
        ax.scatter(
            [0, 1],
            [baseline.loc[seed], sca.loc[seed]],
            s=17,
            color=[BASELINE, SCA],
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
    means = [float(baseline.mean()), float(sca.mean())]
    ax.plot([0, 1], means, color=INK, lw=1.3, zorder=3)
    ax.scatter(
        [0, 1],
        means,
        marker="D",
        s=42,
        color=[BASELINE, SCA],
        edgecolor=INK,
        linewidth=0.5,
        zorder=4,
    )
    reduction = 100.0 * (1.0 - means[1] / means[0])
    sign_recovery = int(
        (
            np.sign(pivot["SCA-Align"].to_numpy(float))
            == np.sign(truth)
        ).sum()
    )
    ax.text(
        0,
        means[0] + 1.0,
        f"{means[0]:.1f} pp",
        ha="center",
        va="bottom",
        fontsize=5.0,
        color=BASELINE,
        fontweight="bold",
    )
    ax.text(
        1,
        means[1] - 1.0,
        f"{means[1]:.1f} pp",
        ha="center",
        va="top",
        fontsize=5.0,
        color=SCA,
        fontweight="bold",
    )
    ax.text(
        0.97,
        0.94,
        f"{reduction:.0f}% lower mean distortion\n"
        f"direction recovered in {sign_recovery}/5 seeds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color=SCA,
        fontweight="bold",
    )
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(10, 30)
    ax.set_xticks([0, 1], ["Baseline", "SCA-Align"])
    ax.set_ylabel("Absolute contrast distortion (pp)", fontsize=5.3)
    style_axis(ax)
    panel_label(ax, "c", "SCA-Align consistently reduces distortion")


def main() -> None:
    case = select_case()
    fig = plt.figure(figsize=(3.42, 4.45), facecolor="white")
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.95, 1.15, 1.05],
        left=0.23,
        right=0.97,
        bottom=0.08,
        top=0.97,
        hspace=0.63,
    )
    panel_a(fig.add_subplot(grid[0]), case)
    panel_b(fig.add_subplot(grid[1]), case)
    panel_c(fig.add_subplot(grid[2]), case)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    case.to_csv(SOURCE_OUTPUT, index=False)
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
    ):
        fig.savefig(OUTPUT.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
