#!/usr/bin/env python3
"""Plot paired SCA-vs-LC effects and support/sample-size sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_LABELS = {"geneformer_v1": "Geneformer", "scgpt_continual": "scGPT"}
CONTEXT_LABELS = {"assay": "assay", "dataset_id": "dataset"}
SPLIT_LABELS = {
    "patient_level_cv": "Observed-context patient-CV",
    "leave_one_context": "Unseen-context leave-one",
}
METHOD_LABELS = {
    "erm_mlp": "ERM",
    "label_context_reweight": "LC-Reweight",
    "sca_lite": "SCA-Align",
}
METHOD_COLORS = {
    "erm_mlp": "#5B6268",
    "label_context_reweight": "#2B7A78",
    "sca_lite": "#C77B30",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=ROOT / "data/cellxgene_support_calibrated_formal",
    )
    parser.add_argument(
        "--support-summary-csv",
        type=Path,
        default=ROOT
        / "data/figure4_support_sample_size_sensitivity/support_identifiability_multiseed_table.csv",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "figures/figure4_intervention_regime",
    )
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "data/figure_source"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def percentile_mean_ci(
    values: np.ndarray, rng: np.random.Generator, samples: int
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError(f"Paired comparison needs at least two seeds; found {len(values)}")
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load_paired_effects(
    root: Path, samples: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    paths = sorted(root.glob("bone_marrow_*/*/*/summary/per_seed_gaps.csv"))
    if len(paths) != 4:
        raise ValueError(f"Expected four bone-marrow tasks; found {len(paths)} under {root}")
    for path in paths:
        task, model, context = path.relative_to(root).parts[:3]
        frame = pd.read_csv(path)
        frame = frame.loc[frame["method"].isin(["label_context_reweight", "sca_lite"])].copy()
        pivot = frame.pivot_table(
            index=["seed", "split_type"],
            columns="method",
            values=["worst_value", "gap"],
            aggfunc="first",
        )
        pivot.columns = [f"{metric}_{method}" for metric, method in pivot.columns]
        pivot = pivot.reset_index()
        pivot["delta_worst_sca_minus_lc"] = (
            pivot["worst_value_sca_lite"] - pivot["worst_value_label_context_reweight"]
        )
        pivot["gap_reduction_sca_vs_lc"] = (
            pivot["gap_label_context_reweight"] - pivot["gap_sca_lite"]
        )
        for row in pivot.to_dict(orient="records"):
            raw_rows.append(
                {
                    "task": task,
                    "model": model,
                    "context_field": context,
                    "task_label": f"{MODEL_LABELS[model]} · {CONTEXT_LABELS[context]}",
                    **row,
                    "source_csv": str(path),
                }
            )
    raw = pd.DataFrame.from_records(raw_rows)
    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    keys = ["task", "model", "context_field", "task_label", "split_type"]
    for key, group in raw.groupby(keys, sort=False):
        if group["seed"].nunique() != 5:
            raise ValueError(f"Expected five paired seeds for {key}; found {group['seed'].nunique()}")
        worst = percentile_mean_ci(
            group["delta_worst_sca_minus_lc"].to_numpy(float), rng, samples
        )
        gap = percentile_mean_ci(
            group["gap_reduction_sca_vs_lc"].to_numpy(float), rng, samples
        )
        summary_rows.append(
            {
                **dict(zip(keys, key)),
                "split_label": SPLIT_LABELS[key[-1]],
                "n_seeds": int(group["seed"].nunique()),
                "delta_worst_mean": worst[0],
                "delta_worst_ci_low": worst[1],
                "delta_worst_ci_high": worst[2],
                "gap_reduction_mean": gap[0],
                "gap_reduction_ci_low": gap[1],
                "gap_reduction_ci_high": gap[2],
                "paired_seeds": ";".join(str(value) for value in sorted(group["seed"].unique())),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows)
    return summary, raw


def row_layout(frame: pd.DataFrame) -> pd.DataFrame:
    order = {
        ("geneformer_v1", "assay"): 0,
        ("geneformer_v1", "dataset_id"): 1,
        ("scgpt_continual", "assay"): 2,
        ("scgpt_continual", "dataset_id"): 3,
    }
    split_order = {"patient_level_cv": 0, "leave_one_context": 1}
    output = frame.copy()
    output["task_order"] = output.apply(
        lambda row: order[(str(row["model"]), str(row["context_field"]))], axis=1
    )
    output["split_order"] = output["split_type"].map(split_order)
    output = output.sort_values(["split_order", "task_order"]).reset_index(drop=True)
    output["y"] = [8, 7, 6, 5, 3, 2, 1, 0]
    return output


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E3E6E8", linewidth=0.55, zorder=0)
    ax.tick_params(labelsize=6.2, length=2.5)
    ax.set_axisbelow(True)


def plot_forest(
    ax: plt.Axes,
    frame: pd.DataFrame,
    mean_col: str,
    low_col: str,
    high_col: str,
    xlabel: str,
    show_labels: bool,
) -> None:
    values = frame[mean_col].to_numpy(float)
    lows = frame[low_col].to_numpy(float)
    highs = frame[high_col].to_numpy(float)
    y = frame["y"].to_numpy(float)
    ax.axvline(0, color="#626A70", linewidth=0.8, zorder=1)
    for split_type, color in (("patient_level_cv", "#2B7A78"), ("leave_one_context", "#C77B30")):
        mask = frame["split_type"].eq(split_type).to_numpy()
        ax.errorbar(
            values[mask],
            y[mask],
            xerr=np.vstack([values[mask] - lows[mask], highs[mask] - values[mask]]),
            fmt="none",
            ecolor=color,
            elinewidth=1.25,
            capsize=2.2,
            zorder=2,
        )
        ax.scatter(
            values[mask],
            y[mask],
            s=27,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    for level in y:
        ax.axhline(level, color="#F0F1F2", linewidth=0.45, zorder=0)
    ax.axhline(4, color="#AEB4B8", linewidth=0.7, zorder=0)
    ax.set_yticks(y)
    if show_labels:
        ax.set_yticklabels(frame["task_label"], fontsize=6.1)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(-0.7, 8.75)
    max_abs = max(abs(float(lows.min())), abs(float(highs.max())), 0.025) * 1.16
    ax.set_xlim(-max_abs, max_abs)
    ax.set_xlabel(xlabel, fontsize=6.7)
    ax.text(
        0.98,
        0.98,
        "SCA-Align better →",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color="#8A5B22",
    )
    style_axis(ax)


def plot_sensitivity(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    low: str,
    high: str,
    ylabel: str,
) -> None:
    ax.axvspan(20, 50, color="#F4E6E4", alpha=0.78, zorder=0)
    ax.axvspan(50, 105, color="#E8F0EA", alpha=0.74, zorder=0)
    ax.axvline(50, color="#8B9297", linestyle="--", linewidth=0.7, zorder=1)
    for method in ("erm_mlp", "label_context_reweight", "sca_lite"):
        subset = frame.loc[frame["method"].eq(method)].sort_values("target_support_fraction")
        if subset.empty:
            raise ValueError(f"Missing {method} from support sensitivity table")
        x = 100.0 * subset["target_support_fraction"].to_numpy(float)
        y = subset[metric].to_numpy(float)
        lower = subset[low].to_numpy(float)
        upper = subset[high].to_numpy(float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - lower, upper - y]),
            color=METHOD_COLORS[method],
            marker="o",
            markersize=3.5,
            linewidth=1.25,
            capsize=2.0,
            label=METHOD_LABELS[method],
            zorder=3,
        )
    ax.set_xlim(20, 105)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xlabel("Cross-context label overlap retained (%)", fontsize=6.7)
    ax.set_ylabel(ylabel, fontsize=6.7)
    ax.text(
        0.02,
        0.04,
        "Lower-overlap regime\nwithhold adaptation claim",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color="#8A4F48",
    )
    ax.text(
        0.98,
        0.04,
        "Comparison more supported\nnot guaranteed valid",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color="#466A50",
    )
    style_axis(ax)


def main() -> None:
    args = parse_args()
    paired, raw = load_paired_effects(args.formal_root, args.bootstrap_samples, args.seed)
    paired = row_layout(paired)
    support = pd.read_csv(args.support_summary_csv)
    support = support.loc[support["method"].isin(METHOD_LABELS)].copy()
    if support["n_seeds"].min() != 5:
        raise ValueError("Support sensitivity panel requires five seeds for every method and level")

    args.source_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.source_dir / "figure4_sca_vs_lc_paired.csv", index=False)
    raw.to_csv(args.source_dir / "figure4_sca_vs_lc_paired_seed_values.csv", index=False)
    support.to_csv(args.source_dir / "figure4_support_sample_size_sensitivity.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )
    comparison_prefix = args.output_prefix.parent / "figure4_intervention_comparison"
    comparison = plt.figure(figsize=(7.20, 2.65))
    comparison_grid = comparison.add_gridspec(
        1,
        2,
        left=0.16,
        right=0.985,
        bottom=0.20,
        top=0.90,
        wspace=0.18,
    )
    comparison_worst = comparison.add_subplot(comparison_grid[0, 0])
    comparison_gap = comparison.add_subplot(
        comparison_grid[0, 1],
        sharey=comparison_worst,
    )
    plot_forest(
        comparison_worst,
        paired,
        "delta_worst_mean",
        "delta_worst_ci_low",
        "delta_worst_ci_high",
        r"$\Delta$ worst-bin BA = SCA − LC",
        True,
    )
    plot_forest(
        comparison_gap,
        paired,
        "gap_reduction_mean",
        "gap_reduction_ci_low",
        "gap_reduction_ci_high",
        r"$-\Delta$ gap = gap$_{LC}$ − gap$_{SCA}$",
        False,
    )
    comparison_worst.set_title(
        "Worst-bin performance",
        loc="left",
        fontsize=7.5,
        fontweight="bold",
        pad=6,
    )
    comparison_gap.set_title(
        "Context-gap reduction",
        loc="left",
        fontsize=7.5,
        fontweight="bold",
        pad=6,
    )
    comparison_worst.text(
        0.01,
        8.52,
        "Observed patient-CV",
        transform=comparison_worst.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=6.0,
        color="#2B7A78",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )
    comparison_worst.text(
        0.01,
        3.63,
        "Unseen leave-one",
        transform=comparison_worst.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=6.0,
        color="#C77B30",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 450})):
        comparison.savefig(comparison_prefix.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(comparison)

    support_prefix = args.output_prefix.parent / "figureA_support_boundary"
    support_figure, support_axes = plt.subplots(
        1,
        2,
        figsize=(7.20, 2.65),
        gridspec_kw={
            "left": 0.09,
            "right": 0.985,
            "bottom": 0.20,
            "top": 0.87,
            "wspace": 0.28,
        },
    )
    plot_sensitivity(
        support_axes[0],
        support,
        "worst_context_ba_mean",
        "worst_context_ba_ci_low",
        "worst_context_ba_ci_high",
        "Worst-bin BA",
    )
    plot_sensitivity(
        support_axes[1],
        support,
        "gap_mean",
        "gap_ci_low",
        "gap_ci_high",
        "Context gap",
    )
    support_axes[0].set_title(
        "Worst-bin performance",
        loc="left",
        fontsize=7.5,
        fontweight="bold",
        pad=6,
    )
    support_axes[1].set_title(
        "Context gap",
        loc="left",
        fontsize=7.5,
        fontweight="bold",
        pad=6,
    )
    support_axes[0].legend(
        loc="upper left",
        fontsize=6.0,
        ncol=3,
        handlelength=1.5,
        columnspacing=0.8,
    )
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 450})):
        support_figure.savefig(support_prefix.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(support_figure)

    fig = plt.figure(figsize=(7.20, 5.15))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.25, 1.0],
        left=0.16,
        right=0.985,
        bottom=0.105,
        top=0.94,
        hspace=0.42,
    )
    top = outer[0].subgridspec(1, 2, wspace=0.18)
    bottom = outer[1].subgridspec(1, 2, wspace=0.30)
    ax_worst = fig.add_subplot(top[0, 0])
    ax_gap = fig.add_subplot(top[0, 1], sharey=ax_worst)
    ax_support_worst = fig.add_subplot(bottom[0, 0])
    ax_support_gap = fig.add_subplot(bottom[0, 1])

    plot_forest(
        ax_worst,
        paired,
        "delta_worst_mean",
        "delta_worst_ci_low",
        "delta_worst_ci_high",
        r"$\Delta$ worst-bin BA = SCA − LC",
        True,
    )
    plot_forest(
        ax_gap,
        paired,
        "gap_reduction_mean",
        "gap_reduction_ci_low",
        "gap_reduction_ci_high",
        r"$-\Delta$ gap = gap$_{LC}$ − gap$_{SCA}$",
        False,
    )
    ax_worst.set_title("Worst-bin performance", loc="left", fontsize=7.5, fontweight="bold", pad=6)
    ax_gap.set_title("Context-gap reduction", loc="left", fontsize=7.5, fontweight="bold", pad=6)
    ax_worst.text(-0.36, 1.07, "a", transform=ax_worst.transAxes, fontsize=9, fontweight="bold")
    ax_worst.text(
        0.01,
        8.52,
        "Observed patient-CV",
        transform=ax_worst.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=6.0,
        color="#2B7A78",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )
    ax_worst.text(
        0.01,
        3.63,
        "Unseen leave-one",
        transform=ax_worst.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=6.0,
        color="#C77B30",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )

    plot_sensitivity(
        ax_support_worst,
        support,
        "worst_context_ba_mean",
        "worst_context_ba_ci_low",
        "worst_context_ba_ci_high",
        "Worst-bin BA",
    )
    plot_sensitivity(
        ax_support_gap,
        support,
        "gap_mean",
        "gap_ci_low",
        "gap_ci_high",
        "Context gap",
    )
    ax_support_worst.set_title("Support-and-sample-size sensitivity", loc="left", fontsize=7.5, fontweight="bold", pad=6)
    ax_support_gap.set_title("Support-and-sample-size sensitivity", loc="left", fontsize=7.5, fontweight="bold", pad=6)
    ax_support_worst.text(-0.22, 1.08, "b", transform=ax_support_worst.transAxes, fontsize=9, fontweight="bold")
    ax_support_gap.text(-0.22, 1.08, "c", transform=ax_support_gap.transAxes, fontsize=9, fontweight="bold")
    ax_support_worst.legend(loc="upper left", fontsize=6.0, ncol=3, handlelength=1.5, columnspacing=0.8)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 450})):
        fig.savefig(args.output_prefix.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {args.output_prefix}.[svg|pdf|png]")
    print(f"paired rows: {len(paired)}; sensitivity rows: {len(support)}")


if __name__ == "__main__":
    main()
