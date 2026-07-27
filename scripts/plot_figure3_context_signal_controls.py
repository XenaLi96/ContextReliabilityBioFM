#!/usr/bin/env python3
"""Plot context recoverability diagnostics and artifact-residualization controls."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DOMAIN_LABELS = {
    "single_cell": "Single-cell",
    "pathology": "Pathology",
    "spatial_pathology": "Spatial pathology",
}
DOMAIN_COLORS = {
    "single_cell": "#2B7A78",
    "pathology": "#4267A9",
    "spatial_pathology": "#B75D45",
}
DOMAIN_MARKERS = {"single_cell": "o", "pathology": "s", "spatial_pathology": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "data/failure_warning_model/failure_warning_predictions.csv",
    )
    parser.add_argument(
        "--residualized-csv",
        type=Path,
        default=ROOT / "data/paper_tables/residualized_control_delta_table.csv",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "figures/figure3_context_signal_controls",
    )
    parser.add_argument(
        "--cellxgene-root",
        type=Path,
        default=ROOT / "data/embeddings/cellxgene",
        help="Root containing <model>_<tissue>/metadata.csv directories.",
    )
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "data/figure_source"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def pathology_cardinalities() -> dict[tuple[str, str, str], int]:
    path = ROOT / "data/paper_tables/tcga_pathology_context_bins.csv"
    frame = pd.read_csv(path)
    result: dict[tuple[str, str, str], int] = {}
    for row in frame.itertuples(index=False):
        match = re.match(r"\s*(\d+)\s+held-out contexts", str(row.note))
        if match:
            key = (f"TCGA:{row.data_task}_status", str(row.embedding).lower(), str(row.context))
            result[key] = int(match.group(1))
    return result


def spatial_cardinalities() -> dict[tuple[str, str], int]:
    path = ROOT / "data/hest51_context_shift_mainline/representation_context_leakage.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["status"].eq("ok")]
    return {
        (str(row.model).lower(), str(row.field)): int(row.n_classes)
        for row in frame.itertuples(index=False)
        if pd.notna(row.n_classes)
    }


def cellxgene_cardinality(
    dataset: str,
    model: str,
    context: str,
    cellxgene_root: Path,
) -> float:
    tissue = dataset.split(":", 1)[1]
    metadata = cellxgene_root / f"{model}_{tissue}" / "metadata.csv"
    if not metadata.exists():
        metadata = cellxgene_root / f"scgpt_continual_{tissue}" / "metadata.csv"
    if not metadata.exists():
        return float("nan")
    values = pd.read_csv(metadata, usecols=[context])[context].dropna().astype(str)
    return float(values.nunique())


def add_context_cardinality(
    frame: pd.DataFrame,
    cellxgene_root: Path,
) -> pd.DataFrame:
    pathology = pathology_cardinalities()
    spatial = spatial_cardinalities()
    output = frame.copy()
    values: list[float] = []
    for row in output.itertuples(index=False):
        value = float(row.n_context_values) if pd.notna(row.n_context_values) else float("nan")
        if np.isfinite(value):
            values.append(value)
            continue
        if row.domain == "single_cell":
            value = cellxgene_cardinality(
                str(row.dataset),
                str(row.model),
                str(row.context),
                cellxgene_root,
            )
        elif row.domain == "pathology":
            value = float(
                pathology.get((str(row.dataset), str(row.model).lower(), str(row.context)), np.nan)
            )
        elif row.domain == "spatial_pathology":
            value = float(spatial.get((str(row.model).lower(), str(row.context)), np.nan))
        values.append(value)
    output["context_cardinality"] = values
    output["probe_lift_above_chance"] = (
        output["context_probe_ba"] - 1.0 / output["context_cardinality"]
    ) / (1.0 - 1.0 / output["context_cardinality"])
    output["support_status"] = np.where(
        pd.to_numeric(output["support_coverage"], errors="coerce").ge(0.8),
        "support met",
        "partial / insufficient",
    )
    return output


def cluster_bootstrap_spearman(
    frame: pd.DataFrame, x: str, y: str, clusters: str, samples: int, seed: int
) -> tuple[float, float, float]:
    rho = float(spearmanr(frame[x], frame[y]).statistic)
    names = frame[clusters].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.full(samples, np.nan)
    groups = {name: frame.loc[frame[clusters].eq(name)] for name in names}
    for index in range(samples):
        sampled = rng.choice(names, size=len(names), replace=True)
        pieces = [groups[name] for name in sampled]
        candidate = pd.concat(pieces, ignore_index=True)
        if candidate[x].nunique() > 1 and candidate[y].nunique() > 1:
            boot[index] = float(spearmanr(candidate[x], candidate[y]).statistic)
    finite = boot[np.isfinite(boot)]
    return rho, float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def diagnostic_source(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for regime, metric in (
        ("Observed-context patient-CV", "patient_cv_worst_context_gap"),
        ("Unseen-context leave-one", "leave_one_context_drop"),
    ):
        subset = frame.copy()
        subset["downstream_gap"] = pd.to_numeric(subset[metric], errors="coerce")
        subset["regime"] = regime
        subset = subset.loc[
            np.isfinite(subset["probe_lift_above_chance"])
            & np.isfinite(subset["downstream_gap"])
            & subset["context_cardinality"].ge(2)
        ]
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def residualized_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selectors = [
        ("LGG IDH H-optimus0", "site", "Unseen-context leave-one", "LGG IDH · H-optimus0 · site"),
        ("LGG IDH CONCH", "site", "Unseen-context leave-one", "LGG IDH · CONCH · site"),
        ("LUAD KRAS H-optimus0", "site", "Unseen-context leave-one", "LUAD KRAS · H-optimus0 · site"),
        ("LUAD KRAS CONCH", "site", "Unseen-context leave-one", "LUAD KRAS · CONCH · site"),
        (
            "CELLxGENE bone marrow geneformer_v1 geneformer_v1",
            "assay",
            "Observed-context patient-CV",
            "Bone marrow · Geneformer · assay",
        ),
        (
            "CELLxGENE bone marrow scgpt_continual scgpt_continual",
            "assay",
            "Observed-context patient-CV",
            "Bone marrow · scGPT · assay",
        ),
    ]
    rows: list[dict[str, object]] = []
    for task, context, regime, label in selectors:
        match = frame.loc[frame["task_label"].eq(task) & frame["context_field"].eq(context)]
        if len(match) != 1:
            raise ValueError(f"Expected one residualized row for {task}/{context}; found {len(match)}")
        row = match.iloc[0]
        observed = regime.startswith("Observed")
        rows.append(
            {
                "label": label,
                "domain": "single_cell" if task.startswith("CELLxGENE") else "pathology",
                "regime": regime,
                "base_probe_ba": row["base_probe_ba"],
                "residualized_probe_ba": row["artifact_resid_probe_ba"],
                "base_gap": row["base_patient_cv_gap" if observed else "base_leave_one_gap"],
                "residualized_gap": row[
                    "artifact_resid_patient_cv_gap" if observed else "artifact_resid_leave_one_gap"
                ],
            }
        )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E6E8EA", linewidth=0.55, zorder=0)
    ax.tick_params(labelsize=6.2, length=2.5)
    ax.set_axisbelow(True)


def plot_diagnostic_panel(
    ax: plt.Axes, frame: pd.DataFrame, title: str, samples: int, seed: int
) -> None:
    subset = frame.loc[frame["regime"].eq(title)].copy()
    for domain in DOMAIN_LABELS:
        domain_rows = subset.loc[subset["domain"].eq(domain)]
        for support, filled in (("support met", True), ("partial / insufficient", False)):
            points = domain_rows.loc[domain_rows["support_status"].eq(support)]
            if points.empty:
                continue
            color = DOMAIN_COLORS[domain]
            ax.scatter(
                points["probe_lift_above_chance"],
                points["downstream_gap"],
                s=24,
                marker=DOMAIN_MARKERS[domain],
                facecolors=color if filled else "white",
                edgecolors=color,
                linewidths=0.75,
                alpha=0.90,
                zorder=3,
            )
    rho, low, high = cluster_bootstrap_spearman(
        subset,
        "probe_lift_above_chance",
        "downstream_gap",
        "dataset",
        samples,
        seed,
    )
    ax.text(
        0.03,
        0.97,
        rf"$\rho={rho:.2f}$ [{low:.2f}, {high:.2f}]" + f"\n{len(subset)} rows",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#30363B",
    )
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", pad=6)
    ax.set_xlabel("Context-probe lift above chance", fontsize=6.7)
    style_axis(ax)


def plot_residualized_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    label_tracks = (0.89, 0.76, 0.40, 0.63, 0.10, 0.23)
    for index, row in enumerate(frame.itertuples(index=False)):
        color = DOMAIN_COLORS[row.domain]
        ax.annotate(
            "",
            xy=(row.residualized_probe_ba, row.residualized_gap),
            xytext=(row.base_probe_ba, row.base_gap),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.25, "mutation_scale": 8},
            zorder=2,
        )
        ax.scatter(row.base_probe_ba, row.base_gap, s=22, color="#545B61", zorder=3)
        ax.scatter(
            row.residualized_probe_ba,
            row.residualized_gap,
            s=27,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        display = str(row.label).replace(" · ", "\n· ", 1)
        ax.annotate(
            display,
            (row.residualized_probe_ba, row.residualized_gap),
            xytext=(0.98, label_tracks[index]),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55, "alpha": 0.65},
            ha="right",
            va="center",
            fontsize=5.7,
            color=color,
            bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    ax.set_title("Artifact-residualized controls", loc="left", fontsize=7.5, fontweight="bold", pad=6)
    ax.set_xlabel("Context-probe BA", fontsize=6.7)
    ax.set_ylabel("Downstream context gap", fontsize=6.7)
    style_axis(ax)
    endpoint_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#545B61", markersize=4, label="Base"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4267A9", markersize=4, label="Artifact-residualized"),
    ]
    ax.legend(
        handles=endpoint_handles,
        loc="lower left",
        ncol=1,
        fontsize=5.7,
        handletextpad=0.25,
        columnspacing=0.7,
    )


def main() -> None:
    args = parse_args()
    audit = diagnostic_source(
        add_context_cardinality(pd.read_csv(args.input_csv), args.cellxgene_root)
    )
    residualized = residualized_source(args.residualized_csv)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.source_dir / "figure3_context_recoverability.csv", index=False)
    residualized.to_csv(args.source_dir / "figure3_artifact_residualized_arrows.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.75,
        }
    )
    fig = plt.figure(figsize=(7.20, 3.25))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.28],
        left=0.075,
        right=0.985,
        bottom=0.17,
        top=0.88,
        wspace=0.38,
    )
    observed = fig.add_subplot(grid[0, 0])
    unseen = fig.add_subplot(grid[0, 1], sharey=observed)
    arrows = fig.add_subplot(grid[0, 2])
    plot_diagnostic_panel(observed, audit, "Observed-context patient-CV", args.bootstrap_samples, args.seed)
    plot_diagnostic_panel(unseen, audit, "Unseen-context leave-one", args.bootstrap_samples, args.seed + 1)
    plot_residualized_panel(arrows, residualized)
    observed.set_ylabel("Downstream context gap", fontsize=6.7)
    unseen.tick_params(labelleft=False)
    observed.text(-0.27, 1.10, "a", transform=observed.transAxes, fontsize=9, fontweight="bold")
    arrows.text(-0.20, 1.10, "b", transform=arrows.transAxes, fontsize=9, fontweight="bold")

    modality_handles = [
        Line2D(
            [0],
            [0],
            marker=DOMAIN_MARKERS[domain],
            color="none",
            markerfacecolor=DOMAIN_COLORS[domain],
            markeredgecolor=DOMAIN_COLORS[domain],
            markersize=4.5,
            label=label,
        )
        for domain, label in DOMAIN_LABELS.items()
    ]
    support_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="#666666", markersize=4.5, label="Support met"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#666666", markersize=4.5, label="Partial / insufficient"),
    ]
    observed.legend(
        handles=modality_handles + support_handles,
        loc="lower right",
        fontsize=5.8,
        handletextpad=0.3,
        borderaxespad=0.3,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 450})):
        fig.savefig(args.output_prefix.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {args.output_prefix}.[svg|pdf|png]")
    print(f"diagnostic rows: {len(audit)}; residualized arrows: {len(residualized)}")


if __name__ == "__main__":
    main()
