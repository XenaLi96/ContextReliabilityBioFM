#!/usr/bin/env python3
"""Build manuscript appendix tables for audit and TP53 robustness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"


def escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def audit_tables() -> None:
    path = ROOT / "data/systematic_audit_robustness/audit_row_bootstrap_fdr.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["foundation_model"] & frame["support_eligible"]].copy()
    frame = frame.sort_values(
        ["domain", "dataset", "model", "context_axis", "deployment_regime"]
    ).reset_index(drop=True)
    chunks = np.array_split(frame, 2)
    output = []
    for index, chunk in enumerate(chunks):
        caption = (
            r"\caption{\textbf{Support-eligible foundation-model audit rows.} "
            r"The fixed observed best/worst bins define clustered-bootstrap CIs and "
            r"the bootstrap-tail screening values shown here; BH correction is across "
            r"all 85 support-eligible audit rows.}"
            if index == 0
            else r"\caption*{\textbf{Support-eligible foundation-model audit rows (continued).}}"
        )
        label = r"\label{tab:audit_failure_prevalence_full}" if index == 0 else ""
        output.extend(
            [
                r"\begin{table*}[p]",
                r"\centering",
                r"\begingroup",
                r"\setlength{\tabcolsep}{2.2pt}",
                r"\renewcommand{\arraystretch}{0.98}",
                r"\tiny",
                caption,
                label,
                r"\begin{tabular}{@{}p{0.15\textwidth}p{0.10\textwidth}p{0.08\textwidth}p{0.09\textwidth}p{0.08\textwidth}p{0.15\textwidth}p{0.07\textwidth}p{0.07\textwidth}p{0.05\textwidth}@{}}",
                r"\toprule",
                r"Dataset & Model & Context & Regime & Gap & 95\% clustered CI & $p_{\mathrm{boot}}$ & $q_{\mathrm{BH}}$ & Screen \\",
                r"\midrule",
            ]
        )
        for row in chunk.itertuples(index=False):
            dataset = escape(str(row.dataset).replace("CELLxGENE:", "CXG:"))
            regime = "observed" if row.deployment_regime == "observed_context" else "unseen"
            context = escape(str(row.context_axis).replace("_", " "))
            output.append(
                f"{dataset} & {escape(row.model)} & {context} & "
                f"{regime} & {row.context_gap:.3f} & "
                f"[{row.gap_ci_low:.3f}, {row.gap_ci_high:.3f}] & "
                f"{row.p_value:.4f} & {row.q_value:.4f} & "
                f"{'yes' if row.significant_fdr_0_05 else 'no'} \\\\"
            )
        output.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\endgroup",
                r"\end{table*}",
                "",
            ]
        )
    (TABLE_DIR / "audit_failure_prevalence_full.tex").write_text(
        "\n".join(output), encoding="utf-8"
    )


def tp53_tables() -> None:
    path = ROOT / "data/pathology_prevalence_robustness/site_pair_robustness.csv"
    frame = pd.read_csv(path)
    tp53 = frame.loc[
        frame["task"].eq("TP53_status")
        & frame["calibration"].eq("selected")
        & frame["estimator"].eq("probability")
    ].copy()
    model_order = ["CONCH", "H-optimus0", "UNI", "Virchow2"]
    output = []
    for index, model in enumerate(model_order):
        chunk = tp53.loc[tp53["model"].eq(model)].sort_values(["site_a", "site_b"])
        caption = (
            r"\caption{\textbf{All support-eligible TP53 site-pair contrasts.} "
            r"Contrasts are site A minus site B in percentage points using the "
            r"training-site-selected probability calibrator; intervals are 5,000-replicate "
            r"patient-bootstrap intervals.}"
            if index == 0
            else rf"\caption*{{\textbf{{All TP53 site-pair contrasts ({escape(model)}; continued).}}}}"
        )
        label = r"\label{tab:tp53_all_pairs_full}" if index == 0 else ""
        output.extend(
            [
                r"\begin{table*}[p]",
                r"\centering",
                r"\begingroup",
                r"\setlength{\tabcolsep}{3pt}",
                r"\renewcommand{\arraystretch}{0.98}",
                r"\scriptsize",
                caption,
                label,
                r"\begin{tabular}{@{}p{0.09\textwidth}p{0.08\textwidth}p{0.08\textwidth}p{0.10\textwidth}p{0.10\textwidth}p{0.10\textwidth}p{0.25\textwidth}p{0.07\textwidth}@{}}",
                r"\toprule",
                r"Model & Site A & Site B & Seq. contrast & Model contrast & Distortion & Distortion 95\% CI & Reversal \\",
                r"\midrule",
            ]
        )
        for row in chunk.itertuples(index=False):
            output.append(
                f"{escape(row.model)} & {escape(row.site_a)} & {escape(row.site_b)} & "
                f"{100 * row.true_difference:+.1f} & "
                f"{100 * row.predicted_difference:+.1f} & "
                f"{100 * row.distortion:+.1f} & "
                f"[{100 * row.distortion_ci_low:+.1f}, {100 * row.distortion_ci_high:+.1f}] & "
                f"{'yes' if row.direction_reversal else 'no'} \\\\"
            )
        output.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\endgroup",
                r"\end{table*}",
                "",
            ]
        )
    (TABLE_DIR / "tp53_site_pair_full_tables.tex").write_text(
        "\n".join(output), encoding="utf-8"
    )

    magnitude = pd.read_csv(
        ROOT
        / "data/pathology_prevalence_robustness/"
        "sequencing_contrast_magnitude_sensitivity.csv"
    )
    selected = magnitude.loc[
        magnitude["task"].eq("TP53_status")
        & magnitude["calibration"].eq("selected")
        & (
            magnitude["estimator"].eq("probability")
            | (
                magnitude["estimator"].eq("hard")
                & magnitude["threshold"].eq(0.5)
            )
        )
    ].copy()
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\begingroup",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        r"\caption{\textbf{TP53 site-pair robustness to estimator and sequencing-contrast magnitude.} "
        r"Rows use the calibrator selected exclusively on non-held-out sites.}",
        r"\label{tab:tp53_robustness_summary}",
        r"\begin{tabular}{@{}p{0.18\textwidth}p{0.16\textwidth}p{0.18\textwidth}p{0.16\textwidth}p{0.20\textwidth}@{}}",
        r"\toprule",
        r"Estimator & Minimum $|\Delta_{\mathrm{seq}}|$ & Model--site pairs & Reversals & Median absolute distortion \\",
        r"\midrule",
    ]
    for row in selected.sort_values(
        ["min_absolute_sequencing_contrast", "estimator"]
    ).itertuples(index=False):
        estimator = (
            "Mean patient probability"
            if row.estimator == "probability"
            else "Hard prediction (0.5)"
        )
        lines.append(
            f"{estimator} & {100 * row.min_absolute_sequencing_contrast:.0f} pp & "
            f"{int(row.n_model_site_pairs)} & "
            f"{int(row.sign_reversals)} ({100 * row.sign_reversal_rate:.1f}\\%) & "
            f"{100 * row.median_absolute_distortion:.1f} pp \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
            r"\end{table*}",
        ]
    )
    (TABLE_DIR / "tp53_robustness_summary.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    aggregate = pd.read_csv(
        ROOT
        / "data/pathology_prevalence_robustness/"
        "site_pair_robustness_summary.csv"
    )
    hard = aggregate.loc[
        aggregate["task"].eq("TP53_status")
        & aggregate["calibration"].eq("selected")
        & aggregate["estimator"].eq("hard")
    ].copy()
    probability = aggregate.loc[
        aggregate["task"].eq("TP53_status")
        & aggregate["estimator"].eq("probability")
        & aggregate["calibration"].isin(["raw", "platt", "isotonic", "selected"])
    ].copy()
    calibration_label = {
        "raw": "Mean probability: raw",
        "platt": "Mean probability: Platt",
        "isotonic": "Mean probability: isotonic",
        "selected": "Mean probability: train-site selected",
    }
    sensitivity_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}",
        r"\scriptsize",
        r"\caption{\textbf{TP53 threshold and calibration sensitivity across 60 supported model--site-pair comparisons.}}",
        r"\label{tab:tp53_threshold_calibration}",
        r"\begin{tabular}{@{}p{0.43\columnwidth}p{0.18\columnwidth}p{0.26\columnwidth}@{}}",
        r"\toprule",
        r"Estimator & Reversals & Median absolute distortion \\",
        r"\midrule",
    ]
    for row in probability.sort_values("calibration").itertuples(index=False):
        sensitivity_lines.append(
            f"{calibration_label[row.calibration]} & "
            f"{100 * row.sign_reversal_rate:.1f}\\% & "
            f"{100 * row.median_absolute_distortion:.1f} pp \\\\"
        )
    sensitivity_lines.append(r"\midrule")
    for row in hard.sort_values("threshold").itertuples(index=False):
        sensitivity_lines.append(
            f"Hard prediction: threshold {row.threshold:.2f} & "
            f"{100 * row.sign_reversal_rate:.1f}\\% & "
            f"{100 * row.median_absolute_distortion:.1f} pp \\\\"
        )
    sensitivity_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
            r"\end{table}",
        ]
    )
    (TABLE_DIR / "tp53_threshold_calibration.tex").write_text(
        "\n".join(sensitivity_lines), encoding="utf-8"
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    audit_tables()
    tp53_tables()
    print("wrote audit and TP53 appendix tables")


if __name__ == "__main__":
    main()
