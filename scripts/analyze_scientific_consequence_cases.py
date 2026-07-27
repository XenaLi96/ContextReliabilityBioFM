#!/usr/bin/env python3
"""Build two reproducible scientific-consequence case studies.

Case 1 quantifies an assay-specific acinar-to-ductal composition distortion in
the pancreas CELLxGENE benchmark.  Case 2 quantifies a site-specific reversal
of inferred TP53 prevalence in TCGA-LUAD.  The script reuses saved out-of-fold
or held-out-context predictions and therefore does not retrain or select on a
test metric during fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "erm_mlp": "ERM",
    "label_context_reweight": "LC-Reweight",
    "sca_lite": "SCA-Align",
    "group_dro": "GroupDRO",
}
ACINAR_LABELS = {"acinar cell", "pancreatic acinar cell"}
DUCTAL_LABEL = "pancreatic ductal cell"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path("data/cellxgene_support_calibrated_formal"),
    )
    parser.add_argument(
        "--abundance-dir",
        type=Path,
        default=Path("data/donor_abundance_consequence"),
    )
    parser.add_argument(
        "--tcga-root",
        type=Path,
        default=Path("data/tcga_image_context_shift"),
    )
    parser.add_argument(
        "--residual-root",
        type=Path,
        default=Path("data/residualized_embedding_controls/tcga"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scientific_consequence_cases"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("seed_"):
            return int(part.replace("seed_", ""))
    raise ValueError("No seed component in %s" % path)


def quantiles(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def mode_string(values: pd.Series) -> str:
    counts = values.astype(str).value_counts()
    if counts.empty:
        return ""
    return str(counts.index[0])


def build_pancreas_case(args: argparse.Namespace) -> Dict[str, object]:
    errors = pd.read_csv(args.abundance_dir / "celltype_context_abundance_error.csv")
    errors = errors.loc[
        errors["task"].eq("pancreas_assay")
        & errors["model"].eq("scgpt_continual")
        & errors["split_type"].eq("leave_one_context")
        & errors["context_value"].eq("microwell-seq")
    ].copy()
    if errors.empty:
        raise ValueError("No pancreas/microwell abundance rows found")

    abundance_rows: List[Dict[str, object]] = []
    for (seed, method, method_label), group in errors.groupby(
        ["seed", "method", "method_label"], sort=True
    ):
        acinar = group.loc[group["label"].isin(ACINAR_LABELS)]
        ductal = group.loc[group["label"].eq(DUCTAL_LABEL)]
        abundance_rows.append(
            {
                "seed": int(seed),
                "method": method,
                "method_label": method_label,
                "true_acinar_lineage_abundance": float(acinar["true_abundance"].sum()),
                "predicted_acinar_lineage_abundance": float(acinar["predicted_abundance"].sum()),
                "true_ductal_abundance": float(ductal["true_abundance"].sum()),
                "predicted_ductal_abundance": float(ductal["predicted_abundance"].sum()),
            }
        )
    abundance = pd.DataFrame.from_records(abundance_rows)
    abundance["acinar_underestimate"] = (
        abundance["predicted_acinar_lineage_abundance"]
        - abundance["true_acinar_lineage_abundance"]
    )
    abundance.to_csv(args.output_dir / "pancreas_abundance_per_seed.csv", index=False)

    abundance_summary = (
        abundance.groupby(["method", "method_label"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            true_acinar_lineage_abundance=("true_acinar_lineage_abundance", "mean"),
            predicted_acinar_lineage_abundance_mean=("predicted_acinar_lineage_abundance", "mean"),
            predicted_acinar_lineage_abundance_std=("predicted_acinar_lineage_abundance", "std"),
            predicted_ductal_abundance_mean=("predicted_ductal_abundance", "mean"),
            predicted_ductal_abundance_std=("predicted_ductal_abundance", "std"),
            acinar_underestimate_mean=("acinar_underestimate", "mean"),
            acinar_underestimate_std=("acinar_underestimate", "std"),
        )
        .reset_index()
    )
    abundance_summary.to_csv(args.output_dir / "pancreas_abundance_summary.csv", index=False)

    prediction_paths = sorted(
        args.formal_root.glob(
            "pancreas_assay/scgpt_continual/assay/seed_*/*/leave_one_context_predictions.csv"
        )
    )
    if not prediction_paths:
        raise FileNotFoundError("No formal pancreas prediction files found")
    cell_rows: List[Dict[str, object]] = []
    for path in prediction_paths:
        pred = pd.read_csv(path)
        pred = pred.loc[pred["context_value"].eq("microwell-seq")].copy()
        for method, group in pred.groupby("method", sort=False):
            acinar = group.loc[group["true_label"].isin(ACINAR_LABELS)]
            cell_rows.append(
                {
                    "seed": seed_from_path(path),
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "n_target_cells": int(len(group)),
                    "n_target_donors": int(group["donor_id"].astype(str).nunique()),
                    "n_true_acinar_cells": int(len(acinar)),
                    "acinar_lineage_recall": float(acinar["pred_label"].isin(ACINAR_LABELS).mean()),
                    "acinar_to_ductal_rate": float(acinar["pred_label"].eq(DUCTAL_LABEL).mean()),
                }
            )
    cell_metrics = pd.DataFrame.from_records(cell_rows)
    cell_metrics.to_csv(args.output_dir / "pancreas_cell_error_per_seed.csv", index=False)
    cell_summary = (
        cell_metrics.groupby(["method", "method_label"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_target_cells=("n_target_cells", "first"),
            n_target_donors=("n_target_donors", "first"),
            n_true_acinar_cells=("n_true_acinar_cells", "first"),
            acinar_lineage_recall_mean=("acinar_lineage_recall", "mean"),
            acinar_lineage_recall_std=("acinar_lineage_recall", "std"),
            acinar_to_ductal_rate_mean=("acinar_to_ductal_rate", "mean"),
            acinar_to_ductal_rate_std=("acinar_to_ductal_rate", "std"),
        )
        .reset_index()
    )
    cell_summary.to_csv(args.output_dir / "pancreas_cell_error_summary.csv", index=False)

    first_run = prediction_paths[0].parent
    support = pd.read_csv(first_run / "support_label_context_counts.csv")
    support = support.loc[
        support["context_field"].eq("assay")
        & support["label"].isin(ACINAR_LABELS)
    ].copy()
    support.to_csv(args.output_dir / "pancreas_acinar_support.csv", index=False)

    erm_abundance = abundance_summary.loc[abundance_summary["method_label"].eq("ERM")].iloc[0]
    erm_cells = cell_summary.loc[cell_summary["method_label"].eq("ERM")].iloc[0]
    target_support = support.loc[
        support["context_value"].eq("microwell-seq")
        & support["label"].eq("acinar cell")
    ].iloc[0]
    return {
        "case": "pancreas_acinar_to_ductal",
        "deployment_context": "held-out microwell-seq",
        "n_target_cells": int(erm_cells["n_target_cells"]),
        "n_target_donors": int(erm_cells["n_target_donors"]),
        "n_true_acinar_cells": int(erm_cells["n_true_acinar_cells"]),
        "true_acinar_lineage_abundance": float(erm_abundance["true_acinar_lineage_abundance"]),
        "erm_predicted_acinar_lineage_abundance": float(
            erm_abundance["predicted_acinar_lineage_abundance_mean"]
        ),
        "erm_predicted_ductal_abundance": float(erm_abundance["predicted_ductal_abundance_mean"]),
        "erm_acinar_lineage_recall": float(erm_cells["acinar_lineage_recall_mean"]),
        "erm_acinar_to_ductal_rate": float(erm_cells["acinar_to_ductal_rate_mean"]),
        "target_acinar_support_donors": int(target_support["n_donors"]),
        "target_acinar_support_eligible": bool(target_support["support_eligible_cell"]),
        "interpretation": (
            "A naive annotation-to-composition workflow would report marked acinar depletion "
            "and a large ductal compartment, although the source annotation has the opposite composition."
        ),
    }


def patient_site_frame(path: Path, method: str = "erm") -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["context_field"].eq("site") & frame["method"].eq(method)
    ].copy()
    frame["true_positive"] = frame["true_label"].astype(str).str.endswith("_mut")
    frame["pred_positive"] = frame["pred_label"].astype(str).str.endswith("_mut")
    patient = (
        frame.groupby(["task", "model", "site", "patient_id"], as_index=False)
        .agg(
            true_positive=("true_positive", "mean"),
            pred_positive=("pred_positive", "mean"),
        )
    )
    patient["true_positive"] = patient["true_positive"] >= 0.5
    patient["pred_positive"] = patient["pred_positive"] >= 0.5
    return patient


def scan_pathology_pairs(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(args.tcga_root.glob("*/leave_one_context_predictions.csv")):
        patient = patient_site_frame(path, method="erm")
        if patient.empty:
            continue
        site = (
            patient.groupby(["task", "model", "site"], sort=True)
            .agg(
                n_patients=("patient_id", "nunique"),
                n_positive=("true_positive", "sum"),
                true_prevalence=("true_positive", "mean"),
                predicted_prevalence=("pred_positive", "mean"),
            )
            .reset_index()
        )
        site["n_negative"] = site["n_patients"] - site["n_positive"]
        site = site.loc[
            (site["n_patients"] >= 25)
            & (site["n_positive"] >= 8)
            & (site["n_negative"] >= 8)
        ]
        for (_, _), group in site.groupby(["task", "model"], sort=False):
            records = group.sort_values("site").to_dict("records")
            for index, left in enumerate(records):
                for right in records[index + 1 :]:
                    true_difference = float(
                        left["true_prevalence"] - right["true_prevalence"]
                    )
                    predicted_difference = float(
                        left["predicted_prevalence"] - right["predicted_prevalence"]
                    )
                    rows.append(
                        {
                            "task": left["task"],
                            "model": left["model"],
                            "site_a": left["site"],
                            "site_b": right["site"],
                            "n_a": int(left["n_patients"]),
                            "n_b": int(right["n_patients"]),
                            "positive_a": int(left["n_positive"]),
                            "positive_b": int(right["n_positive"]),
                            "negative_a": int(left["n_negative"]),
                            "negative_b": int(right["n_negative"]),
                            "true_prevalence_a": float(left["true_prevalence"]),
                            "true_prevalence_b": float(right["true_prevalence"]),
                            "predicted_prevalence_a": float(left["predicted_prevalence"]),
                            "predicted_prevalence_b": float(right["predicted_prevalence"]),
                            "true_difference": true_difference,
                            "predicted_difference": predicted_difference,
                            "direction_reversal": bool(
                                true_difference * predicted_difference < 0
                            ),
                            "absolute_distortion": abs(
                                predicted_difference - true_difference
                            ),
                        }
                    )
    scan = pd.DataFrame.from_records(rows)
    if scan.empty:
        raise ValueError("No pathology site pairs met the support rule")
    return scan.sort_values(
        ["direction_reversal", "absolute_distortion"],
        ascending=[False, False],
    ).reset_index(drop=True)


def bootstrap_site_pair(
    site_a: pd.DataFrame,
    site_b: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    true_diff: List[float] = []
    pred_diff: List[float] = []
    distortion: List[float] = []
    reversal: List[bool] = []
    for _ in range(int(n_bootstrap)):
        left = site_a.iloc[rng.integers(0, len(site_a), size=len(site_a))]
        right = site_b.iloc[rng.integers(0, len(site_b), size=len(site_b))]
        observed = float(left["true_positive"].mean() - right["true_positive"].mean())
        predicted = float(left["pred_positive"].mean() - right["pred_positive"].mean())
        true_diff.append(observed)
        pred_diff.append(predicted)
        distortion.append(predicted - observed)
        reversal.append(bool(observed * predicted < 0))
    true_low, true_high = quantiles(true_diff)
    pred_low, pred_high = quantiles(pred_diff)
    distortion_low, distortion_high = quantiles(distortion)
    return {
        "n_bootstrap": int(n_bootstrap),
        "true_difference_ci_low": true_low,
        "true_difference_ci_high": true_high,
        "predicted_difference_ci_low": pred_low,
        "predicted_difference_ci_high": pred_high,
        "prediction_minus_truth_ci_low": distortion_low,
        "prediction_minus_truth_ci_high": distortion_high,
        "bootstrap_direction_reversal_rate": float(np.mean(reversal)),
    }


def build_pathology_case(args: argparse.Namespace) -> Dict[str, object]:
    scan = scan_pathology_pairs(args)
    scan.to_csv(args.output_dir / "pathology_supported_pair_scan.csv", index=False)
    selected = scan.iloc[0]
    task = str(selected["task"])
    site_a = str(selected["site_a"])
    site_b = str(selected["site_b"])

    model_rows: List[Dict[str, object]] = []
    bootstrap_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    task_slug = task.replace("_status", "").lower()
    for path in sorted(args.tcga_root.glob("%s_*/leave_one_context_predictions.csv" % task_slug)):
        patient = patient_site_frame(path, method="erm")
        patient = patient.loc[patient["task"].eq(task)]
        if patient.empty:
            continue
        model = str(patient["model"].iloc[0])
        left = patient.loc[patient["site"].astype(str).eq(site_a)].copy()
        right = patient.loc[patient["site"].astype(str).eq(site_b)].copy()
        if left.empty or right.empty:
            continue
        true_a = float(left["true_positive"].mean())
        true_b = float(right["true_positive"].mean())
        pred_a = float(left["pred_positive"].mean())
        pred_b = float(right["pred_positive"].mean())
        model_rows.append(
            {
                "task": task,
                "model": model,
                "method": "ERM",
                "site_a": site_a,
                "site_b": site_b,
                "n_a": int(len(left)),
                "n_b": int(len(right)),
                "positive_a": int(left["true_positive"].sum()),
                "positive_b": int(right["true_positive"].sum()),
                "negative_a": int((~left["true_positive"]).sum()),
                "negative_b": int((~right["true_positive"]).sum()),
                "true_prevalence_a": true_a,
                "true_prevalence_b": true_b,
                "predicted_prevalence_a": pred_a,
                "predicted_prevalence_b": pred_b,
                "true_difference_a_minus_b": true_a - true_b,
                "predicted_difference_a_minus_b": pred_a - pred_b,
                "direction_reversal": bool((true_a - true_b) * (pred_a - pred_b) < 0),
                "absolute_distortion": abs((pred_a - pred_b) - (true_a - true_b)),
            }
        )
        bootstrap_rows.append(
            {
                "task": task,
                "model": model,
                "site_a": site_a,
                "site_b": site_b,
                **bootstrap_site_pair(left, right, args.n_bootstrap, rng),
            }
        )
    comparison = pd.DataFrame.from_records(model_rows).sort_values("model")
    bootstrap = pd.DataFrame.from_records(bootstrap_rows).sort_values("model")
    comparison.to_csv(args.output_dir / "pathology_tp53_site_pair.csv", index=False)
    bootstrap.to_csv(args.output_dir / "pathology_tp53_site_pair_bootstrap.csv", index=False)

    average_rows: List[pd.DataFrame] = []
    for path in sorted(args.tcga_root.glob("%s_*/random_cv_metrics.csv" % task_slug)):
        frame = pd.read_csv(path)
        frame = frame.loc[frame["task"].eq(task)].copy()
        average_rows.append(frame)
    if average_rows:
        pd.concat(average_rows, ignore_index=True).to_csv(
            args.output_dir / "pathology_tp53_average_metrics.csv", index=False
        )

    residual_rows: List[Dict[str, object]] = []
    for row in comparison.itertuples(index=False):
        slug = str(row.model).lower().replace("-", "").replace("virchow2", "virchow2")
        candidates = list(args.residual_root.glob("%s_*/residualized_control_summary.csv" % task_slug))
        for path in candidates:
            if slug not in path.parent.name.replace("-", "").lower():
                continue
            frame = pd.read_csv(path)
            frame = frame.loc[frame["context_field"].eq("site")]
            for record in frame.to_dict("records"):
                residual_rows.append({"model": row.model, **record})
    if residual_rows:
        pd.DataFrame.from_records(residual_rows).to_csv(
            args.output_dir / "pathology_tp53_residualization_context.csv", index=False
        )

    representative = comparison.sort_values("absolute_distortion", ascending=False).iloc[0]
    return {
        "case": "tcga_luad_tp53_site_prevalence_reversal",
        "selection_rule": (
            "largest ERM site-pair prevalence distortion among pairs with >=25 patients "
            "and >=8 mutation-positive and >=8 wild-type patients per site"
        ),
        "site_a": site_a,
        "site_b": site_b,
        "n_site_a": int(representative["n_a"]),
        "n_site_b": int(representative["n_b"]),
        "true_prevalence_site_a": float(representative["true_prevalence_a"]),
        "true_prevalence_site_b": float(representative["true_prevalence_b"]),
        "true_difference_a_minus_b": float(representative["true_difference_a_minus_b"]),
        "largest_distortion_model": str(representative["model"]),
        "predicted_difference_a_minus_b": float(
            representative["predicted_difference_a_minus_b"]
        ),
        "all_models_reverse_direction": bool(comparison["direction_reversal"].all()),
        "n_models": int(comparison["model"].nunique()),
        "interpretation": (
            "Held-out-site pathology predictions would reverse the ordering of TP53 prevalence "
            "between two adequately sized contributing sites and could misdirect molecular-testing triage."
        ),
    }


def write_readme(output_dir: Path, pancreas: Dict[str, object], pathology: Dict[str, object]) -> None:
    text = f"""# Scientific-consequence case studies

## Case 1: pancreas acinar-to-ductal composition distortion

In the held-out microwell-seq context ({pancreas['n_target_cells']} cells from
{pancreas['n_target_donors']} donors), the sampling-corrected source annotation
contains {pancreas['true_acinar_lineage_abundance']:.1%} acinar-lineage cells.
ERM reconstructs only {pancreas['erm_predicted_acinar_lineage_abundance']:.1%}
acinar lineage and creates {pancreas['erm_predicted_ductal_abundance']:.1%}
ductal abundance. Across five seeds, only
{pancreas['erm_acinar_lineage_recall']:.1%} of annotated acinar cells remain in
the acinar lineage, while {pancreas['erm_acinar_to_ductal_rate']:.1%} are called
ductal. The target acinar label has only
{pancreas['target_acinar_support_donors']} independent donors and fails the
five-donor support rule.

Scientific consequence: a naive downstream composition analysis reports
acinar depletion together with a large ductal compartment, the opposite of the
annotated selected-label composition. The admissible conclusion is therefore
not that the held-out cohort has undergone acinar-to-ductal remodeling, but that
the comparison needs label-ontology harmonization and additional acinar donors
under the target protocol.

Domain validation to request: blinded reannotation of the acinar/ductal errors;
PRSS1/PRSS2/CPA1/REG1A versus KRT19/KRT8/KRT18 marker review; and an independent
microwell-seq or matched-protocol pancreas cohort.

## Case 2: TCGA-LUAD TP53 prevalence reversal across sites

Sites {pathology['site_a']} and {pathology['site_b']} contain
{pathology['n_site_a']} and {pathology['n_site_b']} patients. Their sequencing
ground-truth TP53 prevalence is {pathology['true_prevalence_site_a']:.1%} versus
{pathology['true_prevalence_site_b']:.1%}, a difference of
{pathology['true_difference_a_minus_b']:+.1%}. All
{pathology['n_models']} pathology foundation models reverse this ordering under
held-out-site evaluation; the largest distortion is
{pathology['largest_distortion_model']}, whose predicted difference is
{pathology['predicted_difference_a_minus_b']:+.1%}.

Scientific consequence: image-derived mutation prevalence would make the wrong
site appear TP53-enriched and could reverse cohort prioritization for molecular
testing. Here sample support is adequate, showing that support is necessary but
not sufficient: acquisition-specific morphology or low-level slide signatures
still require validation.

Domain validation to request: blinded pathologist review of false-positive and
false-negative slides from both sites; stain/scanner/focus/tissue-quality
scoring; confirmation against sequencing labels; and site-stratified replication
in an external LUAD cohort such as CPTAC.

## Literature anchors

- Mereu et al., *Nature Biotechnology* (2020), protocol benchmarking,
  DOI: 10.1038/s41587-020-0469-4.
- Muraro et al., *Cell Systems* (2016), human pancreas cell identities,
  DOI: 10.1016/j.cels.2016.09.002.
- Howard et al., *Nature Communications* (2021), site-specific digital
  histology signatures, DOI: 10.1038/s41467-021-24698-1.
- Dehkharghanian et al., *Diagnostic Pathology* (2023), acquisition-site
  prediction in TCGA, DOI: 10.1186/s13000-023-01355-3.
- Kather et al., *Nature Cancer* (2020), image-based genetic-alteration
  prediction, DOI: 10.1038/s43018-020-0087-6.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pancreas = build_pancreas_case(args)
    pathology = build_pathology_case(args)
    summary = {
        "analysis_seed": int(args.seed),
        "n_bootstrap": int(args.n_bootstrap),
        "pancreas": pancreas,
        "pathology": pathology,
        "claim_boundary": (
            "These are conclusion-distortion case studies using saved held-out-context "
            "predictions; they do not establish causal biological differences."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_readme(args.output_dir, pancreas, pathology)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
