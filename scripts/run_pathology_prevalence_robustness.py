#!/usr/bin/env python3
"""Systematic site-pair robustness for image-derived molecular prevalence.

For every TCGA task/model and support-eligible tissue-source site, fit a
leave-one-site logistic head on frozen embeddings.  Calibration uses only
non-held-out sites.  The output compares hard prevalence across thresholds and
mean predicted probability across every eligible site pair, with patient
bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/tcga_image_context_shift"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pathology_prevalence_robustness"),
    )
    parser.add_argument("--tasks", nargs="+", default=["TP53", "KRAS", "IDH"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--min-patients", type=int, default=25)
    parser.add_argument("--min-positive", type=int, default=8)
    parser.add_argument("--min-negative", type=int, default=8)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def sample_key(value: str) -> str:
    return Path(value).stem


def load_run(summary_path: Path) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = pd.read_csv(summary["metadata_csv"])
    path_column = "slide_file_name"
    metadata["sample_key"] = metadata[path_column].astype(str).map(sample_key)
    features_dir = Path(str(summary["features_dir"]))
    suffix = str(summary["embedding_suffix"])
    rows = []
    embeddings = []
    for row in metadata.itertuples(index=False):
        key = str(getattr(row, "sample_key"))
        path = features_dir / f"{key}{suffix}"
        if not path.exists():
            continue
        payload = np.load(path, allow_pickle=True)
        embeddings.append(np.asarray(payload["embedding"], dtype=np.float32))
        rows.append(row._asdict())
    frame = pd.DataFrame(rows).reset_index(drop=True)
    if not len(frame):
        raise RuntimeError(f"No embeddings for {summary_path}")
    x = np.stack(embeddings, axis=0)
    return frame, x, summary


def make_head(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def split_fit_calibration(
    y: np.ndarray, fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=fraction, random_state=seed
    )
    fit, calibration = next(splitter.split(np.zeros(len(y)), y))
    return fit.astype(int), calibration.astype(int)


def fit_calibrators(
    raw: np.ndarray, y: np.ndarray, seed: int
) -> Tuple[Dict[str, object], List[Dict[str, object]], str]:
    eps = 1e-6
    raw = np.clip(raw.astype(float), eps, 1.0 - eps)
    logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    platt = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    platt.fit(logit, y)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw, y)
    calibrated = {
        "raw": raw,
        "platt": platt.predict_proba(logit)[:, 1],
        "isotonic": isotonic.predict(raw),
    }
    quality = []
    for name, values in calibrated.items():
        quality.append(
            {
                "calibration": name,
                "brier": float(brier_score_loss(y, values)),
            }
        )
    selected = min(quality, key=lambda row: (row["brier"], row["calibration"]))[
        "calibration"
    ]
    return {"platt": platt, "isotonic": isotonic}, quality, str(selected)


def apply_calibrator(name: str, calibrators: Dict[str, object], raw: np.ndarray) -> np.ndarray:
    eps = 1e-6
    raw = np.clip(raw.astype(float), eps, 1.0 - eps)
    if name == "raw":
        return raw
    if name == "platt":
        logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)
        return calibrators["platt"].predict_proba(logit)[:, 1]
    if name == "isotonic":
        return calibrators["isotonic"].predict(raw)
    raise ValueError(name)


def leave_one_site_predictions(
    frame: pd.DataFrame,
    x: np.ndarray,
    summary: Dict[str, object],
    args: argparse.Namespace,
    run_index: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_column = str(summary["label_column"])
    keep = frame[label_column].notna().to_numpy()
    frame = frame.loc[keep].copy().reset_index(drop=True)
    x = x[keep]
    y = frame[label_column].astype(str).str.endswith("_mut").astype(int).to_numpy()
    prediction_rows: List[Dict[str, object]] = []
    quality_rows: List[Dict[str, object]] = []
    for site_index, site in enumerate(sorted(frame["site"].astype(str).unique())):
        test_mask = frame["site"].astype(str).eq(site).to_numpy()
        train_mask = ~test_mask
        if test_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2:
            continue
        try:
            fit_local, cal_local = split_fit_calibration(
                y[train_mask],
                args.calibration_fraction,
                args.seed + 1000 * run_index + site_index,
            )
        except ValueError:
            continue
        train_indices = np.flatnonzero(train_mask)
        fit_idx = train_indices[fit_local]
        cal_idx = train_indices[cal_local]
        test_idx = np.flatnonzero(test_mask)
        head = make_head(args.seed + 1000 * run_index + site_index)
        head.fit(x[fit_idx], y[fit_idx])
        cal_raw = head.predict_proba(x[cal_idx])[:, 1]
        test_raw = head.predict_proba(x[test_idx])[:, 1]
        calibrators, quality, selected = fit_calibrators(
            cal_raw,
            y[cal_idx],
            args.seed + 2000 * run_index + site_index,
        )
        for row in quality:
            quality_rows.append(
                {
                    "task": label_column,
                    "model": summary["model_name"],
                    "heldout_site": site,
                    "n_fit": int(len(fit_idx)),
                    "n_calibration": int(len(cal_idx)),
                    "selected_calibration": selected,
                    **row,
                }
            )
        for calibration in ("raw", "platt", "isotonic", "selected"):
            actual = selected if calibration == "selected" else calibration
            prob = apply_calibrator(actual, calibrators, test_raw)
            for local, global_index in enumerate(test_idx):
                row = frame.iloc[int(global_index)]
                prediction_rows.append(
                    {
                        "task": label_column,
                        "model": summary["model_name"],
                        "heldout_site": site,
                        "patient_id": row.get("patient_id", row["sample_key"]),
                        "sample_key": row["sample_key"],
                        "true_positive": int(y[global_index]),
                        "calibration": calibration,
                        "actual_calibrator": actual,
                        "predicted_probability": float(prob[local]),
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        return predictions, pd.DataFrame(quality_rows)
    patient = (
        predictions.groupby(
            [
                "task",
                "model",
                "heldout_site",
                "patient_id",
                "calibration",
                "actual_calibrator",
            ],
            as_index=False,
        )
        .agg(
            true_positive=("true_positive", "mean"),
            predicted_probability=("predicted_probability", "mean"),
            n_slides=("sample_key", "nunique"),
        )
    )
    patient["true_positive"] = patient["true_positive"].ge(0.5).astype(int)
    return patient, pd.DataFrame(quality_rows)


def eligible_sites(
    patient: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    base = patient.loc[patient["calibration"].eq("raw")].copy()
    support = (
        base.groupby(["task", "model", "heldout_site"], as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            n_positive=("true_positive", "sum"),
        )
    )
    support["n_negative"] = support["n_patients"] - support["n_positive"]
    support["support_eligible"] = (
        support["n_patients"].ge(args.min_patients)
        & support["n_positive"].ge(args.min_positive)
        & support["n_negative"].ge(args.min_negative)
    )
    return support


def percentile(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def bootstrap_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    estimator: str,
    threshold: float | None,
    draws: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    left_truth = left["true_positive"].to_numpy(float)
    right_truth = right["true_positive"].to_numpy(float)
    if estimator == "probability":
        left_prediction = left["predicted_probability"].to_numpy(float)
        right_prediction = right["predicted_probability"].to_numpy(float)
    else:
        left_prediction = left["predicted_probability"].ge(float(threshold)).to_numpy(float)
        right_prediction = right["predicted_probability"].ge(float(threshold)).to_numpy(float)
    left_indices = rng.integers(0, len(left), size=(draws, len(left)))
    right_indices = rng.integers(0, len(right), size=(draws, len(right)))
    truth = left_truth[left_indices].mean(axis=1) - right_truth[right_indices].mean(axis=1)
    predicted = (
        left_prediction[left_indices].mean(axis=1)
        - right_prediction[right_indices].mean(axis=1)
    )
    distortion = predicted - truth
    reversal = truth * predicted < 0
    truth_low, truth_high = percentile(truth)
    pred_low, pred_high = percentile(predicted)
    dist_low, dist_high = percentile(distortion)
    return {
        "truth_ci_low": truth_low,
        "truth_ci_high": truth_high,
        "prediction_ci_low": pred_low,
        "prediction_ci_high": pred_high,
        "distortion_ci_low": dist_low,
        "distortion_ci_high": dist_high,
        "bootstrap_reversal_rate": float(np.mean(reversal)),
    }


def site_pair_rows(
    patient: pd.DataFrame,
    support: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    for (task, model, calibration), group in patient.groupby(
        ["task", "model", "calibration"], sort=True
    ):
        eligible = support.loc[
            support["task"].eq(task)
            & support["model"].eq(model)
            & support["support_eligible"],
            "heldout_site",
        ].astype(str)
        sites = sorted(set(group["heldout_site"].astype(str)) & set(eligible))
        for i, site_a in enumerate(sites):
            for site_b in sites[i + 1 :]:
                left = group.loc[group["heldout_site"].astype(str).eq(site_a)]
                right = group.loc[group["heldout_site"].astype(str).eq(site_b)]
                true_difference = float(
                    left["true_positive"].mean() - right["true_positive"].mean()
                )
                estimators = [("probability", None)] + [
                    ("hard", float(value)) for value in args.thresholds
                ]
                for estimator, threshold in estimators:
                    if estimator == "probability":
                        predicted = float(
                            left["predicted_probability"].mean()
                            - right["predicted_probability"].mean()
                        )
                    else:
                        predicted = float(
                            left["predicted_probability"].ge(float(threshold)).mean()
                            - right["predicted_probability"].ge(float(threshold)).mean()
                        )
                    boot = bootstrap_pair(
                        left,
                        right,
                        estimator,
                        threshold,
                        args.n_bootstrap,
                        rng,
                    )
                    rows.append(
                        {
                            "task": task,
                            "model": model,
                            "calibration": calibration,
                            "site_a": site_a,
                            "site_b": site_b,
                            "n_a": int(len(left)),
                            "n_b": int(len(right)),
                            "estimator": estimator,
                            "threshold": threshold,
                            "true_difference": true_difference,
                            "predicted_difference": predicted,
                            "distortion": predicted - true_difference,
                            "absolute_distortion": abs(predicted - true_difference),
                            "direction_reversal": bool(true_difference * predicted < 0),
                            **boot,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        rows.groupby(["task", "calibration", "estimator", "threshold"], dropna=False)
        .agg(
            n_model_site_pairs=("direction_reversal", "size"),
            n_models=("model", "nunique"),
            n_site_pairs=("site_a", "size"),
            sign_reversal_rate=("direction_reversal", "mean"),
            median_absolute_distortion=("absolute_distortion", "median"),
            median_bootstrap_reversal_rate=("bootstrap_reversal_rate", "median"),
        )
        .reset_index()
    )
    return grouped


def contrast_magnitude_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for minimum in (0.0, 0.05, 0.10):
        subset = rows.loc[rows["true_difference"].abs().ge(minimum)]
        for key, group in subset.groupby(
            ["task", "calibration", "estimator", "threshold"],
            dropna=False,
        ):
            output.append(
                {
                    "task": key[0],
                    "calibration": key[1],
                    "estimator": key[2],
                    "threshold": key[3],
                    "min_absolute_sequencing_contrast": minimum,
                    "n_model_site_pairs": int(len(group)),
                    "n_site_pairs": int(
                        group[["site_a", "site_b"]].drop_duplicates().shape[0]
                    ),
                    "sign_reversals": int(group["direction_reversal"].sum()),
                    "sign_reversal_rate": float(group["direction_reversal"].mean()),
                    "median_absolute_distortion": float(
                        group["absolute_distortion"].median()
                    ),
                }
            )
    return pd.DataFrame(output)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_slugs = {value.lower() for value in args.tasks}
    prediction_frames = []
    quality_frames = []
    run_index = 0
    for summary_path in sorted(args.input_root.glob("*/summary.json")):
        slug = summary_path.parent.name.split("_", 1)[0].lower()
        if slug not in task_slugs:
            continue
        frame, x, summary = load_run(summary_path)
        if args.models is not None and str(summary["model_name"]) not in set(args.models):
            continue
        patient, quality = leave_one_site_predictions(
            frame, x, summary, args, run_index
        )
        if not patient.empty:
            prediction_frames.append(patient)
            quality_frames.append(quality)
            run_index += 1
    if not prediction_frames:
        raise RuntimeError("No pathology robustness predictions generated")
    patient = pd.concat(prediction_frames, ignore_index=True)
    quality = pd.concat(quality_frames, ignore_index=True)
    support = eligible_sites(patient, args)
    pairs = site_pair_rows(patient, support, args)
    summary = summarize_pairs(pairs)
    magnitude = contrast_magnitude_summary(pairs)
    patient.to_csv(args.output_dir / "patient_leave_one_site_probabilities.csv", index=False)
    quality.to_csv(args.output_dir / "calibration_quality.csv", index=False)
    support.to_csv(args.output_dir / "site_support.csv", index=False)
    pairs.to_csv(args.output_dir / "site_pair_robustness.csv", index=False)
    summary.to_csv(args.output_dir / "site_pair_robustness_summary.csv", index=False)
    magnitude.to_csv(
        args.output_dir / "sequencing_contrast_magnitude_sensitivity.csv",
        index=False,
    )

    selected = pairs.loc[pairs["calibration"].eq("selected")]
    payload = {
        "n_task_model_runs": int(
            patient[["task", "model"]].drop_duplicates().shape[0]
        ),
        "n_support_eligible_site_rows": int(support["support_eligible"].sum()),
        "n_supported_model_site_pairs": int(
            selected[["task", "model", "site_a", "site_b"]]
            .drop_duplicates()
            .shape[0]
        ),
        "selected_calibration_probability": {
            "sign_reversal_rate": float(
                selected.loc[selected["estimator"].eq("probability"), "direction_reversal"].mean()
            ),
            "median_absolute_distortion": float(
                selected.loc[
                    selected["estimator"].eq("probability"), "absolute_distortion"
                ].median()
            ),
        },
        "selected_calibration_hard_0_5": {
            "sign_reversal_rate": float(
                selected.loc[
                    selected["estimator"].eq("hard")
                    & selected["threshold"].eq(0.5),
                    "direction_reversal",
                ].mean()
            ),
            "median_absolute_distortion": float(
                selected.loc[
                    selected["estimator"].eq("hard")
                    & selected["threshold"].eq(0.5),
                    "absolute_distortion",
                ].median()
            ),
        },
        "calibration_rule": (
            "Platt versus isotonic selected by Brier score using only non-held-out-site "
            "calibration patients; raw probabilities are retained as a sensitivity."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
