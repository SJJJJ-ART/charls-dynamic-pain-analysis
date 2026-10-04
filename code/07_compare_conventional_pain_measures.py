#!/usr/bin/env python3
"""Compare pain transitions with conventional pain-burden measures.

This analysis uses the same outcome-observed repeated-landmark risk set,
imputation seeds, response weights, and Model 1 covariates as the primary
analysis. Only disclosure-safe aggregate tables are written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from charls_core import STATE_LABELS, TRANSITION_LABELS, normalize_2011_id
from design import outcome_design
from imputation import completed_dataset
from models import make_ipcw
from statistical import modified_poisson, poisson_coefficients, rubin_pool, weighted_logistic


WINDOW_WAVES = {1: (2011, 2013), 2: (2013, 2015), 3: (2015, 2018)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--primary-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imputations", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=20260718)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def pain_site_count(raw_dir: Path, wave: int) -> pd.DataFrame:
    filename = "health_status_and_functioning.dta" if wave == 2011 else "Health_Status_and_Functioning.dta"
    prefix = "da042_s" if wave == 2018 else "da042s"
    columns = ["ID", *[f"{prefix}{index}" for index in range(1, 16)]]
    frame = pd.read_stata(raw_dir / str(wave) / filename, columns=columns, convert_categoricals=False)
    if wave == 2011:
        frame["ID"] = normalize_2011_id(frame["ID"])
    selected = np.column_stack(
        [frame[f"{prefix}{index}"].eq(index) for index in range(1, 16)]
    )
    return frame[["ID"]].assign(**{f"site_count_{wave}": selected.sum(axis=1).astype(int)})


def attach_site_counts(frame: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    result = frame.copy()
    counts = {wave: pain_site_count(raw_dir, wave) for wave in (2011, 2013, 2015, 2018)}
    for wave, values in counts.items():
        result = result.merge(values, on="ID", how="left", validate="many_to_one")
    result["start_site_count"] = np.nan
    result["current_site_count"] = np.nan
    for window, (start_wave, landmark_wave) in WINDOW_WAVES.items():
        mask = result["window"].eq(window)
        result.loc[mask, "start_site_count"] = result.loc[mask, f"site_count_{start_wave}"]
        result.loc[mask, "current_site_count"] = result.loc[mask, f"site_count_{landmark_wave}"]
    expected_zero = result["landmark_state"].eq(0)
    if not result.loc[expected_zero, "current_site_count"].eq(0).all():
        raise AssertionError("Pain-free landmark intervals must have zero recorded pain sites")
    if result["current_site_count"].isna().any():
        raise AssertionError("Missing current site count in an observed pain-state interval")
    return result


def base_model1_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    design, terms = outcome_design(frame, model=1)
    keep = [0, *range(8, design.shape[1])]
    return design[:, keep], [terms[index] for index in keep]


def predictor_design(frame: pd.DataFrame, model_name: str) -> tuple[np.ndarray, list[str]]:
    base, base_terms = base_model1_design(frame)
    current_pain = frame["landmark_state"].ne(0).astype(float).to_numpy()
    site_count = frame["current_site_count"].to_numpy(float)

    if model_name == "Covariates only":
        return base, base_terms
    if model_name == "Current pain (binary)":
        return np.column_stack([base, current_pain]), [*base_terms, "Current pain"]
    if model_name == "Current site count":
        category = np.minimum(site_count.astype(int), 4)
        additions = [category == value for value in range(1, 5)]
        terms = ["1 site", "2 sites", "3 sites", "4+ sites"]
        return np.column_stack([base, *additions]).astype(float), [*base_terms, *terms]
    if model_name == "Any-pain persistence pattern":
        start = frame["start_state"].ne(0).astype(int).to_numpy()
        current = frame["landmark_state"].ne(0).astype(int).to_numpy()
        pattern = 2 * start + current
        additions = [pattern == value for value in (1, 2, 3)]
        terms = ["Incident any pain", "Remitted any pain", "Persistent any pain"]
        return np.column_stack([base, *additions]).astype(float), [*base_terms, *terms]
    if model_name == "Current four-state pain distribution":
        additions = [frame["landmark_state"].eq(value).to_numpy() for value in (1, 2, 3)]
        terms = [STATE_LABELS[value] for value in (1, 2, 3)]
        return np.column_stack([base, *additions]).astype(float), [*base_terms, *terms]
    if model_name in {"Eight pain transitions", "Eight transitions plus exact site count"}:
        design, terms = outcome_design(frame, model=1)
        if model_name.endswith("exact site count"):
            design = np.column_stack([design, site_count])
            terms = [*terms, "Current site count, per site"]
        return design.astype(float), terms
    if model_name == "Eight transitions plus flexible site count":
        design, terms = outcome_design(frame, model=1)
        category = np.minimum(site_count.astype(int), 5)
        additions = [category == value for value in range(1, 6)]
        labels = ["1 site", "2 sites", "3 sites", "4 sites", "5+ sites"]
        return np.column_stack([design, *additions]).astype(float), [*terms, *labels]
    raise ValueError(f"Unknown model: {model_name}")


MODEL_NAMES = [
    "Covariates only",
    "Current pain (binary)",
    "Current site count",
    "Any-pain persistence pattern",
    "Current four-state pain distribution",
    "Eight pain transitions",
    "Eight transitions plus exact site count",
]


def site_count_distribution(observed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    count = observed["current_site_count"].astype(int)
    for code in range(8):
        part = observed.loc[observed["transition"].eq(code)].copy()
        values = part["current_site_count"].astype(int)
        rows.append(
            {
                "transition": TRANSITION_LABELS[code],
                "intervals": len(part),
                "mean_sites": values.mean(),
                "sd_sites": values.std(ddof=1),
                "median_sites": values.median(),
                "q1_sites": values.quantile(0.25),
                "q3_sites": values.quantile(0.75),
                "zero_sites_pct": 100 * values.eq(0).mean(),
                "one_site_pct": 100 * values.eq(1).mean(),
                "two_sites_pct": 100 * values.eq(2).mean(),
                "three_sites_pct": 100 * values.eq(3).mean(),
                "four_sites_pct": 100 * values.eq(4).mean(),
                "five_plus_sites_pct": 100 * values.ge(5).mean(),
            }
        )
    if count.min() < 0 or count.max() > 15:
        raise AssertionError("Site count must lie between 0 and 15")
    return pd.DataFrame(rows)


def cross_validated_metrics(observed: pd.DataFrame, folds: int) -> list[dict[str, float | str]]:
    splitter = GroupKFold(n_splits=folds)
    groups = observed["communityID"].astype("string").to_numpy()
    outcome = observed["event"].to_numpy(float)
    weight = observed["combined_weight"].to_numpy(float)
    rows: list[dict[str, float | str]] = []
    for model_name in MODEL_NAMES:
        design, _ = predictor_design(observed, model_name)
        prediction = np.full(len(observed), np.nan)
        for train, test in splitter.split(design, outcome, groups):
            coefficients = poisson_coefficients(design[train], outcome[train], weight[train])
            prediction[test] = np.exp(np.clip(design[test] @ coefficients, -20, 20))
        probability = np.clip(prediction, 1e-6, 1 - 1e-6)
        auc = roc_auc_score(outcome, probability, sample_weight=weight)
        brier = np.average(np.square(outcome - probability), weights=weight)
        log_loss = -np.average(
            outcome * np.log(probability) + (1 - outcome) * np.log(1 - probability),
            weights=weight,
        )
        calibration = weighted_logistic(
            np.column_stack([np.ones(len(observed)), logit(probability)]), outcome, weight
        )
        rows.append(
            {
                "model": model_name,
                "auc": auc,
                "brier": brier,
                "log_loss": log_loss,
                "calibration_intercept": calibration.coefficients[0],
                "calibration_slope": calibration.coefficients[1],
            }
        )
    return rows


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_NAMES:
        part = metrics.loc[metrics["model"].eq(model_name)]
        row: dict[str, float | int | str] = {"model": model_name, "imputations": len(part)}
        for metric in ("auc", "brier", "log_loss", "calibration_intercept", "calibration_slope"):
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_min"] = part[metric].min()
            row[f"{metric}_max"] = part[metric].max()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(index="imputation", columns="model")
    transition = "Eight pain transitions"
    rows = []
    for comparator in MODEL_NAMES[:-2]:
        row: dict[str, float | str] = {"comparison": f"{transition} minus {comparator}"}
        for metric in ("auc", "brier", "log_loss"):
            delta = wide[metric][transition] - wide[metric][comparator]
            row[f"delta_{metric}_mean"] = delta.mean()
            row[f"delta_{metric}_min"] = delta.min()
            row[f"delta_{metric}_max"] = delta.max()
        rows.append(row)
    return pd.DataFrame(rows)


def pool_adjusted_contrast(fits: list, adjustment: str) -> dict[str, float | int | str]:
    estimates = np.stack([fit.coefficients for fit in fits])
    covariances = np.stack([fit.covariance for fit in fits])
    _, _, _, total = rubin_pool(estimates, covariances)
    pooled = estimates.mean(axis=0)
    contrast = np.zeros(estimates.shape[1])
    contrast[6] = 1.0  # Persistent multisite
    contrast[5] = -1.0  # Incident multisite
    log_rr = float(contrast @ pooled)
    variance = float(contrast @ total @ contrast)
    se = np.sqrt(variance)
    return {
        "comparison": "Persistent vs incident axial multisite pain",
        "adjustment": adjustment,
        "rr": np.exp(log_rr),
        "ci_low": np.exp(log_rr - 1.96 * se),
        "ci_high": np.exp(log_rr + 1.96 * se),
        "log_rr": log_rr,
        "se": se,
        "imputations": len(fits),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.primary_results_dir / "private" / "eligible_with_covariates.pkl"
    eligible = attach_site_counts(pd.read_pickle(source), args.raw_dir)
    observed = eligible.loc[eligible["response"].eq(1)].copy()
    site_count_distribution(observed).to_csv(
        args.output_dir / "pain_site_count_distribution.csv", index=False
    )

    metric_rows = []
    augmented_fits = []
    flexible_fits = []
    for imputation, seed in enumerate(range(args.seed_start, args.seed_start + args.imputations), 1):
        completed = completed_dataset(eligible, seed)
        weighted = make_ipcw(completed)
        current = weighted.loc[weighted["response"].eq(1)].copy()
        for row in cross_validated_metrics(current, args.folds):
            metric_rows.append({"imputation": imputation, "seed": seed, **row})
        design, _ = predictor_design(current, "Eight transitions plus exact site count")
        augmented_fits.append(
            modified_poisson(
                design,
                current["event"].to_numpy(float),
                current["combined_weight"].to_numpy(float),
                current["communityID"].to_numpy(),
            )
        )
        flexible_design, _ = predictor_design(current, "Eight transitions plus flexible site count")
        flexible_fits.append(
            modified_poisson(
                flexible_design,
                current["event"].to_numpy(float),
                current["combined_weight"].to_numpy(float),
                current["communityID"].to_numpy(),
            )
        )
        print(f"Completed comparison imputation {imputation}/{args.imputations}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "model_comparison_by_imputation.csv", index=False)
    summarize_metrics(metrics).to_csv(
        args.output_dir / "model_comparison_cross_validated.csv", index=False
    )
    summarize_deltas(metrics).to_csv(args.output_dir / "model_comparison_deltas.csv", index=False)
    contrasts = [
        pool_adjusted_contrast(
            augmented_fits, "Model 1 covariates plus exact current pain-site count"
        ),
        pool_adjusted_contrast(
            flexible_fits, "Model 1 covariates plus current site-count categories (0/1/2/3/4/5+)"
        ),
    ]
    pd.DataFrame(contrasts).to_csv(args.output_dir / "site_count_adjusted_contrast.csv", index=False)


if __name__ == "__main__":
    main()
