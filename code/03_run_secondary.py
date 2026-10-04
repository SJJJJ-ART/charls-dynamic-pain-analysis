#!/usr/bin/env python3
"""Run standardization, PHC interaction, contrasts, and core sensitivities."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.special import expit, logit

from imputation import completed_dataset
from secondary import (
    descriptive_summary,
    direct_contrasts,
    factor_and_window_interaction,
    normal_summary,
    phc_analysis,
    sensitivity_estimates,
    start_state_contrasts,
    transition_standardization,
    window_standardization,
)
from statistical import rubin_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imputations", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=20260718)
    return parser.parse_args()


def _one(seed: int, eligible_path: str) -> dict[str, object]:
    frame = pd.read_pickle(eligible_path)
    completed = completed_dataset(frame, seed)
    risks, differences = transition_standardization(completed)
    window_risks, window_differences = window_standardization(completed)
    phc_risks, phc_interactions = phc_analysis(completed)
    _, phc_window1_interactions = phc_analysis(completed.loc[completed["window"].eq(1)].copy())
    return {
        "seed": seed,
        "risks": risks,
        "differences": differences,
        "window_risks": window_risks,
        "window_differences": window_differences,
        "contrasts": direct_contrasts(completed),
        "start_state_contrasts": start_state_contrasts(completed),
        "phc_risks": phc_risks,
        "phc_interactions": phc_interactions,
        "phc_window1_interactions": phc_window1_interactions,
        "sensitivities": sensitivity_estimates(completed),
        "descriptives": descriptive_summary(completed),
        "factor_interaction": factor_and_window_interaction(completed),
    }


def _pool_rows(results: list[dict[str, object]], key: str, group_columns: list[str], exponentiate: bool = False) -> pd.DataFrame:
    long_rows: list[dict[str, object]] = []
    for imputation, result in enumerate(results, 1):
        for row in result[key]:
            long_rows.append({"imputation": imputation, "seed": result["seed"], **row})
    long = pd.DataFrame(long_rows)
    pooled_rows = []
    for group_values, part in long.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        estimates = part["estimate"].to_numpy(float)[:, None]
        covariances = part["variance"].to_numpy(float)[:, None, None]
        q_bar, within, between, total = rubin_pool(estimates, covariances)
        summary = normal_summary(float(q_bar[0]), float(total[0, 0]), exponentiate=exponentiate)
        pooled = dict(zip(group_columns, group_values))
        pooled.update(summary)
        pooled["within_variance"] = float(within[0, 0])
        pooled["between_variance"] = float(between[0, 0])
        pooled["total_variance"] = float(total[0, 0])
        for column in ("n", "events"):
            if column in part:
                pooled[column] = int(round(part[column].mean()))
        pooled_rows.append(pooled)
    return pd.DataFrame(pooled_rows), long


def _bound_probability_intervals(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    invalid = result["ci_low"].lt(0) | result["ci_high"].gt(1)
    for index in result.index[invalid]:
        probability = float(np.clip(result.at[index, "estimate"], 1e-6, 1 - 1e-6))
        probability_se = float(result.at[index, "se"])
        logit_se = probability_se / (probability * (1 - probability))
        result.at[index, "ci_low"] = expit(logit(probability) - 1.96 * logit_se)
        result.at[index, "ci_high"] = expit(logit(probability) + 1.96 * logit_se)
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eligible_path = args.primary_results_dir / "private" / "eligible_with_covariates.pkl"
    if not eligible_path.exists():
        raise FileNotFoundError(
            "The restricted eligible file was not found. Run code/02_run_analysis.py first."
        )
    seeds = list(range(args.seed_start, args.seed_start + args.imputations))
    returned: dict[int, dict[str, object]] = {}
    if args.jobs == 1:
        for seed in seeds:
            returned[seed] = _one(seed, str(eligible_path))
            print(f"Completed secondary seed {seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_one, seed, str(eligible_path)): seed for seed in seeds}
            for future in as_completed(futures):
                result = future.result()
                returned[int(result["seed"])] = result
                print(f"Completed secondary seed {result['seed']}", flush=True)
    results = [returned[seed] for seed in seeds]

    risks, risk_long = _pool_rows(results, "risks", ["transition_code", "transition"])
    risks = _bound_probability_intervals(risks)
    differences, difference_long = _pool_rows(results, "differences", ["transition_code", "transition"])
    risk_output = risks.rename(columns={"estimate": "adjusted_risk", "ci_low": "risk_ci_low", "ci_high": "risk_ci_high"})
    difference_output = differences.rename(columns={"estimate": "risk_difference", "ci_low": "rd_ci_low", "ci_high": "rd_ci_high"})
    combined = risk_output.merge(
        difference_output[["transition_code", "transition", "risk_difference", "rd_ci_low", "rd_ci_high"]],
        on=["transition_code", "transition"],
        validate="one_to_one",
    )
    combined.to_csv(args.output_dir / "standardized_risks_and_differences.csv", index=False)
    risk_long.to_csv(args.output_dir / "standardized_risks_by_imputation.csv", index=False)
    difference_long.to_csv(args.output_dir / "risk_differences_by_imputation.csv", index=False)

    window_risks, window_risk_long = _pool_rows(
        results, "window_risks", ["window", "transition_code", "transition"]
    )
    window_risks = _bound_probability_intervals(window_risks)
    window_differences, window_difference_long = _pool_rows(
        results, "window_differences", ["window", "transition_code", "transition"]
    )
    window_combined = window_risks.rename(
        columns={"estimate": "adjusted_risk", "ci_low": "risk_ci_low", "ci_high": "risk_ci_high"}
    ).merge(
        window_differences[["window", "transition_code", "transition", "estimate", "ci_low", "ci_high"]].rename(
            columns={"estimate": "risk_difference", "ci_low": "rd_ci_low", "ci_high": "rd_ci_high"}
        ),
        on=["window", "transition_code", "transition"],
        validate="one_to_one",
    )
    window_combined.to_csv(args.output_dir / "window_standardized_risks.csv", index=False)
    window_risk_long.to_csv(args.output_dir / "window_risks_by_imputation.csv", index=False)
    window_difference_long.to_csv(args.output_dir / "window_risk_differences_by_imputation.csv", index=False)

    contrasts, contrast_long = _pool_rows(results, "contrasts", ["contrast"], exponentiate=True)
    contrasts.to_csv(args.output_dir / "direct_contrasts.csv", index=False)
    contrast_long.to_csv(args.output_dir / "direct_contrasts_by_imputation.csv", index=False)

    start_contrasts, start_contrast_long = _pool_rows(
        results, "start_state_contrasts", ["stratum", "contrast"], exponentiate=True
    )
    start_contrasts.to_csv(args.output_dir / "start_state_contrasts.csv", index=False)
    start_contrast_long.to_csv(args.output_dir / "start_state_contrasts_by_imputation.csv", index=False)

    phc_risks, phc_risk_long = _pool_rows(results, "phc_risks", ["pain_group", "phc_present"])
    phc_risks = phc_risks.rename(columns={"estimate": "adjusted_risk"})
    phc_risks.to_csv(args.output_dir / "phc_standardized_risks.csv", index=False)
    phc_risk_long.to_csv(args.output_dir / "phc_risks_by_imputation.csv", index=False)

    interactions, interaction_long = _pool_rows(results, "phc_interactions", ["interaction"])
    multiplicative = interactions["interaction"].str.contains("multiplicative")
    interactions.loc[multiplicative, "ratio_of_rr"] = np.exp(interactions.loc[multiplicative, "estimate"])
    interactions.loc[multiplicative, "ratio_ci_low"] = np.exp(interactions.loc[multiplicative, "ci_low"])
    interactions.loc[multiplicative, "ratio_ci_high"] = np.exp(interactions.loc[multiplicative, "ci_high"])
    interactions.to_csv(args.output_dir / "phc_interactions.csv", index=False)
    interaction_long.to_csv(args.output_dir / "phc_interactions_by_imputation.csv", index=False)

    window1_interactions, window1_interaction_long = _pool_rows(
        results, "phc_window1_interactions", ["interaction"]
    )
    window1_multiplicative = window1_interactions["interaction"].str.contains("multiplicative")
    window1_interactions.loc[window1_multiplicative, "ratio_of_rr"] = np.exp(
        window1_interactions.loc[window1_multiplicative, "estimate"]
    )
    window1_interactions.loc[window1_multiplicative, "ratio_ci_low"] = np.exp(
        window1_interactions.loc[window1_multiplicative, "ci_low"]
    )
    window1_interactions.loc[window1_multiplicative, "ratio_ci_high"] = np.exp(
        window1_interactions.loc[window1_multiplicative, "ci_high"]
    )
    window1_interactions.to_csv(args.output_dir / "phc_window1_interactions.csv", index=False)
    window1_interaction_long.to_csv(
        args.output_dir / "phc_window1_interactions_by_imputation.csv", index=False
    )

    sensitivities, sensitivity_long = _pool_rows(results, "sensitivities", ["analysis"], exponentiate=True)
    sensitivities.to_csv(args.output_dir / "core_sensitivity_analyses.csv", index=False)
    sensitivity_long.to_csv(args.output_dir / "core_sensitivities_by_imputation.csv", index=False)

    descriptive_rows = []
    for imputation, result in enumerate(results, 1):
        for row in result["descriptives"]:
            descriptive_rows.append({"imputation": imputation, "seed": result["seed"], **row})
    descriptive_long = pd.DataFrame(descriptive_rows)
    fixed = descriptive_long.groupby(["transition_code", "transition"], as_index=False)[["n", "events"]].first()
    averaged = descriptive_long.groupby(["transition_code", "transition"], as_index=False)[
        ["event", "age_landmark", "female", "chronic_count", "cesd10", "mobility_count"]
    ].mean()
    fixed.merge(averaged, on=["transition_code", "transition"], validate="one_to_one").to_csv(
        args.output_dir / "table1_descriptive_summary.csv", index=False
    )
    descriptive_long.to_csv(args.output_dir / "table1_descriptives_by_imputation.csv", index=False)

    factor_blocks = [result["factor_interaction"] for result in results]
    for prefix, terms_key, estimate_key, covariance_key in (
        ("transition_factor", "transition_terms", "transition_estimates", "transition_covariance"),
        ("transition_by_window", "interaction_terms", "interaction_estimates", "interaction_covariance"),
    ):
        estimates = np.stack([block[estimate_key] for block in factor_blocks])
        covariances = np.stack([block[covariance_key] for block in factor_blocks])
        q_bar, within, between, total = rubin_pool(estimates, covariances)
        statistic = float(q_bar @ np.linalg.pinv(total) @ q_bar)
        df = len(q_bar)
        pd.DataFrame(
            {
                "test": [prefix],
                "chi_square": [statistic],
                "df": [df],
                "p_value": [float(chi2.sf(statistic, df))],
            }
        ).to_csv(args.output_dir / f"{prefix}_wald_test.csv", index=False)
        pd.DataFrame(
            {
                "term": factor_blocks[0][terms_key],
                "estimate": q_bar,
                "se": np.sqrt(np.diag(total)),
            }
        ).to_csv(args.output_dir / f"{prefix}_coefficients.csv", index=False)


if __name__ == "__main__":
    main()
