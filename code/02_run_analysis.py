#!/usr/bin/env python3
"""Run 30 imputations, IPCW models, and Rubin pooling."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from covariates import add_covariates, missingness_table
from imputation import IMPUTED_COLUMNS, completed_dataset
from models import ImputationAnalysis, analyze_imputation, pool_model_fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imputations", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=20260718)
    return parser.parse_args()


def _one(seed: int, eligible_path: str) -> tuple[int, ImputationAnalysis, dict[str, float]]:
    frame = pd.read_pickle(eligible_path)
    completed = completed_dataset(frame, seed)
    analysis = analyze_imputation(completed)
    diagnostics: dict[str, float] = {}
    for column in IMPUTED_COLUMNS:
        missing = frame[column].isna()
        diagnostics[f"{column}__completed_mean"] = float(completed[column].mean())
        diagnostics[f"{column}__imputed_only_mean"] = (
            float(completed.loc[missing, column].mean()) if missing.any() else np.nan
        )
    return seed, analysis, diagnostics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)

    alive = pd.read_pickle(args.derived_dir / "alive_eligible_person_intervals.pkl")
    eligible = add_covariates(args.raw_dir, alive)
    eligible_path = private_dir / "eligible_with_covariates.pkl"
    eligible.to_pickle(eligible_path)
    observed = eligible.loc[eligible["response"].eq(1)]
    missingness_table(observed).to_csv(args.output_dir / "missingness_analysis.csv", index=False)
    missingness_table(eligible).to_csv(args.output_dir / "missingness_alive_eligible.csv", index=False)

    seeds = list(range(args.seed_start, args.seed_start + args.imputations))
    results: dict[int, ImputationAnalysis] = {}
    diagnostics: dict[int, dict[str, float]] = {}
    if args.jobs == 1:
        for seed in seeds:
            returned_seed, analysis, means = _one(seed, str(eligible_path))
            results[returned_seed] = analysis
            diagnostics[returned_seed] = means
            print(f"Completed seed {seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_one, seed, str(eligible_path)): seed for seed in seeds}
            for future in as_completed(futures):
                returned_seed, analysis, means = future.result()
                results[returned_seed] = analysis
                diagnostics[returned_seed] = means
                print(f"Completed seed {returned_seed}", flush=True)

    analyses = [results[seed] for seed in seeds]
    pooled_tables = []
    component_rows = []
    imputation_rows = []
    for model in (0, 1, 2):
        pooled, components = pool_model_fits(analyses, model)
        pooled_tables.append(pooled)
        terms = analyses[0].model_terms[model]
        for index, term in enumerate(terms):
            component_rows.append(
                {
                    "model": f"Model {model}",
                    "term": term,
                    "m": len(analyses),
                    "q_bar": components["q_bar"][index],
                    "within_variance": components["within"][index, index],
                    "between_variance": components["between"][index, index],
                    "total_variance": components["total"][index, index],
                }
            )
        for imputation, analysis in enumerate(analyses, 1):
            fit = analysis.model_fits[model]
            for index, term in enumerate(terms):
                imputation_rows.append(
                    {
                        "imputation": imputation,
                        "seed": seeds[imputation - 1],
                        "model": f"Model {model}",
                        "term": term,
                        "log_rr": fit.coefficients[index],
                        "within_variance": fit.covariance[index, index],
                    }
                )

    pd.concat(pooled_tables, ignore_index=True).to_csv(args.output_dir / "pooled_models.csv", index=False)
    pd.DataFrame(component_rows).to_csv(args.output_dir / "rubin_components.csv", index=False)
    pd.DataFrame(imputation_rows).to_csv(args.output_dir / "models_by_imputation.csv", index=False)
    pd.DataFrame.from_dict(diagnostics, orient="index").rename_axis("seed").reset_index().to_csv(
        args.output_dir / "imputation_diagnostics.csv", index=False
    )

    weight_parts = []
    summary_parts = []
    for imputation, analysis in enumerate(analyses, 1):
        weights = analysis.response_weights.copy()
        weights["imputation"] = imputation
        weight_parts.append(weights)
        for window, part in weights.groupby("window"):
            responders = part.loc[part["response"].eq(1)]
            combined = responders["combined_weight"].to_numpy()
            summary_parts.append(
                {
                    "imputation": imputation,
                    "window": int(window),
                    "alive_eligible": len(part),
                    "respondents": int(part["response"].sum()),
                    "ipcw_p50": responders["ipcw"].median(),
                    "ipcw_p99": responders["ipcw"].quantile(0.99),
                    "ipcw_max": responders["ipcw"].max(),
                    "combined_max": responders["combined_weight"].max(),
                    "ess": combined.sum() ** 2 / np.square(combined).sum(),
                }
            )
    weights_long = pd.concat(weight_parts, ignore_index=True)
    weights_long.to_csv(private_dir / "ipcw_individual_weights_long.csv.gz", index=False, compression="gzip")
    pd.DataFrame(summary_parts).to_csv(args.output_dir / "ipcw_diagnostics_by_imputation.csv", index=False)
    (
        weights_long.groupby(["ID", "window"], as_index=False)[["ipcw", "combined_weight"]]
        .mean()
        .to_csv(private_dir / "ipcw_individual_weights_mean.csv", index=False)
    )
    metadata = {
        "seeds": seeds,
        "imputations": len(seeds),
        "iterations": 15,
        "person_intervals": int(observed.shape[0]),
        "events": int(observed["event"].sum()),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
