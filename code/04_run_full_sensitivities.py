#!/usr/bin/env python3
"""Run outcome, threshold, proxy, window, and bootstrap sensitivities."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from covariates import MODEL1_COVARIATES, MODEL2_ADDITIONAL
from design import outcome_design
from imputation import completed_dataset
from models import make_ipcw
from secondary import normal_summary
from sensitivity_cohorts import (
    attach_proxy_exclusion,
    attach_severe_badl,
    build_death_composite_cohort,
    build_iadl_cohort,
    build_strict_pain_cohort,
)
from statistical import modified_poisson, poisson_coefficients, rubin_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--primary-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imputations", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=20260718)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    return parser.parse_args()


def _fit_persistent(frame: pd.DataFrame, cluster: str = "communityID") -> dict[str, float]:
    weighted = make_ipcw(frame)
    observed = weighted.loc[weighted["response"].eq(1)].copy()
    design, terms = outcome_design(observed, model=1)
    fit = modified_poisson(
        design,
        observed["event"].to_numpy(float),
        observed["combined_weight"].to_numpy(float),
        observed[cluster].to_numpy(),
    )
    position = terms.index("Persistent multisite")
    return {
        "estimate": float(fit.coefficients[position]),
        "variance": float(fit.covariance[position, position]),
        "n": len(observed),
        "events": int(observed["event"].sum()),
    }


def _one(seed: int, paths: dict[str, str]) -> dict[str, object]:
    frames = {name: completed_dataset(pd.read_pickle(path), seed) for name, path in paths.items()}
    rows = []
    for name in ("death", "strict", "iadl"):
        rows.append({"analysis": {"death": "BADL or death", "strict": "Pain threshold ≥ Some/Somewhat", "iadl": "Next-wave IADL limitation"}[name], **_fit_persistent(frames[name])})

    main = frames["main"]
    severe = main.copy()
    severe["event"] = severe["severe_event"]
    rows.append({"analysis": "Severe BADL", **_fit_persistent(severe)})

    no_proxy = main.loc[~main["proxy_exclude"]].copy()
    rows.append({"analysis": "Exclude proxy interviews", **_fit_persistent(no_proxy)})

    for window in (1, 2, 3):
        rows.append({"analysis": f"Window {window}", **_fit_persistent(main.loc[main["window"].eq(window)].copy())})

    return {"seed": seed, "rows": rows}


def _pool(results: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for imputation, result in enumerate(results, 1):
        for row in result["rows"]:
            rows.append({"imputation": imputation, "seed": result["seed"], **row})
    long = pd.DataFrame(rows)
    pooled = []
    for analysis, part in long.groupby("analysis", sort=False):
        q, within, between, total = rubin_pool(
            part["estimate"].to_numpy()[:, None],
            part["variance"].to_numpy()[:, None, None],
        )
        summary = normal_summary(float(q[0]), float(total[0, 0]), exponentiate=True)
        pooled.append(
            {
                "analysis": analysis,
                **summary,
                "n": int(round(part["n"].mean())),
                "events": int(round(part["events"].mean())),
                "within_variance": float(within[0, 0]),
                "between_variance": float(between[0, 0]),
                "total_variance": float(total[0, 0]),
            }
        )
    return pd.DataFrame(pooled), long


def _complete_case(frame: pd.DataFrame) -> dict[str, float]:
    complete = frame.dropna(subset=MODEL1_COVARIATES + MODEL2_ADDITIONAL).copy()
    result = _fit_persistent(complete)
    return {"analysis": "Complete cases for all Model 1 and Model 2 covariates", **normal_summary(result["estimate"], result["variance"], exponentiate=True), "n": result["n"], "events": result["events"]}


def _community_bootstrap(frame: pd.DataFrame, replicates: int, seed: int = 20260718) -> tuple[pd.DataFrame, dict[str, float]]:
    completed = completed_dataset(frame, seed)
    weighted = make_ipcw(completed)
    observed = weighted.loc[weighted["response"].eq(1)].copy()
    design, terms = outcome_design(observed, model=1)
    position = terms.index("Persistent multisite")
    clusters = observed["communityID"].drop_duplicates().to_numpy()
    row_indices = {cluster: np.flatnonzero(observed["communityID"].to_numpy() == cluster) for cluster in clusters}
    rng = np.random.default_rng(20260719)
    estimates = []
    for replicate in range(1, replicates + 1):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([row_indices[cluster] for cluster in sampled])
        beta = poisson_coefficients(
            design[indices],
            observed["event"].to_numpy(float)[indices],
            observed["combined_weight"].to_numpy(float)[indices],
        )
        estimates.append({"replicate": replicate, "log_rr": beta[position], "rr": np.exp(beta[position])})
    table = pd.DataFrame(estimates)
    summary = {
        "analysis": f"Community bootstrap ({replicates} replicates)",
        "estimate": float(table["log_rr"].median()),
        "rr": float(table["rr"].median()),
        "ci_low": float(table["rr"].quantile(0.025)),
        "ci_high": float(table["rr"].quantile(0.975)),
        "n": len(observed),
        "events": int(observed["event"].sum()),
    }
    return table, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private = args.output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    main_frame = pd.read_pickle(args.primary_results_dir / "private" / "eligible_with_covariates.pkl")
    main_frame = attach_proxy_exclusion(args.raw_dir, attach_severe_badl(args.raw_dir, main_frame))
    frames = {
        "main": main_frame,
        "death": build_death_composite_cohort(args.raw_dir),
        "strict": build_strict_pain_cohort(args.raw_dir),
        "iadl": build_iadl_cohort(args.raw_dir),
    }
    paths = {}
    for name, frame in frames.items():
        path = private / f"{name}_eligible.pkl"
        frame.to_pickle(path)
        paths[name] = str(path)

    seeds = list(range(args.seed_start, args.seed_start + args.imputations))
    returned = {}
    if args.jobs == 1:
        for seed in seeds:
            returned[seed] = _one(seed, paths)
            print(f"Completed full sensitivity seed {seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_one, seed, paths): seed for seed in seeds}
            for future in as_completed(futures):
                result = future.result()
                returned[int(result["seed"])] = result
                print(f"Completed full sensitivity seed {result['seed']}", flush=True)
    pooled, long = _pool([returned[seed] for seed in seeds])
    complete_case = _complete_case(main_frame)
    bootstrap, bootstrap_summary = _community_bootstrap(main_frame, args.bootstrap_replicates)
    output = pd.concat([pooled, pd.DataFrame([complete_case, bootstrap_summary])], ignore_index=True, sort=False)
    output.to_csv(args.output_dir / "full_sensitivity_analyses.csv", index=False)
    long.to_csv(args.output_dir / "full_sensitivities_by_imputation.csv", index=False)
    bootstrap.to_csv(args.output_dir / "community_bootstrap_replicates.csv", index=False)


if __name__ == "__main__":
    main()
