"""IPCW construction, primary models, and Rubin pooling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from design import outcome_design, response_design
from statistical import FitResult, modified_poisson, rubin_pool, weighted_logistic


@dataclass
class ImputationAnalysis:
    model_fits: dict[int, FitResult]
    model_terms: dict[int, list[str]]
    response_weights: pd.DataFrame


def make_ipcw(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for window in sorted(frame["window"].dropna().astype(int).unique()):
        part = frame.loc[frame["window"].eq(window)].copy()
        survey_weight = part["baseline_weight"].to_numpy(float)
        response = part["response"].to_numpy(float)
        numerator = np.average(response, weights=survey_weight)
        design = response_design(part)
        fit = weighted_logistic(design, response, survey_weight)
        probability = np.clip(fit.fitted, 0.02, 0.98)
        stabilized = numerator / probability
        normalized_survey = survey_weight / survey_weight.mean()
        part["ipcw"] = stabilized
        part["combined_weight"] = stabilized * normalized_survey
        pieces.append(part)
    return pd.concat(pieces).sort_index()


def analyze_imputation(frame: pd.DataFrame, cluster_column: str = "communityID") -> ImputationAnalysis:
    weighted = make_ipcw(frame)
    observed = weighted.loc[weighted["response"].eq(1)].copy()
    fits: dict[int, FitResult] = {}
    terms: dict[int, list[str]] = {}
    for model in (0, 1, 2):
        design, labels = outcome_design(observed, model)
        fits[model] = modified_poisson(
            design,
            observed["event"].to_numpy(float),
            observed["combined_weight"].to_numpy(float),
            observed[cluster_column].to_numpy(),
        )
        terms[model] = labels
    weights = weighted[["ID", "window", "response", "ipcw", "combined_weight"]].copy()
    return ImputationAnalysis(fits, terms, weights)


def pool_model_fits(analyses: list[ImputationAnalysis], model: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    estimates = np.stack([analysis.model_fits[model].coefficients for analysis in analyses])
    covariances = np.stack([analysis.model_fits[model].covariance for analysis in analyses])
    q_bar, within, between, total = rubin_pool(estimates, covariances)
    terms = analyses[0].model_terms[model]
    se = np.sqrt(np.diag(total))
    table = pd.DataFrame(
        {
            "model": f"Model {model}",
            "term": terms,
            "log_rr": q_bar,
            "se": se,
            "rr": np.exp(q_bar),
            "ci_low": np.exp(q_bar - 1.96 * se),
            "ci_high": np.exp(q_bar + 1.96 * se),
        }
    )
    components = {"q_bar": q_bar, "within": within, "between": between, "total": total}
    return table, components
