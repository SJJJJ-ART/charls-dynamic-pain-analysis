"""Secondary estimands derived from one completed CHARLS data set.

The functions in this module never write participant-level data.  They return
aggregate estimates and covariance components that can be pooled with Rubin's
rules by the command-line scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from charls_core import TRANSITION_LABELS
from design import outcome_design
from models import make_ipcw
from statistical import FitResult, modified_poisson


@dataclass
class ScalarEstimate:
    estimate: float
    variance: float


def _fit_model1(observed: pd.DataFrame, cluster: str = "communityID") -> tuple[FitResult, list[str]]:
    design, terms = outcome_design(observed, model=1)
    fit = modified_poisson(
        design,
        observed["event"].to_numpy(float),
        observed["combined_weight"].to_numpy(float),
        observed[cluster].to_numpy(),
    )
    return fit, terms


def _standardized_risk(
    observed: pd.DataFrame,
    fit: FitResult,
    transition: int,
) -> tuple[float, np.ndarray]:
    counterfactual = observed.copy()
    counterfactual["transition"] = transition
    design, _ = outcome_design(counterfactual, model=1)
    mean = np.exp(np.clip(design @ fit.coefficients, -20, 20))
    weight = observed["combined_weight"].to_numpy(float)
    denominator = weight.sum()
    risk = float(np.sum(weight * mean) / denominator)
    gradient = np.sum((weight * mean)[:, None] * design, axis=0) / denominator
    return risk, gradient


def transition_standardization(frame: pd.DataFrame) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return Model 1 standardized risks, risk differences, and variances."""

    weighted = make_ipcw(frame)
    observed = weighted.loc[weighted["response"].eq(1)].copy()
    fit, _ = _fit_model1(observed)
    risks: dict[int, tuple[float, np.ndarray]] = {
        code: _standardized_risk(observed, fit, code) for code in range(8)
    }
    risk_rows: list[dict[str, float]] = []
    rd_rows: list[dict[str, float]] = []
    reference, reference_gradient = risks[0]
    for code in range(8):
        risk, gradient = risks[code]
        risk_rows.append(
            {
                "transition_code": code,
                "transition": TRANSITION_LABELS[code],
                "estimate": risk,
                "variance": float(gradient @ fit.covariance @ gradient),
            }
        )
        difference_gradient = gradient - reference_gradient
        rd_rows.append(
            {
                "transition_code": code,
                "transition": TRANSITION_LABELS[code],
                "estimate": risk - reference,
                "variance": float(difference_gradient @ fit.covariance @ difference_gradient),
            }
        )
    return risk_rows, rd_rows


def window_standardization(frame: pd.DataFrame) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return Model 1 standardized risks and differences within each window."""

    risk_rows: list[dict[str, float]] = []
    rd_rows: list[dict[str, float]] = []
    for window in sorted(frame["window"].unique()):
        weighted = make_ipcw(frame.loc[frame["window"].eq(window)].copy())
        observed = weighted.loc[weighted["response"].eq(1)].copy()
        full_design, _ = outcome_design(observed, model=1)
        design = np.delete(full_design, [8, 9], axis=1)
        fit = modified_poisson(
            design,
            observed["event"].to_numpy(float),
            observed["combined_weight"].to_numpy(float),
            observed["communityID"].to_numpy(),
        )
        risks: dict[int, tuple[float, np.ndarray]] = {}
        for code in range(8):
            counterfactual = observed.copy()
            counterfactual["transition"] = code
            full_counterfactual, _ = outcome_design(counterfactual, model=1)
            counterfactual_design = np.delete(full_counterfactual, [8, 9], axis=1)
            mean = np.exp(np.clip(counterfactual_design @ fit.coefficients, -20, 20))
            weight = observed["combined_weight"].to_numpy(float)
            denominator = weight.sum()
            risk = float(np.sum(weight * mean) / denominator)
            gradient = np.sum((weight * mean)[:, None] * counterfactual_design, axis=0) / denominator
            risks[code] = (risk, gradient)
        reference, reference_gradient = risks[0]
        for code, (risk, gradient) in risks.items():
            risk_rows.append(
                {
                    "window": int(window),
                    "transition_code": code,
                    "transition": TRANSITION_LABELS[code],
                    "estimate": risk,
                    "variance": float(gradient @ fit.covariance @ gradient),
                }
            )
            difference_gradient = gradient - reference_gradient
            rd_rows.append(
                {
                    "window": int(window),
                    "transition_code": code,
                    "transition": TRANSITION_LABELS[code],
                    "estimate": risk - reference,
                    "variance": float(difference_gradient @ fit.covariance @ difference_gradient),
                }
            )
    return risk_rows, rd_rows


def factor_and_window_interaction(frame: pd.DataFrame) -> dict[str, object]:
    """Return transition-factor and transition-by-window coefficient blocks."""

    weighted = make_ipcw(frame)
    observed = weighted.loc[weighted["response"].eq(1)].copy()
    main, terms = outcome_design(observed, model=1)
    main_fit = modified_poisson(
        main,
        observed["event"].to_numpy(float),
        observed["combined_weight"].to_numpy(float),
        observed["communityID"].to_numpy(),
    )
    interactions = []
    interaction_terms = []
    for window in (2, 3):
        window_indicator = observed["window"].eq(window).astype(float).to_numpy()
        for transition in range(1, 8):
            interactions.append(observed["transition"].eq(transition).astype(float).to_numpy() * window_indicator)
            interaction_terms.append(f"{TRANSITION_LABELS[transition]} × Window {window}")
    design = np.column_stack([main, *interactions])
    fit = modified_poisson(
        design,
        observed["event"].to_numpy(float),
        observed["combined_weight"].to_numpy(float),
        observed["communityID"].to_numpy(),
    )
    transition_index = np.arange(1, 8)
    interaction_index = np.arange(main.shape[1], design.shape[1])
    return {
        "transition_terms": [TRANSITION_LABELS[code] for code in range(1, 8)],
        "transition_estimates": main_fit.coefficients[transition_index],
        "transition_covariance": main_fit.covariance[np.ix_(transition_index, transition_index)],
        "interaction_terms": interaction_terms,
        "interaction_estimates": fit.coefficients[interaction_index],
        "interaction_covariance": fit.covariance[np.ix_(interaction_index, interaction_index)],
    }


def direct_contrasts(frame: pd.DataFrame) -> list[dict[str, float]]:
    """Return the two prespecified Model 1 log-RR contrasts."""

    weighted = make_ipcw(frame)
    observed = weighted.loc[weighted["response"].eq(1)].copy()
    fit, terms = _fit_model1(observed)
    index = {term: position for position, term in enumerate(terms)}
    specifications = {
        "Persistent vs incident multisite": ("Persistent multisite", "Incident multisite"),
        "Spread vs persistent isolated": ("Spread to multisite", "Persistent isolated"),
    }
    rows: list[dict[str, float]] = []
    for label, (left, right) in specifications.items():
        contrast = np.zeros(len(terms))
        contrast[index[left]] = 1
        contrast[index[right]] = -1
        rows.append(
            {
                "contrast": label,
                "estimate": float(contrast @ fit.coefficients),
                "variance": float(contrast @ fit.covariance @ contrast),
            }
        )
    return rows


def _base_model1_without_transition(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    full, terms = outcome_design(frame, model=1)
    return full[:, [0, *range(8, full.shape[1])]], [terms[0], *terms[8:]]


def start_state_contrasts(frame: pd.DataFrame) -> list[dict[str, float]]:
    """Return isolated-start and axial-multisite-start explanatory contrasts."""

    weighted = make_ipcw(frame)
    observed = weighted.loc[weighted["response"].eq(1)].copy()
    rows: list[dict[str, float]] = []

    isolated = observed.loc[observed["start_state"].eq(2)].copy()
    base, base_terms = _base_model1_without_transition(isolated)
    persistent = isolated["landmark_state"].eq(2).astype(float).to_numpy()
    spread = isolated["landmark_state"].eq(3).astype(float).to_numpy()
    design = np.column_stack([base[:, 0], persistent, spread, base[:, 1:]])
    terms = ["Intercept", "Persistent isolated", "Spread", *base_terms[1:]]
    fit = modified_poisson(
        design,
        isolated["event"].to_numpy(float),
        isolated["combined_weight"].to_numpy(float),
        isolated["communityID"].to_numpy(),
    )
    for label, term in (
        ("Spread vs recovery/contraction among isolated-start intervals", "Spread"),
        ("Persistence vs recovery/contraction among isolated-start intervals", "Persistent isolated"),
    ):
        position = terms.index(term)
        rows.append(
            {
                "stratum": "Isolated axial start",
                "contrast": label,
                "estimate": float(fit.coefficients[position]),
                "variance": float(fit.covariance[position, position]),
            }
        )

    multisite = observed.loc[observed["start_state"].eq(3)].copy()
    base, base_terms = _base_model1_without_transition(multisite)
    nonaxial = multisite["landmark_state"].eq(1).astype(float).to_numpy()
    isolated_state = multisite["landmark_state"].eq(2).astype(float).to_numpy()
    persistent_state = multisite["landmark_state"].eq(3).astype(float).to_numpy()
    design = np.column_stack([base[:, 0], nonaxial, isolated_state, persistent_state, base[:, 1:]])
    terms = ["Intercept", "Landmark non-axial", "Landmark isolated axial", "Landmark axial multisite", *base_terms[1:]]
    fit = modified_poisson(
        design,
        multisite["event"].to_numpy(float),
        multisite["combined_weight"].to_numpy(float),
        multisite["communityID"].to_numpy(),
    )
    for label, term in (
        ("Landmark non-axial vs pain-free among axial-multisite-start intervals", "Landmark non-axial"),
        ("Landmark isolated axial vs pain-free among axial-multisite-start intervals", "Landmark isolated axial"),
        ("Landmark axial multisite vs pain-free among axial-multisite-start intervals", "Landmark axial multisite"),
    ):
        position = terms.index(term)
        rows.append(
            {
                "stratum": "Axial multisite start",
                "contrast": label,
                "estimate": float(fit.coefficients[position]),
                "variance": float(fit.covariance[position, position]),
            }
        )
    return rows


def _phc_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    base, base_terms = outcome_design(frame, model=1)
    other = frame["transition"].isin([1, 2, 3, 5, 7]).astype(float).to_numpy()
    high = frame["transition"].isin([4, 6]).astype(float).to_numpy()
    phc = frame["phc_present"].to_numpy(float)
    # Remove the seven transition indicators from the ordinary Model 1 design.
    retained = base[:, [0, *range(8, base.shape[1])]]
    retained_terms = [base_terms[0], *base_terms[8:]]
    columns = np.column_stack([retained[:, 0], other, high, phc, other * phc, high * phc, retained[:, 1:]])
    terms = [
        "Intercept",
        "Other transition",
        "High-risk pattern",
        "PHC present",
        "Other × PHC",
        "High-risk × PHC",
        *retained_terms[1:],
    ]
    return columns, terms


def _phc_standardized_risk(
    observed: pd.DataFrame,
    fit: FitResult,
    pain_group: int,
    phc_present: int,
) -> tuple[float, np.ndarray]:
    counterfactual = observed.copy()
    counterfactual["transition"] = {0: 0, 1: 1, 2: 6}[pain_group]
    counterfactual["phc_present"] = phc_present
    design, _ = _phc_design(counterfactual)
    mean = np.exp(np.clip(design @ fit.coefficients, -20, 20))
    weight = observed["combined_weight"].to_numpy(float)
    denominator = weight.sum()
    risk = float(np.sum(weight * mean) / denominator)
    gradient = np.sum((weight * mean)[:, None] * design, axis=0) / denominator
    return risk, gradient


def phc_analysis(frame: pd.DataFrame) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return six standardized PHC risks and multiplicative/additive interactions."""

    weighted = make_ipcw(frame)
    observed = weighted.loc[weighted["response"].eq(1)].copy()
    design, terms = _phc_design(observed)
    fit = modified_poisson(
        design,
        observed["event"].to_numpy(float),
        observed["combined_weight"].to_numpy(float),
        observed["communityID"].to_numpy(),
    )
    group_labels = {0: "Stable pain-free", 1: "Other transition", 2: "High-risk pattern"}
    cells: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    risk_rows: list[dict[str, float]] = []
    for group in range(3):
        for phc in (0, 1):
            risk, gradient = _phc_standardized_risk(observed, fit, group, phc)
            cells[(group, phc)] = (risk, gradient)
            risk_rows.append(
                {
                    "pain_group": group_labels[group],
                    "phc_present": phc,
                    "estimate": risk,
                    "variance": float(gradient @ fit.covariance @ gradient),
                }
            )

    high_interaction_index = terms.index("High-risk × PHC")
    interaction = np.zeros(len(terms))
    interaction[high_interaction_index] = 1

    r11, g11 = cells[(2, 1)]
    r10, g10 = cells[(2, 0)]
    r01, g01 = cells[(0, 1)]
    r00, g00 = cells[(0, 0)]
    additive_gradient = g11 - g10 - g01 + g00
    interactions = [
        {
            "interaction": "High-risk multiplicative interaction, log ratio of RRs",
            "estimate": float(interaction @ fit.coefficients),
            "variance": float(interaction @ fit.covariance @ interaction),
        },
        {
            "interaction": "High-risk additive interaction, risk scale",
            "estimate": float(r11 - r10 - r01 + r00),
            "variance": float(additive_gradient @ fit.covariance @ additive_gradient),
        },
    ]
    return risk_rows, interactions


def sensitivity_estimates(frame: pd.DataFrame) -> list[dict[str, float]]:
    """Return weight and clustering sensitivity estimates for one imputation."""

    weighted = make_ipcw(frame)
    observed = weighted.loc[weighted["response"].eq(1)].copy()
    specifications: list[tuple[str, pd.DataFrame, np.ndarray, str]] = []
    specifications.append(("Primary IPCW × baseline weight", observed, observed["combined_weight"].to_numpy(), "communityID"))
    specifications.append(("Unweighted", observed, np.ones(len(observed)), "communityID"))
    baseline = observed["baseline_weight"].to_numpy(float)
    specifications.append(("Baseline weight only", observed, baseline / baseline.mean(), "communityID"))
    specifications.append(("Participant-clustered SE", observed, observed["combined_weight"].to_numpy(), "ID"))
    specifications.append(("Exclude 2018–2020 window", observed.loc[observed["window"].ne(3)].copy(), observed.loc[observed["window"].ne(3), "combined_weight"].to_numpy(), "communityID"))

    for lower, upper, label in (
        (0.01, 0.99, "IPCW truncated at 1st/99th percentiles"),
        (0.005, 0.995, "IPCW truncated at 0.5th/99.5th percentiles"),
    ):
        truncated = observed.copy()
        pieces = []
        for _, part in truncated.groupby("window", observed=True):
            lo, hi = part["ipcw"].quantile([lower, upper])
            part = part.copy()
            part["ipcw_truncated"] = part["ipcw"].clip(lo, hi)
            survey = part["baseline_weight"] / part["baseline_weight"].mean()
            part["weight_truncated"] = part["ipcw_truncated"] * survey
            pieces.append(part)
        truncated = pd.concat(pieces).sort_index()
        specifications.append((label, truncated, truncated["weight_truncated"].to_numpy(), "communityID"))

    rows: list[dict[str, float]] = []
    for label, subset, weights, cluster in specifications:
        design, terms = outcome_design(subset, model=1)
        fit = modified_poisson(
            design,
            subset["event"].to_numpy(float),
            weights,
            subset[cluster].to_numpy(),
        )
        position = terms.index("Persistent multisite")
        rows.append(
            {
                "analysis": label,
                "estimate": float(fit.coefficients[position]),
                "variance": float(fit.covariance[position, position]),
                "n": len(subset),
                "events": int(subset["event"].sum()),
            }
        )
    return rows


def descriptive_summary(frame: pd.DataFrame) -> list[dict[str, float]]:
    """Return combined-weight descriptive estimates by transition."""

    weighted = make_ipcw(frame)
    observed = weighted.loc[weighted["response"].eq(1)].copy()
    variables = ["event", "age_landmark", "female", "chronic_count", "cesd10", "mobility_count"]
    rows = []
    for code, part in observed.groupby("transition", observed=True):
        weight = part["combined_weight"].to_numpy(float)
        row: dict[str, float] = {
            "transition_code": int(code),
            "transition": TRANSITION_LABELS[int(code)],
            "n": len(part),
            "events": int(part["event"].sum()),
        }
        for variable in variables:
            row[variable] = float(np.average(part[variable].to_numpy(float), weights=weight))
        rows.append(row)
    return rows


def normal_summary(estimate: float, variance: float, exponentiate: bool = False) -> dict[str, float]:
    standard_error = float(np.sqrt(max(variance, 0)))
    lower = estimate - 1.96 * standard_error
    upper = estimate + 1.96 * standard_error
    p_value = float(2 * norm.sf(abs(estimate / standard_error))) if standard_error > 0 else np.nan
    if exponentiate:
        return {
            "estimate": estimate,
            "se": standard_error,
            "rr": float(np.exp(estimate)),
            "ci_low": float(np.exp(lower)),
            "ci_high": float(np.exp(upper)),
            "p_value": p_value,
        }
    return {
        "estimate": estimate,
        "se": standard_error,
        "ci_low": lower,
        "ci_high": upper,
        "p_value": p_value,
    }
