"""Small, dependency-light estimators used by the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


@dataclass
class FitResult:
    coefficients: np.ndarray
    covariance: np.ndarray
    fitted: np.ndarray
    converged: bool
    iterations: int


def _safe_solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ vector


def weighted_logistic(
    design: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> FitResult:
    """Fit a survey-weighted logistic response model by Fisher scoring."""

    x = np.asarray(design, dtype=float)
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(weight, dtype=float)
    beta = np.zeros(x.shape[1])
    converged = False
    for iteration in range(1, max_iter + 1):
        probability = expit(np.clip(x @ beta, -30, 30))
        variance_weight = w * probability * (1 - probability)
        information = x.T @ (variance_weight[:, None] * x)
        score = x.T @ (w * (y - probability))
        step = _safe_solve(information, score)
        beta += step
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break
    probability = expit(np.clip(x @ beta, -30, 30))
    covariance = np.linalg.pinv(x.T @ ((w * probability * (1 - probability))[:, None] * x))
    return FitResult(beta, covariance, probability, converged, iteration)


def modified_poisson(
    design: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
    cluster: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-10,
    small_sample_correction: bool = True,
) -> FitResult:
    """Fit log-link Poisson regression with cluster-robust covariance."""

    x = np.asarray(design, dtype=float)
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(weight, dtype=float)
    beta = np.zeros(x.shape[1])
    converged = False
    for iteration in range(1, max_iter + 1):
        mean = np.exp(np.clip(x @ beta, -20, 20))
        information = x.T @ ((w * mean)[:, None] * x)
        score = x.T @ (w * (y - mean))
        step = _safe_solve(information, score)
        beta += step
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break

    mean = np.exp(np.clip(x @ beta, -20, 20))
    bread = np.linalg.pinv(x.T @ ((w * mean)[:, None] * x))
    score_rows = x * (w * (y - mean))[:, None]
    score_frame = pd.DataFrame(score_rows)
    score_frame["cluster"] = pd.Series(cluster).astype("string").to_numpy()
    cluster_scores = score_frame.groupby("cluster", observed=True).sum().to_numpy()
    meat = cluster_scores.T @ cluster_scores
    covariance = bread @ meat @ bread

    if small_sample_correction:
        groups = cluster_scores.shape[0]
        observations, parameters = x.shape
        if groups > 1 and observations > parameters:
            covariance *= (groups / (groups - 1)) * ((observations - 1) / (observations - parameters))
    return FitResult(beta, covariance, mean, converged, iteration)


def poisson_coefficients(
    design: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Return weighted log-link Poisson coefficients without a covariance matrix."""

    x = np.asarray(design, dtype=float)
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(weight, dtype=float)
    beta = np.zeros(x.shape[1])
    for _ in range(max_iter):
        mean = np.exp(np.clip(x @ beta, -20, 20))
        information = x.T @ ((w * mean)[:, None] * x)
        score = x.T @ (w * (y - mean))
        step = _safe_solve(information, score)
        beta += step
        if np.max(np.abs(step)) < tolerance:
            break
    return beta


def rubin_pool(estimates: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool a vector estimate and its full covariance with Rubin's rules."""

    q = np.asarray(estimates, dtype=float)
    u = np.asarray(covariances, dtype=float)
    m = q.shape[0]
    q_bar = q.mean(axis=0)
    within = u.mean(axis=0)
    between = np.cov(q, rowvar=False, ddof=1)
    if q.shape[1] == 1:
        between = np.asarray([[float(between)]])
    total = within + (1 + 1 / m) * between
    return q_bar, within, between, total


def coefficient_table(
    terms: list[str], coefficients: np.ndarray, covariance: np.ndarray
) -> pd.DataFrame:
    standard_error = np.sqrt(np.diag(covariance))
    z_value = coefficients / standard_error
    return pd.DataFrame(
        {
            "term": terms,
            "log_rr": coefficients,
            "se": standard_error,
            "rr": np.exp(coefficients),
            "ci_low": np.exp(coefficients - 1.96 * standard_error),
            "ci_high": np.exp(coefficients + 1.96 * standard_error),
            "p_value": 2 * norm.sf(np.abs(z_value)),
        }
    )
