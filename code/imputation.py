"""Thirty stochastic chained-regression imputations and postprocessing."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from covariates import MODEL1_COVARIATES, MODEL2_ADDITIONAL


IMPUTED_COLUMNS = MODEL1_COVARIATES + MODEL2_ADDITIONAL

LEGAL_RANGES = {
    "age_landmark": (45, 120, False),
    "female": (0, 1, True),
    "education": (0, 3, True),
    "married": (0, 1, True),
    "rural": (0, 1, True),
    "agricultural_hukou": (0, 1, True),
    "bmi": (10, 60, False),
    "chronic_count": (0, 13, True),
    "arthritis": (0, 1, True),
    "stroke": (0, 1, True),
    "cesd10": (0, 30, False),
    "sleep_hours": (0, 24, False),
    "smoking": (0, 2, True),
    "drinking": (1, 3, True),
    "fair_poor_health": (0, 1, True),
    "mobility_count": (0, 9, True),
    "iadl_count": (0, 5, True),
}


def _matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    matrix = frame[IMPUTED_COLUMNS].copy()
    for transition in range(1, 8):
        matrix[f"aux_transition_{transition}"] = frame["transition"].eq(transition).astype(int)
    for window in (2, 3):
        matrix[f"aux_window_{window}"] = frame["window"].eq(window).astype(int)
    matrix["aux_response"] = frame["response"].astype(int)
    matrix["aux_phc"] = frame["phc_present"].astype(int)
    matrix["aux_log_weight"] = np.log(frame["baseline_weight"])
    return matrix, list(matrix.columns)


def completed_dataset(frame: pd.DataFrame, seed: int, iterations: int = 15) -> pd.DataFrame:
    matrix, columns = _matrix(frame)
    imputer = IterativeImputer(
        max_iter=iterations,
        sample_posterior=True,
        random_state=seed,
        initial_strategy="mean",
        skip_complete=True,
        tol=1e-3,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Early stopping criterion not reached")
        transformed = imputer.fit_transform(matrix)
    completed = pd.DataFrame(transformed, columns=columns, index=frame.index)
    result = frame.copy()
    for column in IMPUTED_COLUMNS:
        lower, upper, integer = LEGAL_RANGES[column]
        values = completed[column].clip(lower, upper)
        if integer:
            values = values.round()
        result[column] = values
    return result
