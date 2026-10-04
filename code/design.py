"""Analysis and response-model design matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from charls_core import TRANSITION_LABELS


def _indicator(series: pd.Series, value: int) -> np.ndarray:
    return series.eq(value).astype(float).to_numpy()


def outcome_design(frame: pd.DataFrame, model: int) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = [np.ones(len(frame))]
    terms = ["Intercept"]
    for transition in range(1, 8):
        columns.append(_indicator(frame["transition"], transition))
        terms.append(TRANSITION_LABELS[transition])
    for window in (2, 3):
        columns.append(_indicator(frame["window"], window))
        terms.append(f"Window {window}")

    if model >= 1:
        columns.extend(
            [
                frame["age_landmark"].to_numpy(float) / 10,
                frame["female"].to_numpy(float),
            ]
        )
        terms.extend(["Age, per 10 y", "Female"])
        for education, label in ((1, "Primary education"), (2, "Middle education"), (3, "High education+")):
            columns.append(_indicator(frame["education"], education))
            terms.append(label)
        columns.extend(
            [
                frame["married"].to_numpy(float),
                frame["rural"].to_numpy(float),
                frame["agricultural_hukou"].to_numpy(float),
                frame["bmi"].to_numpy(float) / 5,
            ]
        )
        terms.extend(["Married", "Rural", "Agricultural hukou", "BMI, per 5 kg/m2"])

    if model >= 2:
        columns.extend(
            [
                frame["chronic_count"].to_numpy(float),
                frame["arthritis"].to_numpy(float),
                frame["stroke"].to_numpy(float),
                frame["cesd10"].to_numpy(float) / 5,
                frame["sleep_hours"].to_numpy(float) / 2,
                _indicator(frame["smoking"], 1),
                _indicator(frame["smoking"], 2),
                _indicator(frame["drinking"], 1),
                _indicator(frame["drinking"], 2),
                frame["fair_poor_health"].to_numpy(float),
                frame["mobility_count"].to_numpy(float),
                frame["iadl_count"].to_numpy(float),
            ]
        )
        terms.extend(
            [
                "Chronic count",
                "Arthritis",
                "Stroke",
                "CES-D, per 5",
                "Sleep, per 2 h",
                "Former smoker",
                "Current smoker",
                "Drinks > monthly",
                "Drinks ≤ monthly",
                "Fair/poor health",
                "Mobility count",
                "IADL count",
            ]
        )
    return np.column_stack(columns).astype(float), terms


def response_design(frame: pd.DataFrame) -> np.ndarray:
    """Return the saturated covariate design used for window-specific IPCW."""

    # outcome_design includes two window indicators. The response model is fit
    # separately by window, so those columns are removed.
    design, _ = outcome_design(frame, model=2)
    return np.delete(design, [8, 9], axis=1)
