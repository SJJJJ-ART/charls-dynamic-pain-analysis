"""Builders for outcome, exposure-threshold, death, and proxy sensitivities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from charls_core import (
    WINDOWS,
    badl_status,
    baseline_cohort,
    build_core_cohorts,
    normalize_2011_id,
    pain_state,
    read_stata_columns,
    transition_code,
)
from covariates import add_covariates


def _functional_status(raw_dir: Path, wave: int, outcome: str) -> pd.DataFrame:
    path = raw_dir / str(wave) / "Health_Status_and_Functioning.dta"
    if outcome in {"badl", "severe_badl"}:
        items = (
            ["db010", "db011", "db012", "db013", "db014", "db015"]
            if wave < 2020
            else ["db001", "db003", "db005", "db007", "db009", "db011"]
        )
    elif outcome == "iadl":
        items = (
            ["db016", "db017", "db018", "db019", "db020"]
            if wave < 2020
            else ["db012", "db014", "db016", "db022", "db020"]
        )
    else:
        raise ValueError(outcome)
    frame = read_stata_columns(path, ["ID", *items])
    values = frame[items]
    complete = values.notna().all(axis=1)
    threshold = 3 if outcome == "severe_badl" else 2
    status = values.ge(threshold).any(axis=1).astype(float).where(complete)
    return frame[["ID"]].assign(status=status)


def attach_severe_badl(raw_dir: Path, frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for window, _, _, outcome_wave in WINDOWS:
        status = _functional_status(raw_dir, outcome_wave, "severe_badl").rename(
            columns={"status": f"severe_event_{window}"}
        )
        result = result.merge(status, on="ID", how="left", validate="many_to_one")
    result["severe_event"] = np.select(
        [result["window"].eq(1), result["window"].eq(2), result["window"].eq(3)],
        [result["severe_event_1"], result["severe_event_2"], result["severe_event_3"]],
        default=np.nan,
    )
    # The primary BADL builder treats a documented questionnaire skip as
    # independence. Those intervals have an observed main event of zero even
    # though all six item fields are blank, so their severe event is also zero.
    result.loc[
        result["response"].eq(1) & result["event"].eq(0) & result["severe_event"].isna(),
        "severe_event",
    ] = 0
    return result


def _wave_proxy(raw_dir: Path, wave: int) -> pd.DataFrame:
    filename = "health_status_and_functioning.dta" if wave == 2011 else "Health_Status_and_Functioning.dta"
    if wave in (2013, 2018):
        column = "db034"
        frame = read_stata_columns(raw_dir / str(wave) / filename, ["ID", column])
        frame["proxy_wave"] = frame[column].notna()
    elif wave in (2011, 2015, 2020):
        column = "proxy"
        frame = read_stata_columns(raw_dir / str(wave) / filename, ["ID", column])
        frame["proxy_wave"] = frame[column].eq(1)
    else:
        raise ValueError(wave)
    if wave == 2011:
        frame["ID"] = normalize_2011_id(frame["ID"])
    return frame[["ID", "proxy_wave"]]


def attach_proxy_exclusion(raw_dir: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Mark proxy use at the landmark or outcome interview.

    CHARLS 2013 and 2018 expose the health-module proxy reason (`db034`),
    while 2015 and 2020 expose the direct proxy flag (`proxy`).  This rule
    reproduces the 1,020 excluded primary-analysis intervals.
    """

    result = frame.copy()
    for wave in (2013, 2015, 2018, 2020):
        proxy = _wave_proxy(raw_dir, wave).rename(columns={"proxy_wave": f"proxy_{wave}"})
        result = result.merge(proxy, on="ID", how="left", validate="many_to_one")
    result["proxy_exclude"] = np.select(
        [result["window"].eq(1), result["window"].eq(2), result["window"].eq(3)],
        [
            result[["proxy_2013", "proxy_2015"]].any(axis=1),
            result[["proxy_2015", "proxy_2018"]].any(axis=1),
            result[["proxy_2018", "proxy_2020"]].any(axis=1),
        ],
        default=False,
    ).astype(bool)
    return result


def build_strict_pain_cohort(raw_dir: Path) -> pd.DataFrame:
    cohort = build_core_cohorts(raw_dir, strict_pain=True)
    return add_covariates(raw_dir, cohort.alive_eligible)


def build_death_composite_cohort(raw_dir: Path) -> pd.DataFrame:
    baseline = baseline_cohort(raw_dir)
    merged = baseline[["ID"]].copy()
    for wave in (2011, 2013, 2015, 2018):
        merged = merged.merge(pain_state(raw_dir, wave), on="ID", how="left")
    for wave in (2013, 2015, 2018, 2020):
        merged = merged.merge(badl_status(raw_dir, wave), on="ID", how="left")

    parts = []
    for window, start_wave, landmark_wave, outcome_wave in WINDOWS:
        complete_pain = merged[[f"pain_{start_wave}", f"pain_{landmark_wave}"]].notna().all(axis=1)
        part = merged.loc[complete_pain & merged[f"badl_{landmark_wave}"].eq(0)].copy()
        sample = read_stata_columns(raw_dir / str(outcome_wave) / "Sample_Infor.dta", ["ID", "died"])
        part = part.merge(sample, on="ID", how="left", validate="one_to_one")
        part["window"] = window
        part["start_state"] = part[f"pain_{start_wave}"].astype(int)
        part["landmark_state"] = part[f"pain_{landmark_wave}"].astype(int)
        part["transition"] = transition_code(part["start_state"], part["landmark_state"])
        part["response"] = (part["died"].eq(1) | part[f"badl_{outcome_wave}"].notna()).astype(int)
        part["event"] = np.where(part["died"].eq(1), 1, part[f"badl_{outcome_wave}"])
        parts.append(part)
    return add_covariates(raw_dir, pd.concat(parts, ignore_index=True))


def build_iadl_cohort(raw_dir: Path) -> pd.DataFrame:
    baseline = baseline_cohort(raw_dir)
    merged = baseline[["ID"]].copy()
    for wave in (2011, 2013, 2015, 2018):
        merged = merged.merge(pain_state(raw_dir, wave), on="ID", how="left")
    for wave in (2013, 2015, 2018, 2020):
        iadl = _functional_status(raw_dir, wave, "iadl").rename(columns={"status": f"iadl_{wave}"})
        merged = merged.merge(iadl, on="ID", how="left")

    parts = []
    for window, start_wave, landmark_wave, outcome_wave in WINDOWS:
        complete_pain = merged[[f"pain_{start_wave}", f"pain_{landmark_wave}"]].notna().all(axis=1)
        part = merged.loc[complete_pain & merged[f"iadl_{landmark_wave}"].eq(0)].copy()
        sample = read_stata_columns(raw_dir / str(outcome_wave) / "Sample_Infor.dta", ["ID", "died"])
        part = part.merge(sample, on="ID", how="left", validate="one_to_one")
        part = part.loc[~part["died"].eq(1)].copy()
        part["window"] = window
        part["start_state"] = part[f"pain_{start_wave}"].astype(int)
        part["landmark_state"] = part[f"pain_{landmark_wave}"].astype(int)
        part["transition"] = transition_code(part["start_state"], part["landmark_state"])
        part["response"] = part[f"iadl_{outcome_wave}"].notna().astype(int)
        part["event"] = part[f"iadl_{outcome_wave}"]
        parts.append(part)
    return add_covariates(raw_dir, pd.concat(parts, ignore_index=True))
