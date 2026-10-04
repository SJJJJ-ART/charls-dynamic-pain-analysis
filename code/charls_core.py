"""Core harmonization and repeated-landmark cohort construction.

The code uses only released CHARLS variable names. It never writes raw or
person-level data into the public output directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PAIN_FREE = 0
NON_AXIAL = 1
ISOLATED_AXIAL = 2
AXIAL_MULTISITE = 3

STATE_LABELS = {
    PAIN_FREE: "Pain-free",
    NON_AXIAL: "Non-axial pain only",
    ISOLATED_AXIAL: "Isolated axial pain",
    AXIAL_MULTISITE: "Axial multisite pain",
}

TRANSITION_LABELS = {
    0: "Stable pain-free",
    1: "Non-axial/no-axial",
    2: "Incident isolated",
    3: "Persistent isolated",
    4: "Spread to multisite",
    5: "Incident multisite",
    6: "Persistent multisite",
    7: "Improvement/contraction",
}

WINDOWS = ((1, 2011, 2013, 2015), (2, 2013, 2015, 2018), (3, 2015, 2018, 2020))


def normalize_2011_id(series: pd.Series) -> pd.Series:
    """Convert the 11-character 2011 identifier to the later 12-character form."""

    text = series.astype("string")
    return text.str[:-2] + "0" + text.str[-2:]


def read_stata_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    return pd.read_stata(path, columns=list(columns), convert_categoricals=False)


def baseline_cohort(raw_dir: Path) -> pd.DataFrame:
    """Return baseline adults aged at least 45 at their actual interview year."""

    demographic = read_stata_columns(
        raw_dir / "2011" / "demographic_background.dta",
        ["ID", "ba002_1", "rgender", "bd001", "be001", "bc001", "communityID"],
    )
    weights = read_stata_columns(
        raw_dir / "2011" / "weight.dta",
        ["ID", "iyear", "ind_weight_ad2"],
    )
    frame = demographic.merge(weights, on="ID", how="left", validate="one_to_one")
    frame["interview_year"] = pd.to_numeric(frame["iyear"], errors="coerce")
    frame["baseline_age"] = frame["interview_year"] - frame["ba002_1"]
    frame = frame.loc[
        frame["baseline_age"].ge(45)
        & frame["ind_weight_ad2"].notna()
        & frame["ind_weight_ad2"].gt(0)
    ].copy()
    frame["ID"] = normalize_2011_id(frame["ID"])
    if len(frame) != 17_289:
        raise AssertionError(f"Expected 17,289 baseline adults, found {len(frame):,}")
    return frame


def pain_state(raw_dir: Path, wave: int, strict: bool = False) -> pd.DataFrame:
    """Harmonize one wave into four mutually exclusive pain states.

    The main definition treats A Little or higher as pain in 2013 and 2018.
    The strict definition treats Some/Somewhat or higher as pain. The 16th
    'other' site is not part of the 15 harmonized anatomic sites.
    """

    filename = "health_status_and_functioning.dta" if wave == 2011 else "Health_Status_and_Functioning.dta"
    screen = {2011: "da041", 2013: "wb16", 2015: "da041", 2018: "da041_w4"}[wave]
    site_prefix = "da042_s" if wave == 2018 else "da042s"
    sites = [f"{site_prefix}{index}" for index in range(1, 16)]
    frame = read_stata_columns(raw_dir / str(wave) / filename, ["ID", screen, *sites])
    if wave == 2011:
        frame["ID"] = normalize_2011_id(frame["ID"])

    selected = np.column_stack([frame[column].eq(index) for index, column in enumerate(sites, 1)])
    any_site = selected.any(axis=1)
    axial = selected[:, 7] | selected[:, 8]
    non_axial = selected[:, :7].any(axis=1) | selected[:, 9:].any(axis=1)

    if wave in (2011, 2015):
        screen_negative = frame[screen].eq(2)
        screen_positive = frame[screen].eq(1)
    else:
        threshold = 3 if strict else 2
        screen_negative = frame[screen].lt(threshold) & frame[screen].ge(1)
        screen_positive = frame[screen].ge(threshold)

    state = np.full(len(frame), np.nan)
    state[screen_negative & ~any_site] = PAIN_FREE
    if strict and wave in (2013, 2018):
        # Sites were collected for lower-severity pain, but the strict analysis
        # classifies an "A Little" response as pain-free. A "None" response
        # with listed sites remains contradictory and is not reclassified.
        state[frame[screen].eq(2)] = PAIN_FREE
    state[screen_positive & any_site & ~axial] = NON_AXIAL
    state[screen_positive & axial & ~non_axial] = ISOLATED_AXIAL
    state[screen_positive & axial & non_axial] = AXIAL_MULTISITE

    return frame[["ID"]].assign(**{f"pain_{wave}": state})


def badl_status(raw_dir: Path, wave: int) -> pd.DataFrame:
    """Construct BADL status and preserve incomplete, non-skip records as missing."""

    path = raw_dir / str(wave) / "Health_Status_and_Functioning.dta"
    if wave < 2020:
        items = ["db010", "db011", "db012", "db013", "db014", "db015"]
        skip_items = ["db001", "db005", "db006", "db007", "db008", "db009"]
    else:
        items = ["db001", "db003", "db005", "db007", "db009", "db011"]
        skip_items = []
    frame = read_stata_columns(path, ["ID", *items, *skip_items])
    values = frame[items]
    complete = values.notna().all(axis=1)
    limited = complete & values.ge(2).any(axis=1)
    independent = complete & values.eq(1).all(axis=1)
    if skip_items:
        documented_skip = values.isna().all(axis=1) & frame[skip_items].eq(1).all(axis=1)
    else:
        documented_skip = np.zeros(len(frame), dtype=bool)
    status = np.full(len(frame), np.nan)
    status[limited] = 1
    status[independent | documented_skip] = 0
    return frame[["ID"]].assign(**{f"badl_{wave}": status})


def transition_code(start: pd.Series, landmark: pd.Series) -> pd.Series:
    """Map the complete 4 by 4 state table to the eight reported transitions."""

    start_array = start.to_numpy()
    landmark_array = landmark.to_numpy()
    result = np.full(len(start), np.nan)
    result[(start_array == 0) & (landmark_array == 0)] = 0
    result[((start_array == 0) | (start_array == 1)) & (landmark_array == 1)] = 1
    result[(start_array == 1) & (landmark_array == 0)] = 1
    result[((start_array == 0) | (start_array == 1)) & (landmark_array == 2)] = 2
    result[(start_array == 2) & (landmark_array == 2)] = 3
    result[(start_array == 2) & (landmark_array == 3)] = 4
    result[((start_array == 0) | (start_array == 1)) & (landmark_array == 3)] = 5
    result[(start_array == 3) & (landmark_array == 3)] = 6
    result[((start_array == 2) | (start_array == 3)) & (landmark_array < start_array)] = 7
    return pd.Series(result, index=start.index, name="transition")


@dataclass
class CohortBuild:
    analysis: pd.DataFrame
    alive_eligible: pd.DataFrame
    flow: pd.DataFrame


def build_core_cohorts(raw_dir: Path, strict_pain: bool = False) -> CohortBuild:
    """Build the alive-eligible and outcome-observed repeated-landmark cohorts."""

    baseline = baseline_cohort(raw_dir)
    merged = baseline[["ID"]].copy()
    for wave in (2011, 2013, 2015, 2018):
        merged = merged.merge(pain_state(raw_dir, wave, strict=strict_pain), on="ID", how="left")
    for wave in (2013, 2015, 2018, 2020):
        merged = merged.merge(badl_status(raw_dir, wave), on="ID", how="left")

    analysis_parts: list[pd.DataFrame] = []
    eligible_parts: list[pd.DataFrame] = []
    flow_rows: list[dict[str, int]] = []
    for window, start_wave, landmark_wave, outcome_wave in WINDOWS:
        pain_complete = merged[[f"pain_{start_wave}", f"pain_{landmark_wave}"]].notna().all(axis=1)
        observed_pain = merged.loc[pain_complete].copy()
        independent = observed_pain.loc[observed_pain[f"badl_{landmark_wave}"].eq(0)].copy()

        sample = read_stata_columns(raw_dir / str(outcome_wave) / "Sample_Infor.dta", ["ID", "died"])
        independent = independent.merge(sample, on="ID", how="left", validate="one_to_one")
        confirmed_death = independent["died"].eq(1)
        eligible = independent.loc[~confirmed_death].copy()
        eligible["window"] = window
        eligible["start_state"] = eligible[f"pain_{start_wave}"].astype(int)
        eligible["landmark_state"] = eligible[f"pain_{landmark_wave}"].astype(int)
        eligible["transition"] = transition_code(eligible["start_state"], eligible["landmark_state"])
        eligible["response"] = eligible[f"badl_{outcome_wave}"].notna().astype(int)
        eligible["event"] = eligible[f"badl_{outcome_wave}"]
        eligible_parts.append(eligible)
        analysis_parts.append(eligible.loc[eligible["response"].eq(1)].copy())

        flow_rows.append(
            {
                "window": window,
                "observed_pain_states": len(observed_pain),
                "badl_independent": len(independent),
                "confirmed_deaths": int(confirmed_death.sum()),
                "alive_eligible": len(eligible),
                "missing_outcome": int(eligible["response"].eq(0).sum()),
                "analysis_intervals": int(eligible["response"].eq(1).sum()),
                "events": int(eligible["event"].eq(1).sum()),
            }
        )

    alive = pd.concat(eligible_parts, ignore_index=True)
    analysis = pd.concat(analysis_parts, ignore_index=True)
    flow = pd.DataFrame(flow_rows)
    expected = {
        "observed_pain_states": [13_324, 11_733, 11_591],
        "badl_independent": [10_923, 9_136, 9_031],
        "confirmed_deaths": [201, 285, 201],
        "alive_eligible": [10_722, 8_851, 8_830],
        "missing_outcome": [801, 578, 450],
        "analysis_intervals": [9_921, 8_273, 8_380],
        "events": [1_489, 1_064, 1_430],
    }
    if not strict_pain:
        for column, values in expected.items():
            if flow[column].tolist() != values:
                raise AssertionError(f"Unexpected {column}: {flow[column].tolist()} != {values}")
        if analysis["ID"].nunique() != 11_947:
            raise AssertionError("Expected 11,947 unique analysis participants")
    return CohortBuild(analysis=analysis, alive_eligible=alive, flow=flow)


def aggregate_transition_cells(analysis: pd.DataFrame) -> pd.DataFrame:
    cells = (
        analysis.groupby(["start_state", "landmark_state"], observed=True)["event"]
        .agg(intervals="size", events="sum")
        .reset_index()
    )
    cells["unweighted_risk"] = cells["events"] / cells["intervals"]
    cells["start_label"] = cells["start_state"].map(STATE_LABELS)
    cells["landmark_label"] = cells["landmark_state"].map(STATE_LABELS)
    return cells
