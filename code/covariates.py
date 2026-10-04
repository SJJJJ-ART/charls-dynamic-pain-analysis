"""Covariate harmonization for the CHARLS repeated-landmark analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from charls_core import normalize_2011_id, read_stata_columns


LANDMARK_WAVES = {1: 2013, 2: 2015, 3: 2018}


def _baseline_data(raw_dir: Path) -> pd.DataFrame:
    demographic = read_stata_columns(
        raw_dir / "2011" / "demographic_background.dta",
        ["ID", "ba002_1", "rgender", "bd001", "bc001", "communityID"],
    )
    weights = read_stata_columns(
        raw_dir / "2011" / "weight.dta", ["ID", "ind_weight_ad2"]
    )
    biomarkers = read_stata_columns(
        raw_dir / "2011" / "biomarkers.dta", ["ID", "qi002", "ql002"]
    )
    for frame in (demographic, weights, biomarkers):
        frame["ID"] = normalize_2011_id(frame["ID"])

    height = biomarkers["qi002"].where(biomarkers["qi002"].between(100, 250))
    weight = biomarkers["ql002"].where(biomarkers["ql002"].between(20, 200))
    biomarkers["bmi"] = weight / (height / 100) ** 2

    psu = read_stata_columns(
        raw_dir / "2011" / "psu.dta", ["communityID", "urban_nbs"]
    )
    result = (
        demographic.merge(weights, on="ID", how="left", validate="one_to_one")
        .merge(biomarkers[["ID", "bmi"]], on="ID", how="left", validate="one_to_one")
        .merge(psu, on="communityID", how="left", validate="many_to_one")
    )
    return result


def _community_exposure(raw_dir: Path) -> pd.DataFrame:
    """Create baseline primary-care presence at the community level."""

    facility_columns = [f"jf002_{index}_" for index in (5, 6, 7, 8)]
    community = read_stata_columns(
        raw_dir / "2011" / "community.dta", ["communityID", *facility_columns]
    )
    community["phc_present_row"] = community[facility_columns].eq(1).any(axis=1).astype(int)
    # Four community identifiers have more than one subcommunity record. A
    # facility in either record establishes presence for that baseline cluster.
    return (
        community.groupby("communityID", as_index=False)["phc_present_row"]
        .max()
        .rename(columns={"phc_present_row": "phc_present"})
    )


def _landmark_marital(raw_dir: Path, intervals: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for window, wave in LANDMARK_WAVES.items():
        demographic = read_stata_columns(
            raw_dir / str(wave) / "Demographic_Background.dta", ["ID", "be001"]
        )
        part = intervals.loc[intervals["window"].eq(window), ["ID", "window"]].merge(
            demographic, on="ID", how="left", validate="many_to_one"
        )
        # Codes 1 and 2 denote married. Codes 3 through 6 denote not married.
        # Code 7 denotes cohabitation in 2013 and 2015. The legacy verification
        # analysis treated code 7 as outside the prespecified 1--6 legal range;
        # a corrected partnered-status sensitivity analysis is provided later.
        part["married"] = part["be001"].isin([1, 2]).where(part["be001"].isin(range(1, 7)))
        part["partnered_corrected"] = part["be001"].isin([1, 2, 7]).where(
            part["be001"].isin(range(1, 8))
        )
        parts.append(part[["ID", "window", "married", "partnered_corrected"]])
    return pd.concat(parts, ignore_index=True)


def _self_rated_health(raw_dir: Path, intervals: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for window, wave in LANDMARK_WAVES.items():
        columns = ["ID", "da002"] + (["da079"] if wave < 2018 else [])
        health = read_stata_columns(
            raw_dir / str(wave) / "Health_Status_and_Functioning.dta", columns
        )
        part = intervals.loc[intervals["window"].eq(window), ["ID", "window"]].merge(
            health, on="ID", how="left", validate="many_to_one"
        )
        rating = part["da002"] if wave == 2018 else part["da079"].combine_first(part["da002"])
        part["fair_poor_health"] = rating.ge(3).where(rating.isin([1, 2, 3, 4, 5]))
        parts.append(part[["ID", "window", "fair_poor_health"]])
    return pd.concat(parts, ignore_index=True)


def _cesd10(raw_dir: Path, intervals: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    items = [f"dc{index:03d}" for index in range(9, 19)]
    for window, wave in LANDMARK_WAVES.items():
        filename = "Cognition.dta" if wave == 2018 else "Health_Status_and_Functioning.dta"
        cognition = read_stata_columns(raw_dir / str(wave) / filename, ["ID", *items])
        part = intervals.loc[intervals["window"].eq(window), ["ID", "window"]].merge(
            cognition, on="ID", how="left", validate="many_to_one"
        )
        values = part[items].where(part[items].isin([1, 2, 3, 4]))
        scored = values - 1
        for positive_item in ("dc013", "dc016"):
            scored[positive_item] = 3 - scored[positive_item]
        valid = scored.notna().sum(axis=1)
        # CHARLS studies commonly prorate the score when at least 8 of 10
        # responses are present.
        part["cesd10"] = (scored.sum(axis=1, min_count=8) * 10 / valid).where(valid.ge(8))
        parts.append(part[["ID", "window", "cesd10"]])
    return pd.concat(parts, ignore_index=True)


def _mobility_and_iadl(raw_dir: Path, intervals: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    mobility_items = [f"db{index:03d}" for index in range(1, 10)]
    iadl_items = ["db016", "db017", "db018", "db019", "db020"]
    for window, wave in LANDMARK_WAVES.items():
        health = read_stata_columns(
            raw_dir / str(wave) / "Health_Status_and_Functioning.dta",
            ["ID", *mobility_items, *iadl_items],
        )
        part = intervals.loc[intervals["window"].eq(window), ["ID", "window"]].merge(
            health, on="ID", how="left", validate="many_to_one"
        )
        mobility = part[mobility_items].copy()
        mobility.loc[mobility["db002"].isna() & mobility["db001"].eq(1), "db002"] = 1
        mobility.loc[
            mobility["db003"].isna()
            & (mobility["db001"].eq(1) | mobility["db002"].eq(1)),
            "db003",
        ] = 1
        mobility_complete = mobility.notna().all(axis=1)
        part["mobility_count"] = mobility.ge(2).sum(axis=1).where(mobility_complete)

        iadl = part[iadl_items]
        iadl_complete = iadl.notna().all(axis=1)
        part["iadl_count"] = iadl.ge(2).sum(axis=1).where(iadl_complete)
        parts.append(part[["ID", "window", "mobility_count", "iadl_count"]])
    return pd.concat(parts, ignore_index=True)


def _sleep_and_drinking(raw_dir: Path, intervals: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for window, wave in LANDMARK_WAVES.items():
        health = read_stata_columns(
            raw_dir / str(wave) / "Health_Status_and_Functioning.dta",
            ["ID", "da049", "da067"],
        )
        part = intervals.loc[intervals["window"].eq(window), ["ID", "window"]].merge(
            health, on="ID", how="left", validate="many_to_one"
        )
        part["sleep_hours"] = part["da049"].where(part["da049"].between(0, 24))
        part["drinking"] = part["da067"].where(part["da067"].isin([1, 2, 3]))
        parts.append(part[["ID", "window", "sleep_hours", "drinking"]])
    return pd.concat(parts, ignore_index=True)


def _smoking(raw_dir: Path, intervals: pd.DataFrame) -> pd.DataFrame:
    """Create never/former/current status by carrying confirmed status forward."""

    baseline = read_stata_columns(
        raw_dir / "2011" / "health_status_and_functioning.dta", ["ID", "da059", "da061"]
    )
    baseline["ID"] = normalize_2011_id(baseline["ID"])
    baseline = baseline.set_index("ID")
    status = pd.Series(np.nan, index=baseline.index)
    status[baseline["da059"].eq(2)] = 0
    status[baseline["da059"].eq(1) & baseline["da061"].eq(2)] = 1
    status[baseline["da059"].eq(1) & baseline["da061"].eq(1)] = 2

    parts: list[pd.DataFrame] = []
    for window, wave in LANDMARK_WAVES.items():
        available = ["ID", "da059", "da061"]
        if wave == 2015:
            available.append("da061_w3")
        if wave == 2018:
            available.append("da061_w4")
        health = read_stata_columns(
            raw_dir / str(wave) / "Health_Status_and_Functioning.dta", available
        ).set_index("ID")
        previous = status.reindex(health.index)
        updated = previous.copy()
        updated[health["da059"].eq(2)] = 0
        still = health["da061"].copy()
        for column in ("da061_w3", "da061_w4"):
            if column in health:
                still = health[column].combine_first(still)
        updated[still.isin([2, 3])] = 1
        updated[still.eq(1)] = 2
        status = updated

        part = intervals.loc[intervals["window"].eq(window), ["ID", "window"]].copy()
        part["smoking"] = part["ID"].map(status)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _chronic_conditions(raw_dir: Path, intervals: pd.DataFrame) -> pd.DataFrame:
    """Construct diagnosed-condition indicators from all information available.

    This implementation retains a prior confirmed diagnosis and adds a later
    confirmed diagnosis. It is intentionally isolated because the cross-wave
    chronic-disease routing is the least transparent part of the public wave
    files. The validation report compares its missingness and distribution with
    the archived analysis diagnostics.
    """

    indices = range(1, 14)
    baseline_columns = ["ID", *[f"da007_{index}_" for index in indices]]
    baseline = read_stata_columns(
        raw_dir / "2011" / "health_status_and_functioning.dta", baseline_columns
    )
    baseline["ID"] = normalize_2011_id(baseline["ID"])
    baseline = baseline.set_index("ID")
    statuses: dict[int, pd.Series] = {}
    for index in indices:
        raw = baseline[f"da007_{index}_"]
        status = pd.Series(np.nan, index=baseline.index)
        status[raw.eq(2)] = 0
        status[raw.eq(1)] = 1
        statuses[index] = status

    parts: list[pd.DataFrame] = []
    for window, wave in LANDMARK_WAVES.items():
        columns = ["ID"]
        for index in indices:
            columns.append(f"da007_{index}_")
            if wave < 2018:
                columns.extend([f"zda007_{index}_", f"da007_w2_2_{index}_"])
        health = read_stata_columns(
            raw_dir / str(wave) / "Health_Status_and_Functioning.dta", columns
        ).set_index("ID")

        for index in indices:
            previous = statuses[index].reindex(health.index)
            current = previous.copy()
            new = health[f"da007_{index}_"]
            if wave < 2018:
                prior_record = health[f"zda007_{index}_"]
                since_last = health[f"da007_w2_2_{index}_"]
                current[(prior_record.eq(2) | since_last.eq(2) | new.eq(2)) & ~previous.eq(1)] = 0
                current[prior_record.eq(1) | since_last.eq(1) | new.eq(1) | previous.eq(1)] = 1
            else:
                current[new.eq(2) & ~previous.eq(1)] = 0
                current[new.eq(1) | previous.eq(1)] = 1
            statuses[index] = current

        part = intervals.loc[intervals["window"].eq(window), ["ID", "window"]].copy()
        condition_columns: list[str] = []
        for index in indices:
            column = f"condition_{index}"
            part[column] = part["ID"].map(statuses[index])
            condition_columns.append(column)
        part["chronic_count"] = part[condition_columns].sum(axis=1, min_count=len(condition_columns))
        part["stroke"] = part["condition_8"]
        part["arthritis"] = part["condition_13"]
        parts.append(part[["ID", "window", "chronic_count", "arthritis", "stroke"]])
    return pd.concat(parts, ignore_index=True)


def add_covariates(raw_dir: Path, intervals: pd.DataFrame) -> pd.DataFrame:
    """Add Model 1, Model 2, survey-weight, and community variables."""

    frame = intervals.copy()
    baseline = _baseline_data(raw_dir)
    frame = frame.merge(baseline, on="ID", how="left", validate="many_to_one")
    frame = frame.merge(_community_exposure(raw_dir), on="communityID", how="left", validate="many_to_one")
    frame = frame.merge(_landmark_marital(raw_dir, frame), on=["ID", "window"], how="left")
    frame = frame.merge(_self_rated_health(raw_dir, frame), on=["ID", "window"], how="left")
    frame = frame.merge(_cesd10(raw_dir, frame), on=["ID", "window"], how="left")
    frame = frame.merge(_mobility_and_iadl(raw_dir, frame), on=["ID", "window"], how="left")
    frame = frame.merge(_sleep_and_drinking(raw_dir, frame), on=["ID", "window"], how="left")
    frame = frame.merge(_smoking(raw_dir, frame), on=["ID", "window"], how="left")
    frame = frame.merge(_chronic_conditions(raw_dir, frame), on=["ID", "window"], how="left")

    landmark_year = frame["window"].map(LANDMARK_WAVES)
    frame["age_landmark"] = landmark_year - frame["ba002_1"]
    frame["female"] = frame["rgender"].eq(2).where(frame["rgender"].isin([1, 2]))
    frame["education"] = np.select(
        [
            frame["bd001"].isin([1, 2, 3]),
            frame["bd001"].eq(4),
            frame["bd001"].eq(5),
            frame["bd001"].ge(6),
        ],
        [0, 1, 2, 3],
        default=np.nan,
    )
    frame["rural"] = frame["urban_nbs"].eq(0).where(frame["urban_nbs"].isin([0, 1]))
    frame["agricultural_hukou"] = frame["bc001"].eq(1).where(frame["bc001"].isin([1, 2, 3, 4]))
    frame["baseline_weight"] = frame["ind_weight_ad2"]
    return frame


MODEL1_COVARIATES = [
    "age_landmark",
    "female",
    "education",
    "married",
    "rural",
    "agricultural_hukou",
    "bmi",
]

MODEL2_ADDITIONAL = [
    "chronic_count",
    "arthritis",
    "stroke",
    "cesd10",
    "sleep_hours",
    "smoking",
    "drinking",
    "fair_poor_health",
    "mobility_count",
    "iadl_count",
]


def missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    variables = MODEL1_COVARIATES + MODEL2_ADDITIONAL + ["phc_present"]
    rows = []
    for variable in variables:
        count = int(frame[variable].isna().sum())
        rows.append(
            {
                "variable": variable,
                "missing_n": count,
                "missing_pct": 100 * count / len(frame),
            }
        )
    return pd.DataFrame(rows)
