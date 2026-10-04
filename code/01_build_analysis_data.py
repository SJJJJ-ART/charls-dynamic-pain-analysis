#!/usr/bin/env python3
"""Build and validate the restricted person-interval analysis files."""

from __future__ import annotations

import argparse
from pathlib import Path

from charls_core import aggregate_transition_cells, build_core_cohorts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohorts = build_core_cohorts(args.raw_dir)
    cohorts.analysis.to_pickle(args.output_dir / "analysis_person_intervals.pkl")
    cohorts.alive_eligible.to_pickle(args.output_dir / "alive_eligible_person_intervals.pkl")
    cohorts.flow.to_csv(args.output_dir / "sample_flow.csv", index=False)
    aggregate_transition_cells(cohorts.analysis).to_csv(
        args.output_dir / "transition_16_cell.csv", index=False
    )
    print(cohorts.flow.to_string(index=False))
    print(f"Analysis intervals: {len(cohorts.analysis):,}")
    print(f"Events: {int(cohorts.analysis['event'].sum()):,}")
    print(f"Participants: {cohorts.analysis['ID'].nunique():,}")


if __name__ == "__main__":
    main()
