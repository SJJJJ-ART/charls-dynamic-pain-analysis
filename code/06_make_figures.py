#!/usr/bin/env python3
"""Create manuscript Figures 2–4 from disclosure-safe aggregate files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = [
    "Stable pain-free",
    "Non-axial/no-axial",
    "Incident isolated",
    "Persistent isolated",
    "Spread to multisite",
    "Incident multisite",
    "Persistent multisite",
    "Improvement/contraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secondary-results-dir", type=Path, required=True)
    parser.add_argument("--full-sensitivity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def figure2(source: pd.DataFrame, output: Path) -> None:
    data = source.set_index("transition").loc[ORDER].reset_index()
    y = np.arange(len(data))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.2), sharey=True, gridspec_kw={"wspace": 0.08})
    colors = ["#4C78A8" if label != "Persistent multisite" else "#C44E52" for label in data["transition"]]

    risk = 100 * data["adjusted_risk"].to_numpy()
    risk_low = 100 * data["risk_ci_low"].to_numpy()
    risk_high = 100 * data["risk_ci_high"].to_numpy()
    axes[0].errorbar(risk, y, xerr=[risk-risk_low, risk_high-risk], fmt="none", ecolor="#5A5A5A", capsize=3, lw=1.2)
    axes[0].scatter(risk, y, c=colors, s=48, zorder=3)
    axes[0].set_yticks(y, data["transition"])
    axes[0].set_xlabel("Standardized next-wave risk (%)")
    axes[0].set_xlim(0, max(38, risk_high.max()+2))
    axes[0].grid(axis="x", color="#E6E6E6", lw=0.8)

    difference = 100 * data["risk_difference"].to_numpy()
    diff_low = 100 * data["rd_ci_low"].to_numpy()
    diff_high = 100 * data["rd_ci_high"].to_numpy()
    axes[1].axvline(0, color="#777777", lw=1, ls="--")
    axes[1].errorbar(difference, y, xerr=[difference-diff_low, diff_high-difference], fmt="none", ecolor="#5A5A5A", capsize=3, lw=1.2)
    axes[1].scatter(difference, y, c=colors, s=48, zorder=3)
    axes[1].set_xlabel("Risk difference vs stable pain-free (percentage points)")
    axes[1].set_xlim(min(-8, diff_low.min()-2), max(28, diff_high.max()+2))
    axes[1].grid(axis="x", color="#E6E6E6", lw=0.8)
    fig.suptitle("Model 1 standardized BADL risks across pooled 2-, 3-, and 2-year windows", y=0.98)
    fig.text(0.02, 0.015, "Points show pooled estimates. Bars show 95% confidence intervals.", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def figure3(source: pd.DataFrame, output: Path) -> None:
    order = ["Stable pain-free", "Other transition", "High-risk pattern"]
    data = source.copy()
    data["pain_group"] = pd.Categorical(data["pain_group"], order, ordered=True)
    data = data.sort_values(["pain_group", "phc_present"])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for phc, label, color, offset in ((0, "PHC absent", "#E07B39", -0.13), (1, "PHC present", "#3B7EA1", 0.13)):
        part = data.loc[data["phc_present"].eq(phc)].set_index("pain_group").loc[order]
        risk = 100 * part["adjusted_risk"].to_numpy()
        low = 100 * part["ci_low"].to_numpy()
        high = 100 * part["ci_high"].to_numpy()
        ax.errorbar(x+offset, risk, yerr=[risk-low, high-risk], fmt="o", color=color, label=label, capsize=4, markersize=7, lw=1.5)
    ax.set_xticks(x, order)
    ax.set_ylabel("Standardized next-wave BADL risk (%)")
    ax.set_ylim(0, max(42, 100*data["ci_high"].max()+3))
    ax.grid(axis="y", color="#E6E6E6", lw=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Exploratory baseline community primary-care interaction")
    fig.text(0.02, 0.015, "The high-risk pattern combines spread and persistent axial multisite pain.", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def figure4(core: pd.DataFrame, full: pd.DataFrame, output: Path) -> pd.DataFrame:
    combined = pd.concat([core, full], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("analysis", keep="last")
    order = [
        "Primary IPCW × baseline weight",
        "Unweighted",
        "Baseline weight only",
        "Exclude proxy interviews",
        "Exclude 2018–2020 window",
        "Severe BADL",
        "Participant-clustered SE",
        "IPCW truncated at 1st/99th percentiles",
        "IPCW truncated at 0.5th/99.5th percentiles",
        "BADL or death",
        "Pain threshold ≥ Some/Somewhat",
        "Next-wave IADL limitation",
        "Window 1",
        "Window 2",
        "Window 3",
        "Community bootstrap (500 replicates)",
        "Complete cases for all Model 1 and Model 2 covariates",
    ]
    available = [label for label in order if label in set(combined["analysis"])]
    data = combined.set_index("analysis").loc[available].reset_index()
    display = {
        "Primary IPCW × baseline weight": "Primary IPCW + baseline weight",
        "IPCW truncated at 1st/99th percentiles": "IPCW truncated (1st/99th)",
        "IPCW truncated at 0.5th/99.5th percentiles": "IPCW truncated (0.5th/99.5th)",
        "Pain threshold ≥ Some/Somewhat": "Pain threshold ≥ Some",
        "Next-wave IADL limitation": "Incident IADL limitation",
        "Community bootstrap (500 replicates)": "Community bootstrap (500)",
        "Complete cases for all Model 1 and Model 2 covariates": "Complete cases",
    }
    data["display_label"] = data["analysis"].replace(display)
    y = np.arange(len(data))[::-1]
    rr = data["rr"].to_numpy(float)
    low = data["ci_low"].to_numpy(float)
    high = data["ci_high"].to_numpy(float)
    colors = ["#C44E52" if label == order[0] else "#4C78A8" for label in data["analysis"]]
    fig, ax = plt.subplots(figsize=(8.4, 9.6))
    ax.axvline(1, color="#777777", lw=1, ls="--")
    ax.errorbar(rr, y, xerr=[rr-low, high-rr], fmt="none", ecolor="#606060", capsize=3, lw=1.2)
    ax.scatter(rr, y, c=colors, s=45, zorder=3)
    ax.set_yticks(y, data["display_label"])
    ax.set_xlabel("Risk ratio for persistent axial multisite pain vs stable pain-free")
    ax.set_xlim(0.5, max(4.75, high.max()+0.35))
    ax.grid(axis="x", color="#E6E6E6", lw=0.8)
    ax.set_title("Sensitivity analyses")
    label_x = ax.get_xlim()[1] - 0.06
    for yi, estimate in zip(y, rr):
        ax.text(label_x, yi, f"{estimate:.2f}", va="center", ha="right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return data


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    standardized = pd.read_csv(args.secondary_results_dir / "standardized_risks_and_differences.csv")
    phc = pd.read_csv(args.secondary_results_dir / "phc_standardized_risks.csv")
    core = pd.read_csv(args.secondary_results_dir / "core_sensitivity_analyses.csv")
    full = pd.read_csv(args.full_sensitivity_dir / "full_sensitivity_analyses.csv")
    standardized.to_csv(args.output_dir / "figure2_source.csv", index=False)
    phc.to_csv(args.output_dir / "figure3_source.csv", index=False)
    figure2(standardized, args.output_dir / "Figure2_standardized_BADL_risk.png")
    figure3(phc, args.output_dir / "Figure3_PHC_interaction.png")
    figure4_data = figure4(core, full, args.output_dir / "Figure4_sensitivity_forest.png")
    figure4_data.to_csv(args.output_dir / "figure4_source.csv", index=False)


if __name__ == "__main__":
    main()
