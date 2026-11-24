"""Apply the frozen acceptance criteria for the guard-140 safety screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_PATIENTS = [f"adolescent#{index:03d}" for index in range(1, 11)]
EXPECTED_MODES = ["ppo_shield", "switch_nogate_shield"]
EXPECTED_REPEATS = 3
METRICS = [
    "cgm_tir",
    "cgm_tbr70",
    "cgm_tbr54",
    "cgm_tar180",
    "cgm_tar250",
    "cgm_mean_bg",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)


def _validate_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient",
        "mode",
        "repeat",
        "seed",
        "terminated_early",
        *METRICS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"summary.csv is missing columns: {missing}")

    wanted = frame[frame["mode"].isin(EXPECTED_MODES)].copy()
    expected_rows = len(EXPECTED_PATIENTS) * len(EXPECTED_MODES) * EXPECTED_REPEATS
    if len(wanted) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(wanted)}")
    if sorted(wanted["patient"].unique()) != EXPECTED_PATIENTS:
        raise ValueError("patient set does not match the frozen ten-adolescent screen")
    if wanted.duplicated(["patient", "mode", "repeat"]).any():
        raise ValueError("duplicate patient/mode/repeat rows found")

    counts = wanted.groupby(["patient", "mode"])["repeat"].nunique()
    if not counts.eq(EXPECTED_REPEATS).all():
        raise ValueError("each patient/mode cell must contain exactly three repeats")

    seed_counts = wanted.groupby(["patient", "repeat"])["seed"].nunique()
    if not seed_counts.eq(1).all():
        raise ValueError("PPO and switch are not paired on identical seeds")

    if wanted[METRICS].isna().any().any():
        raise ValueError("non-finite screen metrics found")
    wanted["terminated_early"] = _as_bool(wanted["terminated_early"])
    return wanted


def analyze_frame(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    wanted = _validate_matrix(frame)

    patient_means = wanted.groupby(["patient", "mode"], sort=True)[METRICS].mean()
    wide = patient_means.unstack("mode")
    paired = pd.DataFrame(index=EXPECTED_PATIENTS)
    paired.index.name = "patient"
    for metric in METRICS:
        paired[f"ppo_{metric}"] = wide[(metric, "ppo_shield")]
        paired[f"switch_{metric}"] = wide[(metric, "switch_nogate_shield")]
        paired[f"delta_{metric}"] = (
            paired[f"switch_{metric}"] - paired[f"ppo_{metric}"]
        )

    termination = wanted.pivot(
        index=["patient", "repeat"], columns="mode", values="terminated_early"
    )
    switch_only = (
        termination["switch_nogate_shield"] & ~termination["ppo_shield"]
    )

    mean_tir_delta = float(paired["delta_cgm_tir"].mean())
    mean_tbr_delta = float(paired["delta_cgm_tbr70"].mean())
    max_patient_tbr_delta = float(paired["delta_cgm_tbr70"].max())
    switch_only_count = int(switch_only.sum())

    criteria = [
        {
            "name": "mean_cgm_tir_improvement",
            "observed": mean_tir_delta,
            "operator": ">=",
            "threshold": 0.75,
            "unit": "percentage_points",
            "passed": mean_tir_delta >= 0.75,
        },
        {
            "name": "mean_cgm_tbr70_increase",
            "observed": mean_tbr_delta,
            "operator": "<=",
            "threshold": 0.20,
            "unit": "percentage_points",
            "passed": mean_tbr_delta <= 0.20,
        },
        {
            "name": "switch_only_early_terminations",
            "observed": switch_only_count,
            "operator": "==",
            "threshold": 0,
            "unit": "runs",
            "passed": switch_only_count == 0,
        },
        {
            "name": "max_patient_cgm_tbr70_increase",
            "observed": max_patient_tbr_delta,
            "operator": "<=",
            "threshold": 0.50,
            "unit": "percentage_points",
            "passed": max_patient_tbr_delta <= 0.50,
        },
    ]

    report = {
        "schema": "guard140-screen-decision-v1",
        "decision": "ADVANCE" if all(item["passed"] for item in criteria) else "REJECT",
        "evidence_scope": "exploratory_24h_three_repeat_screen",
        "statistical_unit": "patient_mean_over_three_repeats",
        "n_patients": len(EXPECTED_PATIENTS),
        "n_paired_runs": len(termination),
        "switch_only_early_terminations": switch_only_count,
        "ppo_early_terminations": int(termination["ppo_shield"].sum()),
        "switch_early_terminations": int(
            termination["switch_nogate_shield"].sum()
        ),
        "criteria": criteria,
        "metric_means": {
            metric: {
                "ppo": float(paired[f"ppo_{metric}"].mean()),
                "switch": float(paired[f"switch_{metric}"].mean()),
                "paired_delta_switch_minus_ppo": float(
                    paired[f"delta_{metric}"].mean()
                ),
            }
            for metric in METRICS
        },
    }
    return report, paired.reset_index()


def analyze_directory(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    frame = pd.read_csv(run_dir / "summary.csv")
    report, paired = analyze_frame(frame)
    paired.to_csv(run_dir / "guard140_patient_paired.csv", index=False)
    (run_dir / "guard140_screen_decision.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_directory(args.input_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
