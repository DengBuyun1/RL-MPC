"""Analyze the frozen reviewer-2 one-factor-at-a-time threshold matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_PATIENTS = [f"adolescent#{index:03d}" for index in range(1, 11)]
METRICS = (
    "cgm_tir", "cgm_tbr70", "cgm_tbr54", "cgm_tar180", "cgm_tar250",
    "cgm_mean_bg", "total_insulin_u",
)


def _boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)


def _validate(
    frame: pd.DataFrame,
    mode: str,
    patients: Sequence[str],
    repeats: int,
    label: str,
) -> pd.DataFrame:
    required = {"patient", "mode", "repeat", "seed", "terminated_early", *METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    wanted = frame.loc[frame["mode"].eq(mode)].copy()
    expected = len(patients) * repeats
    if len(wanted) != expected:
        raise ValueError(f"{label}: expected {expected} rows, found {len(wanted)}")
    if sorted(wanted["patient"].unique()) != sorted(patients):
        raise ValueError(f"{label}: patient set differs from frozen design")
    if wanted.duplicated(["patient", "repeat"]).any():
        raise ValueError(f"{label}: duplicate patient/repeat rows")
    counts = wanted.groupby("patient")["repeat"].nunique()
    if not counts.eq(repeats).all():
        raise ValueError(f"{label}: each patient must contain {repeats} repeats")
    if not np.isfinite(wanted[list(METRICS)].to_numpy(float)).all():
        raise ValueError(f"{label}: non-finite metric found")
    wanted["terminated_early"] = _boolean(wanted["terminated_early"])
    return wanted


def _paired_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, zero_method="wilcox", method="auto").pvalue)


def _bootstrap_ci(values: np.ndarray, seed: int = 20260919) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def analyze_frames(
    reference: pd.DataFrame,
    settings: Mapping[str, pd.DataFrame],
    *,
    patients: Sequence[str] = DEFAULT_PATIENTS,
    repeats: int = 3,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    reference = _validate(reference, "ppo_shield", patients, repeats, "ppo_reference")
    ref_keys = reference[["patient", "repeat", "seed"]].sort_values(
        ["patient", "repeat"]
    ).reset_index(drop=True)
    ref_means = reference.groupby("patient", sort=True)[list(METRICS)].mean()

    summaries = []
    paired_frames = []
    for setting, raw in settings.items():
        switch = _validate(
            raw, "switch_nogate_shield", patients, repeats, setting
        )
        switch_keys = switch[["patient", "repeat", "seed"]].sort_values(
            ["patient", "repeat"]
        ).reset_index(drop=True)
        if not ref_keys.equals(switch_keys):
            raise ValueError(f"{setting}: reference and switch seeds are not paired")

        switch_means = switch.groupby("patient", sort=True)[list(METRICS)].mean()
        paired = pd.DataFrame({"patient": sorted(patients), "setting": setting})
        for metric in METRICS:
            paired[f"ppo_{metric}"] = ref_means.loc[sorted(patients), metric].to_numpy()
            paired[f"switch_{metric}"] = switch_means.loc[sorted(patients), metric].to_numpy()
            paired[f"delta_{metric}"] = paired[f"switch_{metric}"] - paired[f"ppo_{metric}"]
        paired_frames.append(paired)

        row = {
            "setting": setting,
            "n_patients": len(patients),
            "n_repeats": repeats,
            "ppo_early_terminations": int(reference["terminated_early"].sum()),
            "switch_early_terminations": int(switch["terminated_early"].sum()),
            "switch_only_early_terminations": int((
                switch.sort_values(["patient", "repeat"])["terminated_early"].to_numpy()
                & ~reference.sort_values(["patient", "repeat"])["terminated_early"].to_numpy()
            ).sum()),
        }
        for metric in METRICS:
            delta = paired[f"delta_{metric}"].to_numpy(float)
            low, high = _bootstrap_ci(delta)
            row[f"ppo_{metric}"] = float(paired[f"ppo_{metric}"].mean())
            row[f"switch_{metric}"] = float(paired[f"switch_{metric}"].mean())
            row[f"delta_{metric}"] = float(delta.mean())
            row[f"delta_{metric}_ci95_low"] = low
            row[f"delta_{metric}_ci95_high"] = high
            row[f"delta_{metric}_wilcoxon_p"] = _paired_pvalue(delta)
        summaries.append(row)

    setting_summary = pd.DataFrame(summaries).sort_values("setting").reset_index(drop=True)
    paired_all = pd.concat(paired_frames, ignore_index=True)
    report = {
        "schema": "reviewer2-threshold-sensitivity-v1",
        "evidence_scope": "exploratory_one_factor_at_a_time_sensitivity",
        "statistical_unit": "patient_mean_over_repeats",
        "n_patients": len(patients),
        "repeats": repeats,
        "settings": sorted(settings),
        "primary_metric": "cgm_tir",
        "safety_metrics": ["cgm_tbr70", "cgm_tbr54", "switch_only_early_terminations"],
        "interpretation_rule": (
            "Use estimates and patient-level uncertainty; do not treat repeated seeds as independent patients."
        ),
    }
    return report, setting_summary, paired_all


def analyze_directory(root: Path) -> dict:
    root = Path(root)
    plan = json.loads((root / "threshold_sensitivity_plan.json").read_text(encoding="utf-8"))
    patients = plan["patients"]
    repeats = int(plan["repeats"])
    reference = pd.read_csv(root / "ppo_reference" / "summary.csv")
    settings = {
        name: pd.read_csv(root / name / "summary.csv")
        for name in plan["settings"]
    }
    report, summary, paired = analyze_frames(
        reference, settings, patients=patients, repeats=repeats
    )
    report["design"] = plan
    summary.to_csv(root / "threshold_setting_summary.csv", index=False)
    paired.to_csv(root / "threshold_patient_paired.csv", index=False)
    (root / "threshold_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_directory(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
