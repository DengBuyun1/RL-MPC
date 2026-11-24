"""Empirical closed-loop safety and regularity analysis for formal trajectories.

This script does not claim Lyapunov or formal stability.  It quantifies boundedness,
termination, excursion recovery, safety intervention, and control-input regularity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SAMPLE_MINUTES = 5.0
MODES = ("ppo_shield", "switch_nogate_shield")


def episode_durations(mask: np.ndarray, sample_minutes: float = SAMPLE_MINUTES) -> list[float]:
    mask = np.asarray(mask, dtype=bool)
    durations: list[float] = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            durations.append((index - start) * sample_minutes)
            start = None
    if start is not None:
        durations.append((len(mask) - start) * sample_minutes)
    return durations


def recovery_durations(
    values: np.ndarray,
    *,
    trigger: Callable[[float], bool],
    recovered: Callable[[float], bool],
    sample_minutes: float = SAMPLE_MINUTES,
) -> tuple[list[float], int]:
    values = np.asarray(values, dtype=float)
    durations: list[float] = []
    censored = 0
    start = None
    for index, value in enumerate(values):
        if start is None and trigger(value):
            start = index
        elif start is not None and recovered(value):
            durations.append((index - start) * sample_minutes)
            start = None
    if start is not None:
        censored = 1
    return durations, censored


def action_regularity(values: np.ndarray, sample_minutes: float = SAMPLE_MINUTES) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {
            "basal_total_variation_per_hour": 0.0,
            "basal_step_change_p95": 0.0,
            "basal_direction_reversals": 0,
            "basal_direction_reversals_per_hour": 0.0,
        }
    changes = np.diff(values)
    hours = len(changes) * sample_minutes / 60.0
    nonzero = np.sign(changes[np.abs(changes) > 1e-12])
    reversals = int(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
    return {
        "basal_total_variation_per_hour": float(np.abs(changes).sum() / hours),
        "basal_step_change_p95": float(np.quantile(np.abs(changes), 0.95)),
        "basal_direction_reversals": reversals,
        "basal_direction_reversals_per_hour": float(reversals / hours),
    }


def _boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)


def _stem(patient: str, mode: str, seed: int) -> str:
    return f"{patient.replace('#', '-')}_{mode}_seed{int(seed)}"


def _run_metrics(run_dir: Path, row: pd.Series) -> dict:
    stem = _stem(str(row["patient"]), str(row["mode"]), int(row["seed"]))
    rollout = pd.read_csv(run_dir / f"{stem}_rollout.csv")
    history = pd.read_csv(run_dir / f"{stem}_policy_history.csv")
    cgm = rollout["CGM"].to_numpy(float)
    basal = history["selected_basal"].to_numpy(float)
    low_durations = episode_durations(cgm < 70.0)
    severe_low_durations = episode_durations(cgm < 54.0)
    high_recovery, high_censored = recovery_durations(
        cgm, trigger=lambda x: x > 250.0, recovered=lambda x: x <= 180.0
    )
    low_recovery, low_censored = recovery_durations(
        cgm, trigger=lambda x: x < 70.0, recovered=lambda x: x >= 70.0
    )
    selected = history["selected"].astype(str).to_numpy()
    hours = max((len(history) - 1) * SAMPLE_MINUTES / 60.0, SAMPLE_MINUTES / 60.0)
    finite = np.isfinite(cgm)
    result = {
        "patient": row["patient"],
        "mode": row["mode"],
        "repeat": int(row["repeat"]),
        "seed": int(row["seed"]),
        "completed_steps": int(row["completed_steps"]),
        "expected_steps": int(row["expected_steps"]),
        "terminated_early": bool(row["terminated_early"]),
        "termination_reason": row.get("termination_reason", ""),
        "finite_cgm_fraction": float(finite.mean()),
        "cgm_min": float(np.nanmin(cgm)),
        "cgm_max": float(np.nanmax(cgm)),
        "cgm_tir": float(row["cgm_tir"]),
        "cgm_tbr70": float(row["cgm_tbr70"]),
        "cgm_tbr54": float(row["cgm_tbr54"]),
        "cgm_tar180": float(row["cgm_tar180"]),
        "cgm_tar250": float(row["cgm_tar250"]),
        "low_episode_count": len(low_durations),
        "low_episode_total_minutes": float(sum(low_durations)),
        "low_episode_max_minutes": float(max(low_durations, default=0.0)),
        "severe_low_episode_count": len(severe_low_durations),
        "severe_low_episode_total_minutes": float(sum(severe_low_durations)),
        "high_recovery_event_count": len(high_recovery),
        "high_recovery_median_minutes": float(np.median(high_recovery)) if high_recovery else np.nan,
        "high_recovery_censored": int(high_censored),
        "low_recovery_event_count": len(low_recovery),
        "low_recovery_median_minutes": float(np.median(low_recovery)) if low_recovery else np.nan,
        "low_recovery_censored": int(low_censored),
        "safety_override_rate": float(_boolean(history["safety_override"]).mean()),
        "controller_switches": int(np.sum(selected[1:] != selected[:-1])),
        "controller_switches_per_hour": float(np.sum(selected[1:] != selected[:-1]) / hours),
        "total_insulin_u": float(row["total_insulin_u"]),
    }
    result.update(action_regularity(basal))
    return result


def _paired_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, zero_method="wilcox", method="auto").pvalue)


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(20260919)
    sampled = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(sampled, [0.025, 0.975]))


def _paired_statistics(run_level: pd.DataFrame, complete_pairs_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = run_level.copy()
    if complete_pairs_only:
        valid = ~data.groupby(["patient", "repeat"])["terminated_early"].transform("any")
        data = data.loc[valid].copy()
    numeric = [
        "cgm_tir", "cgm_tbr70", "cgm_tbr54", "cgm_tar180", "cgm_tar250",
        "cgm_min", "cgm_max", "low_episode_count", "low_episode_total_minutes",
        "low_episode_max_minutes", "severe_low_episode_count",
        "severe_low_episode_total_minutes", "safety_override_rate",
        "controller_switches_per_hour", "basal_total_variation_per_hour",
        "basal_step_change_p95", "basal_direction_reversals_per_hour",
        "total_insulin_u",
    ]
    patient = data.groupby(["patient", "mode"], sort=True)[numeric].mean().unstack("mode")
    paired = pd.DataFrame(index=patient.index)
    rows = []
    for metric in numeric:
        ppo = patient[(metric, "ppo_shield")]
        switch = patient[(metric, "switch_nogate_shield")]
        delta = switch - ppo
        paired[f"ppo_{metric}"] = ppo
        paired[f"switch_{metric}"] = switch
        paired[f"delta_{metric}"] = delta
        low, high = _bootstrap_ci(delta.to_numpy())
        rows.append({
            "metric": metric,
            "complete_pairs_only": complete_pairs_only,
            "n_patients": int(delta.notna().sum()),
            "ppo_mean": float(ppo.mean()),
            "switch_mean": float(switch.mean()),
            "mean_delta_switch_minus_ppo": float(delta.mean()),
            "median_delta": float(delta.median()),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "wilcoxon_p": _paired_pvalue(delta.to_numpy()),
        })
    return pd.DataFrame(rows), paired.reset_index()


def analyze_directory(input_dir: Path, output_dir: Path) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(input_dir / "summary.csv")
    summary["terminated_early"] = _boolean(summary["terminated_early"])
    wanted = summary.loc[summary["mode"].isin(MODES)].copy()
    if len(wanted) != 100 or wanted["patient"].nunique() != 10:
        raise ValueError("expected the formal 10-patient x 2-mode x 5-repeat matrix")
    records = [_run_metrics(input_dir, row) for _, row in wanted.iterrows()]
    run_level = pd.DataFrame(records)
    if not np.isfinite(run_level["finite_cgm_fraction"]).all():
        raise ValueError("non-finite boundedness metric")

    all_stats, all_paired = _paired_statistics(run_level, complete_pairs_only=False)
    complete_stats, complete_paired = _paired_statistics(run_level, complete_pairs_only=True)
    run_level.to_csv(output_dir / "run_level_safety_stability.csv", index=False)
    all_paired.to_csv(output_dir / "patient_level_paired_all_runs.csv", index=False)
    complete_paired.to_csv(output_dir / "patient_level_paired_complete_pairs.csv", index=False)
    pd.concat([all_stats, complete_stats], ignore_index=True).to_csv(
        output_dir / "paired_statistics.csv", index=False
    )

    term = run_level.groupby("mode")["terminated_early"].sum().to_dict()
    complete_pair_count = int((
        ~run_level.groupby(["patient", "repeat"])["terminated_early"].any()
    ).sum())
    report = {
        "schema": "reviewer2-empirical-safety-stability-v1",
        "source": str(input_dir),
        "scope": "empirical_closed_loop_safety_boundedness_and_input_regularity",
        "claim_boundary": (
            "These finite-horizon simulation diagnostics are not a Lyapunov, ISS, or formal stability proof."
        ),
        "statistical_unit": "patient_mean_over_repeats",
        "runs": len(run_level),
        "patients": int(run_level["patient"].nunique()),
        "paired_runs": int(len(run_level) / 2),
        "complete_paired_runs": complete_pair_count,
        "early_terminations": {key: int(value) for key, value in term.items()},
        "all_cgm_samples_finite": bool(run_level["finite_cgm_fraction"].eq(1.0).all()),
        "primary_safety_metrics": ["cgm_tbr70", "cgm_tbr54", "low_episode_max_minutes"],
        "regularity_metrics": [
            "controller_switches_per_hour", "basal_total_variation_per_hour",
            "basal_step_change_p95", "basal_direction_reversals_per_hour",
        ],
        "outputs": [
            "run_level_safety_stability.csv", "patient_level_paired_all_runs.csv",
            "patient_level_paired_complete_pairs.csv", "paired_statistics.csv",
        ],
    }
    (output_dir / "safety_stability_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_directory(args.input_dir, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
