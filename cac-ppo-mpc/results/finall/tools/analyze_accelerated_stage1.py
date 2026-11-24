"""Analyze the frozen accelerated safety screen without post-hoc retuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATES = ("h120_only", "h120_taper")
PATIENTS = ["adolescent#002", "adolescent#004", "adolescent#009", "adolescent#010"]
MODES = ("ppo_shield", "switch_nogate_shield")
REPEATS = 2
METRICS = (
    "cgm_tir",
    "cgm_tbr70",
    "cgm_tbr54",
    "cgm_tar180",
    "cgm_tar250",
    "cgm_mean_bg",
    "total_insulin_u",
)


def _boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def _validate(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    required = {"patient", "mode", "repeat", "seed", "terminated_early", *METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{candidate}: missing columns {missing}")
    wanted = frame[frame["mode"].isin(MODES)].copy()
    expected = len(PATIENTS) * len(MODES) * REPEATS
    if len(wanted) != expected:
        raise ValueError(f"{candidate}: expected {expected} rows, found {len(wanted)}")
    if sorted(wanted["patient"].unique()) != PATIENTS:
        raise ValueError(f"{candidate}: patient set differs from frozen stage-1 set")
    if wanted.duplicated(["patient", "mode", "repeat"]).any():
        raise ValueError(f"{candidate}: duplicate patient/mode/repeat rows")
    counts = wanted.groupby(["patient", "mode"])["repeat"].nunique()
    if not counts.eq(REPEATS).all():
        raise ValueError(f"{candidate}: each cell must contain two repeats")
    seed_counts = wanted.groupby(["patient", "repeat"])["seed"].nunique()
    if not seed_counts.eq(1).all():
        raise ValueError(f"{candidate}: PPO and switch seeds are not paired")
    if not np.isfinite(wanted[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError(f"{candidate}: non-finite metric found")
    wanted["terminated_early"] = _boolean(wanted["terminated_early"])
    return wanted


def _candidate_report(frame: pd.DataFrame, candidate: str) -> tuple[dict, pd.DataFrame]:
    wanted = _validate(frame, candidate)
    means = wanted.groupby(["patient", "mode"], sort=True)[list(METRICS)].mean()
    wide = means.unstack("mode")
    paired = pd.DataFrame(index=PATIENTS)
    paired.index.name = "patient"
    for metric in METRICS:
        paired[f"ppo_{metric}"] = wide[(metric, "ppo_shield")]
        paired[f"switch_{metric}"] = wide[(metric, "switch_nogate_shield")]
        paired[f"delta_{metric}"] = paired[f"switch_{metric}"] - paired[f"ppo_{metric}"]

    termination = wanted.pivot(
        index=["patient", "repeat"], columns="mode", values="terminated_early"
    )
    switch_only = int(
        (termination["switch_nogate_shield"] & ~termination["ppo_shield"]).sum()
    )
    observed = {
        "mean_cgm_tir_delta": float(paired["delta_cgm_tir"].mean()),
        "mean_cgm_tbr70_delta": float(paired["delta_cgm_tbr70"].mean()),
        "mean_cgm_tbr54_delta": float(paired["delta_cgm_tbr54"].mean()),
        "max_patient_cgm_tbr70_delta": float(paired["delta_cgm_tbr70"].max()),
        "switch_only_early_terminations": switch_only,
    }
    criteria = [
        {"name": "mean_cgm_tir_delta", "operator": ">=", "threshold": 0.25,
         "observed": observed["mean_cgm_tir_delta"],
         "passed": observed["mean_cgm_tir_delta"] >= 0.25},
        {"name": "mean_cgm_tbr70_delta", "operator": "<=", "threshold": 0.20,
         "observed": observed["mean_cgm_tbr70_delta"],
         "passed": observed["mean_cgm_tbr70_delta"] <= 0.20},
        {"name": "mean_cgm_tbr54_delta", "operator": "<=", "threshold": 0.10,
         "observed": observed["mean_cgm_tbr54_delta"],
         "passed": observed["mean_cgm_tbr54_delta"] <= 0.10},
        {"name": "max_patient_cgm_tbr70_delta", "operator": "<=", "threshold": 0.50,
         "observed": observed["max_patient_cgm_tbr70_delta"],
         "passed": observed["max_patient_cgm_tbr70_delta"] <= 0.50},
        {"name": "switch_only_early_terminations", "operator": "==", "threshold": 0,
         "observed": switch_only, "passed": switch_only == 0},
    ]
    report = {
        "candidate": candidate,
        "passed": all(item["passed"] for item in criteria),
        "criteria": criteria,
        "observed": observed,
        "ppo_early_terminations": int(termination["ppo_shield"].sum()),
        "switch_early_terminations": int(termination["switch_nogate_shield"].sum()),
    }
    paired.insert(0, "candidate", candidate)
    return report, paired.reset_index()


def analyze_frames(frames: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    if set(frames) != set(CANDIDATES):
        raise ValueError(f"expected candidates {CANDIDATES}")
    reports = {}
    paired_frames = []
    validated = {}
    for candidate in CANDIDATES:
        validated[candidate] = _validate(frames[candidate], candidate)
        reports[candidate], paired = _candidate_report(frames[candidate], candidate)
        paired_frames.append(paired)

    key = ["patient", "repeat", "seed"]
    base_columns = key + [*METRICS, "terminated_early"]
    ppo_a = validated[CANDIDATES[0]].query("mode == 'ppo_shield'")[base_columns].sort_values(key)
    ppo_b = validated[CANDIDATES[1]].query("mode == 'ppo_shield'")[base_columns].sort_values(key)
    keys_equal = ppo_a[key].reset_index(drop=True).equals(ppo_b[key].reset_index(drop=True))
    metrics_equal = np.allclose(
        ppo_a[list(METRICS)].to_numpy(float),
        ppo_b[list(METRICS)].to_numpy(float),
        rtol=0.0,
        atol=1e-9,
    )
    term_equal = np.array_equal(
        _boolean(ppo_a["terminated_early"]).to_numpy(),
        _boolean(ppo_b["terminated_early"]).to_numpy(),
    )
    reproducible = bool(keys_equal and metrics_equal and term_equal)

    passed = [name for name in CANDIDATES if reports[name]["passed"]]
    if not reproducible or not passed:
        decision = "REJECT_ALL"
    else:
        selected = min(
            passed,
            key=lambda name: (
                reports[name]["observed"]["max_patient_cgm_tbr70_delta"],
                reports[name]["observed"]["mean_cgm_tbr70_delta"],
                -reports[name]["observed"]["mean_cgm_tir_delta"],
            ),
        )
        decision = f"PROMOTE_{selected.upper()}"

    report = {
        "schema": "accelerated-safety-stage1-v1",
        "decision": decision,
        "evidence_scope": "exploratory_challenge_patient_screen",
        "statistical_unit": "patient_mean_over_two_repeats",
        "selection_rule": (
            "pass all frozen criteria; then minimize maximum patient TBR70 increase, "
            "then mean TBR70 increase, then maximize mean TIR improvement"
        ),
        "ppo_reproducible_across_candidates": reproducible,
        "candidates": reports,
    }
    return report, pd.concat(paired_frames, ignore_index=True)


def analyze_directory(root: Path) -> dict:
    root = Path(root)
    frames = {name: pd.read_csv(root / name / "summary.csv") for name in CANDIDATES}
    report, paired = analyze_frames(frames)
    paired.to_csv(root / "stage1_patient_paired.csv", index=False)
    (root / "stage1_decision.json").write_text(
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
