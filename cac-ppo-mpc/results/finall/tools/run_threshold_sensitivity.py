"""Run the frozen reviewer-2 threshold sensitivity matrix on the server."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from analyze_threshold_sensitivity import analyze_directory


PROJECT = Path(__file__).resolve().parents[3]
RUN_ROOT = PROJECT / "results" / "finall" / "reviewer2_threshold_sensitivity_72h_r3"
PATIENTS = [f"adolescent#{index:03d}" for index in range(1, 11)]
SETTINGS = {
    "current": {},
    "low_threshold_120": {"risk_current": 120, "risk_projected": 120},
    "low_threshold_140": {"risk_current": 140, "risk_projected": 140},
    "fast_fall_m1p5": {"fast_fall": -1.5},
    "fast_fall_m2p5": {"fast_fall": -2.5},
    "high_threshold_150_170": {"high_current": 150, "high_projected": 170},
    "high_threshold_170_190": {"high_current": 170, "high_projected": 190},
    "iob_guard_120": {"iob_guard": 120},
    "iob_guard_140": {"iob_guard": 140},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def common_command(mode: str, output_dir: Path) -> list[str]:
    return [
        sys.executable, "scripts/evaluate_ppo_mpc.py",
        "--checkpoint", "runs/ppo_adult001_005_unann_tanh_v2_2m/model_final.pt",
        "--patients", ",".join(PATIENTS),
        "--modes", mode,
        "--scenario", "A",
        "--meal-announcement", "unannounced",
        "--hours", "72",
        "--repeats", "3",
        "--seed", "110400",
        "--start-time", "2018-01-01 00:00",
        "--device", "cpu",
        "--arx-param-file", "output/arx_rls_training_30/population_theta_adult001_005.json",
        "--requirements-lock", "requirements-lock-server-py310-cu126.txt",
        "--output-dir", str(output_dir.relative_to(PROJECT)),
    ]


def switch_command(name: str) -> list[str]:
    values = {
        "risk_current": 130,
        "risk_projected": 130,
        "fast_fall": -2.0,
        "high_current": 160,
        "high_projected": 180,
        "iob_guard": 130,
    }
    values.update(SETTINGS[name])
    command = common_command("switch_nogate_shield", RUN_ROOT / name)
    command.extend([
        "--switch-risk-current-glucose", str(values["risk_current"]),
        "--switch-risk-projected-glucose", str(values["risk_projected"]),
        "--switch-risk-fast-fall-rate", str(values["fast_fall"]),
        "--enable-switch-high-risk-arbitration",
        "--switch-high-risk-glucose", str(values["high_current"]),
        "--switch-high-risk-projected-glucose", str(values["high_projected"]),
        "--switch-high-risk-rising-rate", "0",
        "--switch-high-risk-strategy", "higher_insulin",
        "--enable-switch-high-risk-iob-prediction",
        "--switch-high-risk-iob-prediction-horizon-min", "30",
        "--switch-high-risk-iob-prediction-effect-fraction", "0.75",
        "--switch-high-risk-iob-prediction-guard-glucose", str(values["iob_guard"]),
        "--enable-switch-high-risk-correction",
        "--switch-high-risk-correction-target-glucose", "120",
        "--switch-high-risk-correction-horizon-min", "90",
        "--switch-high-risk-correction-max-basal-multiplier", "5",
    ])
    return command


def _run(name: str, argv: list[str], environment: dict[str, str]) -> tuple[str, int]:
    output_dir = RUN_ROOT / name
    summary = output_dir / "summary.csv"
    if summary.exists():
        return name, 0
    output_dir.mkdir(parents=True, exist_ok=True)
    leftovers = [path for path in output_dir.iterdir() if path.name != "runner.log"]
    if leftovers:
        raise RuntimeError(f"{name}: incomplete non-empty directory; preserve it and move it before retry")
    with (output_dir / "runner.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            argv, cwd=PROJECT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
    return name, int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 4:
        raise ValueError("--max-workers must be between 1 and 4")
    arx = PROJECT / "mpc_assets" / "output" / "arx_rls_training_30" / "population_theta_adult001_005.json"
    if not arx.exists():
        raise FileNotFoundError(f"missing ARX parameter file: {arx}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    commands = {"ppo_reference": common_command("ppo_shield", RUN_ROOT / "ppo_reference")}
    commands.update({name: switch_command(name) for name in SETTINGS})
    plan = {
        "schema": "reviewer2-threshold-sensitivity-plan-v1",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "purpose": "one-factor-at-a-time threshold sensitivity requested by reviewer 2",
        "patients": PATIENTS,
        "hours": 72,
        "repeats": 3,
        "seed": 110400,
        "statistical_unit": "patient_mean_over_three_repeats",
        "reference": "ppo_reference",
        "settings": SETTINGS,
        "fixed_parameters": {
            "risk_current": 130, "risk_projected": 130, "fast_fall": -2.0,
            "high_current": 160, "high_projected": 180, "iob_guard": 130,
            "iob_prediction_horizon_min": 30, "iob_effect_fraction": 0.75,
            "correction_target": 120, "correction_horizon_min": 90,
            "correction_max_basal_multiplier": 5,
        },
        "commands": commands,
        "source_hashes": {
            "controllers.py": sha256(PROJECT / "ppo_mpc" / "controllers.py"),
            "evaluate_ppo_mpc.py": sha256(PROJECT / "scripts" / "evaluate_ppo_mpc.py"),
            "analysis.py": sha256(Path(__file__).with_name("analyze_threshold_sensitivity.py")),
        },
        "claim_boundary": "exploratory sensitivity; repeated seeds are not independent patients",
    }
    plan_path = RUN_ROOT / "threshold_sensitivity_plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in old.items() if k != "frozen_at"}
        current = {k: v for k, v in plan.items() if k != "frozen_at"}
        if comparable != current:
            raise RuntimeError("existing frozen plan differs from current code; refusing plan drift")
    else:
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    failures = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_run, name, argv, environment): name for name, argv in commands.items()}
        for future in as_completed(futures):
            name, return_code = future.result()
            print(f"{name}: return_code={return_code}", flush=True)
            if return_code != 0:
                failures[name] = return_code
    if failures:
        raise RuntimeError(f"threshold sensitivity failures: {failures}")
    print(json.dumps(analyze_directory(RUN_ROOT), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
