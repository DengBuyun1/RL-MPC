"""Run two short safety candidates concurrently, then apply frozen gates."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from analyze_accelerated_stage1 import analyze_directory


PROJECT = Path(__file__).resolve().parents[3]
RUN_ROOT = PROJECT / "results" / "finall" / "accelerated_safety_stage1_24h_r2"
PATIENTS = "adolescent#002,adolescent#004,adolescent#009,adolescent#010"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(output_dir: Path, taper: bool) -> list[str]:
    argv = [
        sys.executable,
        "scripts/evaluate_ppo_mpc.py",
        "--checkpoint", "runs/ppo_adult001_005_unann_tanh_v2_2m/model_final.pt",
        "--patients", PATIENTS,
        "--modes", "ppo_shield,switch_nogate_shield",
        "--scenario", "A",
        "--meal-announcement", "unannounced",
        "--hours", "24",
        "--repeats", "2",
        "--seed", "107400",
        "--start-time", "2018-01-01 00:00",
        "--device", "cpu",
        "--arx-param-file", "output/arx_rls_training_30/population_theta_adult001_005.json",
        "--requirements-lock", "requirements-lock-server-py310-cu126.txt",
        "--enable-switch-high-risk-arbitration",
        "--switch-high-risk-glucose", "160",
        "--switch-high-risk-projected-glucose", "180",
        "--switch-high-risk-rising-rate", "0",
        "--switch-high-risk-strategy", "higher_insulin",
        "--enable-switch-high-risk-iob-prediction",
        "--switch-high-risk-iob-prediction-horizon-min", "30",
        "--switch-high-risk-iob-prediction-effect-fraction", "0.75",
        "--switch-high-risk-iob-prediction-guard-glucose", "130",
        "--enable-switch-high-risk-correction",
        "--switch-high-risk-correction-target-glucose", "120",
        "--switch-high-risk-correction-horizon-min", "120",
        "--switch-high-risk-correction-max-basal-multiplier", "5",
    ]
    if taper:
        argv.extend([
            "--switch-high-risk-iob-taper-start-ratio", "1.5",
            "--switch-high-risk-iob-max-ratio", "2.5",
        ])
    argv.extend(["--output-dir", str(output_dir.relative_to(PROJECT))])
    return argv


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = {
        "h120_only": command(RUN_ROOT / "h120_only", taper=False),
        "h120_taper": command(RUN_ROOT / "h120_taper", taper=True),
    }
    for name in candidates:
        summary = RUN_ROOT / name / "summary.csv"
        if summary.exists():
            raise FileExistsError(f"refusing to overwrite completed result: {summary}")
        (RUN_ROOT / name).mkdir(parents=True, exist_ok=True)

    plan = {
        "schema": "accelerated-safety-stage1-plan-v1",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "purpose": "screen correction horizon and causal IOB taper before ten-patient validation",
        "patients": PATIENTS.split(","),
        "hours": 24,
        "repeats": 2,
        "seed": 107400,
        "commands": candidates,
        "source_hashes": {
            "controllers.py": sha256(PROJECT / "ppo_mpc" / "controllers.py"),
            "evaluate_ppo_mpc.py": sha256(PROJECT / "scripts" / "evaluate_ppo_mpc.py"),
            "analysis.py": sha256(Path(__file__).with_name("analyze_accelerated_stage1.py")),
        },
        "frozen_criteria": {
            "mean_cgm_tir_delta_min_pp": 0.25,
            "mean_cgm_tbr70_delta_max_pp": 0.20,
            "mean_cgm_tbr54_delta_max_pp": 0.10,
            "max_patient_cgm_tbr70_delta_max_pp": 0.50,
            "switch_only_early_terminations": 0,
        },
    }
    (RUN_ROOT / "stage1_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    processes = {}
    logs = {}
    try:
        for name, argv in candidates.items():
            log = (RUN_ROOT / name / "runner.log").open("w", encoding="utf-8")
            logs[name] = log
            processes[name] = subprocess.Popen(
                argv,
                cwd=PROJECT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return_codes = {name: process.wait() for name, process in processes.items()}
    finally:
        for log in logs.values():
            log.close()

    if any(code != 0 for code in return_codes.values()):
        raise RuntimeError(f"candidate process failed: {return_codes}")
    report = analyze_directory(RUN_ROOT)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
