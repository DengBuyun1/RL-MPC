"""Run the archived CAC simulation settings against the bundled checkpoint."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
FORMAL_CONFIG = (
    ROOT / "results/finall/formal_iob_prediction_adolescent001_010_72h_r5/config.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run both modes for one hour on one virtual patient.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command as a JSON argument list.")
    parser.add_argument("--output-dir", type=Path, help="Write results to a new directory.")
    parser.add_argument("--smoke-lock", type=Path, help="Use a separately pinned environment for a smoke run.")
    args = parser.parse_args()
    if args.smoke_lock and not args.smoke:
        parser.error("--smoke-lock requires --smoke")

    config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    run_name = "smoke" if args.smoke else "formal"
    output_dir = args.output_dir or ROOT / "reproductions" / run_name
    output_dir = output_dir.resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")

    command = [
        sys.executable, "-B", "-m", "scripts.evaluate_ppo_mpc",
        "--checkpoint", config["checkpoint"],
        "--patients", config["patients"].split(",")[0] if args.smoke else config["patients"],
        "--modes", config["modes"],
        "--repeats", str(1 if args.smoke else config["repeats"]),
        "--seed", str(config["seed"]),
        "--hours", str(1 if args.smoke else config["hours"]),
        "--device", config["device"],
        "--output-dir", str(output_dir),
        "--requirements-lock", str(args.smoke_lock.resolve()) if args.smoke_lock else config["requirements_lock"],
        "--expected-source-tree-sha256", config["evaluation_source_tree_sha256"],
        "--expected-checkpoint-sha256", config["checkpoint_sha256"],
        "--expected-arx-param-sha256", config["arx_param_sha256"],
        "--enable-switch-high-risk-arbitration",
        "--switch-high-risk-glucose", str(config["switch_high_risk_glucose"]),
        "--switch-high-risk-projected-glucose", str(config["switch_high_risk_projected_glucose"]),
        "--switch-high-risk-rising-rate", str(config["switch_high_risk_rising_rate"]),
        "--enable-switch-high-risk-iob-prediction",
        "--switch-high-risk-iob-prediction-horizon-min", str(config["switch_high_risk_iob_prediction_horizon_min"]),
        "--switch-high-risk-iob-prediction-effect-fraction", str(config["switch_high_risk_iob_prediction_effect_fraction"]),
        "--enable-switch-high-risk-correction",
        "--switch-high-risk-correction-target-glucose", str(config["switch_high_risk_correction_target_glucose"]),
        "--switch-high-risk-correction-horizon-min", str(config["switch_high_risk_correction_horizon_min"]),
        "--switch-high-risk-correction-max-basal-multiplier", str(config["switch_high_risk_correction_max_basal_multiplier"]),
    ]
    if not args.smoke_lock:
        command.extend(["--expected-requirements-lock-sha256", config["requirements_lock_sha256"]])

    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
