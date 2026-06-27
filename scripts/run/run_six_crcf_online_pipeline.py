from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = ROOT / "src" / "envs" / "simglucose_mpc"
SIM_SCRIPT = ROOT / "src" / "dt" / "sim" / "222_crcf_online.py"
TWINS_SCRIPT = ROOT / "src" / "dt" / "twins_crcf_online.py"

PATIENT_NAMES = [
    "adolescent#005",
    "adolescent#008",
    "adult#003",
    "adult#006",
    "child#010",
    "child#003",
]


def load_sim_module():
    if str(SIM_ROOT) not in sys.path:
        sys.path.insert(0, str(SIM_ROOT))
    spec = importlib.util.spec_from_file_location("sim222_crcf_online", SIM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SIM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim222_crcf_online"] = module
    spec.loader.exec_module(module)
    return module


def run_command(command: list[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def parse_patients(text: str) -> list[str]:
    if text.strip().lower() == "default":
        return list(PATIENT_NAMES)
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 6-patient 3-day online CR/CF simulation and ReplayBG identification."
    )
    parser.add_argument("--patients", default=",".join(PATIENT_NAMES), help="'default' or comma-separated patient names.")
    parser.add_argument("--simulation-days", type=int, default=3)
    parser.add_argument("--sensor", default="GuardianRT")
    parser.add_argument("--pump", default="Insulet")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-fitting", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patients = parse_patients(args.patients)

    print("Six-patient online CR/CF multi-day pipeline")
    print(f"Patients: {', '.join(patients)}")
    print(f"Simulation days: {args.simulation_days}")
    print("Outputs:")
    print(f"- CR/CF-online simulation CSVs: {ROOT / 'outputs' / 'replaybg_crcf_online'}")
    print(f"- raw CR/CF-online simulation CSVs: {ROOT / 'outputs' / 'raw_crcf_online'}")
    print(f"- ReplayBG results: {ROOT / 'outputs' / 'py_replay_bg' / 'results'}")

    if not args.skip_simulation:
        sim_module = load_sim_module()
        for patient_name in patients:
            sim_module.simulate_patient(
                patient_name=patient_name,
                simulation_days=args.simulation_days,
                sensor_name=args.sensor,
                pump_name=args.pump,
            )

    if not args.skip_fitting:
        run_command([sys.executable, str(TWINS_SCRIPT)], cwd=ROOT)

    print("\nPipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
