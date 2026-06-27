from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from collections import namedtuple
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = ROOT / "src" / "envs" / "simglucose_mpc"
SIM_SCRIPT = ROOT / "src" / "dt" / "sim" / "222.py"
TWINS_SCRIPT = ROOT / "src" / "dt" / "twins.py"
CRCF_BRIDGE_SCRIPT = ROOT / "src" / "dt" / "replaybg_crcf_bridge.py"
CRCF_REPLAY_SCRIPT = ROOT / "src" / "dt" / "replaybg_crcf_replay.py"

PATIENT_NAMES = [
    "adolescent#005",
    "adolescent#008",
    "adult#003",
    "adult#006",
    "child#010",
    "child#003",
]

Action = namedtuple("ctrller_action", ["basal", "bolus"])


def load_sim222_module():
    if str(SIM_ROOT) not in sys.path:
        sys.path.insert(0, str(SIM_ROOT))
    spec = importlib.util.spec_from_file_location("sim222_multimeal", SIM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SIM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim222_multimeal"] = module
    spec.loader.exec_module(module)
    return module


def simulate_patient(sim222, patient_name: str, simulation_days: int, sensor_name: str, pump_name: str) -> None:
    patient = sim222.T1DPatient.withName(patient_name)
    sensor = sim222.CGMSensor.withName(sensor_name, seed=1)
    pump = sim222.InsulinPump.withName(pump_name)
    controller = sim222.LocalBBController()

    patient_row = controller.patient_params.loc[
        controller.patient_params["Name"] == patient_name
    ].iloc[0]
    bw = float(patient_row["BW"])
    start_time = sim222.build_start_time()
    scenario = sim222.CustomScenario(
        start_time=start_time,
        scenario=sim222.build_scenario(simulation_days),
    )
    env = sim222.T1DSimEnv(patient, sensor, pump, scenario)

    step_data = env.reset()
    obs = step_data.observation
    reward = step_data.reward
    done = step_data.done
    info = step_data.info
    sample_time = float(info.get("sample_time", sensor.sample_time))
    total_steps = int(round(simulation_days * 24 * 60 / sample_time))

    raw_rows = []
    print(
        f"[{patient_name}] simulation start={start_time}, days={simulation_days}, "
        f"sample_time={sample_time}, steps={total_steps}",
        flush=True,
    )

    for _ in range(total_steps):
        if done:
            break

        action_cmd = controller.policy(obs, reward, done, **info)
        basal = float(pump.basal(action_cmd.basal))
        bolus = float(pump.bolus(action_cmd.bolus))
        applied_action = Action(basal=basal, bolus=bolus)

        step_data = env.step(applied_action)
        obs = step_data.observation
        reward = step_data.reward
        done = step_data.done
        info = step_data.info

        raw_rows.append(
            {
                "patient_name": patient_name,
                "timestamp": info["time"] - timedelta(minutes=sample_time),
                "cgm": float(getattr(obs, "CGM")),
                "bg_true": float(getattr(obs, "BG", info.get("bg", float("nan")))),
                "meal": float(info.get("meal", 0.0)),
                "basal": basal,
                "bolus": bolus,
                "iob": float(info.get("iob", 0.0)),
                "cob": float(info.get("cob", 0.0)),
                "bw": bw,
            }
        )

    sim222.export_results(
        raw_rows=raw_rows,
        patient_name=patient_name,
        sample_time_min=sample_time,
    )
    print(f"[{patient_name}] simulation finished", flush=True)


def run_command(command: list[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click 6-patient 1-day multi-meal simulation + ReplayBG fitting pipeline."
    )
    parser.add_argument("--patients", default=",".join(PATIENT_NAMES), help="'default' or comma-separated patient names.")
    parser.add_argument("--simulation-days", type=int, default=1)
    parser.add_argument("--sensor", default="GuardianRT")
    parser.add_argument("--pump", default="Insulet")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-crcf-bridge", action="store_true")
    parser.add_argument("--skip-fitting", action="store_true")
    parser.add_argument("--run-crcf-replay", action="store_true")
    parser.add_argument("--skip-missing-crcf-replay", action="store_true")
    return parser.parse_args()


def parse_patients(text: str) -> list[str]:
    if text.strip().lower() == "default":
        return list(PATIENT_NAMES)
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    patients = parse_patients(args.patients)

    print("Six-patient multi-meal pipeline")
    print(f"Patients: {', '.join(patients)}")
    print("Outputs:")
    print(f"- simulation CSVs: {ROOT / 'outputs' / 'replaybg'}")
    print(f"- ReplayBG results: {ROOT / 'outputs' / 'py_replay_bg' / 'results'}")
    print(f"- CR/CF bridge CSVs: {ROOT / 'outputs' / 'replaybg_crcf'}")

    if not args.skip_simulation:
        sim222 = load_sim222_module()
        for patient_name in patients:
            simulate_patient(
                sim222=sim222,
                patient_name=patient_name,
                simulation_days=args.simulation_days,
                sensor_name=args.sensor,
                pump_name=args.pump,
            )

    if not args.skip_crcf_bridge:
        run_command(
            [
                sys.executable,
                str(CRCF_BRIDGE_SCRIPT),
                "--input-dir",
                str(ROOT / "outputs" / "replaybg"),
                "--output-dir",
                str(ROOT / "outputs" / "replaybg_crcf"),
                "--pattern",
                "*_day_1.csv",
            ],
            cwd=ROOT,
        )

    if not args.skip_fitting:
        run_command([sys.executable, str(TWINS_SCRIPT)], cwd=ROOT)

    if args.run_crcf_replay:
        command = [
            sys.executable,
            str(CRCF_REPLAY_SCRIPT),
            "--therapy-dir",
            str(ROOT / "outputs" / "replaybg_crcf"),
            "--patient-info-dir",
            str(ROOT / "outputs" / "replaybg"),
        ]
        if args.skip_missing_crcf_replay:
            command.append("--skip-missing")
        run_command(command, cwd=ROOT)

    print("\nPipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
