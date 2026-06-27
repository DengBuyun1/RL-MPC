from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path
import math
import pickle
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "dt" / "py_replay_bg"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from py_replay_bg.analyzer import Analyzer
from py_replay_bg.py_replay_bg import ReplayBG


SAVE_ROOT = PROJECT_ROOT / "outputs" / "py_replay_bg"
REPLAYBG_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "replaybg"
PATIENT_NAMES = [
    "adolescent#005",
    "adolescent#008",
    "adult#003",
    "adult#006",
    "child#010",
    "child#003",
]
MAX_WORKERS = 6
SAVE_NAME_SUFFIX = "mcmc_auto"
DAY_INDEX = 1

EXPLORATORY_STEPS = 600
EXPLORATORY_RUNS = 2
TAU_MULTIPLIER = 50
MIN_MAIN_STEPS = 2000
MAX_MAIN_STEPS = 50000
N_REPLAY = 100
PARALLELIZE = False


def patient_slug(patient_name: str) -> str:
    return patient_name.replace("#", "_")


def data_path_for(patient_name: str) -> Path:
    return REPLAYBG_OUTPUT_ROOT / f"{patient_slug(patient_name)}_day_{DAY_INDEX}.csv"


def patient_info_path_for(patient_name: str) -> Path:
    return REPLAYBG_OUTPUT_ROOT / f"patient_info_{patient_slug(patient_name)}.csv"


def save_name_for(patient_name: str) -> str:
    return f"{patient_slug(patient_name)}_day_{DAY_INDEX}_{SAVE_NAME_SUFFIX}"


def load_inputs(patient_name: str) -> tuple[pd.DataFrame, float, float]:
    data_path = data_path_for(patient_name)
    patient_info_path = patient_info_path_for(patient_name)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not patient_info_path.exists():
        raise FileNotFoundError(f"Missing patient info file: {patient_info_path}")

    data = pd.read_csv(data_path)
    data["t"] = pd.to_datetime(data["t"], errors="raise")

    patient_info = pd.read_csv(patient_info_path)
    bw = float(patient_info.loc[0, "bw"])
    u2ss = float(patient_info.loc[0, "u2ss"])
    return data, bw, u2ss


def build_rbg() -> ReplayBG:
    return ReplayBG(
        blueprint="multi-meal",
        save_folder=str(SAVE_ROOT),
        yts=5,
        exercise=False,
        seed=1,
        verbose=False,
        plot_mode=False,
    )


def load_twinning_results(save_name: str) -> dict:
    twin_path = SAVE_ROOT / "results" / "mcmc" / f"mcmc_{save_name}.pkl"
    with open(twin_path, "rb") as handle:
        return pickle.load(handle)


def estimate_main_steps(exploratory_results: list[dict]) -> int:
    tau_values = []
    for result in exploratory_results:
        tau = result.get("tau")
        if tau is None:
            continue
        tau_array = np.asarray(tau, dtype=float)
        tau_array = tau_array[np.isfinite(tau_array)]
        if tau_array.size:
            tau_values.append(float(np.max(tau_array)))

    if not tau_values:
        return MIN_MAIN_STEPS

    representative_tau = max(tau_values)
    estimated_steps = int(math.ceil(TAU_MULTIPLIER * representative_tau))
    estimated_steps = max(MIN_MAIN_STEPS, estimated_steps)
    estimated_steps = min(MAX_MAIN_STEPS, estimated_steps)
    return estimated_steps


def run_exploratory_twins(patient_name: str, data: pd.DataFrame, bw: float, u2ss: float) -> list[dict]:
    exploratory_results = []
    base_save_name = save_name_for(patient_name)

    for run_idx in range(1, EXPLORATORY_RUNS + 1):
        exploratory_save_name = f"{base_save_name}_explore_{run_idx}"
        print(
            f"[{patient_name}] exploratory twin {run_idx}/{EXPLORATORY_RUNS} "
            f"with {EXPLORATORY_STEPS} steps",
            flush=True,
        )
        rbg = build_rbg()
        rbg.twin(
            data=data,
            bw=bw,
            save_name=exploratory_save_name,
            twinning_method="mcmc",
            parallelize=PARALLELIZE,
            n_steps=EXPLORATORY_STEPS,
            save_chains=True,
            u2ss=u2ss,
        )
        exploratory_results.append(load_twinning_results(exploratory_save_name))

    return exploratory_results


def run_patient(patient_name: str) -> dict:
    save_name = save_name_for(patient_name)
    data_path = data_path_for(patient_name)
    patient_info_path = patient_info_path_for(patient_name)

    print(f"[{patient_name}] starting", flush=True)
    data, bw, u2ss = load_inputs(patient_name)
    exploratory_results = run_exploratory_twins(patient_name=patient_name, data=data, bw=bw, u2ss=u2ss)
    main_steps = estimate_main_steps(exploratory_results)
    print(f"[{patient_name}] estimated main MCMC steps: {main_steps}", flush=True)

    rbg = build_rbg()
    rbg.twin(
        data=data,
        bw=bw,
        save_name=save_name,
        twinning_method="mcmc",
        parallelize=PARALLELIZE,
        n_steps=main_steps,
        save_chains=True,
        u2ss=u2ss,
    )

    replay_results = rbg.replay(
        data=data,
        bw=bw,
        save_name=save_name,
        twinning_method="mcmc",
        n_replay=N_REPLAY,
        save_workspace=True,
        save_suffix="_baseline",
    )

    analysis = Analyzer.analyze_replay_results(replay_results, data=data)
    twin_path = SAVE_ROOT / "results" / "mcmc" / f"mcmc_{save_name}.pkl"
    workspace_path = SAVE_ROOT / "results" / "workspaces" / f"{save_name}_baseline.pkl"

    return {
        "patient_name": patient_name,
        "data_path": str(data_path),
        "patient_info_path": str(patient_info_path),
        "main_steps": main_steps,
        "fit_mard": float(analysis["median"]["twin"]["mard"]),
        "mean_glucose": float(analysis["median"]["glucose"]["variability"]["mean_glucose"]),
        "twin_path": str(twin_path),
        "workspace_path": str(workspace_path),
    }


def main() -> None:
    print(f"Running batch Python MCMC twin for {len(PATIENT_NAMES)} patients")
    print(f"Patients: {', '.join(PATIENT_NAMES)}")
    print(
        f"exploratory_steps={EXPLORATORY_STEPS}, exploratory_runs={EXPLORATORY_RUNS}, "
        f"tau_multiplier={TAU_MULTIPLIER}, min_main_steps={MIN_MAIN_STEPS}, "
        f"max_main_steps={MAX_MAIN_STEPS}, max_workers={MAX_WORKERS}"
    )

    missing_inputs = []
    for patient_name in PATIENT_NAMES:
        data_path = data_path_for(patient_name)
        patient_info_path = patient_info_path_for(patient_name)
        if not data_path.exists():
            missing_inputs.append(str(data_path))
        if not patient_info_path.exists():
            missing_inputs.append(str(patient_info_path))

    if missing_inputs:
        missing_message = "\n".join(missing_inputs)
        raise FileNotFoundError(f"Missing required input files:\n{missing_message}")


    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_patient = {
            executor.submit(run_patient, patient_name): patient_name for patient_name in PATIENT_NAMES
        }
        for future in as_completed(future_to_patient):
            patient_name = future_to_patient[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append((patient_name, exc, traceback.format_exc()))
                print(f"[{patient_name}] failed: {exc}", flush=True)
                continue

            results.append(result)
            print(
                f"[{patient_name}] done: Fit MARD={result['fit_mard']:.2f}%, "
                f"Mean glucose={result['mean_glucose']:.2f} mg/dL",
                flush=True,
            )

    print("\nCompleted patients:")
    for result in sorted(results, key=lambda item: item["patient_name"]):
        print(
            f"- {result['patient_name']}: steps={result['main_steps']}, "
            f"mard={result['fit_mard']:.2f}%, twin={result['twin_path']}, "
            f"workspace={result['workspace_path']}"
        )

    if failures:
        print("\nFailures:")
        for patient_name, exc, tb in failures:
            print(f"- {patient_name}: {exc}")
            print(tb)
        raise SystemExit(1)


if __name__ == "__main__":
    freeze_support()
    main()
