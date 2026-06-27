from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "dt" / "py_replay_bg"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from py_replay_bg.analyzer import Analyzer
from py_replay_bg.py_replay_bg import ReplayBG


DEFAULT_THERAPY_DIR = PROJECT_ROOT / "outputs" / "replaybg_crcf"
DEFAULT_PATIENT_INFO_DIR = PROJECT_ROOT / "outputs" / "replaybg"
DEFAULT_SAVE_ROOT = PROJECT_ROOT / "outputs" / "py_replay_bg"


def patient_slug_from_therapy_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_crcf"):
        stem = stem[: -len("_crcf")]
    if "_day_" in stem:
        return stem.split("_day_")[0]
    return stem


def save_name_for(slug: str, day_index: int, save_name_suffix: str) -> str:
    return f"{slug}_day_{day_index}_{save_name_suffix}"


def load_inputs(therapy_path: Path, patient_info_dir: Path) -> tuple[pd.DataFrame, float, float]:
    slug = patient_slug_from_therapy_path(therapy_path)
    patient_info_path = patient_info_dir / f"patient_info_{slug}.csv"
    if not patient_info_path.exists():
        raise FileNotFoundError(f"Missing patient info file: {patient_info_path}")

    data = pd.read_csv(therapy_path)
    data["t"] = pd.to_datetime(data["t"], errors="raise")
    patient_info = pd.read_csv(patient_info_path)
    bw = float(patient_info.loc[0, "bw"])
    u2ss = float(patient_info.loc[0, "u2ss"])
    return data, bw, u2ss


def build_rbg(save_root: Path, n_replay_seed: int) -> ReplayBG:
    return ReplayBG(
        blueprint="multi-meal",
        save_folder=str(save_root),
        yts=5,
        exercise=False,
        seed=n_replay_seed,
        verbose=False,
        plot_mode=False,
    )


def run_patient(
    therapy_path: Path,
    patient_info_dir: Path,
    save_root: Path,
    day_index: int,
    save_name_suffix: str,
    n_replay: int,
    seed: int,
) -> dict:
    slug = patient_slug_from_therapy_path(therapy_path)
    save_name = save_name_for(slug, day_index, save_name_suffix)
    twin_path = save_root / "results" / "mcmc" / f"mcmc_{save_name}.pkl"
    if not twin_path.exists():
        raise FileNotFoundError(
            f"Missing multi-meal twin result: {twin_path}. "
            "Run twins.py for the 1-day multi-meal data first."
        )

    data, bw, u2ss = load_inputs(therapy_path, patient_info_dir)
    rbg = build_rbg(save_root=save_root, n_replay_seed=seed)
    replay_results = rbg.replay(
        data=data,
        bw=bw,
        save_name=save_name,
        twinning_method="mcmc",
        bolus_source="data",
        basal_source="data",
        cho_source="data",
        n_replay=n_replay,
        save_workspace=True,
        save_suffix="_crcf",
    )
    analysis = Analyzer.analyze_replay_results(replay_results, data=data)
    workspace_path = save_root / "results" / "workspaces" / f"{save_name}_crcf.pkl"
    return {
        "patient_slug": slug,
        "therapy_path": str(therapy_path),
        "twin_path": str(twin_path),
        "workspace_path": str(workspace_path),
        "mard": float(analysis["median"]["twin"]["mard"]),
        "mean_glucose": float(analysis["median"]["glucose"]["variability"]["mean_glucose"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReplayBG multi-meal replay using CR/CF alternative therapy CSVs.")
    parser.add_argument("--therapy-dir", type=Path, default=DEFAULT_THERAPY_DIR)
    parser.add_argument("--patient-info-dir", type=Path, default=DEFAULT_PATIENT_INFO_DIR)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--pattern", default="*_day_1_crcf.csv")
    parser.add_argument("--day-index", type=int, default=1)
    parser.add_argument("--save-name-suffix", default="mcmc_auto")
    parser.add_argument("--n-replay", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    therapy_paths = sorted(args.therapy_dir.glob(args.pattern))
    if not therapy_paths:
        raise FileNotFoundError(f"No therapy files matched {args.therapy_dir / args.pattern}")

    rows = []
    failures = []
    for therapy_path in therapy_paths:
        print(f"Replaying {therapy_path.name}", flush=True)
        try:
            rows.append(
                run_patient(
                    therapy_path=therapy_path,
                    patient_info_dir=args.patient_info_dir,
                    save_root=args.save_root,
                    day_index=args.day_index,
                    save_name_suffix=args.save_name_suffix,
                    n_replay=args.n_replay,
                    seed=args.seed,
                )
            )
        except Exception as exc:
            failures.append({"therapy_path": str(therapy_path), "error": str(exc)})
            print(f"Failed {therapy_path.name}: {exc}", flush=True)
            if not args.skip_missing:
                raise

    output_path = args.therapy_dir / "crcf_replaybg_replay_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.therapy_dir / "crcf_replaybg_replay_failures.csv", index=False)
    print(f"Saved replay summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
