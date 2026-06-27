from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
CRCF_SEARCH_PATHS = [
    PROJECT_ROOT / "src" / "dt" / "sim",
    PROJECT_ROOT / "src" / "dt" / "111",
    ROOT,
]
for crcf_path in CRCF_SEARCH_PATHS:
    if crcf_path.exists() and str(crcf_path) not in sys.path:
        sys.path.insert(0, str(crcf_path))

from cr_cf_estimator import AdaptiveCRCFConfig, AdaptiveCRCFEstimator, normalize_meal_label


DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "replaybg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "replaybg_crcf"


def infer_sample_minutes(data: pd.DataFrame) -> float:
    times = pd.to_datetime(data["t"])
    diffs = times.diff().dropna().dt.total_seconds() / 60.0
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.empty:
        return 5.0
    return float(diffs.median())


def patient_slug_from_data_path(path: Path) -> str:
    stem = path.stem
    if "_day_" not in stem:
        return stem
    return stem.split("_day_")[0]


def patient_info_path(input_dir: Path, slug: str) -> Path:
    return input_dir / f"patient_info_{slug}.csv"


def estimate_tdd_from_replaybg_data(data: pd.DataFrame, sample_minutes: float) -> float:
    basal = pd.to_numeric(data.get("basal", 0.0), errors="coerce").fillna(0.0)
    bolus_rate = pd.to_numeric(data.get("bolus", 0.0), errors="coerce").fillna(0.0)
    total = float(((basal + bolus_rate) * sample_minutes).sum())
    if total <= 0:
        raise ValueError("Cannot estimate TDD: total basal+bolus insulin is not positive.")
    return total


def prepare_estimation_dataframe(data: pd.DataFrame, sample_minutes: float) -> pd.DataFrame:
    prepared = pd.DataFrame()
    prepared["t"] = pd.to_datetime(data["t"], errors="raise")
    prepared["glucose"] = pd.to_numeric(data["glucose"], errors="coerce")
    # The CSVs generated from local simglucose store CHO as a rate over the
    # sampling interval. Convert it back to the meal amount delivered at that
    # sample for bolus-calculator calculations.
    prepared["cho"] = pd.to_numeric(data["cho"], errors="coerce").fillna(0.0) * sample_minutes
    prepared["bolus"] = pd.to_numeric(data["bolus"], errors="coerce").fillna(0.0) * sample_minutes
    prepared["cho_label"] = data.get("cho_label", "").fillna("").astype(str)
    return prepared


def fit_crcf(
    data: pd.DataFrame,
    sample_minutes: float,
    target_glucose: float,
    cr_method: str,
) -> tuple[AdaptiveCRCFEstimator, pd.DataFrame, float]:
    tdd = estimate_tdd_from_replaybg_data(data, sample_minutes)
    config = AdaptiveCRCFConfig(target_glucose=target_glucose, cr_update_method=cr_method)
    estimator = AdaptiveCRCFEstimator.from_tdd(tdd=tdd, config=config)
    updates = estimator.update_from_dataframe(prepare_estimation_dataframe(data, sample_minutes))
    return estimator, updates, tdd


def generate_crcf_therapy(
    data: pd.DataFrame,
    estimator: AdaptiveCRCFEstimator,
    sample_minutes: float,
    target_glucose: float,
    correction_threshold: float,
    allow_correction_only: bool,
) -> pd.DataFrame:
    out = data.copy()
    out["bolus_original"] = pd.to_numeric(out["bolus"], errors="coerce").fillna(0.0)
    out["crcf_meal_bolus_u"] = 0.0
    out["crcf_correction_bolus_u"] = 0.0
    out["crcf_total_bolus_u"] = 0.0
    out["crcf_cr_used"] = np.nan
    out["crcf_cf_used"] = estimator.cf
    out["crcf_label"] = ""

    cho_rate = pd.to_numeric(out["cho"], errors="coerce").fillna(0.0)
    glucose = pd.to_numeric(out["glucose"], errors="coerce")
    labels = out.get("cho_label", pd.Series([""] * len(out))).fillna("").astype(str)

    for idx in out.index:
        meal_g = float(cho_rate.loc[idx]) * sample_minutes
        g = float(glucose.loc[idx]) if np.isfinite(glucose.loc[idx]) else np.nan
        label = normalize_meal_label(labels.loc[idx]) if meal_g > 0 else ""

        meal_bolus = 0.0
        correction_bolus = 0.0
        cr_used = np.nan

        if meal_g > 0:
            cr_used = estimator.cr[label]
            meal_bolus = max(0.0, meal_g / cr_used)
            if np.isfinite(g):
                correction_bolus = max(0.0, (g - target_glucose) / estimator.cf)
        elif allow_correction_only and np.isfinite(g) and g >= correction_threshold:
            correction_bolus = max(0.0, (g - target_glucose) / estimator.cf)

        total_bolus = meal_bolus + correction_bolus
        out.loc[idx, "crcf_meal_bolus_u"] = meal_bolus
        out.loc[idx, "crcf_correction_bolus_u"] = correction_bolus
        out.loc[idx, "crcf_total_bolus_u"] = total_bolus
        out.loc[idx, "crcf_cr_used"] = cr_used
        out.loc[idx, "crcf_label"] = label

    out["bolus"] = out["crcf_total_bolus_u"] / sample_minutes
    return out


def process_file(
    data_path: Path,
    output_dir: Path,
    target_glucose: float,
    cr_method: str,
    correction_threshold: float,
    allow_correction_only: bool,
) -> tuple[dict, pd.DataFrame]:
    slug = patient_slug_from_data_path(data_path)
    data = pd.read_csv(data_path)
    data["t"] = pd.to_datetime(data["t"], errors="raise")
    sample_minutes = infer_sample_minutes(data)

    estimator, updates, tdd = fit_crcf(
        data=data,
        sample_minutes=sample_minutes,
        target_glucose=target_glucose,
        cr_method=cr_method,
    )
    therapy = generate_crcf_therapy(
        data=data,
        estimator=estimator,
        sample_minutes=sample_minutes,
        target_glucose=target_glucose,
        correction_threshold=correction_threshold,
        allow_correction_only=allow_correction_only,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    therapy_path = output_dir / f"{data_path.stem}_crcf.csv"
    therapy.to_csv(therapy_path, index=False)

    info = pd.DataFrame()
    info_path = patient_info_path(data_path.parent, slug)
    bw = np.nan
    u2ss = np.nan
    if info_path.exists():
        info = pd.read_csv(info_path)
        bw = float(info["bw"].iloc[0]) if "bw" in info.columns else np.nan
        u2ss = float(info["u2ss"].iloc[0]) if "u2ss" in info.columns else np.nan

    n_cr_updates = int((updates["type"] == "CR").sum()) if not updates.empty and "type" in updates.columns else 0
    n_cf_updates = int((updates["type"] == "CF").sum()) if not updates.empty and "type" in updates.columns else 0
    cf_method = "correction_response" if n_cf_updates > 0 else "init_1800_over_tdd"

    summary = {
        "patient_slug": slug,
        "data_path": str(data_path),
        "therapy_path": str(therapy_path),
        "patient_info_path": str(info_path) if info_path.exists() else "",
        "sample_minutes": sample_minutes,
        "estimated_tdd_u": tdd,
        "bw": bw,
        "u2ss": u2ss,
        "target_glucose": target_glucose,
        "cr_method": cr_method,
        "cf_method": cf_method,
        "cr_B": estimator.cr["B"],
        "cr_L": estimator.cr["L"],
        "cr_D": estimator.cr["D"],
        "cf": estimator.cf,
        "n_cr_updates": n_cr_updates,
        "n_cf_updates": n_cf_updates,
        "total_bolus_original_u": float((therapy["bolus_original"] * sample_minutes).sum()),
        "total_bolus_crcf_u": float(therapy["crcf_total_bolus_u"].sum()),
        "total_meal_bolus_crcf_u": float(therapy["crcf_meal_bolus_u"].sum()),
        "total_correction_bolus_crcf_u": float(therapy["crcf_correction_bolus_u"].sum()),
        "total_cho_g": float(pd.to_numeric(therapy["cho"], errors="coerce").fillna(0).sum() * sample_minutes),
    }

    if not updates.empty:
        updates = updates.copy()
        updates.insert(0, "patient_slug", slug)
    return summary, updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate CR/CF from ReplayBG 1-day multi-meal CSVs and export alternative bolus therapy CSVs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="*_day_1.csv")
    parser.add_argument("--target", type=float, default=110.0)
    parser.add_argument("--cr-method", choices=["bolus_adjusted", "proportional"], default="bolus_adjusted")
    parser.add_argument("--correction-threshold", type=float, default=180.0)
    parser.add_argument("--allow-correction-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_paths = sorted(
        path for path in args.input_dir.glob(args.pattern) if not path.name.startswith("patient_info_")
    )
    if not data_paths:
        raise FileNotFoundError(f"No data files matched {args.input_dir / args.pattern}")

    summaries = []
    update_frames = []
    for data_path in data_paths:
        print(f"Processing {data_path.name}", flush=True)
        summary, updates = process_file(
            data_path=data_path,
            output_dir=args.output_dir,
            target_glucose=args.target,
            cr_method=args.cr_method,
            correction_threshold=args.correction_threshold,
            allow_correction_only=args.allow_correction_only,
        )
        summaries.append(summary)
        if not updates.empty:
            update_frames.append(updates)

    summary_df = pd.DataFrame(summaries).sort_values("patient_slug").reset_index(drop=True)
    updates_df = pd.concat(update_frames, ignore_index=True) if update_frames else pd.DataFrame()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "crcf_replaybg_params.csv"
    updates_path = args.output_dir / "crcf_replaybg_updates.csv"
    json_path = args.output_dir / "crcf_replaybg_summary.json"
    summary_df.to_csv(summary_path, index=False)
    updates_df.to_csv(updates_path, index=False)
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(f"Saved params: {summary_path}")
    print(f"Saved updates: {updates_path}")
    print(f"Saved therapy CSVs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
