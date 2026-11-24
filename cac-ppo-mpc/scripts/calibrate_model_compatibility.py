from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "causal_arx_compat_v1"
ALLOWED_SOURCE_PATIENTS = {f"adult#{idx:03d}" for idx in range(1, 6)}


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def singleton(values: list[str], label: str) -> str:
    unique = sorted({str(value) for value in values if str(value)})
    if len(unique) != 1:
        raise ValueError(f"Calibration requires one {label}; observed {unique}")
    return unique[0]


def ewma(values: np.ndarray, *, sample_time_min: float, tau_min: float) -> np.ndarray:
    alpha = 1.0 - math.exp(-float(sample_time_min) / float(tau_min))
    result = np.empty_like(values, dtype=float)
    state = float("nan")
    for index, value in enumerate(values):
        state = float(value) if not math.isfinite(state) else float(
            (1.0 - alpha) * state + alpha * float(value)
        )
        result[index] = state
    return result


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a causal ARX compatibility gate from development trajectories."
    )
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--sample-time-min", type=float, default=5.0)
    parser.add_argument("--ewma-tau-min", type=float, default=30.0)
    parser.add_argument("--enter-quantile", type=float, default=0.80)
    parser.add_argument("--exit-quantile", type=float, default=0.95)
    parser.add_argument("--minimum-trajectories", type=int, default=15)
    parser.add_argument("--minimum-valid-samples", type=int, default=12)
    parser.add_argument("--enter-confirm-steps", type=int, default=6)
    parser.add_argument("--exit-confirm-steps", type=int, default=2)
    parser.add_argument("--min-off-steps", type=int, default=6)
    parser.add_argument("--z-clip", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    if not 0.0 < float(args.enter_quantile) < float(args.exit_quantile) < 1.0:
        raise SystemExit("Require 0 < enter_quantile < exit_quantile < 1.")
    if float(args.sample_time_min) <= 0.0 or float(args.ewma_tau_min) <= 0.0:
        raise SystemExit("Sample time and EWMA tau must be positive.")
    if (
        int(args.minimum_trajectories) < 1
        or int(args.minimum_valid_samples) < 1
        or int(args.enter_confirm_steps) < 1
        or int(args.exit_confirm_steps) < 1
        or int(args.min_off_steps) < 1
    ):
        raise SystemExit("Trajectory, sample, confirmation, and off counts must be positive.")

    manifest_path = suite_dir / "manifest.json"
    manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_rows, list):
        raise SystemExit("Calibration manifest must be a JSON list.")
    calibration_rows = [
        row for row in manifest_rows if str(row.get("role", "")) == "calibration"
    ]
    if not calibration_rows:
        raise SystemExit("No role=calibration experiment found in manifest.")

    source_patients: set[str] = set()
    source_experiments: list[str] = []
    expected_history_files: list[Path] = []
    provenance_values: dict[str, list[str]] = {
        "checkpoint_sha256": [],
        "source_tree_sha256": [],
        "arx_param_sha256": [],
        "requirements_lock_sha256": [],
        "runtime_fingerprint": [],
    }
    for row in calibration_rows:
        experiment = str(row["name"])
        source_experiments.append(experiment)
        if str(row.get("protocol_version", "")) != "causal_model_compat_v4":
            raise SystemExit("Calibration manifest must use causal_model_compat_v4.")
        if str(row.get("split_label", "")) != "development":
            raise SystemExit("Calibration manifest must use split_label=development.")
        patients_ordered = [
            item for item in str(row.get("patients", "")).split(",") if item
        ]
        patients = set(patients_ordered)
        if not patients or not patients.issubset(ALLOWED_SOURCE_PATIENTS):
            raise SystemExit(
                f"Calibration patients must be a subset of adult#001-005: {sorted(patients)}"
            )
        source_patients.update(patients)
        modes_ordered = [
            item for item in str(row.get("modes", "")).split(",") if item
        ]
        modes = set(modes_ordered)
        if modes != {"switch_nogate_shield"}:
            raise SystemExit(
                "Calibration must use only switch_nogate_shield to avoid circular gating."
            )
        if str(row.get("meal_announcement", "")) != "unannounced":
            raise SystemExit("Calibration requires unannounced meals.")
        repeats = int(row.get("repeats", 0))
        if repeats < 1:
            raise SystemExit("Calibration repeats must be positive.")
        expected_runs = len(patients_ordered) * len(modes_ordered) * repeats
        if int(row.get("expected_rows", expected_runs)) != expected_runs:
            raise SystemExit(f"Calibration manifest expected_rows mismatch: {experiment}")

        exp_dir = suite_dir / experiment
        config = read_json(exp_dir / "config.json")
        completion = read_json(exp_dir / "completion.json")
        fingerprint = str(row.get("protocol_fingerprint", ""))
        config_checks = {
            "protocol_version": "causal_model_compat_v4",
            "protocol_fingerprint": fingerprint,
            "checkpoint_sha256": str(row.get("checkpoint_sha256", "")),
            "source_tree_sha256": str(row.get("source_tree_sha256", "")),
            "arx_param_sha256": str(row.get("arx_param_sha256", "")),
            "requirements_lock_sha256": str(
                row.get("requirements_lock_sha256", "")
            ),
            "runtime_fingerprint_actual": str(
                row.get("runtime_fingerprint", "")
            ),
        }
        if not fingerprint or any(
            not expected or str(config.get(key, "")) != expected
            for key, expected in config_checks.items()
        ):
            raise SystemExit(f"Calibration config provenance mismatch: {experiment}")
        if set(config.get("patients_resolved", [])) != patients:
            raise SystemExit(f"Calibration config patient mismatch: {experiment}")
        if set(config.get("modes_resolved", [])) != modes:
            raise SystemExit(f"Calibration config mode mismatch: {experiment}")
        completion_ok = (
            bool(completion.get("execution_complete", completion.get("complete")))
            and bool(completion.get("claim_valid"))
            and int(completion.get("expected_runs", -1)) == expected_runs
            and int(completion.get("completed_runs", -1)) == expected_runs
            and int(completion.get("invalid_runs", -1)) == 0
            and str(completion.get("protocol_version", ""))
            == "causal_model_compat_v4"
            and str(completion.get("protocol_fingerprint", "")) == fingerprint
        )
        if not completion_ok:
            raise SystemExit(
                f"Calibration completion marker is incomplete or stale: {experiment}"
            )

        summary_path = exp_dir / "summary.csv"
        if not summary_path.exists():
            raise SystemExit(f"Calibration summary is missing: {summary_path}")
        summary = pd.read_csv(summary_path)
        required_summary = {"patient", "mode", "repeat", "seed", "claim_valid"}
        missing_summary = sorted(required_summary - set(summary.columns))
        if missing_summary:
            raise SystemExit(
                f"Calibration summary lacks {missing_summary}: {summary_path}"
            )
        expected_keys = {
            (
                patient,
                mode,
                repeat,
                int(row.get("seed", 0))
                + 1000 * repeat
                + int(patient.split("#")[-1]),
            )
            for patient in patients_ordered
            for mode in modes_ordered
            for repeat in range(repeats)
        }
        actual_keys: list[tuple[str, str, int, int]] = []
        for item in summary.itertuples(index=False):
            actual_keys.append(
                (
                    str(item.patient),
                    str(item.mode),
                    int(item.repeat),
                    int(item.seed),
                )
            )
        claim_valid = (
            summary["claim_valid"].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            )
        )
        if (
            len(actual_keys) != expected_runs
            or len(set(actual_keys)) != expected_runs
            or set(actual_keys) != expected_keys
            or not bool(claim_valid.all())
        ):
            raise SystemExit(
                f"Calibration summary key set is incomplete or stale: {experiment}"
            )

        expected_for_experiment = {
            exp_dir
            / (
                f"{patient.replace('#', '-')}_{mode}_seed{seed}"
                "_policy_history.csv"
            )
            for patient, mode, _repeat, seed in expected_keys
        }
        actual_for_experiment = set(
            exp_dir.glob("*_switch_nogate_shield_seed*_policy_history.csv")
        )
        if actual_for_experiment != expected_for_experiment:
            missing = sorted(
                path.name for path in expected_for_experiment - actual_for_experiment
            )
            extra = sorted(
                path.name for path in actual_for_experiment - expected_for_experiment
            )
            raise SystemExit(
                f"Calibration policy-history set mismatch for {experiment}; "
                f"missing={missing}, extra={extra}"
            )
        expected_history_files.extend(sorted(expected_for_experiment))
        for key in provenance_values:
            provenance_values[key].append(str(row.get(key, "")))

    if source_patients != ALLOWED_SOURCE_PATIENTS:
        raise SystemExit(
            "Calibration must contain exactly adult#001-005; observed "
            + ", ".join(sorted(source_patients))
        )

    if len(expected_history_files) != len(set(expected_history_files)):
        raise SystemExit("Calibration manifest resolves duplicate policy histories.")

    trajectory_rows: list[dict] = []
    residual_by_file: list[tuple[Path, np.ndarray]] = []
    for path in sorted(expected_history_files):
        experiment = path.parent.name
        frame = pd.read_csv(path)
        required_columns = {
            "model_residual_raw_mgdl",
            "model_bg_pred_mgdl",
            "model_bg_meas_mgdl",
        }
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise SystemExit(
                f"Missing causal residual audit columns {missing_columns}: {path}"
            )
        raw = pd.to_numeric(
            frame["model_residual_raw_mgdl"], errors="coerce"
        ).to_numpy(dtype=float)
        predicted = pd.to_numeric(
            frame["model_bg_pred_mgdl"], errors="coerce"
        ).to_numpy(dtype=float)
        measured = pd.to_numeric(
            frame["model_bg_meas_mgdl"], errors="coerce"
        ).to_numpy(dtype=float)
        raw_finite = np.isfinite(raw)
        aligned = raw_finite & np.isfinite(predicted) & np.isfinite(measured)
        if bool(np.any(raw_finite & ~aligned)):
            raise SystemExit(
                f"Finite residual lacks aligned prediction/measurement: {path}"
            )
        consistency = np.abs(measured[aligned] - predicted[aligned] - raw[aligned])
        consistency_max = (
            float(np.max(consistency)) if consistency.size else float("nan")
        )
        if not math.isfinite(consistency_max) or consistency_max > 1e-4:
            raise SystemExit(
                f"Residual identity audit failed ({consistency_max} mg/dL): {path}"
            )
        values = raw[aligned]
        if values.size < int(args.minimum_valid_samples):
            raise SystemExit(
                f"Calibration trajectory has only {values.size} valid residuals: {path}"
            )
        if values.size:
            residual_by_file.append((path, values))
            trajectory_rows.append(
                {
                    "experiment": experiment,
                    "file": str(path),
                    "valid_residuals": int(values.size),
                    "residual_median_mgdl": float(np.median(values)),
                    "residual_mae_mgdl": float(np.mean(np.abs(values))),
                    "residual_consistency_max_mgdl": consistency_max,
                }
            )
    if len(residual_by_file) != len(expected_history_files):
        raise SystemExit(
            "Every expected calibration history must contain finite aligned residuals; "
            f"expected {len(expected_history_files)}, found {len(residual_by_file)}."
        )
    if len(residual_by_file) < int(args.minimum_trajectories):
        raise SystemExit(
            f"Need at least {args.minimum_trajectories} calibration trajectories; "
            f"found {len(residual_by_file)}."
        )

    pooled = np.concatenate([values for _, values in residual_by_file])
    center = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - center)))
    scale = float(max(1.4826 * mad, 1.0))
    ewma_values: list[np.ndarray] = []
    for _, values in residual_by_file:
        npe = np.clip(
            np.abs(values - center) / scale,
            0.0,
            float(args.z_clip),
        )
        smoothed = ewma(
            npe,
            sample_time_min=float(args.sample_time_min),
            tau_min=float(args.ewma_tau_min),
        )
        start = max(int(args.minimum_valid_samples) - 1, 0)
        if smoothed.size > start:
            ewma_values.append(smoothed[start:])
    if not ewma_values:
        raise SystemExit("No post-warmup EWMA samples were available.")
    pooled_ewma = np.concatenate(ewma_values)
    enter = float(np.quantile(pooled_ewma, float(args.enter_quantile)))
    raw_exit = float(np.quantile(pooled_ewma, float(args.exit_quantile)))
    exit_threshold = float(max(raw_exit, enter + max(0.1, 0.05 * enter)))
    if not math.isfinite(enter) or enter <= 0.0:
        raise SystemExit("Calibrated enter threshold is invalid.")

    provenance = {
        key: singleton(values, key) for key, values in provenance_values.items()
    }
    gate = {
        "schema_version": SCHEMA_VERSION,
        "residual_center_mgdl": center,
        "residual_scale_mgdl": scale,
        "ewma_tau_min": float(args.ewma_tau_min),
        "enter_threshold_npe": enter,
        "exit_threshold_npe": exit_threshold,
        "min_valid_samples": int(args.minimum_valid_samples),
        "enter_confirm_steps": int(args.enter_confirm_steps),
        "exit_confirm_steps": int(args.exit_confirm_steps),
        "min_off_steps": int(args.min_off_steps),
        "z_clip": float(args.z_clip),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_patients": sorted(source_patients),
        "source_experiments": sorted(source_experiments),
        "source_mode": "switch_nogate_shield",
        "split_label": "development",
        "residual_definition": (
            "current causal CGM minus the pre-update one-step ARX prediction"
        ),
        "calibration_rule": {
            "center": "pooled median",
            "scale": "max(1.4826*MAD, 1 mg/dL)",
            "smoother": "per-trajectory EWMA of clipped normalized absolute residual",
            "enter_quantile": float(args.enter_quantile),
            "exit_quantile": float(args.exit_quantile),
        },
        "sample_time_min": float(args.sample_time_min),
        "n_trajectories": int(len(residual_by_file)),
        "n_raw_residuals": int(pooled.size),
        "n_post_warmup_ewma": int(pooled_ewma.size),
        "gate": gate,
        "provenance": provenance,
    }
    calibration_id = "compat-" + canonical_sha256(payload)[:16]
    payload["calibration_id"] = calibration_id
    payload["gate"]["calibration_id"] = calibration_id

    output = (
        Path(args.output).resolve()
        if args.output
        else suite_dir / "analysis" / "compatibility_calibration.json"
    )
    write_atomic(output, payload)
    audit_path = output.with_name("compatibility_calibration_audit.csv")
    pd.DataFrame(trajectory_rows).to_csv(audit_path, index=False)
    print(f"Saved compatibility calibration: {output}")
    print(f"Calibration id: {calibration_id}")
    print(
        f"center={center:.6f}, scale={scale:.6f}, "
        f"enter={enter:.6f}, exit={exit_threshold:.6f}"
    )


if __name__ == "__main__":
    main()
