from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy may be absent on minimal servers
    stats = None

METRICS = [
    "tir",
    "tbr70",
    "tbr54",
    "tar180",
    "tar250",
    "mean_bg",
    "sd_bg",
    "total_insulin_u",
    "ppo_takeover_rate",
    "mpc_takeover_rate",
    "switch_count",
    "ppo_authorized_rate",
    "ppo_effective_rate",
    "shield_override_rate",
    "model_residual_valid_rate",
    "model_residual_mae_mgdl",
    "model_residual_rmse_mgdl",
    "model_residual_consistency_max_mgdl",
    "model_npe_mean",
    "model_npe_p95",
    "model_npe_ewma_mean",
    "model_npe_ewma_p95",
    "model_compatibility_score_mean",
    "model_compatibility_score_p95",
    "model_compatible_rate",
    "mpc_model_authorized_rate",
    "mpc_candidate_valid_rate",
    "compat_gate_transition_count",
    "mpc_authorization_violations",
]
PAPER_METRICS = ["tir", "tbr70", "tbr54", "tar180", "tar250", "mean_bg", "total_insulin_u"]
PAIRWISE = [
    ("switch", "mpc"),
    ("switch", "ppo"),
    ("switch_shield", "mpc_shield"),
    ("switch_shield", "ppo_shield"),
    ("ppo_shield", "ppo"),
    ("mpc_shield", "mpc"),
    ("switch_shield", "switch"),
    ("switch_shield", "switch_nogate_shield"),
]


def read_json(path: Path, *, required: bool = False) -> dict:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if required:
            raise ValueError(f"invalid JSON: {path}") from exc
        return {}


def load_manifest(suite_dir: Path) -> dict[str, dict]:
    manifest = suite_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest is required: {manifest}")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    result = {row["name"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("manifest contains duplicate experiment names")
    return result


def load_summaries(
    suite_dir: Path, manifest: dict[str, dict], *, allow_partial: bool
) -> pd.DataFrame:
    rows = []
    for exp, meta in manifest.items():
        summary_path = suite_dir / exp / "summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        cfg = read_json(
            summary_path.parent / "config.json", required=not allow_partial
        )
        df["experiment"] = exp
        df["split_label"] = meta.get("split_label", "")
        df["role"] = meta.get("role", "")
        df["model_condition"] = cfg.get(
            "model_condition", meta.get("model_condition", "nominal")
        )
        df["arx_prior_label"] = cfg.get(
            "arx_prior_label", meta.get("arx_prior_label", "")
        )
        df["protocol_version"] = cfg.get(
            "protocol_version", meta.get("protocol_version", "causal_fair_v3")
        )
        df["meal_announcement"] = cfg.get("meal_announcement", meta.get("meal_announcement", ""))
        df["hours"] = cfg.get("hours", meta.get("hours", np.nan))
        df["output_dir"] = str(summary_path.parent)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def validate_suite_completeness(
    suite_dir: Path,
    manifest: dict[str, dict],
    df: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    issues: list[str] = []
    records: list[dict] = []
    missing_rows: list[dict] = []
    discovered = {path.parent.name for path in suite_dir.glob("*/summary.csv")}
    extras = sorted(discovered - set(manifest))
    if extras:
        issues.append(f"summary directories absent from manifest: {extras}")

    required_metrics = ["tir", "tbr70", "tbr54", "tar180", "tar250"]
    v4_formal_experiments: set[str] = set()
    suite_calibration_hashes: set[str] = set()
    suite_calibration_ids: set[str] = set()
    suite_reference_arx_hashes: set[str] = set()
    suite_nominal_arx_hashes: set[str] = set()
    for exp, meta in manifest.items():
        diagnostic_role = str(meta.get("role", "")) == "diagnostic"
        calibration_role = str(meta.get("role", "")) == "calibration"
        patients = [item for item in str(meta.get("patients", "")).split(",") if item]
        modes = [item for item in str(meta.get("modes", "")).split(",") if item]
        repeats = int(meta.get("repeats", 0))
        expected = {
            (patient, mode, repeat)
            for patient in patients
            for mode in modes
            for repeat in range(repeats)
        }
        sub = df[df["experiment"].eq(exp)].copy() if not df.empty else pd.DataFrame()
        actual: set[tuple[str, str, int]] = set()
        duplicates = 0
        if not sub.empty and {"patient", "mode", "repeat"}.issubset(sub.columns):
            keys = [
                (str(row.patient), str(row.mode), int(row.repeat))
                for row in sub[["patient", "mode", "repeat"]].itertuples(index=False)
            ]
            actual = set(keys)
            duplicates = len(keys) - len(actual)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        for patient, mode, repeat in missing:
            missing_rows.append(
                {
                    "experiment": exp,
                    "patient": patient,
                    "mode": mode,
                    "repeat": repeat,
                }
            )

        marker = read_json(suite_dir / exp / "completion.json")
        config = read_json(suite_dir / exp / "config.json")
        expected_fingerprint = str(meta.get("protocol_fingerprint", ""))
        protocol_version = str(
            meta.get("protocol_version")
            or config.get("protocol_version")
            or "causal_fair_v3"
        )
        marker_ok = (
            bool(marker.get("execution_complete", marker.get("complete")))
            and int(marker.get("completed_runs", -1)) == len(expected)
            and int(marker.get("expected_runs", -1)) == len(expected)
            and bool(expected_fingerprint)
            and marker.get("protocol_fingerprint") == expected_fingerprint
        )
        config_ok = (
            config.get("protocol_version") == protocol_version
            and config.get("protocol_fingerprint") == expected_fingerprint
        )
        provenance_ok = True
        compatibility_ok = True
        if protocol_version == "causal_model_compat_v4":
            if not calibration_role:
                v4_formal_experiments.add(exp)
                suite_calibration_hashes.add(
                    str(meta.get("compatibility_calibration_sha256", ""))
                )
                suite_reference_arx_hashes.add(
                    str(meta.get("compatibility_reference_arx_sha256", ""))
                )
                model_condition = str(
                    meta.get(
                        "model_condition",
                        config.get("model_condition", ""),
                    )
                )
                if model_condition == "nominal":
                    suite_nominal_arx_hashes.add(
                        str(meta.get("arx_param_sha256", ""))
                    )
            provenance_fields = [
                ("checkpoint_sha256", "checkpoint_sha256"),
                ("arx_param_sha256", "arx_param_sha256"),
                ("source_tree_sha256", "source_tree_sha256"),
                ("requirements_lock_sha256", "requirements_lock_sha256"),
                ("runtime_fingerprint", "runtime_fingerprint_actual"),
            ]
            for manifest_key, config_key in provenance_fields:
                expected_value = str(meta.get(manifest_key, ""))
                actual_value = str(config.get(config_key, ""))
                if not expected_value or actual_value != expected_value:
                    provenance_ok = False
            expected_reference_arx = str(
                meta.get("compatibility_reference_arx_sha256", "")
            )
            if expected_reference_arx:
                if str(config.get("compatibility_reference_arx_sha256", "")) != expected_reference_arx:
                    provenance_ok = False
            overlap_override = str(
                config.get("allow_compatibility_source_overlap", False)
            ).lower() in {"true", "1", "yes"}
            if overlap_override and str(meta.get("split_label", "")) != "smoke":
                provenance_ok = False
            if str(meta.get("role", "")) != "calibration":
                expected_calibration = str(
                    meta.get("compatibility_calibration_sha256", "")
                )
                actual_calibration = str(
                    config.get("compatibility_calibration_sha256", "")
                )
                if (
                    not expected_calibration
                    or expected_calibration == "MISSING"
                    or actual_calibration != expected_calibration
                ):
                    provenance_ok = False

            gate_modes = {"switch", "switch_shield"}.intersection(modes)
            if gate_modes:
                required_gate_columns = [
                    "model_residual_valid_rate",
                    "model_residual_consistency_max_mgdl",
                    "model_npe_ewma_mean",
                    "model_compatibility_score_mean",
                    "model_compatible_rate",
                    "mpc_model_authorized_rate",
                    "mpc_authorization_violations",
                    "compatibility_gate_enabled",
                    "compatibility_calibration_id",
                ]
                for mode in sorted(gate_modes):
                    mode_sub = sub[sub["mode"].eq(mode)]
                    missing_columns = [
                        column
                        for column in required_gate_columns
                        if column not in mode_sub.columns
                    ]
                    if missing_columns:
                        compatibility_ok = False
                        continue
                    gate_enabled = mode_sub[
                        "compatibility_gate_enabled"
                    ].astype(str).str.lower().isin(["true", "1", "yes"])
                    if not bool(gate_enabled.all()):
                        compatibility_ok = False
                    ids = mode_sub["compatibility_calibration_id"].fillna("").astype(str)
                    if bool(ids.eq("").any()):
                        compatibility_ok = False
                    if not calibration_role:
                        suite_calibration_ids.update(
                            value for value in ids if value
                        )
                    for column in [
                        "model_residual_valid_rate",
                        "model_residual_consistency_max_mgdl",
                        "model_npe_ewma_mean",
                        "model_compatibility_score_mean",
                        "model_compatible_rate",
                        "mpc_model_authorized_rate",
                    ]:
                        values = pd.to_numeric(mode_sub[column], errors="coerce")
                        if bool((~np.isfinite(values)).any()):
                            compatibility_ok = False
                    violations = pd.to_numeric(
                        mode_sub["mpc_authorization_violations"], errors="coerce"
                    )
                    if bool((~np.isfinite(violations) | (violations != 0)).any()):
                        compatibility_ok = False
                    consistency = pd.to_numeric(
                        mode_sub["model_residual_consistency_max_mgdl"], errors="coerce"
                    )
                    if bool(
                        (~np.isfinite(consistency) | (consistency > 1e-4)).any()
                    ):
                        compatibility_ok = False

            nogate_modes = {
                "switch_nogate", "switch_nogate_shield"
            }.intersection(modes)
            for mode in sorted(nogate_modes):
                mode_sub = sub[sub["mode"].eq(mode)]
                required_nogate_columns = [
                    "model_residual_valid_rate",
                    "model_residual_consistency_max_mgdl",
                    "mpc_authorization_violations",
                    "compatibility_gate_enabled",
                ]
                if not calibration_role:
                    required_nogate_columns.append(
                        "compatibility_calibration_id"
                    )
                if any(
                    column not in mode_sub.columns
                    for column in required_nogate_columns
                ):
                    compatibility_ok = False
                    continue
                gate_enabled = mode_sub[
                    "compatibility_gate_enabled"
                ].astype(str).str.lower().isin(["true", "1", "yes"])
                if bool(gate_enabled.any()):
                    compatibility_ok = False
                if not calibration_role:
                    ids = (
                        mode_sub["compatibility_calibration_id"]
                        .fillna("")
                        .astype(str)
                    )
                    if bool(ids.eq("").any()):
                        compatibility_ok = False
                    suite_calibration_ids.update(
                        value for value in ids if value
                    )
                residual_rate = pd.to_numeric(
                    mode_sub["model_residual_valid_rate"], errors="coerce"
                )
                if bool(
                    (
                        ~np.isfinite(residual_rate)
                        | (residual_rate <= 0.0)
                        | (residual_rate > 1.0)
                    ).any()
                ):
                    compatibility_ok = False
                consistency = pd.to_numeric(
                    mode_sub["model_residual_consistency_max_mgdl"],
                    errors="coerce",
                )
                if bool(
                    (~np.isfinite(consistency) | (consistency > 1e-4)).any()
                ):
                    compatibility_ok = False
                violations = pd.to_numeric(
                    mode_sub["mpc_authorization_violations"], errors="coerce"
                )
                if bool((~np.isfinite(violations) | (violations != 0)).any()):
                    compatibility_ok = False

        invalid_rows = 0
        if not sub.empty:
            for metric in required_metrics:
                if metric not in sub.columns:
                    invalid_rows = len(sub)
                    issues.append(f"{exp}: missing required metric {metric}")
                    continue
                values = pd.to_numeric(sub[metric], errors="coerce")
                invalid_rows += int(
                    (~np.isfinite(values) | (values < 0) | (values > 100)).sum()
                )
            if {"tbr54", "tbr70"}.issubset(sub.columns):
                invalid_rows += int(
                    (sub["tbr54"] > sub["tbr70"] + 1e-9).sum()
                )
            if {"tar250", "tar180"}.issubset(sub.columns):
                invalid_rows += int(
                    (sub["tar250"] > sub["tar180"] + 1e-9).sum()
                )
            if "nonfinite_steps" in sub.columns:
                invalid_rows += int(
                    (
                        pd.to_numeric(
                            sub["nonfinite_steps"], errors="coerce"
                        )
                        > 0
                    ).sum()
                )
            else:
                invalid_rows += len(sub)
            if "cgm_nonfinite_steps" in sub.columns:
                invalid_rows += int(
                    (
                        pd.to_numeric(
                            sub["cgm_nonfinite_steps"], errors="coerce"
                        )
                        > 0
                    ).sum()
                )
            else:
                invalid_rows += len(sub)
            if {"raw_steps", "completed_steps"}.issubset(sub.columns):
                invalid_rows += int(
                    (sub["raw_steps"] != sub["completed_steps"]).sum()
                )
            else:
                invalid_rows += len(sub)
            if not diagnostic_role:
                if {"completed_steps", "expected_steps"}.issubset(sub.columns):
                    invalid_rows += int(
                        (sub["completed_steps"] != sub["expected_steps"]).sum()
                    )
                else:
                    invalid_rows += len(sub)
                if "terminated_early" in sub.columns:
                    early = sub["terminated_early"].astype(str).str.lower().isin(
                        ["true", "1", "yes"]
                    )
                    invalid_rows += int(early.sum())
                else:
                    invalid_rows += len(sub)
                if "terminal_event" in sub.columns:
                    terminal = sub["terminal_event"].astype(str).str.lower().isin(
                        ["true", "1", "yes"]
                    )
                    invalid_rows += int(terminal.sum())
                else:
                    invalid_rows += len(sub)
                if "claim_valid" in sub.columns:
                    valid = sub["claim_valid"].astype(str).str.lower().isin(
                        ["true", "1", "yes"]
                    )
                    invalid_rows += int((~valid).sum())
                else:
                    invalid_rows += len(sub)

        marker_claim_valid = bool(marker.get("claim_valid")) and int(
            marker.get("invalid_runs", -1)
        ) == 0
        complete = (
            not missing
            and not unexpected
            and duplicates == 0
            and marker_ok
            and (marker_claim_valid or diagnostic_role)
            and config_ok
            and provenance_ok
            and compatibility_ok
            and invalid_rows == 0
        )
        records.append(
            {
                "experiment": exp,
                "protocol_version": protocol_version,
                "expected_rows": len(expected),
                "actual_rows": len(sub),
                "missing_rows": len(missing),
                "unexpected_rows": len(unexpected),
                "duplicate_rows": duplicates,
                "invalid_checks": invalid_rows,
                "completion_marker_valid": marker_ok,
                "marker_claim_valid": marker_claim_valid,
                "claim_required": not diagnostic_role,
                "config_fingerprint_valid": config_ok,
                "provenance_valid": provenance_ok,
                "compatibility_evidence_valid": compatibility_ok,
                "suite_invariants_valid": True,
                "complete": complete,
            }
        )
        if not complete:
            issues.append(
                f"{exp}: incomplete or invalid; see suite_completeness.csv"
            )

    suite_invariant_issues: list[str] = []
    if v4_formal_experiments:
        if (
            len(suite_calibration_hashes) != 1
            or not next(iter(suite_calibration_hashes), "")
            or "MISSING" in suite_calibration_hashes
        ):
            suite_invariant_issues.append(
                "v4 formal experiments must share one non-missing calibration SHA256"
            )
        if (
            len(suite_reference_arx_hashes) != 1
            or not next(iter(suite_reference_arx_hashes), "")
            or "MISSING" in suite_reference_arx_hashes
        ):
            suite_invariant_issues.append(
                "v4 formal experiments must share one non-missing reference ARX SHA256"
            )
        if (
            len(suite_nominal_arx_hashes) != 1
            or not next(iter(suite_nominal_arx_hashes), "")
            or "MISSING" in suite_nominal_arx_hashes
        ):
            suite_invariant_issues.append(
                "v4 suite must declare one nominal ARX SHA256"
            )
        if (
            len(suite_reference_arx_hashes) == 1
            and len(suite_nominal_arx_hashes) == 1
            and suite_reference_arx_hashes != suite_nominal_arx_hashes
        ):
            suite_invariant_issues.append(
                "v4 reference ARX SHA256 must equal the nominal adult-prior SHA256"
            )
        if len(suite_calibration_ids) != 1 or not next(
            iter(suite_calibration_ids), ""
        ):
            suite_invariant_issues.append(
                "v4 Switch and Switch-no-gate rows must share one calibration ID"
            )
    if suite_invariant_issues:
        issues.extend(suite_invariant_issues)
        for record in records:
            if record["experiment"] in v4_formal_experiments:
                record["suite_invariants_valid"] = False
                record["complete"] = False

    pd.DataFrame(records).to_csv(
        out_dir / "suite_completeness.csv", index=False
    )
    pd.DataFrame(missing_rows).to_csv(
        out_dir / "suite_missing_runs.csv", index=False
    )
    return issues


def _boolean_series(df: pd.DataFrame, name: str, default: bool) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[name].astype(str).str.lower().isin(["true", "1", "yes"])


def write_partial_diagnostics(
    df: pd.DataFrame, manifest: dict[str, dict], out_dir: Path
) -> dict[str, int]:
    diagnostic = df.copy()
    completed = pd.to_numeric(diagnostic.get("completed_steps"), errors="coerce")
    expected = pd.to_numeric(diagnostic.get("expected_steps"), errors="coerce")
    diagnostic["completion_fraction"] = completed / expected.replace(0, np.nan)
    terminal = _boolean_series(diagnostic, "terminal_event", False)
    valid = _boolean_series(diagnostic, "claim_valid", False)
    diagnostic["censored"] = terminal | ~valid | (completed != expected)
    identity_columns = [
        "experiment",
        "patient",
        "mode",
        "repeat",
        "seed",
        "completed_steps",
        "expected_steps",
        "completion_fraction",
        "terminated_early",
        "terminal_event",
        "termination_reason",
        "terminal_bg",
        "nonfinite_steps",
        "cgm_nonfinite_steps",
        "claim_valid",
        "censored",
    ]
    diagnostic[[name for name in identity_columns if name in diagnostic.columns]].to_csv(
        out_dir / "diagnostic_run_validity.csv", index=False
    )

    rows: list[dict] = []
    for experiment, meta in manifest.items():
        patients = [item for item in str(meta.get("patients", "")).split(",") if item]
        repeats = int(meta.get("repeats", 0))
        for mode in [item for item in str(meta.get("modes", "")).split(",") if item]:
            sub = diagnostic[
                diagnostic["experiment"].eq(experiment) & diagnostic["mode"].eq(mode)
            ]
            expected_runs = len(patients) * repeats
            complete_runs = int(
                (
                    pd.to_numeric(sub.get("completed_steps"), errors="coerce")
                    == pd.to_numeric(sub.get("expected_steps"), errors="coerce")
                ).sum()
            )
            terminal_runs = int(_boolean_series(sub, "terminal_event", False).sum())
            invalid_runs = int((~_boolean_series(sub, "claim_valid", False)).sum())
            reasons = sub.loc[
                _boolean_series(sub, "terminal_event", False), "termination_reason"
            ].value_counts(dropna=False)
            if reasons.empty:
                reasons = pd.Series({"none": 0})
            for reason, count in reasons.items():
                rows.append(
                    {
                        "experiment": experiment,
                        "mode": mode,
                        "termination_reason": reason,
                        "expected_runs": expected_runs,
                        "observed_runs": len(sub),
                        "complete_runs": complete_runs,
                        "terminal_runs": terminal_runs,
                        "invalid_runs": invalid_runs,
                        "reason_count": int(count),
                        "terminal_rate_expected": (
                            terminal_runs / expected_runs if expected_runs else np.nan
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(out_dir / "diagnostic_failure_counts.csv", index=False)
    return {
        "terminal_runs": int(terminal.sum()),
        "invalid_runs": int((~valid).sum()),
        "censored_runs": int(diagnostic["censored"].sum()),
    }


def ci95(series: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    mean = float(np.mean(x))
    if x.size == 1:
        return (float("nan"), float("nan"))
    sem = float(np.std(x, ddof=1) / math.sqrt(x.size))
    crit = float(stats.t.ppf(0.975, x.size - 1)) if stats is not None else 1.96
    return (mean - crit * sem, mean + crit * sem)


def aggregate(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    records = []
    group_cols = ["experiment", "split_label", "role", "meal_announcement"]
    for optional in ["model_condition", "arx_prior_label"]:
        if optional in df.columns:
            group_cols.append(optional)
    group_cols.append("mode")
    for keys, group in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        base["n"] = int(len(group))
        base["n_patients"] = int(group["patient"].nunique()) if "patient" in group else int(len(group))
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            low, high = ci95(values)
            base[f"{metric}_mean"] = float(values.mean())
            base[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            base[f"{metric}_ci95_low"] = low
            base[f"{metric}_ci95_high"] = high
            base[f"{metric}_min"] = float(values.min())
            base[f"{metric}_max"] = float(values.max())
        records.append(base)
    result = pd.DataFrame(records)
    result.to_csv(out_dir / "overall_by_experiment_mode.csv", index=False)
    return result


def compatibility_behavior(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    identity = ["experiment"]
    for optional in ["split_label", "role", "model_condition", "arx_prior_label"]:
        if optional in df.columns:
            identity.append(optional)
    identity.append("mode")
    metric_columns = [
        "model_residual_valid_rate",
        "model_residual_mae_mgdl",
        "model_residual_rmse_mgdl",
        "model_residual_consistency_max_mgdl",
        "model_npe_ewma_mean",
        "model_npe_ewma_p95",
        "model_compatibility_score_mean",
        "model_compatible_rate",
        "mpc_model_authorized_rate",
        "mpc_candidate_valid_rate",
        "compat_gate_transition_count",
        "mpc_authorization_violations",
    ]
    if df.empty or "mode" not in df.columns:
        result = pd.DataFrame()
    else:
        present = [column for column in metric_columns if column in df.columns]
        result = (
            df.groupby(identity, dropna=False)[present]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [
                "_".join(str(part) for part in column if str(part))
                for column in result.columns
            ]
    result.to_csv(out_dir / "compatibility_behavior.csv", index=False)
    return result


def patient_means(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    keep_metrics = [m for m in METRICS if m in df.columns]
    group_cols = ["experiment", "split_label", "role", "meal_announcement"]
    for optional in ["model_condition", "arx_prior_label"]:
        if optional in df.columns:
            group_cols.append(optional)
    group_cols.extend(["patient", "mode"])
    result = (
        df.groupby(group_cols, as_index=False)[keep_metrics]
        .mean(numeric_only=True)
        .sort_values(["experiment", "patient", "mode"])
    )
    result.to_csv(out_dir / "patient_mode_mean.csv", index=False)
    return result


def holm_adjust(values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    finite = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    total = len(finite)
    running = 0.0
    for rank, (idx, value) in enumerate(finite.items()):
        candidate = min(1.0, float(value) * float(total - rank))
        running = max(running, candidate)
        adjusted.loc[idx] = running
    return adjusted


def paired_tests(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    records = []
    for exp, sub in df.groupby("experiment"):
        for metric in PAPER_METRICS:
            if metric not in sub.columns:
                continue
            wide = sub.pivot_table(index="patient", columns="mode", values=metric, aggfunc="mean")
            for left, right in PAIRWISE:
                if left not in wide.columns or right not in wide.columns:
                    continue
                paired = wide[[left, right]].dropna()
                if paired.empty:
                    continue
                diff = paired[left] - paired[right]
                low, high = ci95(diff)
                sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
                p_t = float("nan")
                p_w = float("nan")
                if stats is not None and len(diff) >= 2:
                    if sd > 0.0:
                        try:
                            p_t = float(
                                stats.ttest_rel(
                                    paired[left], paired[right], nan_policy="omit"
                                ).pvalue
                            )
                        except Exception:
                            pass
                    if np.allclose(diff.to_numpy(dtype=float), 0.0):
                        p_w = 1.0
                    else:
                        try:
                            p_w = float(stats.wilcoxon(diff).pvalue)
                        except Exception:
                            pass
                records.append(
                    {
                        "experiment": exp,
                        "metric": metric,
                        "comparison": f"{left}_minus_{right}",
                        "n_pairs": int(len(diff)),
                        "positive_pairs": int((diff > 0).sum()),
                        "negative_pairs": int((diff < 0).sum()),
                        "zero_pairs": int((diff == 0).sum()),
                        "mean_diff": float(diff.mean()),
                        "std_diff": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                        "paired_t_p": p_t,
                        "wilcoxon_p": p_w,
                        "cohens_dz": float(diff.mean() / sd) if sd > 0 else float("nan"),
                    }
                )
    result = pd.DataFrame(records)
    if not result.empty:
        result["paired_t_p_holm"] = result.groupby("experiment", group_keys=False)[
            "paired_t_p"
        ].apply(holm_adjust)
        result["wilcoxon_p_holm"] = result.groupby("experiment", group_keys=False)[
            "wilcoxon_p"
        ].apply(holm_adjust)
    result.to_csv(out_dir / "paired_tests.csv", index=False)
    return result


def find_primary_safety(df: pd.DataFrame) -> str | None:
    primary = df[
        df["role"].eq("primary") & df["meal_announcement"].eq("unannounced")
    ]["experiment"].dropna().unique()
    if len(primary) == 1:
        return str(primary[0])
    if len(primary) > 1:
        raise ValueError(f"multiple unannounced primary experiments found: {sorted(primary)}")
    return None


def statistical_design_audit(
    patient_df: pd.DataFrame,
    out_dir: Path,
    baseline_name: str | None,
) -> dict:
    warnings: list[str] = []
    if baseline_name is None:
        payload = {
            "status": "INSUFFICIENT",
            "claim_gate_passed": False,
            "n_independent_patients": 0,
            "warnings": ["No unique unannounced primary experiment was found."],
        }
    else:
        primary = patient_df[patient_df["experiment"].eq(baseline_name)]
        n_patients = int(primary["patient"].nunique())
        minimum_two_sided_exact_p = (
            float(2.0 / (2.0**n_patients)) if n_patients > 0 else float("nan")
        )
        splits = sorted(primary["split_label"].dropna().astype(str).unique())
        if n_patients < 6:
            warnings.append(
                "Fewer than 6 independent patients: a two-sided exact sign/Wilcoxon "
                "test cannot attain unadjusted p<0.05 even if every patient agrees."
            )
        if "test" not in splits:
            warnings.append(
                "Primary results are not labeled as a locked held-out test; treat them "
                "as validation evidence only."
            )
        warnings.append(
            "The suite uses one PPO checkpoint; run prespecified independent training "
            "seeds as a robustness analysis before strong RL reproducibility claims."
        )
        payload = {
            "status": "LIMITED" if warnings else "ADEQUATE",
            "claim_gate_passed": n_patients >= 6 and "test" in splits,
            "baseline_experiment": baseline_name,
            "n_independent_patients": n_patients,
            "repeat_handling": "averaged within patient before inference",
            "minimum_two_sided_exact_p": minimum_two_sided_exact_p,
            "split_labels": splits,
            "warnings": warnings,
        }
    (out_dir / "statistical_design_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def ablation_deltas(df: pd.DataFrame, out_dir: Path, baseline_name: str | None) -> pd.DataFrame:
    if baseline_name is None:
        result = pd.DataFrame()
        result.to_csv(out_dir / "ablation_delta_vs_primary.csv", index=False)
        return result
    baseline = df[
        (df["experiment"] == baseline_name) & (df["mode"] == "switch_shield")
    ].copy()
    records = []
    variants = [name for name in sorted(df["experiment"].unique()) if name.startswith("ablation_")]
    for variant in variants:
        sub = df[
            (df["experiment"] == variant) & (df["mode"] == "switch_shield")
        ].copy()
        merged = sub.merge(
            baseline,
            on=["patient"],
            suffixes=("_variant", "_baseline"),
        )
        if merged.empty:
            continue
        for metric in PAPER_METRICS + ["ppo_takeover_rate", "switch_count"]:
            left = f"{metric}_variant"
            right = f"{metric}_baseline"
            if left not in merged.columns or right not in merged.columns:
                continue
            diff = pd.to_numeric(merged[left], errors="coerce") - pd.to_numeric(
                merged[right], errors="coerce"
            )
            diff = diff.dropna()
            if diff.empty:
                continue
            low, high = ci95(diff)
            records.append(
                {
                    "baseline": baseline_name,
                    "variant": variant,
                    "metric": metric,
                    "n_patients": int(len(diff)),
                    "positive_patients": int((diff > 0).sum()),
                    "negative_patients": int((diff < 0).sum()),
                    "mean_delta_variant_minus_baseline": float(diff.mean()),
                    "std_delta": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    result = pd.DataFrame(records)
    result.to_csv(out_dir / "ablation_delta_vs_primary.csv", index=False)
    return result


def load_switch_histories(suite_dir: Path) -> pd.DataFrame:
    records = []
    pattern = re.compile(
        r"(.+?)_(ppo(?:_shield)?|mpc(?:_shield)?|switch_nogate(?:_shield)?|switch(?:_shield)?)_seed(\d+)_policy_history\.csv$"
    )
    for path in sorted(suite_dir.glob("*/*_policy_history.csv")):
        exp = path.parent.name
        match = pattern.match(path.name)
        if not match:
            continue
        patient = match.group(1).replace("-", "#")
        mode = match.group(2)
        seed = int(match.group(3))
        df = pd.read_csv(path)
        rows = max(len(df), 1)
        selected = df.get("selected", pd.Series(dtype=str)).astype(str)
        base = {
            "experiment": exp,
            "patient": patient,
            "mode": mode,
            "seed": seed,
            "steps": int(len(df)),
            "ppo_rate": float((selected == "ppo").mean()) if len(df) else float("nan"),
            "mpc_rate": float((selected == "mpc").mean()) if len(df) else float("nan"),
        }
        if "safety_override" in df.columns:
            safety = df["safety_override"].astype(str).str.lower().isin(["true", "1", "yes"])
            base["safety_override_rate"] = float(safety.mean())
        else:
            base["safety_override_rate"] = float("nan")
        if "ppo_authorized" in df.columns:
            authorized = df["ppo_authorized"].astype(str).str.lower().isin(["true", "1", "yes"])
            base["ppo_authorized_rate"] = float(authorized.mean())
        if "ppo_effective" in df.columns:
            effective = df["ppo_effective"].astype(str).str.lower().isin(["true", "1", "yes"])
            base["ppo_effective_rate"] = float(effective.mean())
        records.append({**base, "reason": "__overall__", "rate": 1.0})
        if "reason" in df.columns:
            for reason, count in df["reason"].astype(str).value_counts().items():
                records.append({**base, "reason": reason, "rate": float(count / rows)})
        if "safety_reason" in df.columns:
            reasons = df["safety_reason"].fillna("").astype(str)
            for reason, count in reasons[reasons.ne("")].value_counts().items():
                records.append({**base, "reason": f"safety:{reason}", "rate": float(count / rows)})
    return pd.DataFrame(records)


def make_plots(overall: pd.DataFrame, deltas: pd.DataFrame, out_dir: Path, baseline_name: str | None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] skip plots: {exc}")
        return

    if baseline_name and not overall.empty:
        sub = overall[overall["experiment"].eq(baseline_name)].copy()
        modes = [
            mode
            for mode in [
                "ppo",
                "ppo_shield",
                "mpc",
                "mpc_shield",
                "switch_nogate_shield",
                "switch",
                "switch_shield",
            ]
            if mode in set(sub["mode"])
        ]
        if modes:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
            x = np.arange(len(modes))
            labels = [mode.replace("_shield", "+shield") for mode in modes]
            for ax, metric, title in [
                (axes[0], "tir", "Time in range (higher is better)"),
                (axes[1], "tbr54", "Time below 54 mg/dL (lower is better)"),
            ]:
                rows = [sub[sub["mode"].eq(mode)].iloc[0] for mode in modes]
                means = np.asarray([row.get(f"{metric}_mean", np.nan) for row in rows])
                lows = np.asarray([row.get(f"{metric}_ci95_low", np.nan) for row in rows])
                highs = np.asarray([row.get(f"{metric}_ci95_high", np.nan) for row in rows])
                errors = np.nan_to_num(
                    np.vstack([means - lows, highs - means]),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                ax.bar(x, means, yerr=errors, capsize=3)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=35, ha="right")
                ax.set_ylabel("Time (%)")
                ax.set_title(title)
                ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "primary_efficacy_safety.pdf")
            fig.savefig(out_dir / "primary_efficacy_safety.png", dpi=220)
            plt.close(fig)

    if not deltas.empty:
        sub = deltas[deltas["metric"].isin(["tir", "tbr70", "tbr54", "tar250"])]
        if not sub.empty:
            pivot = sub.pivot_table(
                index="variant",
                columns="metric",
                values="mean_delta_variant_minus_baseline",
                aggfunc="mean",
            ).fillna(0.0)
            fig, ax = plt.subplots(figsize=(11, max(4.5, 0.35 * len(pivot))))
            pivot[[c for c in ["tir", "tbr70", "tbr54", "tar250"] if c in pivot.columns]].plot.barh(ax=ax)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("Delta vs primary Switch+shield")
            ax.set_title("Switch ablation deltas")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "ablation_deltas.png", dpi=220)
            plt.close(fig)


def paper_table(overall: pd.DataFrame, out_dir: Path, baseline_name: str | None) -> pd.DataFrame:
    if baseline_name is None or overall.empty:
        result = pd.DataFrame()
        result.to_csv(out_dir / "paper_primary_table.csv", index=False)
        return result
    sub = overall[overall["experiment"].eq(baseline_name)].copy()
    rows = []
    for _, row in sub.iterrows():
        item = {"mode": row["mode"], "n": int(row["n"])}
        for metric in PAPER_METRICS:
            mean = row.get(f"{metric}_mean", np.nan)
            std = row.get(f"{metric}_std", np.nan)
            low = row.get(f"{metric}_ci95_low", np.nan)
            high = row.get(f"{metric}_ci95_high", np.nan)
            item[f"{metric}_mean_std"] = f"{mean:.3f}±{std:.3f}" if pd.notna(mean) else ""
            item[f"{metric}_ci95"] = f"[{low:.3f}, {high:.3f}]" if pd.notna(low) else ""
        rows.append(item)
    result = pd.DataFrame(rows).sort_values("mode")
    result.to_csv(out_dir / "paper_primary_table.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze upgraded PPO-MPC paper experiment suite.")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--baseline-experiment", default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Generate diagnostic outputs despite completeness failures; never use for paper tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    if not suite_dir.is_absolute():
        suite_dir = Path.cwd() / suite_dir
    analysis_dir = suite_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    claim_artifacts = [
        analysis_dir / "all_summary_rows.csv",
        analysis_dir / "patient_mode_mean.csv",
        analysis_dir / "overall_by_experiment_mode.csv",
        analysis_dir / "paired_tests.csv",
        analysis_dir / "ablation_delta_vs_primary.csv",
        analysis_dir / "statistical_design_audit.json",
        analysis_dir / "switch_reason_rates.csv",
        analysis_dir / "compatibility_behavior.csv",
        analysis_dir / "paper_primary_table.csv",
        analysis_dir / "primary_efficacy_safety.pdf",
        analysis_dir / "primary_efficacy_safety.png",
        analysis_dir / "ablation_deltas.png",
    ]
    withheld_marker = analysis_dir / "claim_outputs_withheld.json"
    for artifact in claim_artifacts:
        artifact.unlink(missing_ok=True)
    withheld_marker.write_text(
        json.dumps(
            {
                "status": "WITHHELD",
                "reason": "analysis has not completed strict validation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(suite_dir)
    df = load_summaries(suite_dir, manifest, allow_partial=args.allow_partial)
    if df.empty:
        raise SystemExit(f"No summary.csv files found under {suite_dir}")
    completeness_issues = validate_suite_completeness(
        suite_dir, manifest, df, analysis_dir
    )
    status = "PARTIAL" if completeness_issues else "COMPLETE"
    diagnostic_counts = write_partial_diagnostics(df, manifest, analysis_dir)
    (analysis_dir / "analysis_status.json").write_text(
        json.dumps(
            {
                "status": status,
                "issues": completeness_issues,
                "claim_outputs": "WITHHELD",
                "diagnostics_generated": True,
                "censored_metrics": bool(completeness_issues),
                **diagnostic_counts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if completeness_issues and not args.allow_partial:
        withheld_marker.write_text(
            json.dumps(
                {
                    "status": "WITHHELD",
                    "reason": "suite completeness validation failed",
                    "issues": completeness_issues,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise SystemExit(
            "Suite completeness validation failed:\n- "
            + "\n- ".join(completeness_issues)
        )
    if completeness_issues:
        withheld_marker.write_text(
            json.dumps(
                {
                    "status": "WITHHELD",
                    "reason": "suite completeness validation failed",
                    "issues": completeness_issues,
                    "diagnostics_only": True,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Analyzed suite: {suite_dir}")
        print("Validation status: PARTIAL")
        print("Claim outputs: WITHHELD (diagnostics only)")
        print(f"Outputs: {analysis_dir}")
        return
    df.to_csv(analysis_dir / "all_summary_rows.csv", index=False)

    claim_df = df[~df["role"].eq("diagnostic")].copy()
    patient_df = patient_means(claim_df, analysis_dir)
    overall = aggregate(patient_df, analysis_dir)
    compatibility_behavior(claim_df, analysis_dir)
    paired_tests(patient_df, analysis_dir)
    baseline = args.baseline_experiment or find_primary_safety(patient_df)
    design_audit = statistical_design_audit(patient_df, analysis_dir, baseline)
    deltas = ablation_deltas(patient_df, analysis_dir, baseline)
    histories = load_switch_histories(suite_dir)
    histories.to_csv(analysis_dir / "switch_reason_rates.csv", index=False)
    claim_gate_passed = not completeness_issues and bool(
        design_audit.get("claim_gate_passed")
    )
    (analysis_dir / "analysis_status.json").write_text(
        json.dumps(
            {
                "status": status,
                "issues": completeness_issues,
                "claim_outputs": "RELEASED" if claim_gate_passed else "WITHHELD",
                "statistical_design_status": design_audit.get("status"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if not claim_gate_passed:
        withheld_marker.write_text(
            json.dumps(
                {
                    "status": "WITHHELD",
                    "reason": (
                        "suite completeness validation failed"
                        if completeness_issues
                        else "statistical claim gate requires locked test data and at least 6 independent patients"
                    ),
                    "issues": completeness_issues,
                    "statistical_design": design_audit,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    else:
        withheld_marker.unlink(missing_ok=True)
        paper_table(overall, analysis_dir, baseline)
        make_plots(overall, deltas, analysis_dir, baseline)

    print(f"Analyzed suite: {suite_dir}")
    print(f"Validation status: {status}")
    print(f"Primary baseline: {baseline}")
    print(f"Claim outputs: {'RELEASED' if claim_gate_passed else 'WITHHELD'}")
    print(f"Outputs: {analysis_dir}")


if __name__ == "__main__":
    main()
