from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ppo_mpc.compatibility import CompatibilityGateConfig  # noqa: E402
from ppo_mpc.controllers import (  # noqa: E402
    ArxMpcConfig,
    PpoBasalController,
    PpoMpcSwitchController,
    SharedSafetyGovernorController,
    SwitchConfig,
    make_arx_rls_zone_mpc,
    require_finite_cgm,
)
from ppo_mpc.metrics import summarize_closed_loop  # noqa: E402
from ppo_mpc.paths import MPC_ASSETS_ROOT, add_asset_paths  # noqa: E402
from ppo_mpc.provenance import (  # noqa: E402
    assert_runtime_matches_lock,
    runtime_fingerprint,
    runtime_identity,
    sha256_file,
    source_tree_sha256,
)
from ppo_mpc.safety import SafetyGovernorConfig  # noqa: E402
from ppo_mpc.scenarios import make_paper_scenario  # noqa: E402

add_asset_paths()

from simglucose.actuator.pump import InsulinPump  # noqa: E402
from simglucose.patient.t1dpatient import T1DPatient  # noqa: E402
from simglucose.sensor.cgm import CGMSensor  # noqa: E402
from simglucose.simulation.env import T1DSimEnv  # noqa: E402


PROTOCOL_VERSION = "causal_model_compat_v4"


def parse_patients(value: str) -> list[str]:
    if value.lower() == "adult":
        return [f"adult#{idx:03d}" for idx in range(1, 11)]
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_modes(value: str) -> list[str]:
    aliases = {
        "ppo+shield": "ppo_shield",
        "mpc+shield": "mpc_shield",
        "switch+shield": "switch_shield",
        "switch_nogate+shield": "switch_nogate_shield",
    }
    supported = {
        "ppo", "mpc", "switch", "switch_nogate",
        "ppo_shield", "mpc_shield", "switch_shield", "switch_nogate_shield",
    }
    modes: list[str] = []
    for item in value.split(","):
        mode = aliases.get(item.strip(), item.strip())
        if not mode:
            continue
        if mode not in supported:
            raise ValueError(f"unsupported mode: {mode}; choose from {sorted(supported)}")
        if mode not in modes:
            modes.append(mode)
    return modes


def parse_positive_float_tuple(value: str) -> tuple[float, ...]:
    parsed = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    if not parsed or any(not math.isfinite(item) or item <= 0.0 for item in parsed):
        raise ValueError("prediction horizons must be finite positive minutes")
    return parsed


def patient_specific_theta_overlap(param_file: str | Path, patients: list[str]) -> set[str]:
    path = Path(param_file)
    if not path.is_absolute():
        path = MPC_ASSETS_ROOT / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot audit ARX theta file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"ARX theta file must contain a JSON object: {path}")
    return set(patients).intersection(payload.keys())


def resolve_local_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_compatibility_calibration(
    path_value: str | Path,
    *,
    expected_sha256: str,
    current_source_sha256: str,
    current_checkpoint_sha256: str,
    patients: list[str],
    reference_arx_sha256: str = "",
    current_requirements_lock_sha256: str = "",
    allow_patient_overlap: bool = False,
) -> tuple[dict, CompatibilityGateConfig, str, Path]:
    path = resolve_local_path(path_value)
    if not path.exists():
        raise SystemExit(f"Compatibility calibration file does not exist: {path}")
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise SystemExit(
            "Compatibility calibration hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Invalid compatibility calibration JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Compatibility calibration must be a JSON object.")
    gate_payload = payload.get("gate", payload.get("gate_config", payload))
    if not isinstance(gate_payload, dict):
        raise SystemExit("Compatibility calibration gate payload is invalid.")
    gate_payload = dict(gate_payload)
    if not gate_payload.get("calibration_id"):
        gate_payload["calibration_id"] = payload.get("calibration_id", "")
    try:
        config = CompatibilityGateConfig.from_mapping(gate_payload)
    except Exception as exc:
        raise SystemExit(f"Invalid compatibility gate configuration: {exc}") from exc
    source_patients = {
        str(item) for item in payload.get("source_patients", [])
        if str(item).strip()
    }
    allowed_source_patients = {
        f"adult#{index:03d}" for index in range(1, 6)
    }
    if source_patients != allowed_source_patients:
        raise SystemExit(
            "Compatibility calibration must use exactly adult#001-005; "
            f"observed {sorted(source_patients)}"
        )
    if str(payload.get("split_label", "")) != "development":
        raise SystemExit("Compatibility calibration must be development-only.")
    if str(payload.get("source_mode", "")) != "switch_nogate_shield":
        raise SystemExit(
            "Compatibility calibration must come from switch_nogate_shield."
        )
    overlap = source_patients.intersection(patients)
    if allow_patient_overlap and not set(patients).issubset(allowed_source_patients):
        raise SystemExit(
            "Compatibility source-overlap override is development-only and "
            "requires evaluation patients adult#001-005."
        )
    if overlap and not allow_patient_overlap:
        raise SystemExit(
            "Compatibility calibration patient overlap detected: "
            + ", ".join(sorted(overlap))
        )
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    checks = [
        ("source tree", provenance.get("source_tree_sha256"), current_source_sha256),
        ("checkpoint", provenance.get("checkpoint_sha256"), current_checkpoint_sha256),
    ]
    if reference_arx_sha256:
        checks.append(
            (
                "reference ARX parameter file",
                provenance.get("arx_param_sha256"),
                reference_arx_sha256,
            )
        )
    if current_requirements_lock_sha256:
        checks.append(
            (
                "requirements lock",
                provenance.get("requirements_lock_sha256"),
                current_requirements_lock_sha256,
            )
        )
    for label, expected, current in checks:
        if not expected or str(expected) != str(current):
            raise SystemExit(
                f"Compatibility calibration provenance mismatch for {label}: "
                f"expected {expected or '<missing>'}, got {current}"
            )
    return payload, config, actual_sha256, path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def causal_controller_info(
    raw_info: dict,
    env: T1DSimEnv,
    meal_announcement: str,
) -> dict:
    """Expose only causal controller inputs and explicitly sanitize meal CHO."""
    announced = meal_announcement == "announced"
    sample_time = max(float(raw_info.get("sample_time", env.sample_time)), 1e-8)
    announced_meal = announced_meal_in_next_sample(env, announced)
    payload = {
        "sample_time": sample_time,
        "patient_name": raw_info.get("patient_name", env.patient.name),
        "time": raw_info.get("time", env.time),
        "bolus_rate": raw_info.get("bolus_rate", 0.0),
        "meal_is_announced": announced,
        "announced_meal": float(announced_meal if announced else 0.0),
        "meal": float(announced_meal / sample_time if announced else 0.0),
    }
    return payload

def announced_meal_in_next_sample(env: T1DSimEnv, enabled: bool) -> float:
    if not enabled:
        return 0.0
    total = 0.0
    for minute_offset in range(int(env.sample_time)):
        action = env.scenario.get_action(env.time + timedelta(minutes=minute_offset))
        total += float(getattr(action, "meal", 0.0))
    return float(total)


def make_env(patient_name: str, args: argparse.Namespace, seed: int) -> tuple[T1DSimEnv, timedelta]:
    patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed + 101)
    sensor = CGMSensor.withName(args.sensor_name, seed=seed + 202)
    pump = InsulinPump.withName(args.pump_name)
    start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M")
    scenario = make_paper_scenario(
        args.scenario,
        start_time,
        seed=seed + 303,
        patient_name=patient_name,
    )
    return T1DSimEnv(patient, sensor, pump, scenario), timedelta(hours=float(args.hours))


def make_mpc_config(args: argparse.Namespace) -> ArxMpcConfig:
    return ArxMpcConfig(
        param_file=args.arx_param_file,
        target_glucose=args.arx_target_glucose,
        pred_horizon_min=args.arx_pred_horizon_min,
        control_horizon_min=args.arx_control_horizon_min,
        P0=args.arx_P0,
        lam_base=args.arx_lam_base,
        kappa=args.arx_kappa,
        rls_err_clip=args.arx_rls_err_clip,
        rls_err_deadzone=args.arx_rls_err_deadzone,
        rls_phi_min=args.arx_rls_phi_min,
        min_order=args.arx_min_order,
        u_delay_steps=args.arx_u_delay_steps,
        m_delay_steps=args.arx_m_delay_steps,
        meal_filter_tau_min=args.arx_meal_filter_tau_min,
        meal_release_g_per_min=args.arx_meal_release_g_per_min,
        meal_c_max=args.arx_meal_c_max,
        meal_forecast_steps=args.arx_meal_forecast_steps,
        enable_meal_bolus=args.meal_announcement == "announced",
        meal_bolus_scale=args.arx_meal_bolus_scale,
        workspace_mpc_variant=args.mpc_variant,
        use_iob_constraint=args.mpc_use_iob_constraint,
        max_insulin_units_per_sample=args.mpc_max_insulin_units_per_sample,
        max_insulin_tdi_fraction=args.mpc_max_insulin_tdi_fraction,
        model_gain_factor=args.mpc_model_gain_factor,
        r_plus_scale=args.mpc_r_plus_scale,
        enable_low_glucose_suspend=args.mpc_low_glucose_suspend,
    )


def make_controller(mode: str, agent: PPOAgent, args: argparse.Namespace):
    shielded = mode.endswith("_shield")
    base_mode = mode.removesuffix("_shield")
    mpc = make_arx_rls_zone_mpc(make_mpc_config(args))
    if base_mode == "mpc":
        controller = mpc
    elif base_mode == "ppo":
        controller = PpoBasalController(
            agent,
            max_basal_multiplier=args.max_basal_multiplier,
            deterministic=True,
            bolus_source=mpc,
        )
    elif base_mode in ("switch", "switch_nogate"):
        gate_enabled = (
            base_mode == "switch"
            and bool(getattr(args, "enable_switch_compatibility_gate", False))
        )
        compatibility_cfg = getattr(args, "_compatibility_config", None)
        switch_cfg = SwitchConfig(
            warmup_steps=args.switch_warmup_steps,
            max_action_gap_basal=args.switch_max_action_gap_basal,
            max_basal_multiplier=args.max_basal_multiplier,
            deterministic_ppo=True,
            ppo_min_glucose=args.switch_ppo_min_glucose,
            ppo_min_projected_glucose=args.switch_ppo_min_projected_glucose,
            ppo_min_rate=args.switch_ppo_min_rate,
            risk_low_glucose=args.switch_risk_low_glucose,
            risk_current_glucose=args.switch_risk_current_glucose,
            risk_projected_glucose=args.switch_risk_projected_glucose,
            risk_fast_fall_rate=args.switch_risk_fast_fall_rate,
            risk_prediction_horizons_min=parse_positive_float_tuple(
                args.switch_risk_horizons_min
            ),
            risk_strategy=args.switch_risk_strategy,
            high_risk_arbitration=args.enable_switch_high_risk_arbitration,
            high_risk_glucose=args.switch_high_risk_glucose,
            high_risk_projected_glucose=args.switch_high_risk_projected_glucose,
            high_risk_rising_rate=args.switch_high_risk_rising_rate,
            high_risk_strategy=args.switch_high_risk_strategy,
            high_risk_iob_taper_start_ratio=(
                args.switch_high_risk_iob_taper_start_ratio
            ),
            high_risk_iob_max_ratio=args.switch_high_risk_iob_max_ratio,
            high_risk_iob_prediction_enabled=(
                args.enable_switch_high_risk_iob_prediction
            ),
            high_risk_iob_prediction_horizon_min=(
                args.switch_high_risk_iob_prediction_horizon_min
            ),
            high_risk_iob_prediction_effect_fraction=(
                args.switch_high_risk_iob_prediction_effect_fraction
            ),
            high_risk_iob_prediction_guard_glucose=(
                args.switch_high_risk_iob_prediction_guard_glucose
            ),
            high_risk_correction_enabled=args.enable_switch_high_risk_correction,
            high_risk_correction_target_glucose=(
                args.switch_high_risk_correction_target_glucose
            ),
            high_risk_correction_horizon_min=(
                args.switch_high_risk_correction_horizon_min
            ),
            high_risk_correction_max_basal_multiplier=(
                args.switch_high_risk_correction_max_basal_multiplier
            ),
            # Fair experiments place the same shield outside every base controller.
            safety_governor=False,
            safety_prediction_horizons_min=parse_positive_float_tuple(
                args.switch_safety_horizons_min
            ),
            safety_suspend_glucose=args.switch_safety_suspend_glucose,
            safety_low_glucose=args.switch_safety_low_glucose,
            safety_projected_low_glucose=args.switch_safety_projected_low_glucose,
            safety_falling_glucose=args.switch_safety_falling_glucose,
            safety_falling_rate=args.switch_safety_falling_rate,
            safety_low_basal_fraction=args.switch_safety_low_basal_fraction,
            safety_falling_basal_fraction=args.switch_safety_falling_basal_fraction,
            safety_iob_excess_ratio=args.switch_safety_iob_excess_ratio,
            safety_iob_guard_projected_glucose=args.switch_safety_iob_guard_projected_glucose,
            safety_iob_guard_basal_fraction=args.switch_safety_iob_guard_basal_fraction,
            compatibility_gate_enabled=gate_enabled,
            compatibility_gate_config=compatibility_cfg,
            compatibility_calibration_id=str(
                getattr(args, "_compatibility_calibration_id", "")
            ),
        )
        controller = PpoMpcSwitchController(agent, mpc, switch_cfg)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    if not shielded:
        return controller
    safety_cfg = SafetyGovernorConfig(
        enabled=True,
        prediction_horizons_min=parse_positive_float_tuple(
            args.switch_safety_horizons_min
        ),
        suspend_glucose=args.switch_safety_suspend_glucose,
        low_glucose=args.switch_safety_low_glucose,
        projected_low_glucose=args.switch_safety_projected_low_glucose,
        falling_glucose=args.switch_safety_falling_glucose,
        falling_rate=args.switch_safety_falling_rate,
        low_basal_fraction=args.switch_safety_low_basal_fraction,
        falling_basal_fraction=args.switch_safety_falling_basal_fraction,
        iob_excess_ratio=args.switch_safety_iob_excess_ratio,
        iob_guard_projected_glucose=args.switch_safety_iob_guard_projected_glucose,
        iob_guard_basal_fraction=args.switch_safety_iob_guard_basal_fraction,
        high_guard_enabled=not args.disable_switch_safety_high_guard,
        high_glucose=args.switch_safety_high_glucose,
        severe_high_glucose=args.switch_safety_severe_high_glucose,
        high_rising_rate=args.switch_safety_high_rising_rate,
        correction_target_glucose=args.switch_safety_correction_target_glucose,
        correction_horizon_min=args.switch_safety_correction_horizon_min,
        high_max_basal_multiplier=args.switch_safety_high_max_basal_multiplier,
    )
    return SharedSafetyGovernorController(controller, safety_cfg, base_mode)


def run_episode(
    mode: str,
    agent: PPOAgent,
    patient_name: str,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
):
    env, sim_time = make_env(patient_name, args, seed)
    controller = make_controller(mode, agent, args)
    controller.reset()
    obs, reward, done, info = env.reset()
    end_time = env.scenario.start_time + sim_time
    completed_steps = 0
    terminated_early = False
    terminal_event = False
    terminal_reason = "horizon"
    terminal_bg = float("nan")
    while env.time < end_time:
        require_finite_cgm(obs)
        controller_info = causal_controller_info(dict(info), env, args.meal_announcement)
        action = controller.policy(obs, reward, done, **controller_info)
        obs, reward, done, info = env.step(action)
        completed_steps += 1
        if done:
            terminal_event = True
            terminal_bg = float(getattr(obs, "BG", float("nan")))
            if not math.isfinite(terminal_bg):
                terminal_reason = "nonfinite_bg"
            elif terminal_bg < 10.0:
                terminal_reason = "bg_below_10"
            elif terminal_bg > 600.0:
                terminal_reason = "bg_above_600"
            else:
                terminal_reason = "environment_terminal"
            terminated_early = env.time < end_time
            break
    df = env.show_history()
    run_stem = f"{patient_name.replace('#', '-')}_{mode}_seed{seed}"
    df.to_csv(output_dir / f"{run_stem}_rollout.csv", index=False)

    switch_df = pd.DataFrame()
    if hasattr(controller, "switch_history"):
        switch_df = controller.switch_history()
    elif hasattr(controller, "history"):
        switch_df = pd.DataFrame(controller.history)
    if not switch_df.empty:
        switch_df.to_csv(output_dir / f"{run_stem}_policy_history.csv", index=False)

    summary = summarize_closed_loop(
        df,
        patient=patient_name,
        mode=mode,
        scenario=args.scenario,
        seed=seed,
        switch_history=switch_df,
    )
    sample_time = float(env.sample_time)
    expected_steps = int(math.ceil(float(args.hours) * 60.0 / sample_time))
    claim_valid = (
        completed_steps == expected_steps
        and not terminal_event
        and int(summary.get("nonfinite_steps", 0)) == 0
        and int(summary.get("cgm_nonfinite_steps", 0)) == 0
        and args.meal_announcement == "unannounced"
    )
    last_rollout = df.iloc[-1] if not df.empty else pd.Series(dtype=float)
    last_policy = switch_df.iloc[-1] if not switch_df.empty else pd.Series(dtype=float)
    summary.update(
        {
            "base_controller": mode.removesuffix("_shield"),
            "shield": "shared" if mode.endswith("_shield") else "none",
            "meal_announcement": args.meal_announcement,
            "completed_steps": int(completed_steps),
            "expected_steps": expected_steps,
            "terminated_early": bool(terminated_early),
            "terminal_event": bool(terminal_event),
            "terminal_bg": terminal_bg,
            "termination_reason": terminal_reason,
            "terminal_step": int(completed_steps) if terminal_event else None,
            "terminal_time": str(env.time) if terminal_event else "",
            "last_endpoint_bg": float(last_rollout.get("BG", float("nan"))),
            "last_endpoint_cgm": float(last_rollout.get("CGM", float("nan"))),
            "last_action_basal": float(
                last_policy.get("selected_basal", float("nan"))
            ),
            "last_action_bolus": float(
                last_policy.get("selected_bolus", float("nan"))
            ),
            "last_selector_reason": str(last_policy.get("reason", "")),
            "last_safety_reason": str(last_policy.get("safety_reason", "")),
            "last_causal_iob_units": float(
                last_policy.get("causal_iob_units", float("nan"))
            ),
            "last_safety_projected_min": float(
                last_policy.get("safety_projected_min", float("nan"))
            ),
            "last_model_residual_raw_mgdl": float(
                last_policy.get("model_residual_raw_mgdl", float("nan"))
            ),
            "last_model_npe_ewma": float(
                last_policy.get("model_npe_ewma", float("nan"))
            ),
            "last_model_compatibility_score": float(
                last_policy.get("model_compatibility_score", float("nan"))
            ),
            "last_model_compatible": bool(
                last_policy.get("model_compatible", False)
            ),
            "last_mpc_model_authorized": bool(
                last_policy.get("mpc_model_authorized", False)
            ),
            "compatibility_gate_enabled": bool(
                last_policy.get("compatibility_gate_enabled", False)
            ),
            "compatibility_calibration_id": str(
                getattr(args, "_compatibility_calibration_id", "")
            ),
            "model_condition": str(getattr(args, "model_condition", "nominal")),
            "arx_prior_label": str(getattr(args, "arx_prior_label", "")),
            "protocol_version": PROTOCOL_VERSION,
            "claim_valid": bool(claim_valid),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO, ARX-RLS-MPC, and PPO-MPC switching.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patients", default="adult#001")
    parser.add_argument(
        "--modes",
        default="ppo,ppo_shield,mpc,mpc_shield,switch,switch_shield",
    )
    parser.add_argument("--scenario", default="A")
    parser.add_argument("--meal-announcement", choices=["announced", "unannounced"], default="unannounced")
    parser.add_argument(
        "--allow-experimental-announced",
        action="store_true",
        help="Diagnostic only; announced runs are never claim-valid in the causal protocol.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=22000)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--start-time", default="2018-01-01 00:00")
    parser.add_argument("--sensor-name", choices=["GuardianRT", "Dexcom", "Navigator"], default="GuardianRT")
    parser.add_argument("--pump-name", default="Insulet")
    parser.add_argument("--max-basal-multiplier", type=float, default=5.0)
    parser.add_argument("--output-dir", default="results/ppo_mpc_eval")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--protocol-fingerprint",
        default="standalone",
        help="Suite-generated identity used to reject stale resume artifacts.",
    )
    parser.add_argument("--expected-source-tree-sha256", default="")
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--expected-arx-param-sha256", default="")
    parser.add_argument("--expected-requirements-lock-sha256", default="")
    parser.add_argument("--expected-runtime-fingerprint", default="")
    parser.add_argument(
        "--compatibility-calibration-file",
        default="",
        help="Frozen development-set calibration JSON for the causal model-compatibility gate.",
    )
    parser.add_argument("--expected-compatibility-calibration-sha256", default="")
    parser.add_argument(
        "--allow-compatibility-source-overlap",
        action="store_true",
        help="Development smoke only; formal validation rejects this override.",
    )
    parser.add_argument(
        "--enable-switch-compatibility-gate",
        action="store_true",
        help="Enable the frozen causal compatibility gate for Switch; fail closed when absent.",
    )
    parser.add_argument("--model-condition", default="nominal")
    parser.add_argument("--arx-prior-label", default="matched_adult_population")
    parser.add_argument("--compatibility-reference-arx-sha256", default="")
    parser.add_argument(
        "--requirements-lock",
        default=str(ROOT / "requirements-lock-server-py310-cu126.txt"),
    )
    parser.add_argument(
        "--allow-legacy-ppo-checkpoint",
        action="store_true",
        help="Diagnostic-only override for checkpoints trained with clipped-Normal PPO.",
    )

    parser.add_argument(
        "--arx-param-file",
        default="output/arx_rls_training_30/population_theta_adult001_005.json",
    )
    parser.add_argument(
        "--allow-patient-specific-theta",
        action="store_true",
        help="Diagnostic only: permit offline theta entries for evaluation patients.",
    )
    parser.add_argument("--arx-target-glucose", type=float, default=110.0)
    parser.add_argument("--arx-pred-horizon-min", type=float, default=120.0)
    parser.add_argument("--arx-control-horizon-min", type=float, default=45.0)
    parser.add_argument("--arx-P0", type=float, default=100.0)
    parser.add_argument("--arx-lam-base", type=float, default=0.999)
    parser.add_argument("--arx-kappa", type=float, default=0.002)
    parser.add_argument("--arx-rls-err-clip", type=float, default=20.0)
    parser.add_argument("--arx-rls-err-deadzone", type=float, default=20.0)
    parser.add_argument("--arx-rls-phi-min", type=float, default=0.001)
    parser.add_argument("--arx-min-order", type=int, default=6)
    parser.add_argument("--arx-u-delay-steps", type=int, default=3)
    parser.add_argument("--arx-m-delay-steps", type=int, default=1)
    parser.add_argument("--arx-meal-filter-tau-min", type=float, default=15.0)
    parser.add_argument("--arx-meal-release-g-per-min", type=float, default=5.0)
    parser.add_argument("--arx-meal-c-max", type=float, default=5.0)
    parser.add_argument("--arx-meal-forecast-steps", type=int, default=8)
    parser.add_argument("--arx-meal-bolus-scale", type=float, default=0.75)

    parser.add_argument("--mpc-variant", choices=["previous", "velocity", "adaptive"], default="adaptive")
    parser.add_argument("--mpc-use-iob-constraint", action="store_true")
    parser.add_argument("--mpc-max-insulin-units-per-sample", type=float, default=1.0)
    parser.add_argument("--mpc-max-insulin-tdi-fraction", type=float, default=None)
    parser.add_argument("--mpc-model-gain-factor", type=float, default=1.0)
    parser.add_argument("--mpc-r-plus-scale", type=float, default=1.0)
    parser.add_argument("--mpc-low-glucose-suspend", action="store_true")

    parser.add_argument("--switch-warmup-steps", type=int, default=12)
    parser.add_argument("--switch-risk-low-glucose", type=float, default=90.0)
    parser.add_argument("--switch-risk-current-glucose", type=float, default=130.0)
    parser.add_argument("--switch-risk-projected-glucose", type=float, default=130.0)
    parser.add_argument("--switch-risk-fast-fall-rate", type=float, default=-2.0)
    parser.add_argument("--switch-risk-horizons-min", default="30,60,90")
    parser.add_argument(
        "--switch-risk-strategy",
        choices=["lower_insulin", "ppo"],
        default="lower_insulin",
    )
    parser.add_argument(
        "--enable-switch-high-risk-arbitration",
        action="store_true",
        help=(
            "Enable the exploratory high/rising branch that selects the higher "
            "authorized PPO/MPC insulin candidate."
        ),
    )
    parser.add_argument("--switch-high-risk-glucose", type=float, default=180.0)
    parser.add_argument(
        "--switch-high-risk-projected-glucose", type=float, default=200.0
    )
    parser.add_argument("--switch-high-risk-rising-rate", type=float, default=0.1)
    parser.add_argument(
        "--switch-high-risk-strategy",
        choices=["higher_insulin"],
        default="higher_insulin",
    )
    parser.add_argument(
        "--switch-high-risk-iob-taper-start-ratio",
        type=float,
        default=1.0e9,
        help="Start linearly tapering the high-risk MPC candidate at this IOB ratio.",
    )
    parser.add_argument(
        "--switch-high-risk-iob-max-ratio",
        type=float,
        default=1.0e9,
        help=(
            "Return the high-risk branch to PPO when causal IOB reaches this "
            "multiple of steady-state basal IOB."
        ),
    )
    parser.add_argument(
        "--enable-switch-high-risk-iob-prediction", action="store_true"
    )
    parser.add_argument(
        "--switch-high-risk-iob-prediction-horizon-min", type=float, default=90.0
    )
    parser.add_argument(
        "--switch-high-risk-iob-prediction-effect-fraction", type=float, default=0.5
    )
    parser.add_argument(
        "--switch-high-risk-iob-prediction-guard-glucose", type=float, default=130.0
    )
    parser.add_argument(
        "--enable-switch-high-risk-correction",
        action="store_true",
        help=(
            "Shape the high-risk MPC candidate with a causal IOB-credited "
            "correction floor before PPO/MPC arbitration."
        ),
    )
    parser.add_argument(
        "--switch-high-risk-correction-target-glucose", type=float, default=110.0
    )
    parser.add_argument(
        "--switch-high-risk-correction-horizon-min", type=float, default=60.0
    )
    parser.add_argument(
        "--switch-high-risk-correction-max-basal-multiplier",
        type=float,
        default=8.0,
    )
    parser.add_argument("--switch-trend-horizon-steps", type=int, default=6, help=argparse.SUPPRESS)
    parser.add_argument("--switch-max-action-gap-basal", type=float, default=1.25, help=argparse.SUPPRESS)
    parser.add_argument("--switch-ppo-min-glucose", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--switch-ppo-min-projected-glucose", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--switch-ppo-min-rate", type=float, default=-999.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--disable-switch-safety-governor",
        dest="legacy_disable_switch_safety_governor",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--switch-safety-suspend-glucose", type=float, default=75.0)
    parser.add_argument("--switch-safety-horizons-min", default="30,60,90")
    parser.add_argument("--switch-safety-low-glucose", type=float, default=90.0)
    parser.add_argument("--switch-safety-projected-low-glucose", type=float, default=90.0)
    parser.add_argument("--switch-safety-falling-glucose", type=float, default=130.0)
    parser.add_argument("--switch-safety-falling-rate", type=float, default=-0.25)
    parser.add_argument("--switch-safety-low-basal-fraction", type=float, default=0.0)
    parser.add_argument("--switch-safety-falling-basal-fraction", type=float, default=0.5)
    parser.add_argument("--switch-safety-iob-excess-ratio", type=float, default=1.25)
    parser.add_argument(
        "--switch-safety-iob-guard-projected-glucose", type=float, default=130.0
    )
    parser.add_argument(
        "--switch-safety-iob-guard-basal-fraction", type=float, default=0.0
    )
    parser.add_argument("--switch-safety-high-glucose", type=float, default=250.0)
    parser.add_argument(
        "--switch-safety-severe-high-glucose", type=float, default=300.0
    )
    parser.add_argument("--switch-safety-high-rising-rate", type=float, default=0.25)
    parser.add_argument(
        "--switch-safety-correction-target-glucose", type=float, default=180.0
    )
    parser.add_argument(
        "--switch-safety-correction-horizon-min", type=float, default=180.0
    )
    parser.add_argument(
        "--switch-safety-high-max-basal-multiplier", type=float, default=5.0
    )
    parser.add_argument("--disable-switch-safety-high-guard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.meal_announcement == "announced" and not args.allow_experimental_announced:
        raise SystemExit(
            "Announced evaluation is withheld in the causal protocol because its "
            "bolus/state contract is not yet shared with training. Use unannounced mode."
        )
    if args.mpc_use_iob_constraint:
        raise SystemExit(
            "--mpc-use-iob-constraint is disabled in the causal protocol because the "
            "simulator IOB is hidden physiological truth. Implement and validate "
            "an action-history estimator before enabling this constraint."
        )
    requirements_lock = Path(args.requirements_lock)
    if not requirements_lock.is_absolute():
        requirements_lock = ROOT / requirements_lock
    try:
        runtime_lock_audit = assert_runtime_matches_lock(requirements_lock)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    current_source_sha256 = source_tree_sha256(str(ROOT))
    current_lock_sha256 = sha256_file(requirements_lock)
    current_runtime_fingerprint = runtime_fingerprint()
    current_checkpoint_sha256 = sha256_file(args.checkpoint)
    arx_path = Path(args.arx_param_file)
    if not arx_path.is_absolute():
        arx_path = MPC_ASSETS_ROOT / arx_path
    current_arx_sha256 = sha256_file(arx_path)
    provenance_checks = [
        ("source tree", args.expected_source_tree_sha256, current_source_sha256),
        ("checkpoint", args.expected_checkpoint_sha256, current_checkpoint_sha256),
        ("ARX parameter file", args.expected_arx_param_sha256, current_arx_sha256),
        ("requirements lock", args.expected_requirements_lock_sha256, current_lock_sha256),
        ("runtime", args.expected_runtime_fingerprint, current_runtime_fingerprint),
    ]
    for label, expected, current in provenance_checks:
        if expected and expected != current:
            raise SystemExit(
                f"Protocol provenance mismatch for {label}: expected {expected}, got {current}"
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from ppo_mpc.ppo import PPOAgent

    agent = PPOAgent.load(args.checkpoint, device=args.device)
    checkpoint_distribution = getattr(
        agent, "checkpoint_action_distribution", "legacy_clipped_normal_v0"
    )
    checkpoint_observation_schema = getattr(
        agent, "checkpoint_observation_schema", "legacy_iob_proxy_v1"
    )
    checkpoint_extra = getattr(agent, "checkpoint_extra", {})
    checkpoint_compatible = (
        checkpoint_distribution == "tanh_normal_v2"
        and checkpoint_observation_schema == "causal_iob_units_v2"
    )
    if not checkpoint_compatible and not args.allow_legacy_ppo_checkpoint:
        raise SystemExit(
            "Checkpoint does not satisfy tanh_normal_v2 + causal_iob_units_v2. "
            "Retrain with the corrected policy/observation schema, or pass "
            "--allow-legacy-ppo-checkpoint for diagnostic-only regression checks."
        )

    patients = parse_patients(args.patients)
    modes = parse_modes(args.modes)
    if not patients or not modes:
        raise SystemExit("At least one patient and one mode are required.")

    args._compatibility_config = None
    args._compatibility_calibration_id = ""
    compatibility_payload = {}
    compatibility_sha256 = ""
    compatibility_path = ""
    calibration_requested = bool(args.compatibility_calibration_file)
    gate_requested = bool(args.enable_switch_compatibility_gate)
    if gate_requested and not any(
        mode in {"switch", "switch_shield"} for mode in modes
    ):
        raise SystemExit(
            "--enable-switch-compatibility-gate requires a switch or switch_shield mode."
        )
    if gate_requested and not calibration_requested:
        raise SystemExit(
            "The compatibility gate is fail-closed: provide "
            "--compatibility-calibration-file from the independent development suite."
        )
    if gate_requested and not args.compatibility_reference_arx_sha256:
        raise SystemExit(
            "The compatibility gate requires "
            "--compatibility-reference-arx-sha256 bound to the calibration prior."
        )
    if calibration_requested:
        (
            compatibility_payload,
            args._compatibility_config,
            compatibility_sha256,
            calibration_path,
        ) = load_compatibility_calibration(
            args.compatibility_calibration_file,
            expected_sha256=args.expected_compatibility_calibration_sha256,
            current_source_sha256=current_source_sha256,
            current_checkpoint_sha256=current_checkpoint_sha256,
            patients=patients,
            reference_arx_sha256=args.compatibility_reference_arx_sha256,
            current_requirements_lock_sha256=current_lock_sha256,
            allow_patient_overlap=bool(
                args.allow_compatibility_source_overlap
            ),
        )
        args._compatibility_calibration_id = str(
            compatibility_payload.get(
                "calibration_id",
                getattr(args._compatibility_config, "calibration_id", ""),
            )
        )
        compatibility_path = str(calibration_path)
    theta_overlap = patient_specific_theta_overlap(args.arx_param_file, patients)
    if theta_overlap and not args.allow_patient_specific_theta:
        raise SystemExit(
            "Patient-specific ARX theta leakage detected for evaluation patients: "
            + ", ".join(sorted(theta_overlap))
            + ". Use a training-only population prior."
        )
    serializable_args = {
        key: value for key, value in vars(args).items() if not key.startswith("_")
    }
    config_payload = {
        **serializable_args,
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_action_distribution": checkpoint_distribution,
        "checkpoint_observation_schema": checkpoint_observation_schema,
        "checkpoint_training_source_sha256": checkpoint_extra.get(
            "source_tree_sha256", "UNKNOWN"
        ),
        "checkpoint_sha256": current_checkpoint_sha256,
        "evaluation_source_tree_sha256": current_source_sha256,
        "source_tree_sha256": current_source_sha256,
        "requirements_lock_path": str(requirements_lock.resolve()),
        "requirements_lock_sha256": current_lock_sha256,
        "runtime_lock_verified": True,
        "runtime_lock_audit": runtime_lock_audit,
        "runtime_identity_actual": runtime_identity(),
        "runtime_fingerprint_actual": current_runtime_fingerprint,
        "arx_param_sha256": current_arx_sha256,
        "arx_prior_label": str(args.arx_prior_label),
        "model_condition": str(args.model_condition),
        "compatibility_calibration_file": compatibility_path,
        "compatibility_calibration_sha256": compatibility_sha256,
        "compatibility_calibration_id": str(args._compatibility_calibration_id),
        "compatibility_reference_arx_sha256": str(
            args.compatibility_reference_arx_sha256
        ),
        "patients_resolved": patients,
        "modes_resolved": modes,
    }
    write_json_atomic(output_dir / "config.json", config_payload)

    expected_runs = len(patients) * len(modes) * int(args.repeats)
    write_json_atomic(
        output_dir / "completion.json",
        {
            "complete": False,
            "execution_complete": False,
            "claim_valid": False,
            "expected_runs": expected_runs,
            "completed_runs": 0,
            "invalid_runs": 0,
            "protocol_fingerprint": args.protocol_fingerprint,
            "protocol_version": PROTOCOL_VERSION,
            "model_condition": str(args.model_condition),
            "arx_prior_label": str(args.arx_prior_label),
            "compatibility_calibration_sha256": compatibility_sha256,
        },
    )
    summaries = []
    for patient_name in patients:
        patient_idx = int(patient_name.split("#")[-1])
        for mode in modes:
            for repeat in range(args.repeats):
                seed = args.seed + 1000 * repeat + patient_idx
                summary = run_episode(mode, agent, patient_name, seed, args, output_dir)
                summary["repeat"] = repeat
                summaries.append(summary)
                write_csv(output_dir / "summary.csv", summaries)
                write_json_atomic(
                    output_dir / "completion.json",
                    {
                        "complete": False,
                        "execution_complete": False,
                        "claim_valid": False,
                        "expected_runs": expected_runs,
                        "completed_runs": len(summaries),
                        "invalid_runs": int(
                            sum(not bool(item.get("claim_valid")) for item in summaries)
                        ),
                        "protocol_fingerprint": args.protocol_fingerprint,
                        "protocol_version": PROTOCOL_VERSION,
                        "model_condition": str(args.model_condition),
                        "arx_prior_label": str(args.arx_prior_label),
                        "compatibility_calibration_sha256": compatibility_sha256,
                    },
                )
                print(
                    f"{patient_name} {mode} repeat={repeat + 1}/{args.repeats} "
                    f"TIR={summary['tir']:.1f}% TBR70={summary['tbr70']:.1f}% "
                    f"TAR250={summary['tar250']:.1f}% mean={summary['mean_bg']:.1f}"
                )

    invalid_runs = int(sum(not bool(item.get("claim_valid")) for item in summaries))
    execution_complete = len(summaries) == expected_runs
    write_json_atomic(
        output_dir / "completion.json",
        {
            "complete": execution_complete,
            "execution_complete": execution_complete,
            "claim_valid": execution_complete and invalid_runs == 0,
            "expected_runs": expected_runs,
            "completed_runs": len(summaries),
            "invalid_runs": invalid_runs,
            "protocol_fingerprint": args.protocol_fingerprint,
            "protocol_version": PROTOCOL_VERSION,
            "model_condition": str(args.model_condition),
            "arx_prior_label": str(args.arx_prior_label),
            "compatibility_calibration_sha256": compatibility_sha256,
        },
    )
    if summaries:
        df = pd.DataFrame(summaries)
        aggregate = (
            df.groupby(["patient", "mode"], as_index=False)
            .agg(
                n=("tir", "count"),
                tir_mean=("tir", "mean"),
                tir_std=("tir", "std"),
                tbr70_mean=("tbr70", "mean"),
                tar250_mean=("tar250", "mean"),
                mean_bg_mean=("mean_bg", "mean"),
                ppo_takeover_rate_mean=("ppo_takeover_rate", "mean")
                if "ppo_takeover_rate" in df.columns
                else ("tir", "count"),
            )
        )
        aggregate.to_csv(output_dir / "aggregate_by_patient_mode.csv", index=False)
    print(f"Saved PPO-MPC evaluation under: {output_dir}")


if __name__ == "__main__":
    main()
