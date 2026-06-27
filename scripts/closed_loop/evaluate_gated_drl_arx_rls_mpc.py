from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAC_ROOT = PROJECT_ROOT / "src" / "controllers" / "rl_sac_mask"
SIM_ROOT = PROJECT_ROOT / "src" / "envs" / "simglucose_mpc"
ARX_CONTROLLER_PATH = PROJECT_ROOT / "src" / "controllers" / "arx_rls" / "arx_mpc_ctrller.py"
DEFAULT_ARX_PARAM_FILE = PROJECT_ROOT / "configs" / "arx" / "best_patient_theta.json"
for path in (SAC_ROOT, SIM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sac_mask_drl import MaskRecurrentSACAgent
from sac_mask_drl.ap_env import (
    APStateTracker,
    basal_multiplier_to_raw_action,
    clinical_metrics,
    make_paper_simglucose_env,
    map_action_to_basal,
    patient_basal_rate,
    zone_reward,
)
from sac_mask_drl.history import HistoryBuffer
from sac_mask_drl.knn_ood_gate import ConditionalActionKnnGate, extract_actor_history_embedding
from sac_mask_drl.residual_gate import OnlineRlsOneStepPredictor, residual_feature_vector, residual_score


Observation = namedtuple("Observation", ["CGM"])


def load_arx_controller_class():
    module_path = ARX_CONTROLLER_PATH
    spec = importlib.util.spec_from_file_location("arx_mpc_ctrller_local_for_gated", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ARX controller module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ArxRlsVffMpcController


ArxRlsVffMpcController = load_arx_controller_class()


def parse_patients(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_modes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_patient_set(value: str) -> set[str] | None:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def patient_run_name(patient_name: str, run_template: str) -> str:
    idx3 = patient_name.split("#")[-1]
    return run_template.format(
        patient=patient_name,
        patient_dash=patient_name.replace("#", "-"),
        idx3=idx3,
        idx=int(idx3),
    )


def patient_file_stem(patient_name: str) -> str:
    return patient_name.replace("#", "-")


def load_run_options(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def load_agent(model_path: Path, device: torch.device) -> tuple[MaskRecurrentSACAgent, object]:
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")
    if "config" not in payload:
        raise KeyError(f"checkpoint has no AgentConfig: {model_path}")
    config = payload["config"]
    agent = MaskRecurrentSACAgent(config, device=device)
    agent.load(model_path, load_optimizers=False)
    agent.actor.eval()
    agent.critic.eval()
    return agent, config


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_arx_mpc(args: argparse.Namespace):
    param_file = Path(args.arx_param_file)
    if not param_file.is_absolute():
        param_file = PROJECT_ROOT / param_file
    return ArxRlsVffMpcController(
        target=args.arx_target_glucose,
        sample_time_min=5.0,
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
        use_meal_filter=True,
        meal_filter_tau_min=args.arx_meal_filter_tau_min,
        use_meal_buffer=True,
        meal_release_g_per_min=args.arx_meal_release_g_per_min,
        meal_c_max=args.arx_meal_c_max,
        meal_forecast_steps=args.arx_meal_forecast_steps,
        theta_input_mode="uhat",
        enable_meal_bolus=bool(args.arx_enable_meal_bolus),
        meal_bolus_scale=args.arx_meal_bolus_scale,
        param_file=str(param_file),
        allow_fallback_theta=False,
        use_workspace_mpc=True,
        workspace_mpc_prediction_model="arx",
        workspace_mpc_variant=args.mpc_variant,
        workspace_mpc_use_iob_constraint=args.mpc_use_iob_constraint,
        workspace_mpc_max_insulin_units_per_sample=args.mpc_max_insulin_units_per_sample,
        workspace_mpc_max_insulin_tdi_fraction=args.mpc_max_insulin_tdi_fraction,
        workspace_mpc_model_gain_factor=args.mpc_model_gain_factor,
        workspace_mpc_r_plus_scale=args.mpc_r_plus_scale,
        workspace_mpc_enable_low_glucose_suspend=args.mpc_low_glucose_suspend,
    )


def available_announced_meal(env) -> float:
    if hasattr(env, "_announced_meal_in_next_sample"):
        return float(env._announced_meal_in_next_sample())
    return 0.0


def make_predictor(args: argparse.Namespace) -> OnlineRlsOneStepPredictor:
    return OnlineRlsOneStepPredictor(
        forgetting=args.residual_rls_forgetting,
        p0=args.residual_rls_p0,
        err_clip=args.residual_rls_err_clip,
    )


def calibrate_residual_threshold(
    agent: MaskRecurrentSACAgent,
    config,
    patient_name: str,
    run_options: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    env_steps = []
    for episode in range(args.calibration_episodes):
        rows, _metrics = evaluate_one(
            agent=agent,
            config=config,
            gate=None,
            residual_threshold=None,
            patient_name=patient_name,
            scenario=args.calibration_scenario,
            meal_announcement=args.calibration_meal_announcement,
            seed=args.calibration_seed + episode,
            mode="drl",
            args=args,
            device=device,
            run_options=run_options,
            write_rows=False,
            rl_eligible_patients=None,
        )
        env_steps.extend(rows)
    score_key = "residual_ewma" if args.residual_mode == "ewma" else "residual_score"
    scores = [
        float(row[score_key])
        for row in env_steps
        if int(row["step"]) >= args.residual_warmup_steps
    ]
    if not scores:
        return float("inf")
    return float(np.quantile(np.asarray(scores, dtype=np.float32), args.residual_threshold_quantile))


def gate_decision(
    gate: ConditionalActionKnnGate | None,
    embedding: np.ndarray,
    raw_action: np.ndarray,
    glucose: float,
    glucose_rate: float,
    previous_residual_score: float,
    previous_residual_ewma: float,
    residual_threshold: float | None,
    args: argparse.Namespace,
) -> dict:
    if gate is None:
        knn_familiar = True
        state_distance = np.nan
        action_distance = np.nan
    else:
        familiar_arr, state_distance_arr, action_distance_arr = gate.is_familiar(embedding, raw_action.reshape(1, -1))
        knn_familiar = bool(familiar_arr[0])
        state_distance = float(state_distance_arr[0])
        action_distance = float(action_distance_arr[0])

    residual_gate_score = previous_residual_ewma if args.residual_mode == "ewma" else previous_residual_score
    residual_pass = True
    if args.enable_residual_gate and residual_threshold is not None:
        residual_pass = residual_gate_score <= float(residual_threshold)

    projected = float(glucose) + max(0.0, float(glucose_rate)) * 5.0 * args.trend_risk_horizon_steps
    risk_pass = True
    if args.enable_risk_guard:
        risk_pass = float(glucose) <= args.max_current_glucose and projected <= args.max_projected_glucose

    return {
        "allow_rl": bool(knn_familiar and residual_pass and risk_pass),
        "knn_familiar": bool(knn_familiar),
        "state_distance": state_distance,
        "action_distance": action_distance,
        "residual_pass": bool(residual_pass),
        "risk_pass": bool(risk_pass),
        "residual_gate_score": float(residual_gate_score),
        "trend_projected_max": projected,
    }


def zone_cost(
    glucose: float,
    low: float = 70.0,
    high: float = 180.0,
    hypo_weight: float = 6.0,
    hyper_weight: float = 1.0,
) -> float:
    glucose = float(glucose)
    hypo = max(0.0, low - glucose)
    hyper = max(0.0, glucose - high)
    return float(hypo_weight * hypo * hypo + hyper_weight * hyper * hyper)


def advantage_decision(
    predictor: OnlineRlsOneStepPredictor,
    glucose: float,
    prev_glucose: float,
    glucose_rate: float,
    drl_raw: np.ndarray,
    mpc_raw: np.ndarray,
    drl_basal: np.ndarray,
    mpc_basal: float,
    mpc_bolus: float,
    basal_rate: float,
    announced_meal: float,
    iob: float,
    sample_time: float,
    args: argparse.Namespace,
) -> dict:
    drl_basal_scalar = float(np.asarray(drl_basal).reshape(-1)[0])
    mpc_basal_scalar = float(mpc_basal)
    mpc_bolus_scalar = float(mpc_bolus)

    phi_rl = residual_feature_vector(
        glucose=glucose,
        prev_glucose=prev_glucose,
        raw_action=float(np.asarray(drl_raw).reshape(-1)[0]),
        basal=drl_basal_scalar,
        basal_rate=basal_rate,
        announced_meal=announced_meal,
        iob=iob,
        sample_time=sample_time,
    )
    phi_mpc = residual_feature_vector(
        glucose=glucose,
        prev_glucose=prev_glucose,
        raw_action=float(np.asarray(mpc_raw).reshape(-1)[0]),
        basal=mpc_basal_scalar,
        basal_rate=basal_rate,
        announced_meal=announced_meal,
        iob=iob,
        sample_time=sample_time,
    )
    predicted_rl = predictor.predict(phi_rl)
    predicted_mpc = predictor.predict(phi_mpc)
    cost_rl = zone_cost(predicted_rl, hypo_weight=args.advantage_hypo_weight)
    cost_mpc = zone_cost(predicted_mpc, hypo_weight=args.advantage_hypo_weight)

    drl_sample_insulin = max(0.0, drl_basal_scalar) * float(sample_time)
    mpc_sample_insulin = max(0.0, mpc_basal_scalar) * float(sample_time) + max(0.0, mpc_bolus_scalar)
    insulin_excess = drl_sample_insulin - mpc_sample_insulin

    low_or_falling = (
        float(glucose) <= args.advantage_hypo_glucose
        or (float(glucose) <= args.advantage_falling_glucose and float(glucose_rate) <= args.advantage_falling_rate)
    )
    insulin_pass = True
    if args.enable_advantage_insulin_guard and low_or_falling:
        insulin_pass = insulin_excess <= args.advantage_max_extra_insulin

    cost_pass = True
    if args.enable_advantage_cost_gate:
        cost_pass = cost_rl <= cost_mpc + args.advantage_cost_margin

    hypo_pass = True
    if args.enable_advantage_hypo_prediction_guard:
        hypo_pass = predicted_rl >= min(predicted_mpc - args.advantage_predicted_hypo_margin, args.advantage_min_predicted_glucose)

    return {
        "advantage_pass": bool(insulin_pass and cost_pass and hypo_pass),
        "advantage_cost_pass": bool(cost_pass),
        "advantage_insulin_pass": bool(insulin_pass),
        "advantage_hypo_pass": bool(hypo_pass),
        "predicted_next_rl": float(predicted_rl),
        "predicted_next_mpc": float(predicted_mpc),
        "advantage_cost_rl": float(cost_rl),
        "advantage_cost_mpc": float(cost_mpc),
        "drl_sample_insulin": float(drl_sample_insulin),
        "mpc_sample_insulin": float(mpc_sample_insulin),
        "advantage_insulin_excess": float(insulin_excess),
    }


def hard_safety_override(
    glucose: float,
    glucose_rate: float,
    predicted_next_glucose: float,
    args: argparse.Namespace,
) -> bool:
    if not args.enable_hard_safety_override:
        return False
    if float(glucose) <= args.hard_suspend_glucose:
        return True
    if (
        float(glucose) <= args.hard_falling_glucose
        and float(glucose_rate) <= args.hard_falling_rate
    ):
        return True
    if float(predicted_next_glucose) <= args.hard_predicted_min_glucose:
        return True
    return False


def evaluate_one(
    agent: MaskRecurrentSACAgent,
    config,
    gate: ConditionalActionKnnGate | None,
    residual_threshold: float | None,
    patient_name: str,
    scenario: str,
    meal_announcement: str,
    seed: int,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    run_options: dict,
    write_rows: bool = True,
    rl_eligible_patients: set[str] | None = None,
) -> tuple[list[dict], dict]:
    action_mapping = str(run_options.get("action_mapping", "paper_basal_centered"))
    max_basal_multiplier = float(run_options.get("max_basal_multiplier", 10.0))
    basal_delta_multiplier = float(run_options.get("basal_delta_multiplier", 1.0))
    state_mode = str(run_options.get("state_mode", "paper"))
    sensor_name = str(run_options.get("sensor_name", "GuardianRT"))

    env, patient = make_paper_simglucose_env(
        patient_name,
        scenario,
        seed,
        args.steps,
        meal_announcement=meal_announcement,
        sensor_name=sensor_name,
    )
    mpc = make_arx_mpc(args)
    predictor = make_predictor(args)
    basal_rate = patient_basal_rate(patient)
    body_weight = float(patient._params["BW"])
    max_basal = float(env.action_space.high[0])
    sample_time = float(getattr(env, "sample_time", 5.0))
    tracker = APStateTracker(body_weight, basal_rate, sample_time=sample_time, state_mode=state_mode)
    history = HistoryBuffer(config)

    obs, _info = env.reset(seed=seed)
    glucose = float(obs[0])
    prev_glucose = glucose
    state = tracker.reset(glucose)
    history.reset(state)
    glucose_trace = [glucose]
    rows: list[dict] = []
    total_reward = 0.0
    last_reward = 0.0
    done = False
    previous_residual_score = 0.0
    previous_residual_ewma = 0.0

    for step in range(args.steps):
        history_tensor = history.tensor(device)
        embedding = extract_actor_history_embedding(agent, history_tensor)
        drl_raw = agent.select_action(history_tensor, deterministic=True).astype(np.float32).reshape(-1)
        drl_basal = map_action_to_basal(
            drl_raw,
            basal_rate,
            max_basal,
            max_basal_multiplier,
            action_mapping=action_mapping,
            basal_delta_multiplier=basal_delta_multiplier,
        )

        known_meal = available_announced_meal(env)
        known_meal_rate = known_meal / max(sample_time, 1e-8)
        mpc_action = mpc.policy(
            Observation(CGM=glucose),
            last_reward,
            done,
            sample_time=sample_time,
            patient_name=patient_name,
            time=env.env.time,
            meal=known_meal_rate,
            iob=tracker.current_iob(),
        )
        mpc_basal = float(mpc_action.basal)
        mpc_bolus = float(getattr(mpc_action, "bolus", 0.0))
        mpc_multiplier = float(np.clip(mpc_basal / max(basal_rate, 1e-8), 0.0, max_basal_multiplier))
        mpc_raw = basal_multiplier_to_raw_action(
            mpc_multiplier,
            max_basal_multiplier,
            action_mapping=action_mapping,
            basal_delta_multiplier=basal_delta_multiplier,
        ).astype(np.float32).reshape(-1)

        glucose_rate = (glucose - prev_glucose) / max(sample_time, 1e-8)
        current_iob = tracker.current_iob()
        decision = gate_decision(
            gate=gate,
            embedding=embedding,
            raw_action=drl_raw,
            glucose=glucose,
            glucose_rate=glucose_rate,
            previous_residual_score=previous_residual_score,
            previous_residual_ewma=previous_residual_ewma,
            residual_threshold=residual_threshold,
            args=args,
        )
        advantage = advantage_decision(
            predictor=predictor,
            glucose=glucose,
            prev_glucose=prev_glucose,
            glucose_rate=glucose_rate,
            drl_raw=drl_raw,
            mpc_raw=mpc_raw,
            drl_basal=drl_basal,
            mpc_basal=mpc_basal,
            mpc_bolus=mpc_bolus,
            basal_rate=basal_rate,
            announced_meal=known_meal,
            iob=current_iob,
            sample_time=sample_time,
            args=args,
        )
        if args.enable_advantage_gate:
            decision["allow_rl"] = bool(decision["allow_rl"] and advantage["advantage_pass"])
        competence_pass = rl_eligible_patients is None or patient_name in rl_eligible_patients
        if args.enable_competence_gate:
            decision["allow_rl"] = bool(decision["allow_rl"] and competence_pass)
        decision["competence_pass"] = bool(competence_pass)
        decision.update(advantage)

        if mode == "drl":
            controller = "drl"
            raw_action = drl_raw
            basal_action = drl_basal
        elif mode == "mpc":
            controller = "mpc"
            raw_action = mpc_raw
            basal_action = np.asarray([mpc_basal, mpc_bolus], dtype=np.float32)
        elif mode == "gated":
            if decision["allow_rl"]:
                controller = "drl"
                raw_action = drl_raw
                basal_action = drl_basal
            else:
                controller = "mpc"
                raw_action = mpc_raw
                basal_action = np.asarray([mpc_basal, mpc_bolus], dtype=np.float32)
        else:
            raise ValueError(f"unsupported mode: {mode}")

        pre_safety_phi = residual_feature_vector(
            glucose=glucose,
            prev_glucose=prev_glucose,
            raw_action=float(raw_action[0]),
            basal=float(basal_action[0]),
            basal_rate=basal_rate,
            announced_meal=known_meal,
            iob=current_iob,
            sample_time=sample_time,
        )
        pre_safety_predicted_next = predictor.predict(pre_safety_phi)
        safety_override = hard_safety_override(
            glucose=glucose,
            glucose_rate=glucose_rate,
            predicted_next_glucose=pre_safety_predicted_next,
            args=args,
        )
        if safety_override:
            controller = "safety"
            raw_action = basal_multiplier_to_raw_action(
                0.0,
                max_basal_multiplier,
                action_mapping=action_mapping,
                basal_delta_multiplier=basal_delta_multiplier,
            ).astype(np.float32).reshape(-1)
            basal_action = np.asarray([0.0, 0.0], dtype=np.float32)

        phi = residual_feature_vector(
            glucose=glucose,
            prev_glucose=prev_glucose,
            raw_action=float(raw_action[0]),
            basal=float(basal_action[0]),
            basal_rate=basal_rate,
            announced_meal=known_meal,
            iob=current_iob,
            sample_time=sample_time,
        )
        predicted_next = predictor.predict(phi)

        next_obs, _, terminated, truncated, info = env.step(basal_action)
        bolus_rate = float(info.get("bolus_rate", 0.0)) if isinstance(info, dict) else 0.0
        meal_grams = float(info.get("meal_grams", 0.0)) if isinstance(info, dict) else 0.0
        actual_meal_rate = float(info.get("actual_meal_rate", 0.0)) if isinstance(info, dict) else 0.0
        tracker.record_action(float(basal_action[0]), bolus_rate=bolus_rate)

        next_glucose = float(next_obs[0])
        reward = zone_reward(next_glucose)
        done = bool(terminated or truncated or next_glucose < 39.0 or next_glucose > 400.0)
        next_state = tracker.make_state(next_glucose, bolus_rate=bolus_rate, meal_grams=meal_grams)
        history.append(next_state, raw_action, reward)

        residual_raw = predictor.update(phi, next_glucose)
        score = residual_score(
            glucose=glucose,
            next_glucose=next_glucose,
            predicted_next_glucose=predicted_next,
            sample_time=sample_time,
            positive_weight=args.residual_positive_weight,
            slope_weight=args.residual_slope_weight,
            absolute_weight=args.residual_absolute_weight,
        )
        residual_ewma = args.residual_ewma_alpha * previous_residual_ewma + (1.0 - args.residual_ewma_alpha) * score

        rows.append(
            {
                "patient": patient_name,
                "scenario": scenario,
                "meal_announcement": meal_announcement,
                "seed": seed,
                "mode": mode,
                "step": step,
                "controller": controller,
                "safety_override": bool(safety_override),
                "allow_rl": decision["allow_rl"],
                "knn_familiar": decision["knn_familiar"],
                "residual_pass": decision["residual_pass"],
                "risk_pass": decision["risk_pass"],
                "competence_pass": decision["competence_pass"],
                "advantage_pass": decision["advantage_pass"],
                "advantage_cost_pass": decision["advantage_cost_pass"],
                "advantage_insulin_pass": decision["advantage_insulin_pass"],
                "advantage_hypo_pass": decision["advantage_hypo_pass"],
                "glucose": glucose,
                "next_glucose": next_glucose,
                "glucose_rate": glucose_rate,
                "reward": reward,
                "raw_action": float(raw_action[0]),
                "drl_raw": float(drl_raw[0]),
                "mpc_raw": float(mpc_raw[0]),
                "basal": float(basal_action[0]),
                "drl_basal": float(drl_basal[0]),
                    "mpc_basal": float(mpc_basal),
                    "mpc_bolus": float(mpc_bolus),
                "bolus": bolus_rate,
                "meal": meal_grams,
                "actual_meal_rate": actual_meal_rate,
                "state_distance": decision["state_distance"],
                "action_distance": decision["action_distance"],
                "residual_gate_score": decision["residual_gate_score"],
                "residual_threshold": np.nan if residual_threshold is None else float(residual_threshold),
                "trend_projected_max": decision["trend_projected_max"],
                "pre_safety_predicted_next_glucose": pre_safety_predicted_next,
                "predicted_next_rl": decision["predicted_next_rl"],
                "predicted_next_mpc": decision["predicted_next_mpc"],
                "advantage_cost_rl": decision["advantage_cost_rl"],
                "advantage_cost_mpc": decision["advantage_cost_mpc"],
                "drl_sample_insulin": decision["drl_sample_insulin"],
                "mpc_sample_insulin": decision["mpc_sample_insulin"],
                "advantage_insulin_excess": decision["advantage_insulin_excess"],
                "predicted_next_glucose": predicted_next,
                "residual_raw": residual_raw,
                "residual_score": score,
                "residual_ewma": residual_ewma,
                "done": done,
            }
        )

        prev_glucose = glucose
        glucose = next_glucose
        glucose_trace.append(next_glucose)
        total_reward += reward
        last_reward = reward
        previous_residual_score = score
        previous_residual_ewma = residual_ewma
        if done:
            break

    metrics = clinical_metrics(glucose_trace)
    controller_flags = [row.get("controller") for row in rows] if write_rows else []
    metrics.update(
        {
            "patient": patient_name,
            "scenario": scenario,
            "meal_announcement": meal_announcement,
            "seed": seed,
            "mode": mode,
            "steps": len(glucose_trace) - 1,
            "total_reward": total_reward,
            "rl_takeover_rate": float(np.mean([c == "drl" for c in controller_flags])) if controller_flags else np.nan,
            "mpc_fallback_rate": float(np.mean([c == "mpc" for c in controller_flags])) if controller_flags else np.nan,
            "switch_count": int(np.sum(np.asarray(controller_flags[1:]) != np.asarray(controller_flags[:-1])))
            if len(controller_flags) > 1
            else 0,
            "mpc_success_rate": float(np.mean(getattr(mpc, "solve_successes", []))) if getattr(mpc, "solve_successes", []) else np.nan,
            "mpc_solve_time_mean": float(np.mean(getattr(mpc, "solve_times", []))) if getattr(mpc, "solve_times", []) else np.nan,
            "residual_threshold": np.nan if residual_threshold is None else float(residual_threshold),
        }
    )
    env.close()
    return rows, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop DRL with gated ARX-RLS-MPC fallback.")
    parser.add_argument("--patients", default="adult#004,adult#005")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--run-template", default="sac_baseline_fixed_{patient_dash}")
    parser.add_argument("--model-name", default="model_final.pth")
    parser.add_argument("--memory-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario", default="A")
    parser.add_argument("--meal-announcement", choices=["announced", "unannounced"], default="unannounced")
    parser.add_argument("--modes", default="drl,mpc,gated")
    parser.add_argument("--gate-label", default="gated")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--seed", type=int, default=22000)
    parser.add_argument("--state-mode", choices=["paper", "legacy"], default="paper")
    parser.add_argument("--sensor-name", choices=["GuardianRT", "Dexcom", "Navigator"], default="GuardianRT")
    parser.add_argument("--calibration-scenario", default="train")
    parser.add_argument("--calibration-meal-announcement", choices=["announced", "unannounced"], default="announced")
    parser.add_argument("--calibration-episodes", type=int, default=20)
    parser.add_argument("--calibration-seed", type=int, default=9000)
    parser.add_argument("--enable-residual-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual-mode", choices=["instant", "ewma"], default="ewma")
    parser.add_argument("--residual-ewma-alpha", type=float, default=0.90)
    parser.add_argument("--residual-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--residual-warmup-steps", type=int, default=12)
    parser.add_argument("--residual-positive-weight", type=float, default=1.0)
    parser.add_argument("--residual-slope-weight", type=float, default=10.0)
    parser.add_argument("--residual-absolute-weight", type=float, default=0.0)
    parser.add_argument("--residual-rls-forgetting", type=float, default=0.995)
    parser.add_argument("--residual-rls-p0", type=float, default=1000.0)
    parser.add_argument("--residual-rls-err-clip", type=float, default=80.0)
    parser.add_argument("--enable-risk-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-current-glucose", type=float, default=250.0)
    parser.add_argument("--max-projected-glucose", type=float, default=250.0)
    parser.add_argument("--trend-risk-horizon-steps", type=int, default=12)
    parser.add_argument("--enable-advantage-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-advantage-cost-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-advantage-insulin-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-advantage-hypo-prediction-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--advantage-cost-margin", type=float, default=0.0)
    parser.add_argument("--advantage-hypo-weight", type=float, default=8.0)
    parser.add_argument("--advantage-hypo-glucose", type=float, default=115.0)
    parser.add_argument("--advantage-falling-glucose", type=float, default=140.0)
    parser.add_argument("--advantage-falling-rate", type=float, default=-0.25)
    parser.add_argument("--advantage-max-extra-insulin", type=float, default=0.0)
    parser.add_argument("--advantage-min-predicted-glucose", type=float, default=85.0)
    parser.add_argument("--advantage-predicted-hypo-margin", type=float, default=5.0)
    parser.add_argument("--enable-competence-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rl-eligible-patients",
        default="",
        help="Comma-separated patients allowed to pass the RL takeover gate when competence gate is enabled.",
    )
    parser.add_argument("--enable-hard-safety-override", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hard-suspend-glucose", type=float, default=85.0)
    parser.add_argument("--hard-falling-glucose", type=float, default=120.0)
    parser.add_argument("--hard-falling-rate", type=float, default=-0.5)
    parser.add_argument("--hard-predicted-min-glucose", type=float, default=80.0)
    parser.add_argument("--arx-param-file", default=str(DEFAULT_ARX_PARAM_FILE))
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
    parser.add_argument("--arx-enable-meal-bolus", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--arx-meal-bolus-scale", type=float, default=0.75)
    parser.add_argument("--mpc-variant", choices=["previous", "velocity", "adaptive"], default="adaptive")
    parser.add_argument("--mpc-use-iob-constraint", action="store_true")
    parser.add_argument("--mpc-max-insulin-units-per-sample", type=float, default=1.0)
    parser.add_argument("--mpc-max-insulin-tdi-fraction", type=float, default=None)
    parser.add_argument("--mpc-model-gain-factor", type=float, default=1.0)
    parser.add_argument("--mpc-r-plus-scale", type=float, default=1.0)
    parser.add_argument("--mpc-low-glucose-suspend", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rl_eligible_patients = parse_patient_set(args.rl_eligible_patients)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for patient_name in parse_patients(args.patients):
        run_dir = Path(args.runs_dir) / patient_run_name(patient_name, args.run_template)
        agent, config = load_agent(run_dir / args.model_name, device)
        run_options = load_run_options(run_dir)
        gate_path = Path(args.memory_dir) / f"{patient_file_stem(patient_name)}_conditional_action_knn_memory.npz"
        gate = ConditionalActionKnnGate.load(gate_path)
        competence_enabled = (
            bool(args.enable_competence_gate)
            and rl_eligible_patients is not None
            and patient_name not in rl_eligible_patients
        )
        if not args.enable_residual_gate:
            residual_threshold = float("inf")
            print(f"{patient_name}: residual gate disabled, residual calibration skipped")
        elif competence_enabled:
            residual_threshold = float("inf")
            print(f"{patient_name}: competence-disabled, residual calibration skipped")
        else:
            residual_threshold = calibrate_residual_threshold(agent, config, patient_name, run_options, args, device)
            print(f"{patient_name}: residual_tau={residual_threshold:.3f}")

        for mode in parse_modes(args.modes):
            for repeat in range(args.repeats):
                seed = args.seed + 1000 * repeat + int(patient_name.split("#")[-1])
                use_gate = gate if mode == "gated" else None
                use_threshold = residual_threshold if mode == "gated" else None
                rows, metrics = evaluate_one(
                    agent=agent,
                    config=config,
                    gate=use_gate,
                    residual_threshold=use_threshold,
                    patient_name=patient_name,
                    scenario=args.scenario,
                    meal_announcement=args.meal_announcement,
                    seed=seed,
                    mode=mode,
                    args=args,
                    device=device,
                    run_options=run_options,
                    write_rows=True,
                    rl_eligible_patients=rl_eligible_patients,
                )
                for row in rows:
                    row["gate_label"] = args.gate_label if mode == "gated" else mode
                    row["repeat"] = repeat
                metrics["gate_label"] = args.gate_label if mode == "gated" else mode
                metrics["repeat"] = repeat
                summaries.append(metrics)
                all_rows.extend(rows)
                print(
                    f"{patient_name} {mode} repeat={repeat + 1}/{args.repeats} "
                    f"TIR={metrics['tir']:.1f}% TAR250={metrics.get('tar250', np.nan):.1f}% "
                    f"mean={metrics['mean_bg']:.1f} RL={metrics['rl_takeover_rate']:.2f}"
                )

    write_csv(output_dir / "rollouts.csv", all_rows)
    write_csv(output_dir / "summary.csv", summaries)
    if summaries:
        import pandas as pd

        df = pd.DataFrame(summaries)
        aggregate = []
        for (patient, gate_label), group in df.groupby(["patient", "gate_label"]):
            row = {"patient": patient, "gate_label": gate_label, "n": int(len(group))}
            for col in [
                "titr",
                "tir",
                "tbr70",
                "tbr54",
                "tar180",
                "tar250",
                "mean_bg",
                "sd_bg",
                "total_reward",
                "rl_takeover_rate",
                "mpc_fallback_rate",
                "switch_count",
            ]:
                if col in group:
                    row[col + "_mean"] = float(group[col].mean())
                    row[col + "_std"] = float(group[col].std())
            aggregate.append(row)
        write_csv(output_dir / "aggregate_by_patient_mode.csv", aggregate)
    print(f"Saved gated closed-loop outputs under: {output_dir}")


if __name__ == "__main__":
    main()
