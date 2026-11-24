from __future__ import annotations

import numpy as np
import pandas as pd


def zone_reward(glucose: float) -> float:
    g = float(glucose)
    if g < 39.0:
        return -20.0
    if g > 400.0:
        return -10.0
    if g < 54.0:
        return -5.0
    if g < 70.0:
        return -1.0 - 0.02 * (70.0 - g)
    if g < 100.0:
        return 0.5 + 0.5 * (g - 70.0) / 30.0
    if g <= 140.0:
        return 1.0
    if g <= 180.0:
        return 0.7 - 0.2 * (g - 140.0) / 40.0
    if g <= 250.0:
        return 0.2 - 0.8 * (g - 180.0) / 70.0
    return -1.0 - 0.01 * min(g - 250.0, 150.0)


def clinical_reward(
    glucose: float,
    basal_rate: float,
    basal_reference: float,
    previous_basal: float | None = None,
) -> float:
    reward = zone_reward(glucose)
    basal_ratio = float(basal_rate) / max(float(basal_reference), 1e-8)
    reward -= 0.005 * max(0.0, basal_ratio - 1.5) ** 2
    if previous_basal is not None:
        delta = abs(float(basal_rate) - float(previous_basal)) / max(float(basal_reference), 1e-8)
        reward -= 0.002 * delta
    return float(reward)


def clinical_metrics(glucose_values) -> dict[str, float]:
    raw = np.asarray(glucose_values, dtype=float)
    finite = np.isfinite(raw)
    arr = raw[finite]
    quality = {
        "raw_steps": int(raw.size),
        "nonfinite_steps": int((~finite).sum()),
        "nonfinite_fraction": float((~finite).mean()) if raw.size else 0.0,
    }
    if arr.size == 0:
        return {
            **quality,
            "steps": 0,
            "titr": float("nan"),
            "tir": float("nan"),
            "tbr70": float("nan"),
            "tbr54": float("nan"),
            "tar180": float("nan"),
            "tar250": float("nan"),
            "mean_bg": float("nan"),
            "sd_bg": float("nan"),
            "min_bg": float("nan"),
            "max_bg": float("nan"),
        }
    return {
        **quality,
        "steps": int(arr.size),
        "titr": float(((arr >= 70.0) & (arr <= 140.0)).mean() * 100.0),
        "tir": float(((arr >= 70.0) & (arr <= 180.0)).mean() * 100.0),
        "tbr70": float((arr < 70.0).mean() * 100.0),
        "tbr54": float((arr < 54.0).mean() * 100.0),
        "tar180": float((arr > 180.0).mean() * 100.0),
        "tar250": float((arr > 250.0).mean() * 100.0),
        "mean_bg": float(arr.mean()),
        "sd_bg": float(arr.std(ddof=0)),
        "min_bg": float(arr.min()),
        "max_bg": float(arr.max()),
    }


def summarize_closed_loop(
    dataframe: pd.DataFrame,
    *,
    patient: str,
    mode: str,
    scenario: str,
    seed: int,
    switch_history: pd.DataFrame | None = None,
) -> dict[str, float | str | int]:
    # show_history() contains the pre-control t=0 observation followed by one
    # glucose endpoint per executed interval.  Clinical percentages use only
    # post-action endpoints so N control steps produce N equally weighted
    # samples, not N+1.
    outcome_frame = dataframe.iloc[1:] if len(dataframe) > 0 else dataframe
    bg_col = "BG" if "BG" in dataframe.columns else "CGM"
    metrics = clinical_metrics(outcome_frame[bg_col].to_numpy())
    metrics["metric_source"] = bg_col
    metrics["metric_sampling"] = "post_action_interval_endpoints"
    if "CGM" in dataframe.columns and bg_col != "CGM":
        cgm_metrics = clinical_metrics(outcome_frame["CGM"].to_numpy())
        for key, value in cgm_metrics.items():
            metrics[f"cgm_{key}"] = value
    metrics.update(
        {
            "patient": patient,
            "mode": mode,
            "scenario": scenario,
            "seed": int(seed),
        }
    )
    if "insulin" in dataframe.columns:
        metrics["total_insulin_u"] = float(dataframe["insulin"].dropna().sum() * 5.0)
    if switch_history is not None and not switch_history.empty:
        selected = switch_history["selected"].astype(str)
        metrics["ppo_takeover_rate"] = float((selected == "ppo").mean())
        metrics["mpc_takeover_rate"] = float((selected == "mpc").mean())
        metrics["switch_count"] = int((selected != selected.shift(1)).sum() - 1)
        if "ppo_authorized" in switch_history.columns:
            metrics["ppo_authorized_rate"] = float(
                switch_history["ppo_authorized"].astype(bool).mean()
            )
        if "ppo_effective" in switch_history.columns:
            metrics["ppo_effective_rate"] = float(
                switch_history["ppo_effective"].astype(bool).mean()
            )
        if "safety_override" in switch_history.columns:
            metrics["shield_override_rate"] = float(
                switch_history["safety_override"].astype(bool).mean()
            )
        if "model_residual_raw_mgdl" in switch_history.columns:
            residual = pd.to_numeric(
                switch_history["model_residual_raw_mgdl"], errors="coerce"
            )
            finite_residual = residual[np.isfinite(residual)]
            metrics["model_residual_valid_rate"] = float(
                finite_residual.size / max(len(switch_history), 1)
            )
            metrics["model_residual_mae_mgdl"] = float(
                finite_residual.abs().mean()
            ) if not finite_residual.empty else float("nan")
            metrics["model_residual_rmse_mgdl"] = float(
                (finite_residual.pow(2).mean()) ** 0.5
            ) if not finite_residual.empty else float("nan")
        if {
            "model_bg_pred_mgdl",
            "model_bg_meas_mgdl",
            "model_residual_raw_mgdl",
        }.issubset(switch_history.columns):
            predicted = pd.to_numeric(
                switch_history["model_bg_pred_mgdl"], errors="coerce"
            )
            measured = pd.to_numeric(
                switch_history["model_bg_meas_mgdl"], errors="coerce"
            )
            raw = pd.to_numeric(
                switch_history["model_residual_raw_mgdl"], errors="coerce"
            )
            consistency = (measured - predicted - raw).abs()
            finite_consistency = consistency[np.isfinite(consistency)]
            metrics["model_residual_consistency_max_mgdl"] = float(
                finite_consistency.max()
            ) if not finite_consistency.empty else float("nan")
        for column in (
            "model_npe",
            "model_npe_ewma",
            "model_compatibility_score",
        ):
            if column in switch_history.columns:
                values = pd.to_numeric(switch_history[column], errors="coerce")
                finite_values = values[np.isfinite(values)]
                metrics[f"{column}_mean"] = float(
                    finite_values.mean()
                ) if not finite_values.empty else float("nan")
                metrics[f"{column}_p95"] = float(
                    finite_values.quantile(0.95)
                ) if not finite_values.empty else float("nan")
        if "model_compatible" in switch_history.columns:
            metrics["model_compatible_rate"] = float(
                switch_history["model_compatible"].astype(bool).mean()
            )
        if "mpc_model_authorized" in switch_history.columns:
            metrics["mpc_model_authorized_rate"] = float(
                switch_history["mpc_model_authorized"].astype(bool).mean()
            )
        if "mpc_candidate_valid" in switch_history.columns:
            metrics["mpc_candidate_valid_rate"] = float(
                switch_history["mpc_candidate_valid"].astype(bool).mean()
            )
        if "compat_gate_transition" in switch_history.columns:
            transitions = switch_history["compat_gate_transition"].astype(str)
            metrics["compat_gate_transition_count"] = int(
                (transitions != "").sum()
            )
        if {"selected", "mpc_model_authorized"}.issubset(switch_history.columns):
            selected_mpc = switch_history["selected"].astype(str).eq("mpc")
            authorized = switch_history["mpc_model_authorized"].astype(bool)
            switch_applicable = mode.removesuffix("_shield") in {
                "switch",
                "switch_nogate",
            }
            metrics["mpc_authorization_violations"] = int(
                (selected_mpc & ~authorized).sum()
            ) if switch_applicable else 0
    return metrics
