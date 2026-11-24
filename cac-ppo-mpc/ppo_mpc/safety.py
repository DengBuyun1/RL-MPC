from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SafetyGovernorConfig:
    enabled: bool = True
    prediction_horizons_min: tuple[float, ...] = (30.0, 60.0, 90.0)
    suspend_glucose: float = 75.0
    low_glucose: float = 90.0
    projected_low_glucose: float = 90.0
    falling_glucose: float = 130.0
    falling_rate: float = -0.25
    low_basal_fraction: float = 0.0
    falling_basal_fraction: float = 0.5
    iob_guard_enabled: bool = True
    iob_excess_ratio: float = 1.25
    iob_guard_projected_glucose: float = 130.0
    iob_guard_basal_fraction: float = 0.0
    high_guard_enabled: bool = True
    high_glucose: float = 250.0
    severe_high_glucose: float = 300.0
    high_rising_rate: float = 0.25
    correction_target_glucose: float = 180.0
    correction_horizon_min: float = 180.0
    high_max_basal_multiplier: float = 5.0


@dataclass(frozen=True)
class SafetyDecision:
    basal: float
    bolus: float
    overridden: bool
    reason: str
    cap_basal: float | None
    floor_basal: float | None
    projected_glucose: float
    projected_glucose_30min: float
    projected_glucose_60min: float
    projected_glucose_90min: float
    projected_glucose_min: float
    causal_iob_units: float | None
    basal_iob_units: float | None
    iob_ratio: float | None
    correction_factor_mgdl_per_u: float | None
    correction_need_units: float | None
    correction_iob_credit_units: float | None
    correction_residual_units: float | None


def _projection_at(
    projections: dict[float, float], horizon_min: float, fallback: float
) -> float:
    if not projections:
        return float(fallback)
    nearest = min(projections, key=lambda value: abs(value - horizon_min))
    return float(projections[nearest])


def apply_safety_governor(
    *,
    glucose: float,
    glucose_rate: float,
    basal: float,
    bolus: float,
    basal_reference: float,
    config: SafetyGovernorConfig,
    sample_time_min: float = 5.0,
    iob_units: float | None = None,
    basal_iob_units: float | None = None,
    correction_factor_mgdl_per_u: float | None = None,
) -> SafetyDecision:
    """Apply a controller-independent shield using causal information only.

    Multi-horizon linear values are conservative screening signals, not a
    physiological prediction model. IOB is reconstructed from executed action
    history; simulator-private IOB/COB is never accepted here.
    """
    numeric = {
        "glucose": glucose,
        "glucose_rate": glucose_rate,
        "basal": basal,
        "bolus": bolus,
        "basal_reference": basal_reference,
        "sample_time_min": sample_time_min,
    }
    nonfinite = [
        name for name, value in numeric.items() if not math.isfinite(float(value))
    ]
    if nonfinite:
        raise ValueError(f"non-finite safety input(s): {', '.join(nonfinite)}")
    if iob_units is not None and not math.isfinite(float(iob_units)):
        raise ValueError("non-finite causal IOB")
    if basal_iob_units is not None and (
        not math.isfinite(float(basal_iob_units)) or float(basal_iob_units) <= 0.0
    ):
        raise ValueError("basal IOB reference must be finite and positive")
    if correction_factor_mgdl_per_u is not None and (
        not math.isfinite(float(correction_factor_mgdl_per_u))
        or float(correction_factor_mgdl_per_u) <= 0.0
    ):
        raise ValueError("correction factor must be finite and positive")

    horizons = tuple(
        sorted(
            {
                float(value)
                for value in config.prediction_horizons_min
                if math.isfinite(float(value)) and float(value) > 0.0
            }
        )
    )
    if not horizons:
        horizons = (30.0,)
    projections = {
        horizon: float(glucose + glucose_rate * horizon) for horizon in horizons
    }
    projected_30 = _projection_at(projections, 30.0, glucose)
    projected_60 = _projection_at(projections, 60.0, projected_30)
    projected_90 = _projection_at(projections, 90.0, projected_60)
    projected_min = float(min(projections.values()))
    iob_ratio = (
        float(iob_units) / float(basal_iob_units)
        if iob_units is not None and basal_iob_units is not None
        else None
    )
    correction_need_units: float | None = None
    correction_iob_credit_units: float | None = None
    correction_residual_units: float | None = None

    def result(
        *,
        basal_value: float,
        bolus_value: float,
        reason: str,
        cap: float | None,
        floor: float | None,
    ) -> SafetyDecision:
        return SafetyDecision(
            basal=float(basal_value),
            bolus=float(bolus_value),
            overridden=(
                not math.isclose(float(basal_value), float(basal), abs_tol=1e-12)
                or not math.isclose(float(bolus_value), float(bolus), abs_tol=1e-12)
            ),
            reason=reason,
            cap_basal=cap,
            floor_basal=floor,
            projected_glucose=projected_30,
            projected_glucose_30min=projected_30,
            projected_glucose_60min=projected_60,
            projected_glucose_90min=projected_90,
            projected_glucose_min=projected_min,
            causal_iob_units=None if iob_units is None else float(iob_units),
            basal_iob_units=(
                None if basal_iob_units is None else float(basal_iob_units)
            ),
            iob_ratio=iob_ratio,
            correction_factor_mgdl_per_u=(
                None
                if correction_factor_mgdl_per_u is None
                else float(correction_factor_mgdl_per_u)
            ),
            correction_need_units=correction_need_units,
            correction_iob_credit_units=correction_iob_credit_units,
            correction_residual_units=correction_residual_units,
        )

    if not config.enabled:
        return result(
            basal_value=basal, bolus_value=bolus, reason="", cap=None, floor=None
        )

    cap: float | None = None
    floor: float | None = None
    reason = ""
    if glucose <= config.suspend_glucose:
        cap = 0.0
        reason = "safety_suspend_current_low"
    elif projected_min <= config.suspend_glucose:
        cap = 0.0
        reason = "safety_suspend_multihorizon_low"
    elif glucose <= config.low_glucose:
        cap = config.low_basal_fraction * basal_reference
        reason = "safety_cap_current_near_low"
    elif projected_min <= config.projected_low_glucose:
        cap = config.low_basal_fraction * basal_reference
        reason = "safety_cap_multihorizon_near_low"
    elif (
        config.iob_guard_enabled
        and iob_ratio is not None
        and iob_ratio >= config.iob_excess_ratio
        and glucose_rate < 0.0
        and projected_min <= config.iob_guard_projected_glucose
    ):
        cap = config.iob_guard_basal_fraction * basal_reference
        reason = "safety_cap_iob_excess_falling"
    elif glucose <= config.falling_glucose and glucose_rate <= config.falling_rate:
        cap = config.falling_basal_fraction * basal_reference
        reason = "safety_cap_falling"
    elif (
        config.high_guard_enabled
        and correction_factor_mgdl_per_u is not None
        and (
            glucose >= config.severe_high_glucose
            or (
                glucose >= config.high_glucose
                and glucose_rate >= config.high_rising_rate
            )
        )
    ):
        correction_need_units = max(
            0.0,
            (float(glucose) - float(config.correction_target_glucose))
            / float(correction_factor_mgdl_per_u),
        )
        correction_iob_credit_units = (
            max(0.0, float(iob_units) - float(basal_iob_units))
            if iob_units is not None and basal_iob_units is not None
            else 0.0
        )
        correction_residual_units = max(
            0.0, correction_need_units - correction_iob_credit_units
        )
        correction_rate = correction_residual_units / max(
            float(config.correction_horizon_min), 1e-8
        )
        floor = min(
            float(config.high_max_basal_multiplier) * float(basal_reference),
            float(basal_reference) + correction_rate,
        )
        reason = (
            "safety_floor_severe_high"
            if glucose >= config.severe_high_glucose
            else "safety_floor_rising_high"
        )

    if cap is None and floor is None:
        return result(
            basal_value=basal, bolus_value=bolus, reason="", cap=None, floor=None
        )

    if floor is not None:
        executed_basal = max(float(basal), max(0.0, float(floor)))
        return result(
            basal_value=executed_basal,
            bolus_value=bolus,
            reason=reason,
            cap=None,
            floor=float(floor),
        )

    cap = float(max(0.0, cap))
    executed_basal = min(float(basal), cap)
    executed_bolus = 0.0 if cap <= float(basal_reference) else float(bolus)
    return result(
        basal_value=executed_basal,
        bolus_value=executed_bolus,
        reason=reason,
        cap=cap,
        floor=None,
    )
