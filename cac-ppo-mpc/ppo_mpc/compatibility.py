from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CompatibilityGateConfig:
    """Frozen development-set calibration for a causal ARX compatibility gate."""

    residual_center_mgdl: float
    residual_scale_mgdl: float
    ewma_tau_min: float
    enter_threshold_npe: float
    exit_threshold_npe: float
    min_valid_samples: int = 12
    enter_confirm_steps: int = 6
    exit_confirm_steps: int = 2
    min_off_steps: int = 6
    z_clip: float = 10.0
    calibration_id: str = ""
    schema_version: str = "causal_arx_compat_v1"

    def __post_init__(self) -> None:
        finite_positive = {
            "residual_scale_mgdl": self.residual_scale_mgdl,
            "ewma_tau_min": self.ewma_tau_min,
            "enter_threshold_npe": self.enter_threshold_npe,
            "exit_threshold_npe": self.exit_threshold_npe,
            "z_clip": self.z_clip,
        }
        if not math.isfinite(float(self.residual_center_mgdl)):
            raise ValueError("residual_center_mgdl must be finite")
        for name, value in finite_positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(self.exit_threshold_npe) <= float(self.enter_threshold_npe):
            raise ValueError("exit_threshold_npe must exceed enter_threshold_npe")
        for name in (
            "min_valid_samples",
            "enter_confirm_steps",
            "exit_confirm_steps",
            "min_off_steps",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least one")
        if self.schema_version != "causal_arx_compat_v1":
            raise ValueError(
                f"unsupported compatibility schema_version: {self.schema_version}"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CompatibilityGateConfig":
        return cls(
            residual_center_mgdl=float(payload["residual_center_mgdl"]),
            residual_scale_mgdl=float(payload["residual_scale_mgdl"]),
            ewma_tau_min=float(payload["ewma_tau_min"]),
            enter_threshold_npe=float(payload["enter_threshold_npe"]),
            exit_threshold_npe=float(payload["exit_threshold_npe"]),
            min_valid_samples=int(payload.get("min_valid_samples", 12)),
            enter_confirm_steps=int(payload.get("enter_confirm_steps", 6)),
            exit_confirm_steps=int(payload.get("exit_confirm_steps", 2)),
            min_off_steps=int(payload.get("min_off_steps", 6)),
            z_clip=float(payload.get("z_clip", 10.0)),
            calibration_id=str(payload.get("calibration_id", "")),
            schema_version=str(
                payload.get("schema_version", "causal_arx_compat_v1")
            ),
        )


@dataclass(frozen=True)
class CompatibilityDecision:
    residual_raw_mgdl: float
    residual_abs_mgdl: float
    residual_scale_mgdl: float
    normalized_prediction_error: float
    normalized_prediction_error_ewma: float
    compatibility_score: float
    compatible: bool
    authorized: bool
    transition: str
    reason: str
    valid_samples: int
    good_streak: int
    bad_streak: int
    off_steps: int


class CausalCompatibilityGate:
    """Fail-closed hysteresis driven only by pre-update one-step ARX residuals."""

    def __init__(self, config: CompatibilityGateConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.compatible = False
        self.valid_samples = 0
        self.good_streak = 0
        self.bad_streak = 0
        self.off_steps = 0
        self.ewma_npe = float("nan")

    def _invalid_decision(self, residual_raw_mgdl: float, reason: str) -> CompatibilityDecision:
        transition = "exit_invalid" if self.compatible else ""
        self.compatible = False
        self.good_streak = 0
        self.bad_streak = 0
        self.off_steps = 0
        return CompatibilityDecision(
            residual_raw_mgdl=float(residual_raw_mgdl),
            residual_abs_mgdl=float("nan"),
            residual_scale_mgdl=float(self.config.residual_scale_mgdl),
            normalized_prediction_error=float("nan"),
            normalized_prediction_error_ewma=float(self.ewma_npe),
            compatibility_score=0.0,
            compatible=False,
            authorized=False,
            transition=transition,
            reason=reason,
            valid_samples=int(self.valid_samples),
            good_streak=0,
            bad_streak=0,
            off_steps=0,
        )

    def update(
        self,
        residual_raw_mgdl: float,
        *,
        sample_time_min: float,
        candidate_valid: bool,
    ) -> CompatibilityDecision:
        residual = float(residual_raw_mgdl)
        sample_time = float(sample_time_min)
        if not candidate_valid:
            return self._invalid_decision(residual, "mpc_candidate_invalid")
        if not math.isfinite(residual):
            return self._invalid_decision(residual, "residual_unavailable")
        if not math.isfinite(sample_time) or sample_time <= 0.0:
            return self._invalid_decision(residual, "sample_time_invalid")

        centered_abs = abs(residual - float(self.config.residual_center_mgdl))
        npe = min(
            centered_abs / float(self.config.residual_scale_mgdl),
            float(self.config.z_clip),
        )
        alpha = 1.0 - math.exp(-sample_time / float(self.config.ewma_tau_min))
        if not math.isfinite(self.ewma_npe):
            self.ewma_npe = float(npe)
        else:
            self.ewma_npe = float((1.0 - alpha) * self.ewma_npe + alpha * npe)

        self.valid_samples += 1
        transition = ""
        reason = "compatible" if self.compatible else "incompatible"

        if self.valid_samples < int(self.config.min_valid_samples):
            self.compatible = False
            self.good_streak = 0
            self.bad_streak = 0
            self.off_steps += 1
            reason = "warmup"
        elif self.compatible:
            self.off_steps = 0
            if self.ewma_npe >= float(self.config.exit_threshold_npe):
                self.bad_streak += 1
            else:
                self.bad_streak = 0
            self.good_streak = 0
            if self.bad_streak >= int(self.config.exit_confirm_steps):
                self.compatible = False
                self.bad_streak = 0
                self.off_steps = 0
                transition = "exit_incompatible"
                reason = "exit_threshold"
            else:
                reason = "compatible_hold"
        else:
            self.off_steps += 1
            self.bad_streak = 0
            if self.ewma_npe <= float(self.config.enter_threshold_npe):
                self.good_streak += 1
            else:
                self.good_streak = 0
            if (
                self.off_steps >= int(self.config.min_off_steps)
                and self.good_streak >= int(self.config.enter_confirm_steps)
            ):
                self.compatible = True
                self.good_streak = 0
                self.off_steps = 0
                transition = "enter_compatible"
                reason = "enter_threshold"
            else:
                reason = "incompatible_hold"

        score = math.exp(-max(float(self.ewma_npe), 0.0))
        return CompatibilityDecision(
            residual_raw_mgdl=residual,
            residual_abs_mgdl=centered_abs,
            residual_scale_mgdl=float(self.config.residual_scale_mgdl),
            normalized_prediction_error=float(npe),
            normalized_prediction_error_ewma=float(self.ewma_npe),
            compatibility_score=float(score),
            compatible=bool(self.compatible),
            authorized=bool(self.compatible and candidate_valid),
            transition=transition,
            reason=reason,
            valid_samples=int(self.valid_samples),
            good_streak=int(self.good_streak),
            bad_streak=int(self.bad_streak),
            off_steps=int(self.off_steps),
        )
