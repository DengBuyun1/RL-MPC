from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def residual_feature_vector(
    glucose: float,
    prev_glucose: float,
    raw_action: float,
    basal: float,
    basal_rate: float,
    announced_meal: float,
    iob: float,
    sample_time: float = 5.0,
) -> np.ndarray:
    dg = (float(glucose) - float(prev_glucose)) / max(float(sample_time), 1e-8)
    basal_scale = max(10.0 * float(basal_rate), 1e-8)
    return np.asarray(
        [
            1.0,
            float(glucose) / 400.0,
            dg / 5.0,
            float(raw_action),
            float(basal) / basal_scale,
            float(announced_meal) / 100.0,
            float(iob) / basal_scale,
        ],
        dtype=np.float64,
    )


@dataclass
class OnlineRlsOneStepPredictor:
    """Lightweight online ARX/RLS predictor for one-step glucose residuals."""

    n_features: int = 7
    forgetting: float = 0.995
    p0: float = 1000.0
    err_clip: float = 80.0

    def __post_init__(self) -> None:
        self.theta = np.zeros(self.n_features, dtype=np.float64)
        self.theta[1] = 1.0
        self.P = np.eye(self.n_features, dtype=np.float64) * float(self.p0)

    def predict(self, phi: np.ndarray) -> float:
        return float(np.dot(self.theta, np.asarray(phi, dtype=np.float64)) * 400.0)

    def update(self, phi: np.ndarray, glucose_next: float) -> float:
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        y = float(glucose_next) / 400.0
        y_hat = float(np.dot(self.theta, phi))
        err = y - y_hat
        if np.isfinite(self.err_clip) and self.err_clip > 0:
            err = float(np.clip(err, -self.err_clip / 400.0, self.err_clip / 400.0))
        denom = float(self.forgetting + phi.T @ self.P @ phi)
        gain = (self.P @ phi) / max(denom, 1e-12)
        self.theta = self.theta + gain * err
        self.P = (self.P - np.outer(gain, phi) @ self.P) / max(self.forgetting, 1e-8)
        return float((y - y_hat) * 400.0)


def residual_score(
    glucose: float,
    next_glucose: float,
    predicted_next_glucose: float,
    sample_time: float = 5.0,
    positive_weight: float = 1.0,
    slope_weight: float = 0.5,
    absolute_weight: float = 0.0,
) -> float:
    err = float(next_glucose) - float(predicted_next_glucose)
    actual_delta = float(next_glucose) - float(glucose)
    predicted_delta = float(predicted_next_glucose) - float(glucose)
    positive = max(0.0, err)
    slope_positive = max(0.0, (actual_delta - predicted_delta) / max(float(sample_time), 1e-8))
    return float(
        positive_weight * positive
        + slope_weight * slope_positive
        + absolute_weight * abs(err)
    )

