"""
ARX + RLS-VFF + MPC controller (Generic Order).

Based on result.json analysis:
- Default Order: 4 (Auto-detected from theta length)
- Global Optimization: Consistent across patients
- [FIXED] Critical bug: IndexOutOfBounds when Pred Horizon > Control Horizon

⚠️ Research / simulation use only. NOT for clinical use.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import pkg_resources

# --- Optional MPC solver ---
try:
    import cvxpy as cp

    _HAS_CVXPY = True
except Exception:
    cp = None
    _HAS_CVXPY = False

# --- Optional fallback optimizer (when cvxpy is unavailable) ---
try:
    from scipy.optimize import minimize

    _HAS_SCIPY = True
except Exception:
    minimize = None
    _HAS_SCIPY = False

# --- simglucose controller base ---
try:
    from simglucose.controller.base import Action, Controller
    from simglucose.controller.basal_bolus_ctrller import BBController
    from simglucose.controller.mpc_ctrller import MPCController
except ImportError:
    from base import Action, Controller
    from basal_bolus_ctrller import BBController
    from mpc_ctrller import MPCController

LOGGER = logging.getLogger(__name__)

# --- simglucose parameter CSV paths ---
try:
    CONTROL_QUEST = pkg_resources.resource_filename("simglucose", "params/Quest.csv")
    PATIENT_PARA_FILE = pkg_resources.resource_filename(
        "simglucose", "params/vpatient_params.csv"
    )
except ImportError:
    CONTROL_QUEST = "params/Quest.csv"
    PATIENT_PARA_FILE = "params/vpatient_params.csv"


# ============================================================================
# 1) Generic ARX Utilities
# ============================================================================
def arx_predict(theta: np.ndarray, phi: np.ndarray) -> float:
    """Dot product for prediction: y = theta^T * phi"""
    return float(np.dot(theta, phi))


def construct_phi_generic(
    g_hist: Deque[float],
    u_hist: Deque[float],
    m_hist: Deque[float],
    order: int,
    *,
    u_delay: int = 0,
    m_delay: int = 0,
) -> np.ndarray:
    """
    Construct regression vector phi for generic order n.
    phi = [y(k), ..., y(k-n+1), u(k-d), ..., u(k-d-n+1), m(k-d), ..., m(k-d-n+1), 1]
    Assumes deque has newest element at [-1].
    """

    def _safe_deque_get(dq: Deque[float], idx: int) -> float:
        if not dq:
            return 0.0
        try:
            return float(dq[idx])
        except Exception:
            try:
                return float(dq[0])
            except Exception:
                return 0.0

    u_delay = int(u_delay)
    m_delay = int(m_delay)

    phi = []
    # AR part (Glucose history): y(k-1..k-order)
    for i in range(order):
        phi.append(_safe_deque_get(g_hist, -(i + 1)))

    # Input B part (Insulin history): u(k-1-u_delay ..)
    for i in range(order):
        phi.append(_safe_deque_get(u_hist, -(u_delay + i + 1)))

    # Input C part (Meal history): m(k-1-m_delay ..)
    for i in range(order):
        phi.append(_safe_deque_get(m_hist, -(m_delay + i + 1)))

    # Bias (Intercept)
    phi.append(1.0)

    return np.array(phi, dtype=float)


def rollout_arx_generic(
    theta: np.ndarray,
    g_hist_list: List[float],
    u_hist_list: List[float],
    m_hist_list: List[float],
    u_future: np.ndarray,
    m_future: np.ndarray,
    order: int,
    *,
    u_delay: int = 0,
    m_delay: int = 0,
) -> np.ndarray:
    """
    Simulate ARX model forward (Grid Search fallback).
    """

    def _safe_hist_value(hist: List[float], idx_time: int) -> float:
        # idx_time is negative for history access, e.g. -1 means last.
        if not hist:
            return 0.0
        if idx_time >= 0:
            raise ValueError("idx_time must be negative for history access")
        if -idx_time > len(hist):
            return float(hist[0])
        return float(hist[idx_time])

    u_delay = int(u_delay)
    m_delay = int(m_delay)

    # Parse theta
    n = order
    theta_a = theta[0:n]
    theta_b = theta[n : 2 * n]
    theta_c = theta[2 * n : 3 * n]
    theta_d = theta[-1]

    pred_len = len(u_future)
    g_pred = np.zeros(pred_len + 1)
    g_pred[0] = g_hist_list[-1]  # current state

    # Temporary buffers for simulation
    curr_g_hist = list(g_hist_list)  # copy
    curr_u_hist = list(u_hist_list)
    curr_m_hist = list(m_hist_list)

    for k in range(pred_len):
        val = float(theta_d)

        # AR terms: y(k-i)
        for i in range(n):
            val += float(theta_a[i]) * float(curr_g_hist[-(i + 1)])

        # B/C terms with delays: u(k-u_delay-i), m(k-m_delay-i)
        for i in range(n):
            idx_u = int(k) - int(u_delay) - int(i)
            if idx_u >= 0:
                u_val_eff = float(u_future[idx_u])
            else:
                u_val_eff = _safe_hist_value(curr_u_hist, idx_u)
            val += float(theta_b[i]) * float(u_val_eff)

            idx_m = int(k) - int(m_delay) - int(i)
            if idx_m >= 0:
                m_val_eff = float(m_future[idx_m]) if idx_m < len(m_future) else 0.0
            else:
                m_val_eff = _safe_hist_value(curr_m_hist, idx_m)
            val += float(theta_c[i]) * float(m_val_eff)

        g_pred[k + 1] = val

        # Update buffers for next step
        curr_g_hist.append(val)
        curr_u_hist.append(float(u_future[k]))
        curr_m_hist.append(float(m_future[k]) if k < len(m_future) else 0.0)

    return g_pred


# ============================================================================
# 2) RLS with Variable Forgetting Factor
# ============================================================================
def vff_lambda(err, lam_base, lam_min, lam_max, kappa, err_ref) -> float:
    frac = min(1.0, abs(float(err)) / max(1e-6, float(err_ref)))
    lam = float(lam_base) - float(kappa) * float(frac)
    return float(np.clip(lam, float(lam_min), float(lam_max)))


def rls_vff_update(theta, P, phi, y_meas, lam) -> Tuple[np.ndarray, np.ndarray, float]:
    y_hat = float(arx_predict(theta, phi))
    err = float(y_meas) - float(y_hat)
    denom = float(lam) + float(phi.T @ P @ phi)
    # Guard against numerical blow-ups if P becomes ill-conditioned/indefinite.
    if denom <= 1e-6:
        return theta, P, err

    K = (P @ phi) / denom
    theta_new = theta + K * err
    P_new = (P - np.outer(K, phi) @ P) / float(lam)
    P_new = 0.5 * (P_new + P_new.T)

    # Robustify covariance:
    # - keep only diagonal (reduces spurious cross-covariances under poor excitation)
    # - enforce a floor to avoid "overconfidence" (phi^T P phi -> tiny => huge updates later)
    try:
        diag = np.diag(P_new).astype(float)
        diag = np.clip(diag, 1.0, 1e6)
        P_new = np.diag(diag)
    except Exception:
        pass
    return theta_new, P_new, err


def ar_spectral_radius_companion(a: np.ndarray) -> float:
    """
    Spectral radius of the AR companion matrix for coefficients a.

    Interprets AR part as: y[k] = a1*y[k-1] + ... + an*y[k-n] + ...
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size == 0:
        return 0.0
    if a.size == 1:
        return float(abs(a[0]))
    comp = np.zeros((a.size, a.size), dtype=float)
    comp[0, :] = a
    comp[1:, :-1] = np.eye(a.size - 1, dtype=float)
    try:
        vals = np.linalg.eigvals(comp)
        return float(np.max(np.abs(vals)))
    except Exception:
        return float("nan")


def project_theta(
    theta: np.ndarray,
    order: int,
    *,
    b_min: float = -5.0,
    b_max: float = 0.0,
    c_min: float = 0.0,
    c_max: float = 1.0,
) -> np.ndarray:
    """
    Stability projection for generic order.
    Loosely clips insulin gains (b) to be non-positive and meal gains (c) to be positive.
    """
    n = order
    new_theta = theta.copy()

    try:
        a = np.asarray(new_theta[:n], dtype=float)
        radius_max = 0.98
        for _ in range(3):
            sr = ar_spectral_radius_companion(a)
            if not np.isfinite(sr) or sr <= radius_max:
                break
            scale = float(radius_max / (sr + 1e-12))
            a = a * scale
        new_theta[:n] = a
    except Exception:
        pass

    # Clip Insulin Gains (b) -> insulin effect should not raise BG in this convention.
    for i in range(n, 2 * n):
        # Also cap magnitude to avoid numerical blow-ups (empirically b is O(1) from result.json).
        new_theta[i] = np.clip(new_theta[i], float(b_min), float(b_max))

    # Clip Meal Gains (c) -> must be positive (carbs raise BG)
    for i in range(2 * n, 3 * n):
        # Also cap magnitude (empirically c is small from result.json).
        new_theta[i] = np.clip(new_theta[i], float(c_min), float(c_max))

    # Clip Intercept (d) -> reasonable bounds
    new_theta[-1] = np.clip(new_theta[-1], -500.0, 500.0)

    return new_theta


def normalize_theta_conventions(
    theta: np.ndarray, order: int
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Normalize offline-identified theta to this controller's sign conventions.

    Empirically, some `result.json` entries encode insulin and/or meal inputs with flipped signs.
    This helper flips the signs of the B (insulin) and C (meal) blocks when their medians imply
    an inconsistent convention, and returns a small metadata dict for logging/debug.
    """
    t = np.asarray(theta, dtype=float).copy()
    n = int(order)
    meta: Dict[str, object] = {"flip_b": False, "flip_c": False}
    if n <= 0 or t.size < (3 * n + 1):
        return t, meta

    b = t[n : 2 * n]
    c = t[2 * n : 3 * n]
    try:
        b_med = float(np.median(b[np.isfinite(b)]))
    except Exception:
        b_med = float("nan")
    try:
        c_med = float(np.median(c[np.isfinite(c)]))
    except Exception:
        c_med = float("nan")

    # Controller convention: uhat > 0 means MORE insulin, which should LOWER BG => B <= 0.
    if np.isfinite(b_med) and b_med > 0.0:
        t[n : 2 * n] = -t[n : 2 * n]
        meta["flip_b"] = True

    # Controller convention: meal > 0 should RAISE BG => C >= 0.
    if np.isfinite(c_med) and c_med < 0.0:
        t[2 * n : 3 * n] = -t[2 * n : 3 * n]
        meta["flip_c"] = True

    return t, meta


# ============================================================================
# 3) Generic MPC solve (CVXPY) - [FIXED CRITICAL INDEX BUG]
# ============================================================================
@dataclass(frozen=True)
class MpcSolveResult:
    u0: float
    g_pred: np.ndarray  # shape: (Hp+1,) including g[0]
    u_seq: np.ndarray  # shape: (Hu,)
    problem_status: str
    solver: str
    num_iters: float
    solve_time: float
    objective: float
    error: str


def solve_mpc_generic(
    theta: np.ndarray,
    g_hist_list: List[float],  # List ending with g_curr
    u_hist_list: List[float],  # List ending with u_prev
    m_hist_list: List[float],  # List ending with m_prev
    m_future: Union[float, np.ndarray],  # meal disturbance seq (len>=Hp) or scalar
    target: float,
    g_bounds: Dict[str, float],
    u_bounds: Dict[str, float],
    weights: Dict[str, float],
    horizons: Dict[str, int],
    *,
    u_delay: int = 0,
    m_delay: int = 0,
    return_rollout: bool = False,
) -> Optional[Union[float, MpcSolveResult]]:

    Hp = horizons["pred"]
    Hu = horizons["control"]
    order = int((len(theta) - 1) // 3)

    # Parse Theta
    theta_a = theta[0:order]
    theta_b = theta[order : 2 * order]
    theta_c = theta[2 * order : 3 * order]
    theta_d = theta[-1]

    u_delay = int(u_delay)
    m_delay = int(m_delay)

    if isinstance(m_future, (float, int)):
        m_seq = np.zeros(int(Hp), dtype=float)
        if int(Hp) > 0:
            m_seq[0] = float(m_future)
    else:
        m_seq = np.asarray(m_future, dtype=float).reshape(-1)
        if m_seq.size < int(Hp):
            m_seq = np.pad(
                m_seq,
                (0, int(Hp) - int(m_seq.size)),
                mode="constant",
                constant_values=0.0,
            )
        else:
            m_seq = m_seq[: int(Hp)]

    def _safe_hist_value(hist: List[float], idx_time: int) -> float:
        if not hist:
            return 0.0
        if idx_time >= 0:
            raise ValueError("idx_time must be negative for history access")
        if -idx_time > len(hist):
            return float(hist[0])
        return float(hist[idx_time])

    if _HAS_CVXPY:
        # Variables
        g = cp.Variable(Hp + 1)  # g[0]...g[Hp]
        u = cp.Variable(Hu)  # u[0]...u[Hu-1]
        s_low = cp.Variable(Hp, nonneg=True)
        s_high = cp.Variable(Hp, nonneg=True)

        constraints = []
        cost = 0.0

        # Initial State
        constraints.append(g[0] == g_hist_list[-1])

        # Dynamics Loop
        for k in range(Hp):
            # --- [FIXED] SAFE INPUT ACCESS ---
            # Determine "current" u for this prediction step
            # If k >= Hu, we hold the last input u[Hu-1] (Input Blocking)
            if k < Hu:
                u_curr = u[k]
            else:
                u_curr = u[Hu - 1]

            # Determine "previous" u for regularization
            if k == 0:
                u_prev = float(u_hist_list[-1])
            elif (k - 1) < Hu:
                u_prev = u[k - 1]
            else:
                u_prev = u[Hu - 1]

            # --- ARX Equation Construction ---
            expr = theta_d

            # A terms: Sum a_i * y(k-i)
            for i in range(order):
                if (k - i) >= 0:
                    expr += theta_a[i] * g[k - i]
                else:
                    hist_idx = -1 - (i - k)
                    expr += theta_a[i] * float(g_hist_list[hist_idx])

            # B terms: Sum b_i * u(k-u_delay-i)
            for i in range(order):
                idx_time = int(k) - int(u_delay) - int(i)
                if idx_time >= 0:
                    if idx_time < Hu:
                        u_var = u[idx_time]
                    else:
                        u_var = u[Hu - 1]
                    expr += theta_b[i] * u_var
                else:
                    expr += theta_b[i] * float(_safe_hist_value(u_hist_list, idx_time))

            # C terms: Sum c_i * m(k-m_delay-i)
            for i in range(order):
                idx_time = int(k) - int(m_delay) - int(i)
                if idx_time >= 0:
                    expr += theta_c[i] * float(m_seq[idx_time])
                else:
                    expr += theta_c[i] * float(_safe_hist_value(m_hist_list, idx_time))

            constraints.append(g[k + 1] == expr)

            # Constraints
            # Only apply bound constraints to real variables (k < Hu)
            if k < Hu:
                constraints.append(u_curr >= u_bounds["min"])
                constraints.append(u_curr <= u_bounds["max"])

            # Soft constraints on Glucose
            constraints.append(g[k + 1] + s_low[k] >= g_bounds["safety"])
            constraints.append(g[k + 1] - s_high[k] <= g_bounds["ceiling"])

            # Cost function
            cost += weights["over"] * cp.square(cp.pos(g[k + 1] - g_bounds["upper"]))
            cost += weights["under"] * cp.square(cp.pos(g_bounds["lower"] - g[k + 1]))
            cost += weights["track_over"] * cp.square(cp.pos(g[k + 1] - target))
            cost += weights["track_under"] * cp.square(cp.pos(target - g[k + 1]))
            cost += weights["safety"] * cp.square(s_low[k])
            cost += weights["ceiling"] * cp.square(s_high[k])

            # Input regularization
            cost += weights["r_u"] * cp.square(u_curr)
            cost += weights["r_du"] * cp.square(u_curr - u_prev)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                max_iter=10000,
                polish=True,
                eps_abs=1e-3,
                eps_rel=1e-3,
            )
        except Exception as e:
            if not return_rollout:
                return None
            return MpcSolveResult(
                u0=float("nan"),
                g_pred=np.full(Hp + 1, np.nan, dtype=float),
                u_seq=np.full(Hu, np.nan, dtype=float),
                problem_status="exception",
                solver="OSQP",
                num_iters=float("nan"),
                solve_time=float("nan"),
                objective=float("nan"),
                error=str(e),
            )

        # CVXPY may return a usable u.value even if status is not "optimal".
        status = str(problem.status)
        usable = (u.value is not None) and status in (
            "optimal",
            "optimal_inaccurate",
            "user_limit",
        )
        if usable:
            u0 = float(u.value[0])
            if not return_rollout:
                return u0

            g_pred = (
                np.asarray(g.value, dtype=float).reshape(-1)
                if g.value is not None
                else None
            )
            u_seq = np.asarray(u.value, dtype=float).reshape(-1)
            if g_pred is None or g_pred.size != (Hp + 1):
                g_pred = np.full(Hp + 1, np.nan, dtype=float)
            if u_seq.size != Hu:
                u_seq = np.full(Hu, np.nan, dtype=float)

            stats = getattr(problem, "solver_stats", None)
            num_iters = (
                float(getattr(stats, "num_iters", float("nan")))
                if stats is not None
                else float("nan")
            )
            solve_time = (
                float(getattr(stats, "solve_time", float("nan")))
                if stats is not None
                else float("nan")
            )
            solver_name = (
                str(getattr(stats, "solver_name", "OSQP"))
                if stats is not None
                else "OSQP"
            )
            obj = float(problem.value) if problem.value is not None else float("nan")

            return MpcSolveResult(
                u0=u0,
                g_pred=g_pred,
                u_seq=u_seq,
                problem_status=status,
                solver=solver_name,
                num_iters=num_iters,
                solve_time=solve_time,
                objective=obj,
                error="",
            )

    # ------------------------------------------------------------------
    # Fallback solver (SciPy) when CVXPY is unavailable.
    # Uses direct simulation rollout + bounded optimization over u[0..Hu-1].
    # ------------------------------------------------------------------
    if not _HAS_SCIPY or minimize is None:
        return None

    try:
        umin = float(u_bounds["min"])
        umax = float(u_bounds["max"])
        if not np.isfinite(umin) or not np.isfinite(umax) or umax < umin:
            raise ValueError("Invalid u bounds")
    except Exception:
        return None

    # Initial guess: keep previous input.
    try:
        u_prev0 = float(u_hist_list[-1]) if u_hist_list else 0.0
    except Exception:
        u_prev0 = 0.0
    u0 = float(np.clip(u_prev0, umin, umax))
    x0 = np.full(int(Hu), u0, dtype=float)

    bounds = [(umin, umax) for _ in range(int(Hu))]

    def _objective(u_seq: np.ndarray) -> float:
        u_seq = np.asarray(u_seq, dtype=float).reshape(-1)
        if u_seq.size != int(Hu):
            return float("inf")

        # Build u_future with input blocking after Hu.
        if int(Hp) <= int(Hu):
            u_future = u_seq[: int(Hp)]
        else:
            tail = np.full(int(Hp) - int(Hu), float(u_seq[-1]), dtype=float)
            u_future = np.concatenate([u_seq, tail])

        try:
            g_roll = rollout_arx_generic(
                theta,
                g_hist_list,
                u_hist_list,
                m_hist_list,
                u_future=u_future,
                m_future=m_seq,
                order=int(order),
                u_delay=int(u_delay),
                m_delay=int(m_delay),
            )
        except Exception:
            return float("inf")

        if not isinstance(g_roll, np.ndarray) or g_roll.size < (int(Hp) + 1):
            return float("inf")

        g_bounds_lower = float(g_bounds["lower"])
        g_bounds_upper = float(g_bounds["upper"])
        g_safety = float(g_bounds["safety"])
        g_ceiling = float(g_bounds["ceiling"])

        w_over = float(weights["over"])
        w_under = float(weights["under"])
        w_track_over = float(weights["track_over"])
        w_track_under = float(weights["track_under"])
        w_safety = float(weights["safety"])
        w_ceiling = float(weights["ceiling"])
        r_u = float(weights["r_u"])
        r_du = float(weights["r_du"])

        target_f = float(target)

        total_cost = 0.0
        u_prev = float(u_prev0)

        for k in range(int(Hp)):
            gk = float(g_roll[k + 1])
            # Piecewise penalties (same shape as CVXPY formulation).
            over = max(0.0, gk - g_bounds_upper)
            under = max(0.0, g_bounds_lower - gk)
            track_over = max(0.0, gk - target_f)
            track_under = max(0.0, target_f - gk)
            safety_slack = max(0.0, g_safety - gk)
            ceiling_slack = max(0.0, gk - g_ceiling)

            total_cost += w_over * (over**2)
            total_cost += w_under * (under**2)
            total_cost += w_track_over * (track_over**2)
            total_cost += w_track_under * (track_under**2)
            total_cost += w_safety * (safety_slack**2)
            total_cost += w_ceiling * (ceiling_slack**2)

            u_curr = float(u_seq[k]) if k < int(Hu) else float(u_seq[-1])
            total_cost += r_u * (u_curr**2)
            total_cost += r_du * ((u_curr - u_prev) ** 2)
            u_prev = u_curr

        return float(total_cost)

    try:
        res = minimize(
            _objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-6},
        )
    except Exception as e:
        if not return_rollout:
            return None
        return MpcSolveResult(
            u0=float("nan"),
            g_pred=np.full(int(Hp) + 1, np.nan, dtype=float),
            u_seq=np.full(int(Hu), np.nan, dtype=float),
            problem_status="exception",
            solver="SCIPY_LBFGS_B",
            num_iters=float("nan"),
            solve_time=float("nan"),
            objective=float("nan"),
            error=str(e),
        )

    if not getattr(res, "success", False) or res.x is None:
        if not return_rollout:
            return None
        return MpcSolveResult(
            u0=float("nan"),
            g_pred=np.full(int(Hp) + 1, np.nan, dtype=float),
            u_seq=np.full(int(Hu), np.nan, dtype=float),
            problem_status="failed",
            solver="SCIPY_LBFGS_B",
            num_iters=float(getattr(res, "nit", float("nan"))),
            solve_time=float("nan"),
            objective=float(getattr(res, "fun", float("nan"))),
            error=str(getattr(res, "message", "")),
        )

    u_seq = np.asarray(res.x, dtype=float).reshape(-1)
    u_seq = np.clip(u_seq, umin, umax)
    u0_opt = float(u_seq[0]) if u_seq.size else float(u0)

    if not return_rollout:
        return u0_opt

    # Export rollout under the optimized u sequence.
    if int(Hp) <= int(Hu):
        u_future = u_seq[: int(Hp)]
    else:
        tail = np.full(int(Hp) - int(Hu), float(u_seq[-1]), dtype=float)
        u_future = np.concatenate([u_seq, tail])
    try:
        g_pred = rollout_arx_generic(
            theta,
            g_hist_list,
            u_hist_list,
            m_hist_list,
            u_future=u_future,
            m_future=m_seq,
            order=int(order),
            u_delay=int(u_delay),
            m_delay=int(m_delay),
        )
        g_pred = np.asarray(g_pred, dtype=float).reshape(-1)
        if g_pred.size != (int(Hp) + 1):
            g_pred = np.full(int(Hp) + 1, np.nan, dtype=float)
    except Exception:
        g_pred = np.full(int(Hp) + 1, np.nan, dtype=float)

    return MpcSolveResult(
        u0=u0_opt,
        g_pred=g_pred,
        u_seq=u_seq,
        problem_status="optimal",
        solver="SCIPY_LBFGS_B",
        num_iters=float(getattr(res, "nit", float("nan"))),
        solve_time=float("nan"),
        objective=float(getattr(res, "fun", float("nan"))),
        error="",
    )


# ============================================================================
# 4) Main controller class
# ============================================================================
class ArxRlsVffMpcController(Controller):
    """
    ARX + RLS-VFF controller with MPC action selection.

    By default, action selection is delegated to the workspace
    `simglucose.controller.mpc_ctrller.MPCController`. Set
    `workspace_mpc_prediction_model="arx"` to keep the workspace Zone-MPC cost
    and bounds while evaluating them on ARX-RLS rollouts. Set
    `use_workspace_mpc=False` only when the legacy local ARX MPC solve path is
    explicitly needed.

    Output modes:
    - use_basal_modulation=True (default): insulin = basal * (1 + uhat_mpc) + optional_bolus
      Traditional BBC mode where MPC output modulates the basal rate.
    - use_basal_modulation=False: insulin = basal + uhat_mpc * basal + optional_bolus
      Pure MPC mode (mathematically equivalent but clearer semantics).

    Note: Both modes actually produce the same result, but the flag is provided for
    clarity and potential future extension to absolute insulin control.
    """

    def __init__(
        self,
        init_state=None,
        target: float = 150.0,
        g_lower: float = 100.0,
        g_upper: float = 180.0,
        g_safety: float = 120.0,
        g_ceiling: float = 300.0,
        sample_time_min: float = 5.0,
        # MPC Horizons
        pred_horizon_min: float = 120.0,
        control_horizon_min: float = 45.0,
        # RLS/VFF
        theta0: Optional[np.ndarray] = None,  # Can pass Order 4 theta here
        P0: Union[float, np.ndarray] = 100.0,
        lam_base: float = 0.995,
        kappa: float = 0.01,
        rls_err_clip: float = 60.0,
        rls_err_deadzone: float = 2.0,
        rls_phi_min: float = 1e-3,
        # ARX structure (literature often prefers ~5th order at 5-min sampling)
        min_order: int = 5,
        u_delay_steps: int = 0,
        m_delay_steps: int = 0,
        use_meal_filter: bool = True,
        meal_filter_tau_min: float = 15.0,
        use_meal_buffer: bool = True,
        meal_release_g_per_min: float = 5.0,
        meal_c_max: float = 5.0,
        theta_input_mode: str = "abs_insulin",
        # Weights
        w_track_over: float = 0.5,
        w_track_under: float = 10.0,
        w_over: float = 5.0,
        w_under: float = 5000.0,
        w_safety: float = 15000.0,
        w_ceiling: float = 20.0,
        r_u: float = 2.0,
        r_du: float = 30.0,
        # Control bounds (for pure MPC mode, these represent absolute insulin rate offsets)
        uhat_min: float = -0.01,
        uhat_max_rel: float = 0.1,
        u_abs_cap: float = 0.15,
        # Optional "boost" caps for high BG / around meals (anti-hyperglycemia)
        uhat_max_rel_boost: float = 6.0,
        u_abs_cap_boost: float = 0.15,
        boost_bg_threshold: float = 180.0,
        boost_on_high_bg: bool = False,
        boost_steps_after_meal: int = 12,
        meal_forecast_steps: int = 8,
        enable_fasting_hold: bool = False,
        fasting_hold_band: float = 40.0,
        enable_meal_bolus: bool = False,
        meal_bolus_scale: float = 0.2,
        meal_uhat_min: float = 0.0,
        # Pure MPC mode (no BBC modulation)
        use_basal_modulation: bool = False,
        # Warmup
        warmup: bool = False,
        warmup_controller: Optional[Controller] = None,
        # File params
        param_file: str = "result.json",
        allow_fallback_theta: bool = False,
        # Reuse the workspace MPC controller for action selection.
        use_workspace_mpc: bool = True,
        workspace_mpc_prediction_model: str = "workspace",
        workspace_mpc_variant: str = "adaptive",
        workspace_mpc_use_iob_constraint: bool = False,
        workspace_mpc_max_insulin_units_per_sample: float = 1.0,
        workspace_mpc_max_insulin_tdi_fraction: Optional[float] = None,
        workspace_mpc_model_gain_factor: float = 1.0,
        workspace_mpc_r_plus_scale: float = 1.0,
        workspace_mpc_enable_low_glucose_suspend: bool = False,
    ) -> None:
        super().__init__(init_state=init_state)

        # Load Simglucose generic params
        try:
            self.quest = pd.read_csv(CONTROL_QUEST)
            self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        except Exception:
            self.patient_params = None

        self.target = target
        self.sample_time_min = sample_time_min

        # Bounds & Weights
        self.g_bounds = {
            "lower": g_lower,
            "upper": g_upper,
            "safety": g_safety,
            "ceiling": g_ceiling,
        }
        self.mpc_weights = {
            "track_over": w_track_over,
            "track_under": w_track_under,
            "over": w_over,
            "under": w_under,
            "safety": w_safety,
            "ceiling": w_ceiling,
            "r_u": r_u,
            "r_du": r_du,
        }

        self.pred_steps = int(pred_horizon_min / sample_time_min)
        self.ctrl_steps = int(control_horizon_min / sample_time_min)
        self.use_workspace_mpc = bool(use_workspace_mpc)
        self.workspace_mpc_prediction_model = str(
            workspace_mpc_prediction_model
        ).strip().lower()
        if self.workspace_mpc_prediction_model not in ("workspace", "arx"):
            raise ValueError(
                "workspace_mpc_prediction_model must be 'workspace' or 'arx'"
            )
        self.uhat_min = uhat_min
        self.uhat_max_rel = uhat_max_rel
        self.u_abs_cap = u_abs_cap
        self.uhat_max_rel_boost = float(uhat_max_rel_boost)
        self.u_abs_cap_boost = float(u_abs_cap_boost)
        self.boost_bg_threshold = float(boost_bg_threshold)
        self.boost_on_high_bg = bool(boost_on_high_bg)
        self.boost_steps_after_meal = int(boost_steps_after_meal)
        self.meal_forecast_steps = int(meal_forecast_steps)
        self.enable_fasting_hold = bool(enable_fasting_hold)
        self.fasting_hold_band = float(fasting_hold_band)
        self.enable_meal_bolus = bool(enable_meal_bolus)
        self.meal_bolus_scale = float(meal_bolus_scale)
        self.meal_uhat_min = float(meal_uhat_min)
        self.use_basal_modulation = bool(use_basal_modulation)
        self._boost_remaining = 0

        # --- RLS Init ---
        self.lam_base = lam_base
        self.kappa = kappa
        self.rls_err_clip = float(rls_err_clip)
        self.rls_err_deadzone = float(rls_err_deadzone)
        self.rls_phi_min = float(rls_phi_min)
        self.min_order = int(min_order)
        self.u_delay_steps = int(u_delay_steps)
        self.m_delay_steps = int(m_delay_steps)
        self.use_meal_filter = bool(use_meal_filter)
        self.meal_filter_tau_min = float(meal_filter_tau_min)
        self.use_meal_buffer = bool(use_meal_buffer)
        self.meal_release_g_per_min = float(meal_release_g_per_min)
        self.meal_c_max = float(meal_c_max)
        self.theta_input_mode = str(theta_input_mode).strip().lower()
        self._meal_effect = 0.0
        self._meal_pending_g = 0.0
        self._meal_raw_prev = 0.0
        self._meal_accum_g = 0.0
        self._theta_scaled_for_basal = False
        self._theta_scaled_basal = None
        self._P0_init = P0
        self._theta_bank = None

        # Try to load theta from result.json if not provided
        self.theta = None
        if theta0 is not None:
            self.theta = np.array(theta0, dtype=float)
        else:
            # Attempt to load from result.json (P01 as default)
            try:
                with open(param_file, "r") as f:
                    data = json.load(f)
                    self._theta_bank = data
                    # Pick P01 or first available key
                    first_key = next(iter(data))
                    if first_key == "__meta__":
                        first_key = list(data.keys())[1]

                    loaded_theta = data[first_key]["theta"]
                    self.theta = np.array(loaded_theta, dtype=float)
                    LOGGER.info(
                        f"Loaded theta from {param_file} (Patient: {first_key}, Order: {data[first_key]['order']})"
                    )
            except Exception as e:
                if not bool(allow_fallback_theta):
                    raise ValueError(
                        "ARX theta initialization requires theta0 or a readable "
                        f"param_file with theta data: {param_file!r}"
                    ) from e
                LOGGER.warning(
                    f"Could not load theta from file: {e}. Using Order 2 fallback."
                )
                # Explicit opt-in fallback for exploratory simulations only.
                self.theta = np.array([1.5, -0.5, -5.0, -2.0, 2.0, 1.0, 0.0])

        if self.theta is None:
            raise ValueError("ARX theta initialization produced no parameter vector")
        if (len(self.theta) - 1) % 3 != 0 or len(self.theta) < 4:
            raise ValueError(
                "ARX theta must be [a1..an, b1..bn, c1..cn, d] with n >= 1"
            )

        # Detect/pad order (minimum order is often ~5 at 5-min sampling for better fit).
        loaded_order = int((len(self.theta) - 1) // 3)
        desired_order = max(int(self.min_order), int(loaded_order))
        if desired_order != loaded_order:
            try:
                th = np.asarray(self.theta, dtype=float).reshape(-1)
                n0 = int(loaded_order)
                n1 = int(desired_order)
                a0 = th[:n0]
                b0 = th[n0 : 2 * n0]
                c0 = th[2 * n0 : 3 * n0]
                d0 = float(th[-1])
                a1 = np.zeros(n1, dtype=float)
                b1 = np.zeros(n1, dtype=float)
                c1 = np.zeros(n1, dtype=float)
                a1[:n0] = a0
                b1[:n0] = b0
                c1[:n0] = c0
                self.theta = np.concatenate([a1, b1, c1, np.array([d0], dtype=float)])
            except Exception:
                pass
        self.order = int((len(self.theta) - 1) // 3)
        LOGGER.info(f"ARX Controller Initialized with Order={self.order}")

        # Normalize theta sign conventions to match this controller.
        self.theta, _meta = normalize_theta_conventions(self.theta, self.order)
        if _meta.get("flip_b") or _meta.get("flip_c"):
            LOGGER.info(
                "Normalized theta conventions"
                + f" (flip_b={_meta.get('flip_b')}, flip_c={_meta.get('flip_c')})"
            )
        self.theta_prior = self.theta.copy()
        self._theta_unscaled = self.theta.copy()

        # P Matrix
        if isinstance(P0, (float, int)):
            # Diagonal P with a floor is much more stable under poor excitation.
            self.P = np.eye(len(self.theta)) * float(P0)
            try:
                diag = np.clip(np.diag(self.P).astype(float), 1.0, 1e6)
                self.P = np.diag(diag)
            except Exception:
                pass
        else:
            self.P = np.array(P0, dtype=float)
        self.P_prior = self.P.copy()

        # Histories must retain all delayed regressors used by phi.
        self.history_len = int(
            max(
                self.order + 1,
                self.order + max(self.u_delay_steps, self.m_delay_steps) + 1,
            )
        )
        self.g_hist = deque(maxlen=self.history_len)
        self.uhat_hist = deque(maxlen=self.history_len)
        self.m_hist = deque(maxlen=self.history_len)

        self.warmup = warmup
        self.teacher = (
            warmup_controller if warmup_controller else BBController(target=target)
        )
        self.workspace_mpc = None
        if self.use_workspace_mpc:
            self.workspace_mpc = MPCController(
                variant=str(workspace_mpc_variant),
                prediction_horizon=max(1, int(self.pred_steps)),
                control_horizon=max(1, min(int(self.ctrl_steps), int(self.pred_steps))),
                target_glucose=float(self.target),
                model_sample_time=float(self.sample_time_min),
                announce_meals=bool(self.enable_meal_bolus),
                use_iob_constraint=bool(workspace_mpc_use_iob_constraint),
                max_insulin_units_per_sample=float(
                    workspace_mpc_max_insulin_units_per_sample
                ),
                max_insulin_tdi_fraction=workspace_mpc_max_insulin_tdi_fraction,
                model_gain_factor=float(workspace_mpc_model_gain_factor),
                r_plus_scale=float(workspace_mpc_r_plus_scale),
                meal_bolus_scale=float(self.meal_bolus_scale),
                enable_low_glucose_suspend=bool(
                    workspace_mpc_enable_low_glucose_suspend
                ),
            )
        self._first_step = True
        self._patient_name = None
        self._basal_u_per_min = 0.015
        self._uhat_max = uhat_max_rel

        # Logging
        self._rls_lam_hist = []
        self._rls_err_hist = []
        self._rls_err_raw_hist = []
        self._rls_theta_hist = []
        self._bg_hat_hist = []
        self._rls_P_trace_hist = []
        self._rls_denom_hist = []
        self._rls_phi_norm_hist = []
        self._rls_P_min_eig_hist = []
        self._rls_ar_rho_hist = []
        self._mpc_g_pred_hist = []  # per-step MPC predicted BG (Hp steps ahead)
        self._mpc_u_seq_hist = []  # per-step MPC planned u sequence (Hu steps)
        self._mpc_status_hist = []  # per-step MPC status string
        self._mpc_problem_status_hist = []  # CVXPY problem.status (or fallback marker)
        self._mpc_solver_hist = []  # solver name
        self._mpc_num_iters_hist = []  # iterations (if available)
        self._mpc_solve_time_hist = []  # seconds (if available)
        self._mpc_objective_hist = []  # objective value (if available)
        self._mpc_error_hist = []  # exception / error message (if any)
        self._meal_raw_hist = []  # per-step observed meal rate (g/min)
        self._meal_model_hist = []  # per-step meal signal used by ARX (raw or filtered)
        self._meal_input_hist = []  # per-step meal input after buffering (g per dt)

    def _load_theta_for_patient(self, patient_name: Optional[str]) -> bool:
        if not patient_name or not isinstance(self._theta_bank, dict):
            return False

        entry = self._theta_bank.get(patient_name)
        if not isinstance(entry, dict) or "theta" not in entry:
            return False

        theta = np.asarray(entry["theta"], dtype=float).reshape(-1)
        if (len(theta) - 1) % 3 != 0 or len(theta) < 4:
            raise ValueError(
                f"Invalid theta for patient {patient_name!r}: expected [a,b,c,d] blocks"
            )

        self.theta = theta
        loaded_order = int((len(self.theta) - 1) // 3)
        desired_order = max(int(self.min_order), int(loaded_order))
        if desired_order != loaded_order:
            try:
                th = np.asarray(self.theta, dtype=float).reshape(-1)
                n0 = int(loaded_order)
                n1 = int(desired_order)
                a0 = th[:n0]
                b0 = th[n0 : 2 * n0]
                c0 = th[2 * n0 : 3 * n0]
                d0 = float(th[-1])
                a1 = np.zeros(n1, dtype=float)
                b1 = np.zeros(n1, dtype=float)
                c1 = np.zeros(n1, dtype=float)
                a1[:n0] = a0
                b1[:n0] = b0
                c1[:n0] = c0
                self.theta = np.concatenate([a1, b1, c1, np.array([d0], dtype=float)])
            except Exception:
                pass

        self.order = int((len(self.theta) - 1) // 3)
        self.theta, _meta = normalize_theta_conventions(self.theta, self.order)
        if _meta.get("flip_b") or _meta.get("flip_c"):
            LOGGER.info(
                "Normalized patient theta conventions"
                + f" (flip_b={_meta.get('flip_b')}, flip_c={_meta.get('flip_c')})"
            )
        self.theta_prior = self.theta.copy()
        self._theta_unscaled = self.theta.copy()

        P0 = self._P0_init
        if isinstance(P0, (float, int)):
            self.P = np.eye(len(self.theta)) * float(P0)
            try:
                diag = np.clip(np.diag(self.P).astype(float), 1.0, 1e6)
                self.P = np.diag(diag)
            except Exception:
                pass
        else:
            self.P = np.array(P0, dtype=float)
        self.P_prior = self.P.copy()

        self.history_len = int(
            max(
                self.order + 1,
                self.order + max(self.u_delay_steps, self.m_delay_steps) + 1,
            )
        )
        self.g_hist = deque(maxlen=self.history_len)
        self.uhat_hist = deque(maxlen=self.history_len)
        self.m_hist = deque(maxlen=self.history_len)

        self._first_step = True
        self._theta_scaled_for_basal = False
        self._theta_scaled_basal = None
        self._meal_effect = 0.0
        self._meal_pending_g = 0.0
        self._meal_raw_prev = 0.0
        self._meal_accum_g = 0.0
        self._boost_remaining = 0
        if self.workspace_mpc is not None:
            self.workspace_mpc.reset()

        LOGGER.info(
            "Loaded patient-specific theta from bank (Patient: %s, Order: %s)",
            patient_name,
            self.order,
        )
        return True

    def _meal_filter_alpha(self) -> float:
        dt = float(self.sample_time_min)
        tau = float(self.meal_filter_tau_min)
        if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(tau) or tau <= 0.0:
            return 0.0
        tau = max(tau, dt)
        return float(np.exp(-dt / tau))

    def _compute_meal_input(self, meal_raw: float) -> float:
        meal_raw = float(max(0.0, float(meal_raw)))
        meal_g = meal_raw * float(self.sample_time_min)
        if not self.use_meal_buffer:
            return float(meal_g)

        self._meal_pending_g = float(self._meal_pending_g) + float(meal_g)
        release_per_step = float(self.meal_release_g_per_min) * float(
            self.sample_time_min
        )
        if release_per_step <= 0.0 or not np.isfinite(release_per_step):
            release_per_step = float(meal_g)

        release = float(min(self._meal_pending_g, release_per_step))
        self._meal_pending_g = float(max(0.0, self._meal_pending_g - release))
        return release

    def _maybe_scale_theta_for_basal(self) -> None:
        if self.theta_input_mode not in ("abs_insulin", "absolute", "abs"):
            return
        if self._theta_scaled_for_basal:
            return
        basal = float(self._basal_u_per_min)
        if not np.isfinite(basal) or basal <= 1e-8:
            return
        try:
            n = int(self.order)
            b = np.asarray(self.theta[n : 2 * n], dtype=float)
            d = float(self.theta[-1])
            b_scaled = b * basal
            d_scaled = d + basal * float(np.sum(b))
            self.theta[n : 2 * n] = b_scaled
            self.theta[-1] = d_scaled
            self._theta_scaled_for_basal = True
            self._theta_scaled_basal = basal
        except Exception:
            return

    def _get_patient_cr(self, patient_name: Optional[str]) -> float:
        try:
            if self.quest is not None and patient_name:
                matches = self.quest[self.quest.Name.str.match(patient_name)]
                if not matches.empty:
                    return float(matches.CR.values[0])
        except Exception:
            pass
        return 15.0

    def _update_meal_effect(self, meal_raw: float) -> float:
        meal_raw = float(max(0.0, float(meal_raw)))
        if not self.use_meal_filter:
            self._meal_effect = 0.0
            return meal_raw

        alpha = self._meal_filter_alpha()
        if alpha <= 0.0 or alpha >= 1.0 or not np.isfinite(alpha):
            self._meal_effect = 0.0
            return meal_raw

        prev = float(self._meal_effect) if np.isfinite(self._meal_effect) else 0.0
        self._meal_effect = alpha * prev + (1.0 - alpha) * meal_raw
        return float(self._meal_effect)

    def _build_m_future(self, meal_raw: float) -> np.ndarray:
        """
        Build an Hp-length meal disturbance sequence for MPC/rollout.

        - Without filtering: assume meal rate continues for `meal_forecast_steps` then zero.
        - With filtering: forecast the filtered meal effect by propagating the filter state.
        """
        Hp = int(self.pred_steps)
        m_future = np.zeros(Hp, dtype=float)
        if Hp <= 0:
            return m_future

        meal_raw = float(max(0.0, float(meal_raw)))
        k_meal = max(1, int(self.meal_forecast_steps))
        k_meal = min(k_meal, Hp)

        if not self.use_meal_filter:
            if meal_raw > 0.0:
                m_future[:k_meal] = meal_raw
            return m_future

        alpha = self._meal_filter_alpha()
        if alpha <= 0.0 or alpha >= 1.0 or not np.isfinite(alpha):
            if meal_raw > 0.0:
                m_future[:k_meal] = meal_raw
            return m_future

        eff = float(self._meal_effect) if np.isfinite(self._meal_effect) else 0.0
        m_future[0] = eff
        for t in range(1, Hp):
            meal_assumed = meal_raw if (t < k_meal) else 0.0
            eff = alpha * eff + (1.0 - alpha) * float(meal_assumed)
            m_future[t] = eff
        return m_future

    def _ensure_patient_params(self, patient_name: Optional[str]) -> None:
        if not patient_name or patient_name == self._patient_name:
            return

        if self._theta_scaled_for_basal:
            # uhat scaling is patient-basal dependent. Do not carry a scaled
            # parameterization across patients.
            self.theta = self._theta_unscaled.copy()
            self.P = self.P_prior.copy()
            self._theta_scaled_for_basal = False
            self._theta_scaled_basal = None

        self._load_theta_for_patient(patient_name)

        # Reset basal info based on simglucose params
        basal = 0.015
        if self.patient_params is not None and patient_name:
            matches = self.patient_params[
                self.patient_params.Name.str.match(patient_name)
            ]
            if not matches.empty:
                bw = float(matches.BW.values[0])
                u2ss = float(matches.u2ss.values[0])
                basal = np.clip((u2ss * bw) / 6000.0, 0.0, 0.2)

        self._basal_u_per_min = basal
        if basal > 1e-8:
            u_cap = self.u_abs_cap / basal - 1.0
            self._uhat_max = max(0.0, min(self.uhat_max_rel, u_cap))
        self._patient_name = patient_name

    def _compute_uhat_max(self, *, u_abs_cap: float, uhat_max_rel: float) -> float:
        basal = float(self._basal_u_per_min)
        if basal <= 1e-8:
            return 0.0
        u_cap = float(u_abs_cap) / basal - 1.0
        return float(max(0.0, min(float(uhat_max_rel), float(u_cap))))

    def _workspace_mpc_uhat_plan(self, first_uhat: float) -> np.ndarray:
        """Convert workspace MPC insulin deviations into this ARX model's uhat units."""
        plan = np.full(int(self.ctrl_steps), float(first_uhat), dtype=float)
        if self.workspace_mpc is None or plan.size == 0:
            return plan

        basal_units = float(self._basal_u_per_min) * float(self.sample_time_min)
        if not np.isfinite(basal_units) or basal_units <= 1e-8:
            return plan

        last_solution = getattr(self.workspace_mpc, "last_solution", None)
        if last_solution is None:
            return plan

        try:
            u_dev = np.asarray(last_solution, dtype=float).reshape(-1)
            n_copy = min(int(plan.size), int(u_dev.size))
            if n_copy > 0:
                plan[:n_copy] = u_dev[:n_copy] / basal_units
        except Exception:
            return plan

        plan[0] = float(first_uhat)
        return plan

    def _update_workspace_zone_observer(
        self, glucose: float, patient_name: Optional[str]
    ) -> None:
        """
        Keep the default Zone-MPC observer alive when ARX rollouts are used for
        optimization.

        This mirrors the observer update sequence in
        `simglucose.controller.mpc_ctrller.ZoneMPCController.policy()` so that
        predictive suspend and any observer-dependent logic use the same state
        estimate as the default Zone-MPC implementation.
        """
        zone = self.workspace_mpc
        if zone is None:
            return

        try:
            zone._ensure_profile(patient_name or "Average")
            zone._update_observer(float(glucose))
        except Exception as exc:
            LOGGER.debug("Workspace Zone-MPC observer update failed: %s", exc)

    def _commit_workspace_zone_control(self, u_dev: float) -> None:
        """
        Store the executed Zone-MPC insulin deviation for the next observer
        prediction step.
        """
        zone = self.workspace_mpc
        if zone is None:
            return

        try:
            zone.last_u_dev = float(u_dev)
        except Exception as exc:
            LOGGER.debug("Workspace Zone-MPC control commit failed: %s", exc)

    def _workspace_zone_arx_objective(
        self,
        u_dev_seq: np.ndarray,
        glucose: float,
        sample_time: float,
        now,
        m_future: np.ndarray,
    ) -> float:
        """Evaluate the workspace Zone-MPC cost on an ARX glucose rollout."""
        zone = self.workspace_mpc
        if zone is None:
            return float("inf")

        u_dev_seq = np.asarray(u_dev_seq, dtype=float).reshape(-1)
        if u_dev_seq.size != int(zone.Nu):
            return float("inf")

        basal_units = float(self._basal_u_per_min) * float(sample_time)
        if not np.isfinite(basal_units) or basal_units <= 1e-8:
            return float("inf")

        uhat_control = u_dev_seq / basal_units
        if int(zone.Ny) <= int(zone.Nu):
            u_future = uhat_control[: int(zone.Ny)]
        else:
            u_future = np.concatenate(
                [
                    uhat_control,
                    np.zeros(int(zone.Ny) - int(zone.Nu), dtype=float),
                ]
            )

        try:
            g_roll = rollout_arx_generic(
                self.theta,
                list(self.g_hist),
                list(self.uhat_hist),
                list(self.m_hist),
                u_future=u_future,
                m_future=np.asarray(m_future, dtype=float)[: int(zone.Ny)],
                order=int(self.order),
                u_delay=int(self.u_delay_steps),
                m_delay=int(self.m_delay_steps),
            )
        except Exception:
            return float("inf")

        g_roll = np.asarray(g_roll, dtype=float).reshape(-1)
        if g_roll.size < int(zone.Ny) + 1 or not np.all(
            np.isfinite(g_roll[: int(zone.Ny) + 1])
        ):
            return float("inf")

        velocity_penalty = (
            float(zone.velocity_penalty_D)
            if zone._velocity_penalty_active(float(glucose))
            else 0.0
        )
        cost = 0.0
        dt = max(float(sample_time), 1e-8)

        for k in range(int(zone.Ny)):
            y_prev = float(g_roll[k])
            y_abs = float(g_roll[k + 1])
            velocity = (y_abs - y_prev) / dt
            lower, upper = zone._target_zone(now, k, sample_time)
            z_low = max(float(lower) - y_abs, 0.0)
            z_high = max(y_abs - float(upper), 0.0)
            if zone.variant in {"velocity", "adaptive"}:
                q = float(zone._velocity_weight(velocity))
            else:
                q = 1.0
            cost += z_low**2 + q * z_high**2
            cost += velocity_penalty * max(velocity, 0.0) ** 2

            if k < int(zone.Nu):
                r_plus, r_minus = zone._input_penalties(
                    y_abs=y_abs,
                    velocity=velocity,
                )
                u_dev = float(u_dev_seq[k])
                cost += float(r_plus) * max(u_dev, 0.0) ** 2
                cost += float(r_minus) * min(u_dev, 0.0) ** 2

        return float(cost)

    def _solve_workspace_zone_mpc_with_arx(
        self,
        glucose: float,
        meal_rate: float,
        meal_input: float,
        sample_time: float,
        patient_name: Optional[str],
        now,
        iob,
    ) -> Dict[str, object]:
        """Solve Zone MPC with its cost/bounds and ARX-RLS glucose predictions."""
        zone = self.workspace_mpc
        if zone is None or minimize is None:
            raise RuntimeError("ARX Zone-MPC solve requires workspace MPC and scipy")

        self._update_workspace_zone_observer(
            glucose=float(glucose),
            patient_name=patient_name,
        )
        bounds = zone._input_bounds(
            sample_time=float(sample_time),
            now=now,
            glucose=float(glucose),
            iob=iob,
        )
        x0 = zone._initial_guess(bounds)
        m_future = self._build_m_future(float(meal_input))[: int(zone.Ny)]

        def objective(u_dev_seq: np.ndarray) -> float:
            return self._workspace_zone_arx_objective(
                u_dev_seq,
                glucose=float(glucose),
                sample_time=float(sample_time),
                now=now,
                m_future=m_future,
            )

        tic = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step.*",
                category=RuntimeWarning,
            )
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                options={
                    "maxiter": int(zone.max_slsqp_iter),
                    "ftol": 1e-5,
                    "disp": False,
                },
            )
        solve_time = time.perf_counter() - tic
        solve_ok = bool(result.success) and np.all(np.isfinite(result.x))
        zone.solve_times.append(float(solve_time))
        zone.solve_successes.append(bool(solve_ok))

        u_dev_seq = np.asarray(result.x if solve_ok else x0, dtype=float).reshape(-1)
        zone.last_solution = u_dev_seq.copy()
        u_dev = float(u_dev_seq[0]) if u_dev_seq.size else 0.0
        u_dev = zone._apply_low_glucose_suspend(
            u_dev,
            glucose=float(glucose),
            sample_time=float(sample_time),
        )
        self._commit_workspace_zone_control(u_dev)
        if u_dev_seq.size:
            u_dev_seq[0] = u_dev

        basal_units = float(zone.profile.basal_rate) * float(sample_time)
        absolute_units = max(basal_units + u_dev, 0.0)
        bolus_rate = zone._meal_bolus_rate(
            meal_rate=float(meal_rate),
            glucose=float(glucose),
            sample_time=float(sample_time),
            now=now,
        )
        action = Action(
            basal=float(absolute_units / max(float(sample_time), 1e-8)),
            bolus=float(bolus_rate),
        )

        basal_units_for_arx = float(self._basal_u_per_min) * float(sample_time)
        if basal_units_for_arx > 1e-8:
            uhat_seq = u_dev_seq / basal_units_for_arx
        else:
            uhat_seq = np.zeros_like(u_dev_seq)
        if int(zone.Ny) <= int(zone.Nu):
            u_future = uhat_seq[: int(zone.Ny)]
        else:
            u_future = np.concatenate(
                [uhat_seq, np.zeros(int(zone.Ny) - int(zone.Nu), dtype=float)]
            )
        try:
            g_roll = rollout_arx_generic(
                self.theta,
                list(self.g_hist),
                list(self.uhat_hist),
                list(self.m_hist),
                u_future=u_future,
                m_future=m_future,
                order=int(self.order),
                u_delay=int(self.u_delay_steps),
                m_delay=int(self.m_delay_steps),
            )
            g_future = np.asarray(g_roll, dtype=float).reshape(-1)[
                1 : 1 + int(zone.Ny)
            ]
        except Exception:
            g_future = np.full(int(zone.Ny), np.nan, dtype=float)

        return {
            "action": action,
            "u_plan": np.asarray(uhat_seq, dtype=float),
            "g_future": np.asarray(g_future, dtype=float),
            "problem_status": "optimal" if solve_ok else "fallback",
            "solver": "workspace_zone_slsqp_arx",
            "num_iters": float(getattr(result, "nit", float("nan"))),
            "solve_time": float(solve_time),
            "objective": float(getattr(result, "fun", objective(u_dev_seq))),
            "error": "" if solve_ok else str(getattr(result, "message", "")),
        }

    def policy(self, observation, reward, done, **kwargs) -> Action:
        if done:
            return Action(basal=0, bolus=0)

        g_meas = observation.CGM
        meal_raw = float(kwargs.get("meal", 0.0) or 0.0)
        # Training data uses carbs aggregated as SUM per dt (g per sample).
        meal_input = self._compute_meal_input(meal_raw)
        patient_name = kwargs.get("patient_name")
        self._ensure_patient_params(patient_name)
        self._maybe_scale_theta_for_basal()

        meal_started = False
        meal_bolus_g = 0.0
        try:
            if float(meal_raw) > 0.0:
                self._meal_accum_g = float(self._meal_accum_g) + float(
                    meal_raw * float(self.sample_time_min)
                )
            if float(self._meal_raw_prev) <= 0.0 and float(meal_raw) > 0.0:
                meal_started = True
                # Pre-bolus on meal start using current meal grams (per step).
                meal_bolus_g = float(meal_raw) * float(self.sample_time_min)
            if float(self._meal_raw_prev) > 0.0 and float(meal_raw) <= 0.0:
                self._meal_accum_g = 0.0
        except Exception:
            meal_started = False

        # Cold start
        if self._first_step:
            for _ in range(self.order + 1):
                self.g_hist.append(g_meas)
                self.uhat_hist.append(0.0)
                self.m_hist.append(0.0)
            self._first_step = False
            self._meal_effect = 0.0
            self._meal_pending_g = 0.0
            self._boost_remaining = 0
            if (
                self.workspace_mpc is not None
                and self.workspace_mpc_prediction_model == "arx"
            ):
                self._update_workspace_zone_observer(g_meas, patient_name)
            if self.warmup:
                return self.teacher.policy(observation, reward, done, **kwargs)
            return Action(basal=self._basal_u_per_min, bolus=0)

        # Update meal history BEFORE the RLS update so phi aligns with the most recent known disturbance.
        meal_model = self._update_meal_effect(meal_input)
        self.m_hist.append(float(meal_model))
        self._meal_raw_hist.append(float(meal_raw))
        self._meal_model_hist.append(float(meal_model))
        self._meal_input_hist.append(float(meal_input))

        # 1. RLS Update
        phi_prev = construct_phi_generic(
            self.g_hist,
            self.uhat_hist,
            self.m_hist,
            self.order,
            u_delay=int(self.u_delay_steps),
            m_delay=int(self.m_delay_steps),
        )

        # Predict
        g_hat = arx_predict(self.theta, phi_prev)
        self._bg_hat_hist.append(g_hat)

        # Update Lambda & Theta
        err_raw = float(g_meas) - float(g_hat)
        lam = vff_lambda(err_raw, self.lam_base, 0.99, 0.9999, self.kappa, 60.0)
        # Clip the innovation to avoid catastrophic parameter jumps under large transient mismatch.
        if np.isfinite(self.rls_err_clip) and self.rls_err_clip > 0:
            err_used = float(np.clip(err_raw, -self.rls_err_clip, self.rls_err_clip))
        else:
            err_used = err_raw
        y_meas_used = float(g_hat) + float(err_used)

        # RLS diagnostics (pre-update)
        try:
            rls_denom = float(lam) + float(phi_prev.T @ self.P @ phi_prev)
        except Exception:
            rls_denom = float("nan")
        try:
            phi_norm = float(np.linalg.norm(phi_prev))
        except Exception:
            phi_norm = float("nan")

        skip_update = False
        if np.isfinite(self.rls_err_deadzone) and abs(err_raw) < float(
            self.rls_err_deadzone
        ):
            skip_update = True
        if np.isfinite(self.rls_phi_min) and float(phi_norm) < float(self.rls_phi_min):
            skip_update = True

        if skip_update:
            err = float(err_raw)
        else:
            theta_new, P_new, err = rls_vff_update(
                self.theta, self.P, phi_prev, y_meas_used, lam
            )
            self.theta = project_theta(
                theta_new,
                self.order,
                c_max=float(self.meal_c_max),
            )
            self.P = P_new

        # Logging
        self._rls_lam_hist.append(lam)
        self._rls_err_hist.append(err)
        self._rls_err_raw_hist.append(err_raw)
        self._rls_theta_hist.append(self.theta.copy())
        self._rls_P_trace_hist.append(np.trace(self.P))
        self._rls_denom_hist.append(rls_denom)
        self._rls_phi_norm_hist.append(phi_norm)
        try:
            self._rls_P_min_eig_hist.append(float(np.min(np.linalg.eigvalsh(self.P))))
        except Exception:
            self._rls_P_min_eig_hist.append(float("nan"))
        try:
            self._rls_ar_rho_hist.append(
                ar_spectral_radius_companion(
                    np.asarray(self.theta[: int(self.order)], dtype=float)
                )
            )
        except Exception:
            self._rls_ar_rho_hist.append(float("nan"))

        # Update g_hist with current measurement NOW
        self.g_hist.append(g_meas)

        # 2. Control Logic
        mpc_g_future = np.full(int(self.pred_steps), np.nan, dtype=float)
        mpc_u_future = np.full(int(self.ctrl_steps), np.nan, dtype=float)
        mpc_status = "n/a"
        mpc_problem_status = ""
        mpc_solver = ""
        mpc_num_iters = float("nan")
        mpc_solve_time = float("nan")
        mpc_objective = float("nan")
        mpc_error = ""
        if self.warmup:
            action = self.teacher.policy(observation, reward, done, **kwargs)
            # Back-calculate uhat
            total_u = action.basal + action.bolus
            denom = max(1e-8, self._basal_u_per_min)
            executed_uhat = (total_u / denom) - 1.0
            mpc_status = "warmup"
            mpc_problem_status = "warmup"
        elif (
            self.workspace_mpc is not None
            and self.workspace_mpc_prediction_model == "arx"
        ):
            zone_result = self._solve_workspace_zone_mpc_with_arx(
                glucose=float(g_meas),
                meal_rate=float(meal_raw),
                meal_input=float(meal_input),
                sample_time=float(self.sample_time_min),
                patient_name=patient_name,
                now=kwargs.get("time"),
                iob=kwargs.get("iob"),
            )
            action = zone_result["action"]
            total_u = float(action.basal) + float(action.bolus)
            denom = max(1e-8, float(self._basal_u_per_min))
            executed_uhat = (float(total_u) / float(denom)) - 1.0
            u_plan = np.asarray(zone_result["u_plan"], dtype=float).reshape(-1)
            g_future = np.asarray(zone_result["g_future"], dtype=float).reshape(-1)
            n_u = min(int(mpc_u_future.size), int(u_plan.size))
            n_g = min(int(mpc_g_future.size), int(g_future.size))
            if n_u > 0:
                mpc_u_future[:n_u] = u_plan[:n_u]
            if n_g > 0:
                mpc_g_future[:n_g] = g_future[:n_g]
            mpc_problem_status = str(zone_result["problem_status"])
            mpc_status = (
                "workspace_zone_arx_optimal"
                if mpc_problem_status == "optimal"
                else "workspace_zone_arx_fallback"
            )
            mpc_solver = str(zone_result["solver"])
            mpc_num_iters = float(zone_result["num_iters"])
            mpc_solve_time = float(zone_result["solve_time"])
            mpc_objective = float(zone_result["objective"])
            mpc_error = str(zone_result["error"])
        elif self.workspace_mpc is not None:
            action = self.workspace_mpc.policy(observation, reward, done, **kwargs)
            total_u = float(action.basal) + float(action.bolus)
            denom = max(1e-8, float(self._basal_u_per_min))
            executed_uhat = (float(total_u) / float(denom)) - 1.0

            solve_successes = getattr(self.workspace_mpc, "solve_successes", [])
            solve_ok = bool(solve_successes[-1]) if solve_successes else False
            mpc_status = (
                "workspace_mpc_optimal" if solve_ok else "workspace_mpc_fallback"
            )
            mpc_problem_status = mpc_status
            mpc_solver = "workspace_mpc_slsqp"
            solve_times = getattr(self.workspace_mpc, "solve_times", [])
            if solve_times:
                mpc_solve_time = float(solve_times[-1])

            try:
                mpc_u_future[:] = self._workspace_mpc_uhat_plan(executed_uhat)
                if int(self.pred_steps) <= int(self.ctrl_steps):
                    u_future = mpc_u_future[: int(self.pred_steps)]
                else:
                    tail = np.zeros(
                        int(self.pred_steps) - int(self.ctrl_steps), dtype=float
                    )
                    u_future = np.concatenate([mpc_u_future, tail])

                m_future = self._build_m_future(float(meal_input))
                g_pred = rollout_arx_generic(
                    self.theta,
                    list(self.g_hist),
                    list(self.uhat_hist),
                    list(self.m_hist),
                    u_future=u_future,
                    m_future=m_future,
                    order=int(self.order),
                    u_delay=int(self.u_delay_steps),
                    m_delay=int(self.m_delay_steps),
                )
                if isinstance(g_pred, np.ndarray) and g_pred.size >= 2:
                    hp = min(int(self.pred_steps), int(g_pred.size - 1))
                    mpc_g_future[:hp] = g_pred[1 : 1 + hp]
            except Exception as e:
                mpc_error = str(e)
        else:
            # MPC
            # Provide a short-horizon meal disturbance sequence (optionally filtered to better match absorption).
            m_future = self._build_m_future(float(meal_input))

            if float(meal_raw) > 0.0:
                self._boost_remaining = max(
                    self._boost_remaining, int(self.boost_steps_after_meal)
                )
            if self._boost_remaining > 0:
                self._boost_remaining -= 1
            use_boost = (self._boost_remaining > 0) or (
                self.boost_on_high_bg
                and (float(g_meas) > float(self.boost_bg_threshold))
            )
            uhat_min_eff = float(self.uhat_min)
            if use_boost and np.isfinite(self.meal_uhat_min):
                uhat_min_eff = max(uhat_min_eff, float(self.meal_uhat_min))
            uhat_max_eff = (
                self._compute_uhat_max(
                    u_abs_cap=float(self.u_abs_cap_boost),
                    uhat_max_rel=float(self.uhat_max_rel_boost),
                )
                if use_boost
                else float(self._uhat_max)
            )

            fasting_hold = (
                self.enable_fasting_hold
                and float(meal_raw) <= 0.0
                and float(self.g_bounds["lower"])
                <= float(g_meas)
                <= float(self.g_bounds["upper"])
                and abs(float(g_meas) - float(self.target))
                <= float(self.fasting_hold_band)
            )

            if fasting_hold:
                executed_uhat = 0.0
                mpc_status = "fasting_hold"
                mpc_problem_status = "fasting_hold"
                try:
                    u_future = np.zeros(int(self.pred_steps), dtype=float)
                    m_future = np.zeros(int(self.pred_steps), dtype=float)
                    g_pred = rollout_arx_generic(
                        self.theta,
                        list(self.g_hist),
                        list(self.uhat_hist),
                        list(self.m_hist),
                        u_future=u_future,
                        m_future=m_future,
                        order=int(self.order),
                        u_delay=int(self.u_delay_steps),
                        m_delay=int(self.m_delay_steps),
                    )
                    if isinstance(g_pred, np.ndarray) and g_pred.size >= 2:
                        mpc_g_future[:] = g_pred[1 : 1 + int(self.pred_steps)]
                    mpc_u_future[:] = 0.0
                except Exception:
                    pass
            elif g_meas < self.g_bounds["safety"]:
                executed_uhat = self.uhat_min
                if float(meal_raw) > 0.0 and np.isfinite(self.meal_uhat_min):
                    executed_uhat = max(float(executed_uhat), float(self.meal_uhat_min))
                mpc_status = "safety_clamp"
                mpc_problem_status = "safety_clamp"
            else:
                sol = solve_mpc_generic(
                    self.theta,
                    list(self.g_hist),
                    list(self.uhat_hist),
                    list(self.m_hist),
                    m_future,
                    self.target,
                    self.g_bounds,
                    {"min": float(uhat_min_eff), "max": float(uhat_max_eff)},
                    self.mpc_weights,
                    {"pred": self.pred_steps, "control": self.ctrl_steps},
                    u_delay=int(self.u_delay_steps),
                    m_delay=int(self.m_delay_steps),
                    return_rollout=True,
                )
                if isinstance(sol, MpcSolveResult):
                    executed_uhat = float(sol.u0)
                    mpc_problem_status = str(sol.problem_status)
                    mpc_solver = str(sol.solver)
                    mpc_num_iters = float(sol.num_iters)
                    mpc_solve_time = float(sol.solve_time)
                    mpc_objective = float(sol.objective)
                    mpc_error = str(sol.error)

                    if mpc_problem_status in ("optimal", "optimal_inaccurate"):
                        mpc_status = "optimal"
                    elif mpc_problem_status == "user_limit":
                        mpc_status = "user_limit"
                    else:
                        mpc_status = "solve_failed"

                    # Always export a consistent rollout using a bounded u sequence.
                    # Reason: when OSQP hits iteration limits, cvxpy may return infeasible/unstable g/u values.
                    try:
                        umin = float(uhat_min_eff)
                        umax = float(uhat_max_eff)
                        u_seq = (
                            np.asarray(sol.u_seq, dtype=float).reshape(-1)
                            if isinstance(sol.u_seq, np.ndarray)
                            else np.full(int(self.ctrl_steps), np.nan, dtype=float)
                        )
                        if u_seq.size < int(self.ctrl_steps):
                            u_seq = np.pad(
                                u_seq,
                                (0, int(self.ctrl_steps) - int(u_seq.size)),
                                mode="constant",
                                constant_values=np.nan,
                            )
                        u_seq = u_seq[: int(self.ctrl_steps)]
                        u_seq = np.where(
                            np.isfinite(u_seq), u_seq, float(executed_uhat)
                        )
                        u_seq = np.clip(u_seq, umin, umax)
                        mpc_u_future[:] = u_seq

                        # Build Hp-length u_future with input-blocking after Hu
                        if int(self.pred_steps) <= int(self.ctrl_steps):
                            u_future = u_seq[: int(self.pred_steps)]
                        else:
                            tail = np.full(
                                int(self.pred_steps) - int(self.ctrl_steps),
                                float(u_seq[-1]),
                                dtype=float,
                            )
                            u_future = np.concatenate([u_seq, tail])

                        # m_future already prepared for MPC; keep as-is

                        g_roll = rollout_arx_generic(
                            self.theta,
                            list(self.g_hist),
                            list(self.uhat_hist),
                            list(self.m_hist),
                            u_future=u_future,
                            m_future=m_future,
                            order=int(self.order),
                            u_delay=int(self.u_delay_steps),
                            m_delay=int(self.m_delay_steps),
                        )
                        if isinstance(g_roll, np.ndarray) and g_roll.size >= 2:
                            hp = min(int(self.pred_steps), int(g_roll.size - 1))
                            mpc_g_future[:hp] = g_roll[1 : 1 + hp]

                        if mpc_problem_status == "user_limit":
                            mpc_status = "user_limit_rollout"
                    except Exception:
                        pass
                else:
                    executed_uhat = 0.0
                    mpc_status = "solve_failed"
                    mpc_problem_status = "no_solution"
                executed_uhat = np.clip(
                    executed_uhat, self.uhat_min, float(uhat_max_eff)
                )

            # If we didn't get an MPC rollout (no solver / failed solve / safety clamp),
            # still export a model rollout under the executed uhat (open-loop).
            if mpc_status in ("solve_failed", "safety_clamp"):
                try:
                    u_future = np.full(
                        int(self.pred_steps), float(executed_uhat), dtype=float
                    )
                    # m_future already prepared for MPC; keep as-is
                    g_pred = rollout_arx_generic(
                        self.theta,
                        list(self.g_hist),
                        list(self.uhat_hist),
                        list(self.m_hist),
                        u_future=u_future,
                        m_future=m_future,
                        order=int(self.order),
                        u_delay=int(self.u_delay_steps),
                        m_delay=int(self.m_delay_steps),
                    )
                    if isinstance(g_pred, np.ndarray) and g_pred.size >= 2:
                        mpc_g_future[:] = g_pred[1 : 1 + int(self.pred_steps)]
                    mpc_u_future[:] = float(executed_uhat)
                    if mpc_status == "solve_failed":
                        mpc_status = "solve_failed_rollout"
                        if not mpc_problem_status:
                            mpc_problem_status = "solve_failed_rollout"
                except Exception:
                    pass

            # 根据use_basal_modulation决定输出模式
            if self.use_basal_modulation:
                # BBC模式：basal * (1 + uhat)，即基于basal的调制
                val = max(
                    0.0, float(self._basal_u_per_min) * (1.0 + float(executed_uhat))
                )
            else:
                # 纯MPC模式：直接使用MPC输出的绝对胰岛素率
                # 将uhat转换为绝对胰岛素率（假设uhat=0对应basal）
                val = max(
                    0.0,
                    float(self._basal_u_per_min)
                    + float(executed_uhat) * float(self._basal_u_per_min),
                )

            bolus_rate = 0.0
            total_u = float(val)
            if (not self.warmup) and self.enable_meal_bolus and meal_started:
                try:
                    cr = float(self._get_patient_cr(patient_name))
                    cr = max(1e-6, cr)
                    bolus_units = float(meal_bolus_g) / cr
                    bolus_rate = bolus_units / float(self.sample_time_min)
                    bolus_rate = max(
                        0.0, float(bolus_rate) * float(self.meal_bolus_scale)
                    )
                except Exception:
                    bolus_rate = 0.0

                # Cap total insulin rate during meals using the boost max (absolute + relative).
                try:
                    uhat_cap = self._compute_uhat_max(
                        u_abs_cap=float(self.u_abs_cap_boost),
                        uhat_max_rel=float(self.uhat_max_rel_boost),
                    )
                    total_u_cap = float(self._basal_u_per_min) * (1.0 + float(uhat_cap))
                except Exception:
                    total_u_cap = float("inf")
                total_u = float(min(float(val) + float(bolus_rate), float(total_u_cap)))
                bolus_rate = max(0.0, float(total_u) - float(val))

            action = Action(basal=float(val), bolus=float(bolus_rate))

        # Store executed u
        denom = max(1e-8, float(self._basal_u_per_min))
        executed_uhat_total = (float(total_u) / float(denom)) - 1.0
        self.uhat_hist.append(float(executed_uhat_total))

        # Store MPC rollouts (aligned per-step)
        self._mpc_g_pred_hist.append(mpc_g_future)
        self._mpc_u_seq_hist.append(mpc_u_future)
        self._mpc_status_hist.append(mpc_status)
        self._mpc_problem_status_hist.append(mpc_problem_status)
        self._mpc_solver_hist.append(mpc_solver)
        self._mpc_num_iters_hist.append(mpc_num_iters)
        self._mpc_solve_time_hist.append(mpc_solve_time)
        self._mpc_objective_hist.append(mpc_objective)
        self._mpc_error_hist.append(mpc_error)

        self._meal_raw_prev = float(meal_raw)

        return action

    def reset(self):
        self._first_step = True
        self.theta = self._theta_unscaled.copy()
        self._theta_scaled_for_basal = False
        self._theta_scaled_basal = None
        self.g_hist.clear()
        self.uhat_hist.clear()
        self.m_hist.clear()
        self.P = self.P_prior.copy()
        self._patient_name = None
        self._meal_effect = 0.0
        self._meal_pending_g = 0.0
        self._meal_raw_prev = 0.0
        self._meal_accum_g = 0.0
        self._boost_remaining = 0
        self._rls_lam_hist.clear()
        self._rls_err_hist.clear()
        self._rls_err_raw_hist.clear()
        self._rls_theta_hist.clear()
        self._bg_hat_hist.clear()
        self._rls_P_trace_hist.clear()
        self._rls_denom_hist.clear()
        self._rls_phi_norm_hist.clear()
        self._rls_P_min_eig_hist.clear()
        self._rls_ar_rho_hist.clear()
        self._mpc_g_pred_hist.clear()
        self._mpc_u_seq_hist.clear()
        self._mpc_status_hist.clear()
        self._mpc_problem_status_hist.clear()
        self._mpc_solver_hist.clear()
        self._mpc_num_iters_hist.clear()
        self._mpc_solve_time_hist.clear()
        self._mpc_objective_hist.clear()
        self._mpc_error_hist.clear()
        self._meal_raw_hist.clear()
        self._meal_model_hist.clear()
        self._meal_input_hist.clear()
        if self.workspace_mpc is not None:
            self.workspace_mpc.reset()

    def get_mpc_rollout(self) -> Dict[str, object]:
        """
        Return the latest MPC rolling prediction (Hp steps ahead) and planned u sequence (Hu steps).

        Keys:
          - bg_pred_future: np.ndarray shape (pred_steps,) -> predicted BG at t+1..t+Hp
          - u_plan: np.ndarray shape (ctrl_steps,) -> planned u[0..Hu-1]
          - status: str
        """
        if not self._mpc_g_pred_hist:
            return {}
        return {
            "bg_pred_future": np.asarray(self._mpc_g_pred_hist[-1], dtype=float).copy(),
            "u_plan": (
                np.asarray(self._mpc_u_seq_hist[-1], dtype=float).copy()
                if self._mpc_u_seq_hist
                else np.asarray([], dtype=float)
            ),
            "status": self._mpc_status_hist[-1] if self._mpc_status_hist else "n/a",
            "problem_status": (
                self._mpc_problem_status_hist[-1]
                if self._mpc_problem_status_hist
                else ""
            ),
            "solver": self._mpc_solver_hist[-1] if self._mpc_solver_hist else "",
            "num_iters": (
                self._mpc_num_iters_hist[-1]
                if self._mpc_num_iters_hist
                else float("nan")
            ),
            "solve_time": (
                self._mpc_solve_time_hist[-1]
                if self._mpc_solve_time_hist
                else float("nan")
            ),
            "objective": (
                self._mpc_objective_hist[-1]
                if self._mpc_objective_hist
                else float("nan")
            ),
            "error": self._mpc_error_hist[-1] if self._mpc_error_hist else "",
        }

    # --- [UPDATED] Export history for analysis ---
    def get_rls_history(self) -> pd.DataFrame:
        """Return per-step history including PREDICTION vs MEASUREMENT."""
        if not self._rls_theta_hist:
            return pd.DataFrame()

        thetas = np.vstack(self._rls_theta_hist)
        order = int(getattr(self, "order", 0) or 0)
        min_len = min(
            len(self._rls_lam_hist),
            len(self._bg_hat_hist),
            len(self._rls_err_hist),
            len(self._rls_err_raw_hist),
            len(self._rls_P_trace_hist),
            len(self._rls_denom_hist),
            len(self._rls_phi_norm_hist),
            len(self._rls_P_min_eig_hist),
            len(self._rls_ar_rho_hist),
            len(self._mpc_g_pred_hist),
            len(self._mpc_u_seq_hist),
            len(self._mpc_status_hist),
            len(self._mpc_problem_status_hist),
            len(self._mpc_solver_hist),
            len(self._mpc_num_iters_hist),
            len(self._mpc_solve_time_hist),
            len(self._mpc_objective_hist),
            len(self._mpc_error_hist),
            len(self._meal_raw_hist),
            len(self._meal_model_hist),
            len(self._meal_input_hist),
            int(thetas.shape[0]),
        )

        bg_meas_reconstructed = np.asarray(self._bg_hat_hist[:min_len]) + np.asarray(
            self._rls_err_hist[:min_len]
        )

        df = pd.DataFrame(
            {
                "time_step": np.arange(min_len),
                "arx_order": int(getattr(self, "order", 0) or 0),
                "u_delay_steps": int(getattr(self, "u_delay_steps", 0) or 0),
                "m_delay_steps": int(getattr(self, "m_delay_steps", 0) or 0),
                "meal_forecast_steps": int(
                    getattr(self, "meal_forecast_steps", 0) or 0
                ),
                "use_meal_filter": bool(getattr(self, "use_meal_filter", False)),
                "meal_filter_tau_min": float(
                    getattr(self, "meal_filter_tau_min", float("nan"))
                ),
                "use_meal_buffer": bool(getattr(self, "use_meal_buffer", False)),
                "meal_release_g_per_min": float(
                    getattr(self, "meal_release_g_per_min", float("nan"))
                ),
                "meal_c_max": float(getattr(self, "meal_c_max", float("nan"))),
                "meal_uhat_min": float(getattr(self, "meal_uhat_min", float("nan"))),
                "theta_input_mode": str(getattr(self, "theta_input_mode", "")),
                "theta_scaled_basal": float(
                    getattr(self, "_theta_scaled_basal", float("nan"))
                    if getattr(self, "_theta_scaled_basal", None) is not None
                    else float("nan")
                ),
                "rls_err_deadzone": float(
                    getattr(self, "rls_err_deadzone", float("nan"))
                ),
                "rls_phi_min": float(getattr(self, "rls_phi_min", float("nan"))),
                "meal_raw": np.asarray(self._meal_raw_hist[:min_len], dtype=float),
                "meal_input": np.asarray(self._meal_input_hist[:min_len], dtype=float),
                "meal_model": np.asarray(self._meal_model_hist[:min_len], dtype=float),
                "bg_pred": np.asarray(self._bg_hat_hist[:min_len], dtype=float),
                "bg_meas": bg_meas_reconstructed,
                "err": np.asarray(self._rls_err_hist[:min_len], dtype=float),
                "err_raw": np.asarray(self._rls_err_raw_hist[:min_len], dtype=float),
                "lam": np.asarray(self._rls_lam_hist[:min_len], dtype=float),
                "P_trace": np.asarray(self._rls_P_trace_hist[:min_len], dtype=float),
                "rls_denom": np.asarray(self._rls_denom_hist[:min_len], dtype=float),
                "phi_norm": np.asarray(self._rls_phi_norm_hist[:min_len], dtype=float),
                "P_min_eig": np.asarray(
                    self._rls_P_min_eig_hist[:min_len], dtype=float
                ),
                "ar_rho": np.asarray(self._rls_ar_rho_hist[:min_len], dtype=float),
                "mpc_status": self._mpc_status_hist[:min_len],
                "mpc_problem_status": self._mpc_problem_status_hist[:min_len],
                "mpc_solver": self._mpc_solver_hist[:min_len],
                "mpc_num_iters": np.asarray(
                    self._mpc_num_iters_hist[:min_len], dtype=float
                ),
                "mpc_solve_time": np.asarray(
                    self._mpc_solve_time_hist[:min_len], dtype=float
                ),
                "mpc_objective": np.asarray(
                    self._mpc_objective_hist[:min_len], dtype=float
                ),
                "mpc_error": self._mpc_error_hist[:min_len],
            }
        )

        # Export full theta blocks (generic order): a[1..n], b[1..n], c[1..n], d
        # theta = [a1..an, b1..bn, c1..cn, d]
        try:
            tdim = int(thetas.shape[1])
            n = int(order)
            for i in range(min(n, tdim)):
                df[f"theta_a{i+1}"] = thetas[:min_len, i]
            for i in range(min(n, max(0, tdim - n))):
                df[f"theta_b{i+1}"] = thetas[:min_len, n + i]
            for i in range(min(n, max(0, tdim - 2 * n))):
                df[f"theta_c{i+1}"] = thetas[:min_len, 2 * n + i]
            d_idx = 3 * n
            if 0 <= d_idx < tdim:
                df["theta_d"] = thetas[:min_len, d_idx]
        except Exception:
            pass

        # Export MPC rolling predicted glucose trajectory: BG(t+1..t+Hp)
        try:
            g_roll = np.vstack(self._mpc_g_pred_hist[:min_len])
            for k in range(g_roll.shape[1]):
                df[f"mpc_bg_pred_{k+1:02d}"] = g_roll[:, k]
        except Exception:
            pass

        # Export MPC planned u sequence: u[0..Hu-1]
        try:
            u_roll = np.vstack(self._mpc_u_seq_hist[:min_len])
            for k in range(u_roll.shape[1]):
                df[f"mpc_u_{k+1:02d}"] = u_roll[:, k]
        except Exception:
            pass

        return df
