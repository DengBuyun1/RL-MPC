from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
CLOSED_LOOP_ROOT = PROJECT_ROOT / "scripts" / "closed_loop"
SAC_ROOT = PROJECT_ROOT / "src" / "controllers" / "rl_sac_mask"
SIM_ROOT = PROJECT_ROOT / "src" / "envs" / "simglucose_mpc"
ARX_CONTROLLER_PATH = PROJECT_ROOT / "src" / "controllers" / "arx_rls" / "arx_mpc_ctrller.py"
DEFAULT_ARX_PARAM_FILE = PROJECT_ROOT / "configs" / "arx" / "best_patient_theta.json"
for path in (CLOSED_LOOP_ROOT, SCRIPTS_ROOT, SAC_ROOT, SIM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_hybrid_drl_mpc as base_eval


def load_arx_controller_class():
    module_path = ARX_CONTROLLER_PATH
    spec = importlib.util.spec_from_file_location("arx_mpc_ctrller_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ARX controller module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ArxRlsVffMpcController


ArxRlsVffMpcController = load_arx_controller_class()


class ArxRlsMpcPrior:
    def __init__(self, args: argparse.Namespace):
        param_file = Path(args.arx_param_file)
        if not param_file.is_absolute():
            param_file = PROJECT_ROOT / param_file
        self.controller = ArxRlsVffMpcController(
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
            enable_meal_bolus=False,
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

    def policy(self, *args, **kwargs):
        return self.controller.policy(*args, **kwargs)

    @property
    def solve_times(self):
        zone = getattr(self.controller, "workspace_mpc", None)
        return getattr(zone, "solve_times", [])

    @property
    def solve_successes(self):
        zone = getattr(self.controller, "workspace_mpc", None)
        return getattr(zone, "solve_successes", [])


def make_mpc(args: argparse.Namespace):
    return ArxRlsMpcPrior(args)


def parse_args():
    arx_parser = argparse.ArgumentParser(add_help=False)
    arx_parser.add_argument(
        "--arx-param-file",
        default=str(DEFAULT_ARX_PARAM_FILE),
    )
    arx_parser.add_argument("--arx-target-glucose", type=float, default=110.0)
    arx_parser.add_argument("--arx-pred-horizon-min", type=float, default=120.0)
    arx_parser.add_argument("--arx-control-horizon-min", type=float, default=45.0)
    arx_parser.add_argument("--arx-P0", type=float, default=100.0)
    arx_parser.add_argument("--arx-lam-base", type=float, default=0.999)
    arx_parser.add_argument("--arx-kappa", type=float, default=0.002)
    arx_parser.add_argument("--arx-rls-err-clip", type=float, default=20.0)
    arx_parser.add_argument("--arx-rls-err-deadzone", type=float, default=20.0)
    arx_parser.add_argument("--arx-rls-phi-min", type=float, default=0.001)
    arx_parser.add_argument("--arx-min-order", type=int, default=6)
    arx_parser.add_argument("--arx-u-delay-steps", type=int, default=3)
    arx_parser.add_argument("--arx-m-delay-steps", type=int, default=1)
    arx_parser.add_argument("--arx-meal-filter-tau-min", type=float, default=15.0)
    arx_parser.add_argument("--arx-meal-release-g-per-min", type=float, default=5.0)
    arx_parser.add_argument("--arx-meal-c-max", type=float, default=5.0)
    arx_parser.add_argument("--arx-meal-forecast-steps", type=int, default=8)

    arx_args, remaining = arx_parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = base_eval.parse_args()
    finally:
        sys.argv = original_argv
    for key, value in vars(arx_args).items():
        setattr(args, key, value)
    return args


if __name__ == "__main__":
    base_eval.make_mpc = make_mpc
    base_eval.evaluate(parse_args())
