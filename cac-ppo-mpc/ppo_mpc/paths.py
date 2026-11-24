from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPC_ASSETS_ROOT = PROJECT_ROOT / "mpc_assets"
ARX_RLS_ROOT = MPC_ASSETS_ROOT / "ARX-RLS"
DEFAULT_ARX_THETA = (
    MPC_ASSETS_ROOT
    / "output"
    / "arx_rls_training_30"
    / "population_theta_adult001_005.json"
)


def add_asset_paths() -> None:
    """Put copied local simglucose and MPC assets ahead of site packages."""
    for path in [MPC_ASSETS_ROOT, ARX_RLS_ROOT]:
        value = str(path)
        if path.exists() and value not in sys.path:
            sys.path.insert(0, value)

def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
