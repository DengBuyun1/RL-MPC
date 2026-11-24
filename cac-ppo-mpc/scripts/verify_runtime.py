from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ppo_mpc.provenance import assert_runtime_matches_lock  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the active paper runtime lock.")
    parser.add_argument(
        "--requirements-lock",
        default=str(ROOT / "requirements-lock-server-py310-cu126.txt"),
    )
    args = parser.parse_args()
    try:
        audit = assert_runtime_matches_lock(args.requirements_lock)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
