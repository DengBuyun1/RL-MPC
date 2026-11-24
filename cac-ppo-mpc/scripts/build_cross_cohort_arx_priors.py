from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "mpc_assets" / "output" / "arx_rls_training_30"
DEFAULT_ADULT = DEFAULT_DIR / "population_theta_adult001_005.json"
DEFAULT_BANK = DEFAULT_DIR / "best_patient_theta.json"
DEFAULT_MILD = DEFAULT_DIR / "population_theta_adult_child_blend_050.json"
DEFAULT_SEVERE = DEFAULT_DIR / "population_theta_child001_005.json"
CHILD_SOURCE_IDS = [f"child#{idx:03d}" for idx in range(1, 6)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def theta_from(payload: dict, key: str, source: Path) -> list[float]:
    entry = payload.get(key)
    if not isinstance(entry, dict) or "theta" not in entry:
        raise ValueError(f"Missing theta entry {key!r} in {source}")
    theta = [float(value) for value in entry["theta"]]
    if not theta or any(not math.isfinite(value) for value in theta):
        raise ValueError(f"Non-finite or empty theta entry {key!r} in {source}")
    return theta


def vector_mean(vectors: list[list[float]]) -> list[float]:
    lengths = {len(vector) for vector in vectors}
    if len(lengths) != 1:
        raise ValueError(f"Theta length mismatch: {sorted(lengths)}")
    return [
        float(sum(vector[index] for vector in vectors) / len(vectors))
        for index in range(len(vectors[0]))
    ]


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def make_population_payload(
    *,
    theta: list[float],
    group: str,
    prior_type: str,
    source_ids: list[str],
    source_files: dict[str, str],
    order: int,
    blend_weight_child: float,
) -> dict:
    return {
        "__meta__": {
            "schema_version": "training_only_population_prior_v1",
            "prior_type": prior_type,
            "source_patient_ids": source_ids,
            "source_files_sha256": source_files,
            "selection_rule": "only explicitly listed source_patient_ids are read",
            "excluded_from_prior": "adult#006-010 and all validation/test trajectories",
            "online_adaptation": "ARX-RLS",
            "theta_input_mode": "uhat",
            "order": int(order),
            "blend_weight_child": float(blend_weight_child),
        },
        "__population__": {
            "group": group,
            "order": int(order),
            "theta": [float(value) for value in theta],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training-only mild and severe cross-cohort ARX priors."
    )
    parser.add_argument("--adult-prior", default=str(DEFAULT_ADULT))
    parser.add_argument("--theta-bank", default=str(DEFAULT_BANK))
    parser.add_argument("--mild-output", default=str(DEFAULT_MILD))
    parser.add_argument("--severe-output", default=str(DEFAULT_SEVERE))
    parser.add_argument("--mild-child-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weight = float(args.mild_child_weight)
    if not 0.0 < weight < 1.0:
        raise SystemExit("--mild-child-weight must be strictly between zero and one.")
    adult_path = Path(args.adult_prior).resolve()
    bank_path = Path(args.theta_bank).resolve()
    adult_payload = read_json(adult_path)
    bank_payload = read_json(bank_path)

    adult_theta = theta_from(adult_payload, "__population__", adult_path)
    child_vectors = [
        theta_from(bank_payload, patient, bank_path) for patient in CHILD_SOURCE_IDS
    ]
    child_theta = vector_mean(child_vectors)
    if len(adult_theta) != len(child_theta):
        raise SystemExit("Adult and child population theta dimensions differ.")
    order = int((len(adult_theta) - 1) // 3)
    if len(adult_theta) != 3 * order + 1 or order < 1:
        raise SystemExit("Theta vectors do not follow [a, b, c, d] ARX layout.")

    mild_theta = [
        float((1.0 - weight) * adult + weight * child)
        for adult, child in zip(adult_theta, child_theta)
    ]
    source_files = {
        "adult_prior_sha256": sha256_file(adult_path),
        "theta_bank_sha256": sha256_file(bank_path),
    }
    severe = make_population_payload(
        theta=child_theta,
        group="child001_005_training_population",
        prior_type="arithmetic_mean_child001_005_training_only",
        source_ids=CHILD_SOURCE_IDS,
        source_files=source_files,
        order=order,
        blend_weight_child=1.0,
    )
    mild = make_population_payload(
        theta=mild_theta,
        group="adult_child_training_blend_050",
        prior_type="convex_blend_adult_population_and_child001_005_mean",
        source_ids=[
            "adult_population_prior(adult#001-005)",
            *CHILD_SOURCE_IDS,
        ],
        source_files=source_files,
        order=order,
        blend_weight_child=weight,
    )
    mild_path = Path(args.mild_output).resolve()
    severe_path = Path(args.severe_output).resolve()
    write_atomic(mild_path, mild)
    write_atomic(severe_path, severe)
    print(f"Saved mild mismatch prior: {mild_path}")
    print(f"Saved severe mismatch prior: {severe_path}")
    print(f"Selected child sources: {','.join(CHILD_SOURCE_IDS)}")


if __name__ == "__main__":
    main()
