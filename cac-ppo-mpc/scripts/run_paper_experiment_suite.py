from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ppo_mpc.provenance import (
    assert_runtime_matches_lock,
    runtime_fingerprint,
    runtime_identity,
    sha256_file,
    source_tree_sha256,
)


DEFAULT_UNANN_CKPT = ROOT / "runs" / "ppo_adult001_005_unann_tanh_v2_2m" / "model_final.pt"
DEFAULT_ANN_CKPT = ROOT / "runs" / "ppo_adult001_005_ann_tanh_v2_2m" / "model_final.pt"
PROTOCOL_VERSION = "causal_model_compat_v4"
DEFAULT_ARX_DIR = ROOT / "mpc_assets" / "output" / "arx_rls_training_30"
DEFAULT_ARX_PARAM = DEFAULT_ARX_DIR / "population_theta_adult001_005.json"
DEFAULT_MILD_ARX_PARAM = (
    DEFAULT_ARX_DIR / "population_theta_adult_child_blend_050.json"
)
DEFAULT_SEVERE_ARX_PARAM = DEFAULT_ARX_DIR / "population_theta_child001_005.json"
DEFAULT_COMPATIBILITY_CALIBRATION = (
    ROOT
    / "results"
    / "paper_compat_v4_calibration"
    / "analysis"
    / "compatibility_calibration.json"
)
REQUIREMENTS_LOCK = ROOT / "requirements-lock-server-py310-cu126.txt"

FAIR_MODES = "ppo_shield,mpc_shield,switch_nogate_shield,switch_shield"
DIAGNOSTIC_MODES = "ppo,mpc,switch_nogate"
MISMATCH_MODES = "ppo_shield,mpc_shield,switch_shield"
CALIBRATION_MODES = "switch_nogate_shield"

FINAL_SWITCH_ARGS = [
    "--switch-risk-low-glucose",
    "90",
    "--switch-risk-current-glucose",
    "130",
    "--switch-risk-projected-glucose",
    "130",
    "--switch-risk-fast-fall-rate",
    "-2.0",
    "--switch-risk-horizons-min",
    "30,60,90",
    "--switch-risk-strategy",
    "lower_insulin",
]

SAFETY_ARGS = [
    "--switch-safety-horizons-min",
    "30,60,90",
    "--switch-safety-suspend-glucose",
    "75",
    "--switch-safety-low-glucose",
    "90",
    "--switch-safety-projected-low-glucose",
    "90",
    "--switch-safety-falling-glucose",
    "130",
    "--switch-safety-falling-rate",
    "-0.25",
    "--switch-safety-low-basal-fraction",
    "0.0",
    "--switch-safety-falling-basal-fraction",
    "0.5",
    "--switch-safety-iob-excess-ratio",
    "1.25",
    "--switch-safety-iob-guard-projected-glucose",
    "130",
    "--switch-safety-iob-guard-basal-fraction",
    "0.0",
    "--switch-safety-high-glucose",
    "250",
    "--switch-safety-severe-high-glucose",
    "300",
    "--switch-safety-high-rising-rate",
    "0.25",
    "--switch-safety-correction-target-glucose",
    "180",
    "--switch-safety-correction-horizon-min",
    "180",
    "--switch-safety-high-max-basal-multiplier",
    "5.0",
]


@dataclass
class Experiment:
    name: str
    checkpoint: Path
    patients: str
    modes: str
    meal_announcement: str = "unannounced"
    hours: float = 72.0
    repeats: int = 5
    seed: int = 61000
    scenario: str = "A"
    start_time: str = "2018-01-01 00:00"
    split_label: str = "validation"
    role: str = "primary"
    model_condition: str = "nominal"
    arx_prior_label: str = "matched_adult_population"
    arx_param_file: Path = DEFAULT_ARX_PARAM
    compatibility_calibration_file: Path | None = None
    compatibility_reference_arx_file: Path | None = None
    requirements_lock: Path = REQUIREMENTS_LOCK
    device: str = "cpu"
    extra_args: list[str] = field(default_factory=list)
    purpose: str = ""

    @property
    def expected_rows(self) -> int:
        patients = [item for item in self.patients.split(",") if item]
        modes = [item for item in self.modes.split(",") if item]
        return len(patients) * len(modes) * int(self.repeats)

    @property
    def checkpoint_sha256(self) -> str:
        return sha256_file(self.checkpoint)

    @property
    def arx_param_sha256(self) -> str:
        return sha256_file(self.arx_param_file)

    @property
    def compatibility_calibration_sha256(self) -> str:
        if self.compatibility_calibration_file is None:
            return ""
        if not self.compatibility_calibration_file.exists():
            return "MISSING"
        return sha256_file(self.compatibility_calibration_file)

    @property
    def compatibility_reference_arx_sha256(self) -> str:
        if self.compatibility_reference_arx_file is None:
            return ""
        if not self.compatibility_reference_arx_file.exists():
            return "MISSING"
        return sha256_file(self.compatibility_reference_arx_file)

    @property
    def gate_enabled(self) -> bool:
        modes = {item for item in self.modes.split(",") if item}
        return bool({"switch", "switch_shield"}.intersection(modes))

    @property
    def source_tree_sha256(self) -> str:
        return source_tree_sha256(str(ROOT))

    @property
    def requirements_lock_sha256(self) -> str:
        return sha256_file(self.requirements_lock)

    @property
    def runtime_fingerprint(self) -> str:
        return runtime_fingerprint()

    @property
    def protocol_fingerprint(self) -> str:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "name": self.name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "arx_param_sha256": self.arx_param_sha256,
            "compatibility_calibration_sha256": (
                self.compatibility_calibration_sha256
            ),
            "compatibility_reference_arx_sha256": (
                self.compatibility_reference_arx_sha256
            ),
            "source_tree_sha256": self.source_tree_sha256,
            "requirements_lock_sha256": self.requirements_lock_sha256,
            "runtime_fingerprint": self.runtime_fingerprint,
            "device": self.device,
            "patients": [item for item in self.patients.split(",") if item],
            "modes": [item for item in self.modes.split(",") if item],
            "meal_announcement": self.meal_announcement,
            "hours": float(self.hours),
            "repeats": int(self.repeats),
            "seed": int(self.seed),
            "scenario": self.scenario,
            "start_time": self.start_time,
            "split_label": self.split_label,
            "role": self.role,
            "model_condition": self.model_condition,
            "arx_prior_label": self.arx_prior_label,
            "extra_args": self.extra_args,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def command(self, suite_dir: Path, device: str) -> list[str]:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_ppo_mpc.py"),
            "--checkpoint",
            str(self.checkpoint),
            "--patients",
            self.patients,
            "--modes",
            self.modes,
            "--scenario",
            self.scenario,
            "--start-time",
            self.start_time,
            "--meal-announcement",
            self.meal_announcement,
            "--hours",
            str(self.hours),
            "--repeats",
            str(self.repeats),
            "--seed",
            str(self.seed),
            "--output-dir",
            str(suite_dir / self.name),
            "--device",
            device,
            "--arx-param-file",
            str(self.arx_param_file),
            "--model-condition",
            self.model_condition,
            "--arx-prior-label",
            self.arx_prior_label,
            "--expected-source-tree-sha256",
            self.source_tree_sha256,
            "--expected-checkpoint-sha256",
            self.checkpoint_sha256,
            "--expected-arx-param-sha256",
            self.arx_param_sha256,
            "--expected-requirements-lock-sha256",
            self.requirements_lock_sha256,
            "--requirements-lock",
            str(self.requirements_lock),
            "--expected-runtime-fingerprint",
            self.runtime_fingerprint,
            "--protocol-fingerprint",
            self.protocol_fingerprint,
        ]
        if self.compatibility_calibration_file is not None:
            command.extend(
                [
                    "--compatibility-calibration-file",
                    str(self.compatibility_calibration_file),
                    "--expected-compatibility-calibration-sha256",
                    self.compatibility_calibration_sha256,
                    "--compatibility-reference-arx-sha256",
                    self.compatibility_reference_arx_sha256,
                ]
            )
        if self.gate_enabled:
            command.append("--enable-switch-compatibility-gate")
        command.extend(self.extra_args)
        return command


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def adult_range(start: int, end: int) -> str:
    return ",".join(f"adult#{idx:03d}" for idx in range(start, end + 1))


def cohort(prefix: str) -> str:
    return ",".join(f"{prefix}#{idx:03d}" for idx in range(1, 11))


def ablation_args(name: str) -> list[str]:
    changes = {
        "no_warmup": ["--switch-warmup-steps", "0"],
        "ppo_only_risk": ["--switch-risk-strategy", "ppo"],
        "no_current_risk": ["--switch-risk-current-glucose", "0"],
        "no_projected_risk": ["--switch-risk-projected-glucose", "-1000000"],
        "no_iob_guard": ["--switch-safety-iob-excess-ratio", "1000000"],
    }
    return changes[name]


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    checkpoint = resolve_root_path(args.checkpoint_unannounced)
    matched_prior = resolve_root_path(args.arx_param_file)
    mild_prior = resolve_root_path(
        getattr(args, "mild_arx_param_file", DEFAULT_MILD_ARX_PARAM)
    )
    severe_prior = resolve_root_path(
        getattr(args, "severe_arx_param_file", DEFAULT_SEVERE_ARX_PARAM)
    )
    calibration_file = resolve_root_path(
        getattr(
            args,
            "compatibility_calibration_file",
            DEFAULT_COMPATIBILITY_CALIBRATION,
        )
    )
    requirements_lock = resolve_root_path(
        getattr(args, "requirements_lock", REQUIREMENTS_LOCK)
    )
    experiments: list[Experiment] = []

    if args.scope == "calibration":
        calibration_hours = float(getattr(args, "calibration_hours", 24.0))
        calibration_repeats = int(getattr(args, "calibration_repeats", 3))
        experiments.append(
            Experiment(
                name=(
                    "compatibility_calibration_development_"
                    f"{int(calibration_hours)}h_r{calibration_repeats}"
                ),
                checkpoint=checkpoint,
                patients=adult_range(1, 5),
                modes=CALIBRATION_MODES,
                hours=calibration_hours,
                repeats=calibration_repeats,
                seed=int(getattr(args, "calibration_seed", 71000)),
                start_time=args.start_time,
                split_label="development",
                role="calibration",
                model_condition="nominal",
                arx_prior_label="matched_adult_population",
                arx_param_file=matched_prior,
                compatibility_calibration_file=None,
                compatibility_reference_arx_file=None,
                requirements_lock=requirements_lock,
                device=str(args.device),
                extra_args=[*FINAL_SWITCH_ARGS, *SAFETY_ARGS],
                purpose=(
                    "Development-only residual collection for a frozen causal "
                    "model-compatibility gate; never used as claim evidence."
                ),
            )
        )
        return experiments

    if args.scope == "smoke":
        hours = float(args.smoke_hours)
        repeats = 1
        # Functional smoke stays inside development subjects so adult#006-010
        # are never inspected before the locked validation suite.
        patients = "adult#001"
        start_time = args.smoke_start_time
        split_label = "smoke"
    else:
        hours = float(args.hours)
        repeats = int(args.repeats)
        patients = ",".join(
            item.strip()
            for item in str(args.patients).split(",")
            if item.strip()
        )
        start_time = args.start_time
        split_label = args.split_label

    common_extra_args = [*FINAL_SWITCH_ARGS, *SAFETY_ARGS]
    if args.scope == "smoke":
        common_extra_args.append("--allow-compatibility-source-overlap")

    common = {
        "checkpoint": checkpoint,
        "patients": patients,
        "hours": hours,
        "repeats": repeats,
        "seed": int(args.seed),
        "start_time": start_time,
        "split_label": split_label,
        "compatibility_calibration_file": calibration_file,
        "compatibility_reference_arx_file": matched_prior,
        "requirements_lock": requirements_lock,
        "device": str(args.device),
        "extra_args": common_extra_args,
    }
    experiments.extend(
        [
            Experiment(
                name=f"compat_nominal_unann_{split_label}_{int(hours)}h_r{repeats}",
                modes=FAIR_MODES,
                role="primary",
                model_condition="nominal",
                arx_prior_label="matched_adult_population",
                arx_param_file=matched_prior,
                purpose=(
                    "Matched-prior comparison with the explicit no-gate ablation "
                    "under one shared causal safety envelope."
                ),
                **common,
            ),
            Experiment(
                name=f"compat_mild_mismatch_unann_{split_label}_{int(hours)}h_r{repeats}",
                modes=MISMATCH_MODES,
                role="robustness",
                model_condition="mild_mismatch",
                arx_prior_label="adult_child_blend_050_training_only",
                arx_param_file=mild_prior,
                purpose=(
                    "Prespecified mild ARX-prior mismatch; compares PPO, MPC, and "
                    "compatibility-gated Switch."
                ),
                **common,
            ),
            Experiment(
                name=f"compat_severe_mismatch_unann_{split_label}_{int(hours)}h_r{repeats}",
                modes=FAIR_MODES,
                role="robustness",
                model_condition="severe_mismatch",
                arx_prior_label="child001_005_population_training_only",
                arx_param_file=severe_prior,
                purpose=(
                    "Prespecified severe cross-cohort ARX-prior mismatch with a "
                    "no-gate Switch ablation."
                ),
                **common,
            ),
        ]
    )

    if getattr(args, "include_ablations", False):
        for item in [
            "no_warmup",
            "ppo_only_risk",
            "no_current_risk",
            "no_projected_risk",
            "no_iob_guard",
        ]:
            experiments.append(
                Experiment(
                    name=f"ablation_{item}_unann_{split_label}_{int(hours)}h_r{repeats}",
                    checkpoint=checkpoint,
                    patients=patients,
                    modes="switch_shield",
                    hours=hours,
                    repeats=repeats,
                    seed=int(args.seed),
                    start_time=start_time,
                    split_label=split_label,
                    role="ablation",
                    model_condition="nominal",
                    arx_prior_label="matched_adult_population",
                    arx_param_file=matched_prior,
                    compatibility_calibration_file=calibration_file,
                    compatibility_reference_arx_file=matched_prior,
                    requirements_lock=requirements_lock,
                    device=str(args.device),
                    extra_args=[
                        *FINAL_SWITCH_ARGS,
                        *SAFETY_ARGS,
                        *ablation_args(item),
                    ],
                    purpose=f"Optional legacy switch ablation: {item}.",
                )
            )

    if args.scope == "full" or getattr(args, "include_generalization", False):
        gen_hours = float(getattr(args, "generalization_hours", 72.0))
        gen_repeats = int(getattr(args, "generalization_repeats", 3))
        for cohort_name, patients_value in [
            ("adolescent01_10", cohort("adolescent")),
            ("child01_10", cohort("child")),
        ]:
            experiments.append(
                Experiment(
                    name=(
                        "generalization_compat_unann_"
                        f"{cohort_name}_{int(gen_hours)}h_r{gen_repeats}"
                    ),
                    checkpoint=checkpoint,
                    patients=patients_value,
                    modes=MISMATCH_MODES,
                    hours=gen_hours,
                    repeats=gen_repeats,
                    seed=int(args.seed) + 20000,
                    start_time=args.start_time,
                    split_label="generalization",
                    role="generalization",
                    model_condition="cross_cohort_generalization",
                    arx_prior_label="matched_adult_population",
                    arx_param_file=matched_prior,
                    compatibility_calibration_file=calibration_file,
                    compatibility_reference_arx_file=matched_prior,
                    requirements_lock=requirements_lock,
                    device=str(args.device),
                    extra_args=[*FINAL_SWITCH_ARGS, *SAFETY_ARGS],
                    purpose="Optional cross-cohort generalization analysis.",
                )
            )

    if getattr(args, "include_announced", False):
        raise NotImplementedError(
            f"Announced-meal paper experiments are withheld in {PROTOCOL_VERSION}."
        )
    return experiments


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def experiment_is_complete(suite_dir: Path, experiment: Experiment) -> bool:
    output_dir = suite_dir / experiment.name
    marker = output_dir / "completion.json"
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.csv"
    if not marker.exists() or not config_path.exists() or not summary_path.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with summary_path.open("r", encoding="utf-8", newline="") as stream:
            summary_rows = list(csv.DictReader(stream))
    except Exception:
        return False

    expected_fingerprint = experiment.protocol_fingerprint
    core_valid = (
        bool(payload.get("execution_complete", payload.get("complete")))
        and int(payload.get("completed_runs", -1)) == experiment.expected_rows
        and int(payload.get("expected_runs", experiment.expected_rows))
        == experiment.expected_rows
        and payload.get("protocol_fingerprint") == expected_fingerprint
    )
    expected_config = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": expected_fingerprint,
        "checkpoint_sha256": experiment.checkpoint_sha256,
        "arx_param_sha256": experiment.arx_param_sha256,
        "source_tree_sha256": experiment.source_tree_sha256,
        "requirements_lock_sha256": experiment.requirements_lock_sha256,
        "runtime_fingerprint_actual": experiment.runtime_fingerprint,
        "compatibility_calibration_sha256": (
            experiment.compatibility_calibration_sha256
        ),
        "compatibility_reference_arx_sha256": (
            experiment.compatibility_reference_arx_sha256
        ),
        "model_condition": experiment.model_condition,
        "arx_prior_label": experiment.arx_prior_label,
    }
    config_valid = all(
        str(config.get(key, "")) == str(value)
        for key, value in expected_config.items()
    )

    expected_keys = {
        (
            patient,
            mode,
            repeat,
            int(experiment.seed) + 1000 * repeat + int(patient.split("#")[-1]),
        )
        for patient in experiment.patients.split(",")
        if patient
        for mode in experiment.modes.split(",")
        if mode
        for repeat in range(int(experiment.repeats))
    }
    actual_keys: list[tuple[str, str, int, int]] = []
    summaries_valid = len(summary_rows) == experiment.expected_rows
    try:
        for row in summary_rows:
            actual_keys.append(
                (
                    str(row["patient"]),
                    str(row["mode"]),
                    int(row["repeat"]),
                    int(row["seed"]),
                )
            )
    except (KeyError, TypeError, ValueError):
        summaries_valid = False
    summaries_valid = (
        summaries_valid
        and len(actual_keys) == len(set(actual_keys))
        and set(actual_keys) == expected_keys
    )
    if experiment.role != "diagnostic":
        summaries_valid = summaries_valid and all(
            str(row.get("claim_valid", "")).lower() in {"true", "1", "yes"}
            for row in summary_rows
        )

    if experiment.role == "diagnostic":
        return core_valid and config_valid and summaries_valid
    return (
        core_valid
        and config_valid
        and summaries_valid
        and bool(payload.get("claim_valid"))
        and int(payload.get("invalid_runs", -1)) == 0
    )


def write_suite_status(suite_dir: Path, payload: dict) -> None:
    path = suite_dir / "suite_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_run_files(
    suite_dir: Path, experiments: list[Experiment], args: argparse.Namespace
) -> None:
    suite_dir.mkdir(parents=True, exist_ok=True)
    commands = [exp.command(suite_dir, args.device) for exp in experiments]
    manifest = [
        {
            "protocol_version": PROTOCOL_VERSION,
            "name": exp.name,
            "purpose": exp.purpose,
            "patients": exp.patients,
            "modes": exp.modes,
            "meal_announcement": exp.meal_announcement,
            "hours": exp.hours,
            "repeats": exp.repeats,
            "seed": exp.seed,
            "start_time": exp.start_time,
            "split_label": exp.split_label,
            "role": exp.role,
            "model_condition": exp.model_condition,
            "arx_prior_label": exp.arx_prior_label,
            "expected_rows": exp.expected_rows,
            "checkpoint": str(exp.checkpoint),
            "checkpoint_sha256": exp.checkpoint_sha256,
            "arx_param_file": str(exp.arx_param_file),
            "arx_param_sha256": exp.arx_param_sha256,
            "compatibility_calibration_file": (
                str(exp.compatibility_calibration_file)
                if exp.compatibility_calibration_file is not None
                else ""
            ),
            "compatibility_calibration_sha256": (
                exp.compatibility_calibration_sha256
            ),
            "compatibility_reference_arx_file": (
                str(exp.compatibility_reference_arx_file)
                if exp.compatibility_reference_arx_file is not None
                else ""
            ),
            "compatibility_reference_arx_sha256": (
                exp.compatibility_reference_arx_sha256
            ),
            "source_tree_sha256": exp.source_tree_sha256,
            "requirements_lock_sha256": exp.requirements_lock_sha256,
            "requirements_lock": str(exp.requirements_lock),
            "runtime_fingerprint": exp.runtime_fingerprint,
            "runtime_identity": runtime_identity(),
            "device": exp.device,
            "protocol_fingerprint": exp.protocol_fingerprint,
            "output_dir": str(suite_dir / exp.name),
            "command": command,
        }
        for exp, command in zip(experiments, commands)
    ]
    (suite_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (suite_dir / "commands.sh").write_text(
        "\n".join(shell_join(cmd) for cmd in commands) + "\n",
        encoding="utf-8",
    )
    (suite_dir / "commands.ps1").write_text(
        "\n".join(
            "& " + " ".join(f'"{part}"' for part in cmd) for cmd in commands
        )
        + "\n",
        encoding="utf-8",
    )
    rerun = [
        sys.executable,
        str(ROOT / "scripts" / "run_paper_experiment_suite.py"),
        "--execute",
        "--scope",
        args.scope,
        "--suite-dir",
        str(suite_dir),
        "--checkpoint-unannounced",
        str(args.checkpoint_unannounced),
        "--checkpoint-announced",
        str(args.checkpoint_announced),
        "--arx-param-file",
        str(args.arx_param_file),
        "--mild-arx-param-file",
        str(args.mild_arx_param_file),
        "--severe-arx-param-file",
        str(args.severe_arx_param_file),
        "--compatibility-calibration-file",
        str(args.compatibility_calibration_file),
        "--requirements-lock",
        str(args.requirements_lock),
        "--device",
        args.device,
        "--hours",
        str(args.hours),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--patients",
        args.patients,
        "--split-label",
        args.split_label,
        "--start-time",
        args.start_time,
        "--smoke-hours",
        str(args.smoke_hours),
        "--smoke-start-time",
        args.smoke_start_time,
        "--calibration-hours",
        str(args.calibration_hours),
        "--calibration-repeats",
        str(args.calibration_repeats),
        "--calibration-seed",
        str(args.calibration_seed),
        "--generalization-hours",
        str(args.generalization_hours),
        "--generalization-repeats",
        str(args.generalization_repeats),
    ]
    if args.resume:
        rerun.append("--resume")
    if args.include_ablations:
        rerun.append("--include-ablations")
    if args.include_generalization:
        rerun.append("--include-generalization")
    if args.include_announced:
        rerun.append("--include-announced")
    run_suite = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(ROOT))}
mkdir -p {shlex.quote(str(suite_dir))}
{shell_join(rerun)}
"""
    (suite_dir / "run_suite.sh").write_text(run_suite, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or execute the causal compatibility PPO-MPC paper suite."
    )
    parser.add_argument(
        "--scope",
        choices=["calibration", "smoke", "paper", "full"],
        default="paper",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Run experiments sequentially."
    )
    parser.add_argument("--suite-dir", default=None)
    parser.add_argument("--checkpoint-unannounced", default=str(DEFAULT_UNANN_CKPT))
    parser.add_argument("--checkpoint-announced", default=str(DEFAULT_ANN_CKPT))
    parser.add_argument("--arx-param-file", default=str(DEFAULT_ARX_PARAM))
    parser.add_argument(
        "--mild-arx-param-file", default=str(DEFAULT_MILD_ARX_PARAM)
    )
    parser.add_argument(
        "--severe-arx-param-file", default=str(DEFAULT_SEVERE_ARX_PARAM)
    )
    parser.add_argument(
        "--compatibility-calibration-file",
        default=str(DEFAULT_COMPATIBILITY_CALIBRATION),
    )
    parser.add_argument("--requirements-lock", default=str(REQUIREMENTS_LOCK))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hours", type=float, default=72.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=61000)
    parser.add_argument("--patients", default=adult_range(6, 10))
    parser.add_argument(
        "--split-label", choices=["validation", "test"], default="validation"
    )
    parser.add_argument("--start-time", default="2018-01-01 00:00")
    parser.add_argument("--smoke-hours", type=float, default=3.0)
    parser.add_argument("--smoke-start-time", default="2018-01-01 08:00")
    parser.add_argument("--calibration-hours", type=float, default=24.0)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--calibration-seed", type=int, default=71000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments with a valid complete marker.",
    )
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--include-generalization", action="store_true")
    parser.add_argument("--include-announced", action="store_true")
    parser.add_argument("--generalization-hours", type=float, default=72.0)
    parser.add_argument("--generalization-repeats", type=int, default=3)
    return parser.parse_args()


def validate_input_files(
    args: argparse.Namespace, experiments: list[Experiment]
) -> None:
    missing = sorted(
        {
            str(path)
            for exp in experiments
            for path in [exp.checkpoint, exp.arx_param_file, exp.requirements_lock]
            if not path.exists()
        }
    )
    if args.scope != "calibration":
        calibration = resolve_root_path(args.compatibility_calibration_file)
        if not calibration.exists():
            missing.append(str(calibration))
    if missing:
        raise SystemExit(
            "Missing required protocol file(s):\n"
            + "\n".join(sorted(set(missing)))
            + (
                "\nRun --scope calibration first and pass its frozen JSON."
                if args.scope != "calibration"
                else ""
            )
        )


def main() -> None:
    args = parse_args()
    if args.suite_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_dir = ROOT / "results" / f"paper_compat_v4_{args.scope}_{stamp}"
    else:
        suite_dir = resolve_root_path(args.suite_dir)

    experiments = build_experiments(args)
    validate_input_files(args, experiments)
    write_run_files(suite_dir, experiments, args)
    print(f"Prepared {len(experiments)} experiments under: {suite_dir}")
    print(f"Expected summary rows: {sum(exp.expected_rows for exp in experiments)}")
    print(f"Manifest: {suite_dir / 'manifest.json'}")
    print(f"Commands: {suite_dir / 'commands.sh'}")
    if not args.execute:
        print("Dry run only. Re-run with --execute to launch the suite.")
        return

    try:
        assert_runtime_matches_lock(experiments[0].requirements_lock)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    completed_names: list[str] = []
    skipped_names: list[str] = []
    write_suite_status(
        suite_dir,
        {
            "status": "running",
            "protocol_version": PROTOCOL_VERSION,
            "completed": completed_names,
            "skipped": skipped_names,
        },
    )
    try:
        for idx, exp in enumerate(experiments, start=1):
            if args.resume and experiment_is_complete(suite_dir, exp):
                skipped_names.append(exp.name)
                print(f"\n[{idx}/{len(experiments)}] skip complete: {exp.name}")
                continue
            command = exp.command(suite_dir, args.device)
            print(f"\n[{idx}/{len(experiments)}] {exp.name}")
            print(shell_join(command))
            subprocess.run(command, cwd=ROOT, check=True)
            if not experiment_is_complete(suite_dir, exp):
                raise RuntimeError(
                    "experiment finished without a claim-valid completion marker: "
                    f"{exp.name}"
                )
            completed_names.append(exp.name)
            write_suite_status(
                suite_dir,
                {
                    "status": "running",
                    "protocol_version": PROTOCOL_VERSION,
                    "completed": completed_names,
                    "skipped": skipped_names,
                },
            )
    except Exception as exc:
        write_suite_status(
            suite_dir,
            {
                "status": "failed",
                "protocol_version": PROTOCOL_VERSION,
                "completed": completed_names,
                "skipped": skipped_names,
                "error": repr(exc),
            },
        )
        raise

    if args.scope == "calibration":
        calibration_command = [
            sys.executable,
            str(ROOT / "scripts" / "calibrate_model_compatibility.py"),
            "--suite-dir",
            str(suite_dir),
            "--minimum-trajectories",
            str(5 * int(args.calibration_repeats)),
        ]
        try:
            print("\n[calibration]")
            print(shell_join(calibration_command))
            subprocess.run(calibration_command, cwd=ROOT, check=True)
        except Exception as exc:
            write_suite_status(
                suite_dir,
                {
                    "status": "failed",
                    "protocol_version": PROTOCOL_VERSION,
                    "stage": "calibration",
                    "completed": completed_names,
                    "skipped": skipped_names,
                    "error": repr(exc),
                },
            )
            raise
        calibration_path = (
            suite_dir / "analysis" / "compatibility_calibration.json"
        )
        write_suite_status(
            suite_dir,
            {
                "status": "complete",
                "protocol_version": PROTOCOL_VERSION,
                "completed": completed_names,
                "skipped": skipped_names,
                "compatibility_calibration_file": str(calibration_path),
                "compatibility_calibration_sha256": sha256_file(calibration_path),
            },
        )
        print(f"Frozen calibration: {calibration_path}")
        return

    analysis = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_paper_experiment_suite.py"),
        "--suite-dir",
        str(suite_dir),
        "--allow-partial",
    ]
    try:
        print("\n[analysis]")
        print(shell_join(analysis))
        subprocess.run(analysis, cwd=ROOT, check=True)
    except Exception as exc:
        write_suite_status(
            suite_dir,
            {
                "status": "failed",
                "protocol_version": PROTOCOL_VERSION,
                "completed": completed_names,
                "skipped": skipped_names,
                "stage": "analysis",
                "error": repr(exc),
            },
        )
        raise
    analysis_status_path = suite_dir / "analysis" / "analysis_status.json"
    analysis_status = json.loads(
        analysis_status_path.read_text(encoding="utf-8")
    )
    final_status = (
        "partial" if analysis_status.get("status") == "PARTIAL" else "complete"
    )
    write_suite_status(
        suite_dir,
        {
            "status": final_status,
            "protocol_version": PROTOCOL_VERSION,
            "completed": completed_names,
            "skipped": skipped_names,
            "analysis_status": analysis_status,
        },
    )


if __name__ == "__main__":
    main()
