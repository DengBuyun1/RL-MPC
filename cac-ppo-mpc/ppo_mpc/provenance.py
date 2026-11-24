from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from functools import lru_cache
from pathlib import Path


HASHED_SUFFIXES = {
    ".py",
    ".json",
    ".csv",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        return "MISSING"
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def source_tree_sha256(root_value: str) -> str:
    root = Path(root_value).resolve()
    files: list[Path] = []
    for folder_name in ["ppo_mpc", "scripts", "mpc_assets"]:
        folder = root / folder_name
        if not folder.exists():
            continue
        files.extend(
            path
            for path in folder.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in HASHED_SUFFIXES
        )
    for path in [root / "requirements.txt", *root.glob("requirements-lock*.txt")]:
        if path.exists():
            files.append(path)

    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def runtime_identity() -> dict[str, str]:
    packages = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "gymnasium",
        "matplotlib",
        "setuptools",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "MISSING"
    try:
        import torch

        # Distribution metadata may strip the CUDA local-version suffix.
        packages["torch"] = str(torch.__version__)
    except Exception:
        pass
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        **packages,
    }


def _canonical_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def parse_exact_requirements_lock(path: str | Path) -> tuple[str | None, dict[str, str]]:
    resolved = Path(path)
    if not resolved.is_file():
        raise RuntimeError(f"requirements lock is missing: {resolved}")
    python_version: str | None = None
    packages: dict[str, str] = {}
    for line_number, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# python-version=="):
            python_version = line.split("==", 1)[1].strip()
            continue
        if line.startswith("#") or line.startswith("--"):
            continue
        if "==" not in line or ";" in line:
            raise RuntimeError(
                f"lock must contain unconditional exact pins; line {line_number}: {raw}"
            )
        name, version = (item.strip() for item in line.split("==", 1))
        canonical = _canonical_package_name(name)
        if not canonical or not version or canonical in packages:
            raise RuntimeError(f"invalid or duplicate lock entry at line {line_number}: {raw}")
        packages[canonical] = version
    if not packages:
        raise RuntimeError(f"requirements lock has no exact package pins: {resolved}")
    return python_version, packages


def assert_runtime_matches_lock(path: str | Path) -> dict[str, object]:
    """Fail closed when the active interpreter differs from the paper lock."""
    resolved = Path(path).resolve()
    expected_python, expected_packages = parse_exact_requirements_lock(resolved)
    identity = runtime_identity()
    mismatches: list[str] = []
    if expected_python and identity["python_version"] != expected_python:
        mismatches.append(
            f"python: expected {expected_python}, got {identity['python_version']}"
        )
    for name, expected in expected_packages.items():
        actual = identity.get(name, "MISSING")
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError(
            f"runtime does not match {resolved.name}:\n- " + "\n- ".join(mismatches)
        )
    return {
        "lock_path": str(resolved),
        "lock_sha256": sha256_file(resolved),
        "expected_python": expected_python,
        "expected_packages": expected_packages,
        "runtime_identity": identity,
    }


def runtime_fingerprint() -> str:
    encoded = json.dumps(
        runtime_identity(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
