"""Hardware, software, source, and configuration provenance for v2 audits."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import to_builtin

__all__ = ["collect_provenance", "sha256_file"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(root: Path, ignored_status_paths: Sequence[Path]) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    try:
        excluded: list[str] = []
        for path in ignored_status_paths:
            try:
                relative = path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            excluded.append(relative.as_posix())
        status_arguments = [
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            ".",
            *(f":(exclude){path}" for path in excluded),
        ]
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run(*status_arguments)),
            "ignored_generated_output_paths": excluded,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "branch": None, "dirty": None}


def collect_provenance(
    *,
    adapter: Mapping[str, object],
    config_path: Path,
    repo_root: Path,
    ignored_status_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Collect reproducibility metadata without recording a private hostname."""

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": _git(repo_root, ignored_status_paths),
        "configuration": {
            "path": config_path.name,
            "sha256": sha256_file(config_path),
        },
        "hardware": {
            "architecture": platform.machine(),
            "processor_family": platform.processor() or None,
            "accelerator_identifier": "not_recorded_in_public_artifacts",
        },
        "software": {
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "python": platform.python_version(),
            "ipfd": _package_version("ipfd"),
            "numpy": _package_version("numpy"),
            "PyYAML": _package_version("PyYAML"),
            "mujoco": _package_version("mujoco"),
            "isaaclab": _package_version("isaaclab"),
        },
        "adapter": to_builtin(adapter),
        "adapter_provenance_sha256": hashlib.sha256(
            json.dumps(to_builtin(adapter), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
