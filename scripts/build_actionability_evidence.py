#!/usr/bin/env python3
"""Derive an actionability evidence artifact from saved simulator rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ipfd import build_report
from ipfd.actionability import evaluate_actionability
from ipfd.replay import load_rollout

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("manifest must be a JSON object")
    return data


def build_evidence(manifest_path: Path) -> dict[str, Any]:
    """Load uniquely identified rollouts and derive their alarm relations."""
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema") != "ipfd.actionability_manifest.v1":
        raise ValueError("manifest.schema must equal 'ipfd.actionability_manifest.v1'")
    task = manifest.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("manifest.task must be a non-empty string")
    checkpoint = manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or _SHA256.fullmatch(checkpoint) is None:
        raise ValueError("manifest.checkpoint_sha256 must be a lowercase SHA-256 digest")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest.cases must be a non-empty list")

    output_cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    rollout_digests: set[str] = set()
    runtime: dict[str, str] | None = None
    software: dict[str, str | bool] | None = None
    for index, case in enumerate(cases):
        prefix = f"manifest.cases[{index}]"
        if not isinstance(case, Mapping):
            raise ValueError(f"{prefix} must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError(f"{prefix}.case_id must be non-empty and unique")
        relative_path = case.get("rollout")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"{prefix}.rollout must be a non-empty path")
        disturbance = case.get("disturbance_onset")
        stride = case.get("probe_stride", 1)
        expected = case.get("expected_relation")
        if isinstance(disturbance, bool) or not isinstance(disturbance, int):
            raise ValueError(f"{prefix}.disturbance_onset must be an integer")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError(f"{prefix}.probe_stride must be an integer >= 1")
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"{prefix}.expected_relation must be a non-empty string")

        rollout_path = Path(relative_path)
        if not rollout_path.is_absolute():
            rollout_path = manifest_path.parent / rollout_path
        try:
            digest = hashlib.sha256(rollout_path.read_bytes()).hexdigest()
            rollout = load_rollout(rollout_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load {prefix}.rollout: {exc}") from exc
        if digest in rollout_digests:
            raise ValueError(f"{prefix}.rollout duplicates another case")
        if rollout.seed is None:
            raise ValueError(f"{prefix}.rollout must record an integer seed")
        if rollout.meta.get("source") != "isaac_lab":
            raise ValueError(f"{prefix}.rollout source must equal 'isaac_lab'")
        if rollout.meta.get("task") != task:
            raise ValueError(f"{prefix}.rollout task must match the manifest")
        if rollout.meta.get("checkpoint_sha256") != checkpoint:
            raise ValueError(f"{prefix}.rollout checkpoint must match the manifest")
        case_runtime = rollout.meta.get("runtime")
        if not isinstance(case_runtime, Mapping) or not all(
            isinstance(case_runtime.get(key), str) and case_runtime.get(key)
            for key in ("isaaclab", "isaacsim", "torch")
        ):
            raise ValueError(f"{prefix}.rollout must record the required runtime fingerprint")
        normalized_runtime = {
            key: str(case_runtime[key]) for key in ("isaaclab", "isaacsim", "torch")
        }
        if runtime is None:
            runtime = normalized_runtime
        elif normalized_runtime != runtime:
            raise ValueError(f"{prefix}.rollout runtime does not match the other cases")
        case_software = rollout.meta.get("software")
        if (
            not isinstance(case_software, Mapping)
            or not isinstance(case_software.get("ipfd_version"), str)
            or not case_software.get("ipfd_version")
            or not isinstance(case_software.get("git_commit"), str)
            or _GIT_COMMIT.fullmatch(str(case_software.get("git_commit"))) is None
            or case_software.get("git_dirty") is not False
        ):
            raise ValueError(f"{prefix}.rollout must record clean source provenance")
        normalized_software: dict[str, str | bool] = {
            "ipfd_version": str(case_software["ipfd_version"]),
            "git_commit": str(case_software["git_commit"]),
            "git_dirty": False,
        }
        if software is None:
            software = normalized_software
        elif normalized_software != software:
            raise ValueError(f"{prefix}.rollout software does not match the other cases")
        derived = evaluate_actionability(
            build_report(rollout),
            disturbance_onset=disturbance,
            probe_stride=stride,
        )
        output_cases.append(
            {
                "case_id": case_id,
                "seed": rollout.seed,
                "rollout_sha256": digest,
                "expected_relation": expected,
                "alarm_relation": derived.alarm_relation,
                "disturbance_onset": disturbance,
                "probe_stride": stride,
                "t_alarm": derived.t_alarm,
                "ponr_earliest": derived.ponr_earliest,
                "ponr_latest": derived.ponr_latest,
            }
        )
        case_ids.add(case_id)
        rollout_digests.add(digest)

    return {
        "schema": "ipfd.actionability.v1",
        "status": "complete",
        "source": "isaac_lab",
        "task": task,
        "checkpoint_sha256": checkpoint,
        "runtime": runtime,
        "software": software,
        "cases": output_cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evidence = build_evidence(args.manifest)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
