"""Configuration loading and fail-closed validation for ``ipfd audit``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["AuditConfig", "BranchState", "load_config"]

_REQUIRED_HORIZONS = {1, 5, 10, 30, 90}


@dataclass(frozen=True)
class BranchState:
    id: str
    step: int
    seed: int
    cluster: str


@dataclass(frozen=True)
class AuditConfig:
    source_path: Path
    adapter: dict[str, Any]
    simulator_version: str
    environment: str
    task: str
    snapshot_protocol: str
    branch_states: tuple[BranchState, ...]
    horizons: tuple[int, ...]
    continuation_mode: str
    action_source: str
    decision_functions: tuple[str, ...]
    tolerances: dict[str, dict[str, float]]
    independent_cluster_key: str
    output_directory: Path
    minimum_independent_clusters: int
    reduction: dict[str, Any]
    regression: dict[str, Any] | None
    raw: dict[str, Any]

    def tolerance(self, category: str) -> tuple[float, float]:
        values = self.tolerances.get(category, self.tolerances.get("default", {}))
        return float(values.get("absolute", 0.0)), float(values.get("relative", 0.0))


def _require_nonempty_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _branch_states(raw: Any) -> tuple[BranchState, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("branch_states must be a non-empty list")
    states: list[BranchState] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"branch_states[{index}] must be a mapping")
        step = item.get("step")
        seed = item.get("seed")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"branch_states[{index}].step must be an integer >= 0")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"branch_states[{index}].seed must be an integer")
        state_id = str(item.get("id", f"seed-{seed}-step-{step}"))
        cluster = str(item.get("cluster", seed))
        states.append(BranchState(id=state_id, step=step, seed=seed, cluster=cluster))
    return tuple(states)


def _horizons(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("horizons must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in raw):
        raise ValueError("every horizon must be an integer >= 1")
    values = tuple(sorted(set(raw)))
    missing = sorted(_REQUIRED_HORIZONS - set(values))
    if missing:
        raise ValueError(f"horizons must include the required control-step horizons {missing}")
    return values


def _tolerances(raw: Any) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("tolerances must be a non-empty mapping")
    result: dict[str, dict[str, float]] = {}
    for category, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"tolerances.{category} must be a mapping")
        absolute = float(value.get("absolute", 0.0))
        relative = float(value.get("relative", 0.0))
        if absolute < 0.0 or relative < 0.0:
            raise ValueError(f"tolerances.{category} values must be non-negative")
        result[str(category)] = {"absolute": absolute, "relative": relative}
    if "default" not in result:
        raise ValueError("tolerances.default is required; IPFD does not supply a universal tolerance")
    return result


def load_config(path: str | Path) -> AuditConfig:
    """Load an audit configuration from YAML without applying hidden defaults."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read audit configuration {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("audit configuration must be a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    adapter = raw.get("adapter")
    if not isinstance(adapter, dict):
        raise ValueError("adapter must be a mapping")
    _require_nonempty_string(adapter, "kind")
    decisions = raw.get("decision_functions")
    if not isinstance(decisions, list) or not decisions or not all(isinstance(item, str) and item for item in decisions):
        raise ValueError("decision_functions must be a non-empty list of names")
    output = Path(_require_nonempty_string(raw, "output_directory"))
    if not output.is_absolute():
        output = (source.parent / output).resolve()
    minimum = raw.get("minimum_independent_clusters", 1)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_independent_clusters must be an integer >= 1")
    return AuditConfig(
        source_path=source,
        adapter=dict(adapter),
        simulator_version=_require_nonempty_string(raw, "simulator_version"),
        environment=_require_nonempty_string(raw, "environment"),
        task=_require_nonempty_string(raw, "task"),
        snapshot_protocol=_require_nonempty_string(raw, "snapshot_protocol"),
        branch_states=_branch_states(raw.get("branch_states")),
        horizons=_horizons(raw.get("horizons")),
        continuation_mode=_require_nonempty_string(raw, "continuation_mode"),
        action_source=_require_nonempty_string(raw, "action_source"),
        decision_functions=tuple(decisions),
        tolerances=_tolerances(raw.get("tolerances")),
        independent_cluster_key=_require_nonempty_string(raw, "independent_cluster_key"),
        output_directory=output,
        minimum_independent_clusters=minimum,
        reduction=dict(raw.get("reduction", {})),
        regression=dict(raw["regression"]) if isinstance(raw.get("regression"), dict) else None,
        raw=dict(raw),
    )
