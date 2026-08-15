"""Public data contract for simulator snapshot-and-restore audits.

The abstraction deliberately carries unavailable state next to captured state.
An adapter is not allowed to imply that an omitted simulator component was saved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

ArrayLike = Any

__all__ = [
    "ArrayLike",
    "ContractVerdict",
    "ObservationRecord",
    "ReplayAdapter",
    "Snapshot",
    "StepRecord",
    "TrajectoryRecord",
    "to_builtin",
]


class ContractVerdict(str, Enum):
    """A scoped empirical conclusion, never a universal simulator verdict."""

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def to_builtin(value: Any) -> Any:
    """Convert nested simulator records into JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return value.numpy().tolist()
        except (RuntimeError, TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass
class Snapshot:
    """Captured state and its explicit support boundary."""

    protocol: str
    values: Mapping[str, Any]
    captured_components: Sequence[str]
    unavailable_components: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "values": to_builtin(self.values),
            "captured_components": list(self.captured_components),
            "unavailable_components": list(self.unavailable_components),
            "metadata": to_builtin(self.metadata),
        }


@dataclass
class ObservationRecord:
    """The exposed state categories inspected at the restoration boundary.

    Array-valued leaves use the leading dimension for environment identity. The
    audit compares one uninterrupted row with one restored row.
    """

    scene_state: Mapping[str, Any]
    policy_observations: Mapping[str, Any]
    privileged_observations: Mapping[str, Any] = field(default_factory=dict)
    task_state: Mapping[str, Any] = field(default_factory=dict)
    controller_targets: Mapping[str, Any] = field(default_factory=dict)
    sensor_state: Mapping[str, Any] = field(default_factory=dict)
    counters: Mapping[str, Any] = field(default_factory=dict)
    unavailable: Sequence[str] = ()

    def categories(self) -> dict[str, Mapping[str, Any]]:
        return {
            "scene_state": self.scene_state,
            "policy_observations": self.policy_observations,
            "privileged_observations": self.privileged_observations,
            "task_state": self.task_state,
            "controller_targets": self.controller_targets,
            "sensor_state": self.sensor_state,
            "counters": self.counters,
        }

    def to_dict(self) -> dict[str, Any]:
        result = {name: to_builtin(values) for name, values in self.categories().items()}
        result["unavailable"] = list(self.unavailable)
        return result


@dataclass
class StepRecord:
    """One control-step result for all audited environments."""

    observation: ObservationRecord
    contact_state: Mapping[str, Any]
    task_outputs: Mapping[str, Any] = field(default_factory=dict)
    terminated: Mapping[str, Any] = field(default_factory=dict)
    reward: Mapping[str, Any] = field(default_factory=dict)
    semantic: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "contact_state": to_builtin(self.contact_state),
            "task_outputs": to_builtin(self.task_outputs),
            "terminated": to_builtin(self.terminated),
            "reward": to_builtin(self.reward),
            "semantic": to_builtin(self.semantic),
        }


@dataclass
class TrajectoryRecord:
    """Finite-horizon continuation used by L2 and L3."""

    steps: list[StepRecord]
    actions: list[Any]
    env_id: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "actions": to_builtin(self.actions),
            "steps": [step.to_dict() for step in self.steps],
            "metadata": to_builtin(self.metadata),
        }


@runtime_checkable
class ReplayAdapter(Protocol):
    """Minimal simulator-neutral state replay interface."""

    def capture(self, env_ids: Sequence[int]) -> Snapshot:
        ...

    def restore(self, snapshot: Snapshot, env_ids: Sequence[int]) -> None:
        ...

    def observe(self, env_ids: Sequence[int]) -> ObservationRecord:
        ...

    def step(self, actions: ArrayLike) -> StepRecord:
        ...

    def decision(self, record: TrajectoryRecord, name: str) -> bool:
        ...

    def provenance(self) -> Mapping[str, object]:
        ...
