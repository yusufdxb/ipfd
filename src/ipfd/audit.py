"""Small public API for one paired counterfactual-fidelity audit.

The configuration-driven runner remains the artifact-production interface.  This
module is the integration surface for a lab that already has a live adapter and
wants the same L0 through L3 semantics without constructing a YAML file.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .fidelity.audit import audit_configuration
from .fidelity.config import AuditConfig, BranchState
from .fidelity.contracts import (
    ContractVerdict,
    ReplayAdapter,
    Snapshot,
    TrajectoryRecord,
    to_builtin,
)

__all__ = ["AuditResult", "DecisionContract", "audit"]


@dataclass(frozen=True)
class DecisionContract:
    """A named Boolean conclusion evaluated separately on each branch.

    The evaluator receives a :class:`TrajectoryRecord` whose ``env_id`` selects
    the uninterrupted or restored row.  It must be deterministic, side-effect
    free, and raise when required evidence is missing.
    """

    name: str
    evaluator: Callable[[TrajectoryRecord], bool]
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    true_label: str = "true"
    false_label: str = "false"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("DecisionContract.name must be a non-empty string")
        if not callable(self.evaluator):
            raise TypeError("DecisionContract.evaluator must be callable")
        if not isinstance(self.description, str):
            raise TypeError("DecisionContract.description must be a string")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("DecisionContract.parameters must be a mapping")
        if not isinstance(self.true_label, str) or not self.true_label.strip():
            raise ValueError("DecisionContract.true_label must be a non-empty string")
        if not isinstance(self.false_label, str) or not self.false_label.strip():
            raise ValueError("DecisionContract.false_label must be a non-empty string")
        if self.true_label == self.false_label:
            raise ValueError("DecisionContract true and false labels must differ")
        try:
            canonical_parameters = json.loads(
                json.dumps(to_builtin(self.parameters), sort_keys=True)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("DecisionContract.parameters must be JSON serializable") from exc
        object.__setattr__(self, "parameters", canonical_parameters)

    def evaluate(self, trajectory: TrajectoryRecord) -> bool:
        """Evaluate the contract and reject ambiguous truthy return values."""

        value = self.evaluator(trajectory)
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"decision {self.name!r} must return bool, got {type(value).__name__}"
            )
        return bool(value)

    def label(self, value: bool) -> str:
        return self.true_label if value else self.false_label

    def provenance(self) -> dict[str, Any]:
        """Return a best-effort implementation identity without claiming source availability."""

        implementation = (
            f"{getattr(self.evaluator, '__module__', '<unknown>')}."
            f"{getattr(self.evaluator, '__qualname__', type(self.evaluator).__qualname__)}"
        )
        try:
            source = inspect.getsource(self.evaluator).encode("utf-8")
        except (OSError, TypeError):
            source_digest = None
        else:
            source_digest = hashlib.sha256(source).hexdigest()
        return {
            "name": self.name,
            "description": self.description,
            "parameters": to_builtin(self.parameters),
            "parameter_provenance_complete": True,
            "implementation": implementation,
            "source_sha256": source_digest,
            "source_available": source_digest is not None,
            "true_label": self.true_label,
            "false_label": self.false_label,
        }


@dataclass(frozen=True)
class AuditResult:
    """Typed handle for the existing machine-readable audit record."""

    _data: Mapping[str, Any]

    @property
    def verdict(self) -> ContractVerdict:
        return ContractVerdict(str(self._data["verdict"]))

    @property
    def levels(self) -> Mapping[int, Mapping[str, Any]]:
        """Return L0 through L3 summaries keyed by tested horizon."""

        return {
            int(item["scope"]["horizon"]): item["levels"]
            for item in self._data["configurations"]
        }

    @property
    def fidelity_frontier(self) -> Mapping[str, Any]:
        return self._data["fidelity_frontier"]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible result."""

        return dict(to_builtin(self._data))


_PROVENANCE_FIELDS = (
    "adapter",
    "simulator",
    "simulator_version",
    "environment",
    "snapshot_protocol",
    "state_components_captured",
    "state_components_unavailable",
    "task_state_captured",
    "controller_or_policy_history_captured",
    "random_state_handling",
    "solver_state_availability",
    "sensor_refresh_behavior",
    "unsupported_restoration_claims",
)


def _provenance_errors(provenance: Any) -> list[str]:
    if not isinstance(provenance, Mapping):
        return ["provenance() must return a mapping"]
    missing = [name for name in _PROVENANCE_FIELDS if name not in provenance]
    errors = [f"missing {name}" for name in missing]
    for name in ("adapter", "simulator", "simulator_version", "environment", "snapshot_protocol"):
        if name in provenance and (
            provenance[name] is None or not str(provenance[name]).strip()
        ):
            errors.append(f"{name} must be non-empty")
    for name in (
        "state_components_captured",
        "state_components_unavailable",
        "task_state_captured",
        "unsupported_restoration_claims",
    ):
        value = provenance.get(name)
        if value is not None and (
            isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
        ):
            errors.append(f"{name} must be a sequence")
    try:
        json.dumps(to_builtin(provenance), sort_keys=True)
    except (TypeError, ValueError):
        errors.append("provenance must be JSON serializable")
    return errors


class _PublicAuditAdapter:
    """Validate public-surface invariants while delegating simulator behavior."""

    def __init__(
        self,
        adapter: ReplayAdapter,
        *,
        protocol: str,
        decision: str | DecisionContract,
    ) -> None:
        self._adapter = adapter
        self._protocol = protocol
        self._decision = decision

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def capture(self, env_ids: Sequence[int]) -> Snapshot:
        snapshot = self._adapter.capture(env_ids)
        if not isinstance(snapshot, Snapshot):
            raise TypeError("adapter.capture() must return Snapshot")
        if snapshot.protocol != self._protocol:
            raise RuntimeError(
                "captured snapshot protocol does not match the audited protocol: "
                f"captured {snapshot.protocol!r}, audited {self._protocol!r}"
            )
        if isinstance(snapshot.captured_components, (str, bytes)):
            raise TypeError("Snapshot.captured_components must be a sequence of names")
        if isinstance(snapshot.unavailable_components, (str, bytes)):
            raise TypeError("Snapshot.unavailable_components must be a sequence of names")
        return snapshot

    def decision(self, record: TrajectoryRecord, name: str) -> bool:
        if isinstance(self._decision, DecisionContract):
            if name != self._decision.name:
                raise KeyError(f"unknown decision contract {name!r}")
            first = self._decision.evaluate(record)
            second = self._decision.evaluate(record)
        else:
            first = self._adapter.decision(record, name)
            second = self._adapter.decision(record, name)
            if not isinstance(first, (bool, np.bool_)) or not isinstance(
                second, (bool, np.bool_)
            ):
                raise TypeError(f"decision {name!r} must return bool")
            first = bool(first)
            second = bool(second)
        if first != second:
            raise RuntimeError(
                f"decision {name!r} is nondeterministic for an unchanged trajectory"
            )
        return first


def _normalized_tolerances(
    tolerances: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if tolerances is None:
        return {"default": {"absolute": 0.0, "relative": 0.0}}
    if "default" not in tolerances:
        raise ValueError("tolerances must contain a 'default' comparison policy")
    result: dict[str, dict[str, Any]] = {}
    for category, values in tolerances.items():
        if not isinstance(category, str) or not category:
            raise ValueError("tolerance category names must be non-empty strings")
        if not isinstance(values, Mapping):
            raise TypeError(f"tolerances[{category!r}] must be a mapping")
        absolute = float(values.get("absolute", 0.0))
        relative = float(values.get("relative", 0.0))
        if not np.isfinite((absolute, relative)).all() or absolute < 0.0 or relative < 0.0:
            raise ValueError("tolerances must be finite and non-negative")
        policy: dict[str, Any] = {"absolute": absolute, "relative": relative}
        fields = values.get("fields", {})
        if not isinstance(fields, Mapping):
            raise TypeError(f"tolerances[{category!r}]['fields'] must be a mapping")
        normalized_fields: dict[str, dict[str, Any]] = {}
        for name, field_values in fields.items():
            if not isinstance(name, str) or not name:
                raise ValueError("field tolerance names must be non-empty strings")
            if not isinstance(field_values, Mapping):
                raise TypeError(f"field tolerance {category}.{name} must be a mapping")
            field_absolute = float(field_values.get("absolute", absolute))
            field_relative = float(field_values.get("relative", relative))
            if (
                not np.isfinite((field_absolute, field_relative)).all()
                or field_absolute < 0.0
                or field_relative < 0.0
            ):
                raise ValueError("field tolerances must be finite and non-negative")
            normalized_fields[name] = {
                "absolute": field_absolute,
                "relative": field_relative,
            }
            unit = field_values.get("unit")
            if unit is not None:
                if not isinstance(unit, str) or not unit.strip():
                    raise ValueError("field tolerance units must be non-empty strings")
                normalized_fields[name]["unit"] = unit.strip()
        if normalized_fields:
            policy["fields"] = normalized_fields
        result[category] = policy
    return result


def _frontier(configurations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(configurations, key=lambda item: int(item["scope"]["horizon"]))
    horizons = [int(item["scope"]["horizon"]) for item in ordered]
    statuses = [ContractVerdict(str(item["result"])) for item in ordered]
    supported = [horizon for horizon, status in zip(horizons, statuses, strict=True) if status is ContractVerdict.SUPPORTED]
    unsupported = [horizon for horizon, status in zip(horizons, statuses, strict=True) if status is ContractVerdict.UNSUPPORTED]
    insufficient = [
        horizon
        for horizon, status in zip(horizons, statuses, strict=True)
        if status is ContractVerdict.INSUFFICIENT_EVIDENCE
    ]
    first_untrusted = min(unsupported + insufficient) if unsupported or insufficient else None
    trusted_before = [value for value in supported if first_untrusted is None or value < first_untrusted]
    supported_after_untrusted = bool(
        first_untrusted is not None and any(value > first_untrusted for value in supported)
    )
    return {
        "tested_horizons": horizons,
        "supported_horizons": supported,
        "unsupported_horizons": unsupported,
        "insufficient_evidence_horizons": insufficient,
        "last_supported_horizon_before_failure": max(trusted_before) if trusted_before else None,
        "first_untrusted_horizon": first_untrusted,
        "non_monotonic": supported_after_untrusted,
    }


def _enrich_l3_records(
    data: dict[str, Any],
    *,
    decision_name: str,
    contract: DecisionContract | None,
) -> None:
    direction_counts: dict[int, dict[str, int]] = {}
    for record in data["records"]:
        decision = record["levels"]["L3"]["decisions"][decision_name]
        reference = bool(decision["reference"])
        restored = bool(decision["restored"])
        if reference == restored:
            direction = "MATCH"
        elif reference:
            direction = "REFERENCE_TRUE_RESTORED_FALSE"
        else:
            direction = "REFERENCE_FALSE_RESTORED_TRUE"
        decision["direction"] = direction
        decision["semantic_labels_declared"] = contract is not None
        decision["reference_label"] = contract.label(reference) if contract else None
        decision["restored_label"] = contract.label(restored) if contract else None
        horizon_counts = direction_counts.setdefault(int(record["horizon"]), {})
        horizon_counts[direction] = horizon_counts.get(direction, 0) + 1
    for configuration in data["configurations"]:
        horizon = int(configuration["scope"]["horizon"])
        configuration["levels"]["L3"]["direction_counts"] = direction_counts.get(
            horizon, {}
        )


def audit(
    *,
    adapter: ReplayAdapter,
    branch_step: int,
    horizons: Sequence[int],
    decision: str | DecisionContract,
    protocol: str | None = None,
    continuation: str = "exact_action",
    action_source: str = "recorded",
    seed: int = 0,
    tolerances: Mapping[str, Mapping[str, Any]] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> AuditResult:
    """Audit one branch under identical future actions.

    ``tolerances=None`` deliberately means exact comparison, not a simulator-
    independent numerical tolerance.  Supply field-specific tolerances when the
    downstream claim permits numerical error.
    """

    required_methods = (
        "reset",
        "action",
        "capture",
        "restore",
        "observe",
        "step",
        "decision",
        "provenance",
    )
    missing_methods = [name for name in required_methods if not callable(getattr(adapter, name, None))]
    if missing_methods:
        raise TypeError(f"adapter is missing required methods: {', '.join(missing_methods)}")
    if isinstance(branch_step, bool) or not isinstance(branch_step, int) or branch_step < 0:
        raise ValueError("branch_step must be an integer >= 0")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(horizons, (str, bytes)):
        raise TypeError("horizons must be a sequence of positive integers")
    requested_horizons = tuple(sorted(set(horizons)))
    if not requested_horizons or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in requested_horizons
    ):
        raise ValueError("horizons must contain positive integers")
    if not isinstance(continuation, str) or not continuation.strip():
        raise ValueError("continuation must be a non-empty string")
    if continuation not in {"exact_action", "open_loop", "restored", "cold"}:
        raise ValueError(
            "public audit continuation must be one of exact_action, open_loop, restored, or cold"
        )
    if not isinstance(action_source, str) or not action_source.strip():
        raise ValueError("action_source must be a non-empty string")
    if isinstance(decision, str):
        if not decision.strip():
            raise ValueError("decision must be a non-empty name")
        decision_name = decision.strip()
        decision_provenance: dict[str, Any] = {
            "name": decision_name,
            "implementation": "adapter.decision",
            "source_sha256": None,
            "source_available": False,
            "parameters": None,
            "parameter_provenance_complete": False,
            "true_label": None,
            "false_label": None,
        }
    elif isinstance(decision, DecisionContract):
        decision_name = decision.name
        decision_provenance = decision.provenance()
    else:
        raise TypeError("decision must be a name or DecisionContract")

    adapter_provenance = adapter.provenance()
    provenance_errors = _provenance_errors(adapter_provenance)
    if provenance_errors:
        raise ValueError("adapter provenance is incomplete: " + "; ".join(provenance_errors))
    assert isinstance(adapter_provenance, Mapping)
    if isinstance(decision, str):
        declared_decisions = adapter_provenance.get("decision_contracts")
        if isinstance(declared_decisions, Mapping):
            declared_decision = declared_decisions.get(decision_name)
            if isinstance(declared_decision, Mapping):
                decision_provenance["description"] = declared_decision.get("definition")
                decision_provenance["parameters"] = {
                    str(key): to_builtin(value)
                    for key, value in declared_decision.items()
                    if key != "definition"
                }
                decision_provenance["parameter_provenance_complete"] = True
    declared_protocol = str(adapter_provenance["snapshot_protocol"])
    selected_protocol = protocol or declared_protocol
    if not isinstance(selected_protocol, str) or not selected_protocol.strip():
        raise ValueError("protocol must be a non-empty string")
    if selected_protocol != declared_protocol:
        raise ValueError(
            f"protocol {selected_protocol!r} does not match adapter provenance {declared_protocol!r}"
        )
    metadata_values = dict(metadata or {})
    task = metadata_values.get("task", str(adapter_provenance.get("task", "<not_declared>")))
    environment = metadata_values.get("environment", str(adapter_provenance["environment"]))
    simulator = metadata_values.get("simulator", str(adapter_provenance["simulator"]))
    if any(not isinstance(value, str) or not value.strip() for value in (task, environment, simulator)):
        raise ValueError("metadata task, environment, and simulator values must be non-empty strings")

    normalized_tolerances = _normalized_tolerances(tolerances)
    config = AuditConfig(
        source_path=Path("<library-api>"),
        adapter={"kind": str(adapter_provenance["adapter"]), "simulator": simulator},
        simulator_version=str(adapter_provenance["simulator_version"]),
        environment=environment,
        task=task,
        snapshot_protocol=selected_protocol,
        branch_states=(BranchState(id=f"seed-{seed}-step-{branch_step}", step=branch_step, seed=seed, cluster=str(seed)),),
        horizons=requested_horizons,
        continuation_mode=continuation,
        action_source=action_source,
        decision_functions=(decision_name,),
        tolerances=normalized_tolerances,
        independent_cluster_key="seed",
        output_directory=Path("."),
        minimum_independent_clusters=1,
        reduction={"enabled": False},
        regression=None,
        raw={},
    )
    checked_adapter = _PublicAuditAdapter(
        adapter,
        protocol=selected_protocol,
        decision=decision,
    )
    data = audit_configuration(checked_adapter, config)
    _enrich_l3_records(
        data,
        decision_name=decision_name,
        contract=decision if isinstance(decision, DecisionContract) else None,
    )
    configurations = data["configurations"]
    verdicts = [ContractVerdict(str(item["result"])) for item in configurations]
    if any(value is ContractVerdict.UNSUPPORTED for value in verdicts):
        verdict = ContractVerdict.UNSUPPORTED
    elif any(value is ContractVerdict.INSUFFICIENT_EVIDENCE for value in verdicts):
        verdict = ContractVerdict.INSUFFICIENT_EVIDENCE
    else:
        verdict = ContractVerdict.SUPPORTED
    for item in configurations:
        item["scope"]["hardware_and_software_provenance"] = "embedded_adapter_provenance"
    data.update(
        {
            "verdict": verdict.value,
            "fidelity_frontier": _frontier(configurations),
            "decision_contract": decision_provenance,
            "provenance": {"adapter": to_builtin(adapter_provenance)},
            "comparison_policy": {
                "tolerances": normalized_tolerances,
                "default_is_exact_comparison": tolerances is None,
            },
        }
    )
    return AuditResult(data)
