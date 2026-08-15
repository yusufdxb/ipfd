"""Strict input and provenance helpers for counterfactual fidelity evidence."""

from __future__ import annotations

import hashlib
import json
import lzma
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from .envelope import BranchComparison, validate_branch_comparisons

__all__ = [
    "EvidenceManifest",
    "LoadedBranchComparisons",
    "build_evidence_manifest",
    "load_branch_comparisons",
    "normalize_branch_comparison",
]


@dataclass(frozen=True)
class LoadedBranchComparisons:
    """Validated records plus immutable source-file identity."""

    records: tuple[BranchComparison, ...]
    source_name: str
    source_sha256: str
    source_bytes: int
    content_sha256: str
    content_bytes: int
    compression: str | None
    input_schema: str


@dataclass(frozen=True)
class EvidenceManifest:
    """Deterministic source, code, and analysis provenance."""

    source: dict[str, Any]
    code: dict[str, Any]
    analysis: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "counterfactual_fidelity_evidence_manifest",
            "source": self.source,
            "code": self.code,
            "analysis": self.analysis,
        }


def _strict_string(value: Any, *, field: str, line_number: int | None = None) -> str:
    location = f" at line {line_number}" if line_number is not None else ""
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise TypeError(f"{field}{location} must be a nonempty string or integer identifier")
    result = str(value)
    if not result.strip():
        raise ValueError(f"{field}{location} must not be empty")
    return result


def _strict_scope(value: Any, *, field: str, line_number: int | None = None) -> str:
    location = f" at line {line_number}" if line_number is not None else ""
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field}{location} must be a nonempty string")
    return value.strip()


def _strict_bool(value: Any, *, field: str, line_number: int | None = None) -> bool:
    location = f" at line {line_number}" if line_number is not None else ""
    if not isinstance(value, bool):
        raise TypeError(f"{field}{location} must be a JSON boolean")
    return value


def _strict_int(
    value: Any,
    *,
    field: str,
    positive: bool,
    line_number: int | None = None,
) -> int:
    location = f" at line {line_number}" if line_number is not None else ""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field}{location} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field}{location} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field}{location} must be nonnegative")
    return value


def _aliased(
    record: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    line_number: int | None,
) -> Any:
    present = [name for name in names if name in record]
    location = f" at line {line_number}" if line_number is not None else ""
    if not present:
        raise ValueError(f"missing required field {names[0]}{location}")
    values = [record[name] for name in present]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"conflicting aliases {', '.join(present)}{location}")
    return values[0]


def normalize_branch_comparison(record: Mapping[str, Any], *, line_number: int | None = None) -> BranchComparison:
    """Normalize one compatible flat record and reject silent coercions."""

    cluster_id = _strict_string(
        _aliased(record, ("cluster_id", "base_seed", "seed", "cluster"), line_number=line_number),
        field="cluster_id",
        line_number=line_number,
    )
    protocol = _strict_scope(
        _aliased(record, ("protocol", "snapshot_protocol"), line_number=line_number),
        field="protocol",
        line_number=line_number,
    )
    continuation = _strict_scope(
        _aliased(record, ("continuation", "continuation_mode"), line_number=line_number),
        field="continuation",
        line_number=line_number,
    )
    predicate = _strict_scope(
        _aliased(record, ("predicate", "decision_function"), line_number=line_number),
        field="predicate",
        line_number=line_number,
    )
    horizon = _strict_int(
        _aliased(record, ("horizon",), line_number=line_number),
        field="horizon",
        positive=True,
        line_number=line_number,
    )
    actual_steps = _strict_int(
        _aliased(record, ("actual_continuation_steps",), line_number=line_number),
        field="actual_continuation_steps",
        positive=True,
        line_number=line_number,
    )
    reference = _strict_bool(
        _aliased(record, ("reference_decision",), line_number=line_number),
        field="reference_decision",
        line_number=line_number,
    )
    candidate = _strict_bool(
        _aliased(record, ("candidate_decision", "restored_decision"), line_number=line_number),
        field="candidate_decision",
        line_number=line_number,
    )
    if "decision_match" in record:
        declared_match = _strict_bool(record["decision_match"], field="decision_match", line_number=line_number)
        if declared_match != (reference == candidate):
            location = f" at line {line_number}" if line_number is not None else ""
            raise ValueError(f"decision_match contradicts the decision values{location}")
    branch_step = None
    if "branch_step" in record and record["branch_step"] is not None:
        branch_step = _strict_int(record["branch_step"], field="branch_step", positive=False, line_number=line_number)
    schedule_id_value = record.get("schedule_id", record.get("schedule_sha256"))
    schedule_id = (
        _strict_string(schedule_id_value, field="schedule_id", line_number=line_number) if schedule_id_value is not None else None
    )
    schedule_equivalent = None
    if "schedule_equivalent" in record:
        schedule_equivalent = _strict_bool(record["schedule_equivalent"], field="schedule_equivalent", line_number=line_number)
    comparison = BranchComparison(
        cluster_id=cluster_id,
        branch_id=_strict_string(
            _aliased(record, ("branch_id",), line_number=line_number),
            field="branch_id",
            line_number=line_number,
        ),
        branch_step=branch_step,
        protocol=protocol,
        continuation=continuation,
        disturbance=_strict_scope(
            _aliased(record, ("disturbance",), line_number=line_number),
            field="disturbance",
            line_number=line_number,
        ),
        phase=_strict_scope(
            _aliased(record, ("phase",), line_number=line_number),
            field="phase",
            line_number=line_number,
        ),
        predicate=predicate,
        horizon=horizon,
        actual_continuation_steps=actual_steps,
        reference_decision=reference,
        candidate_decision=candidate,
        schedule_id=schedule_id,
        schedule_equivalent=schedule_equivalent,
        replicate_id=_strict_string(record.get("replicate_id", "0"), field="replicate_id", line_number=line_number),
    )
    if comparison.actual_continuation_steps != comparison.horizon:
        location = f" at line {line_number}" if line_number is not None else ""
        raise ValueError(f"actual_continuation_steps does not equal horizon{location}")
    if comparison.schedule_equivalent is False:
        location = f" at line {line_number}" if line_number is not None else ""
        raise ValueError(f"disturbance schedule is not equivalent{location}")
    return comparison


def _normalize_nested_l3_record(record: Mapping[str, Any], *, line_number: int) -> list[BranchComparison]:
    """Expand one native L0-L3 audit row into decision-level comparisons."""

    context = f" at line {line_number}"
    levels = record.get("levels")
    if not isinstance(levels, Mapping):
        raise TypeError(f"levels{context} must be an object")
    l3 = levels.get("L3")
    if not isinstance(l3, Mapping):
        raise ValueError(f"missing L3 decision evidence{context}")
    decisions = l3.get("decisions")
    if not isinstance(decisions, Mapping) or not decisions:
        raise ValueError(f"L3 decisions{context} must be a nonempty object")
    cluster_value = record.get("cluster", record.get("seed"))
    cluster_id = _strict_string(cluster_value, field="cluster_id", line_number=line_number)
    horizon = _strict_int(record.get("horizon"), field="horizon", positive=True, line_number=line_number)
    branch_step = _strict_int(record.get("branch_step"), field="branch_step", positive=False, line_number=line_number)
    branch_id = _strict_string(record.get("branch_id"), field="branch_id", line_number=line_number)
    protocol = _strict_scope(record.get("snapshot_protocol"), field="protocol", line_number=line_number)
    continuation = _strict_scope(record.get("continuation_mode"), field="continuation", line_number=line_number)
    disturbance = _strict_scope(
        record.get("disturbance", "UNSPECIFIED"),
        field="disturbance",
        line_number=line_number,
    )
    phase = _strict_scope(record.get("phase", "UNSPECIFIED"), field="phase", line_number=line_number)
    normalized: list[BranchComparison] = []
    for predicate, decision in sorted(decisions.items(), key=lambda item: str(item[0])):
        if not isinstance(decision, Mapping):
            raise TypeError(f"L3 decision {predicate!r}{context} must be an object")
        reference = _strict_bool(decision.get("reference"), field="reference_decision", line_number=line_number)
        candidate = _strict_bool(decision.get("restored"), field="candidate_decision", line_number=line_number)
        if "agreement" in decision:
            agreement = _strict_bool(decision["agreement"], field="decision_match", line_number=line_number)
            if agreement != (reference == candidate):
                raise ValueError(f"L3 agreement contradicts decision values{context}")
        normalized.append(
            BranchComparison(
                cluster_id=cluster_id,
                branch_id=branch_id,
                branch_step=branch_step,
                protocol=protocol,
                continuation=continuation,
                disturbance=disturbance,
                phase=phase,
                predicate=_strict_scope(str(predicate), field="predicate", line_number=line_number),
                horizon=horizon,
                actual_continuation_steps=horizon,
                reference_decision=reference,
                candidate_decision=candidate,
            )
        )
    return normalized


def load_branch_comparisons(path: Path | str) -> LoadedBranchComparisons:
    """Load a strict JSONL evidence file and verify semantic uniqueness."""

    source = Path(path)
    source_payload = source.read_bytes()
    if source.suffix == ".xz":
        try:
            payload = lzma.decompress(source_payload)
        except lzma.LZMAError as exc:
            raise ValueError(f"invalid xz-compressed evidence: {exc}") from exc
        compression = "xz"
    else:
        payload = source_payload
        compression = None
    records: list[BranchComparison] = []
    schemas: set[str] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"blank JSONL record at line {line_number}")
        try:
            value = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"JSONL record at line {line_number} is not an object")
        if "levels" in value:
            schemas.add("native_l3_audit_v1")
            records.extend(_normalize_nested_l3_record(value, line_number=line_number))
        else:
            schemas.add("flat_branch_comparison_v1")
            records.append(normalize_branch_comparison(value, line_number=line_number))
    if len(schemas) > 1:
        raise ValueError("mixed branch-comparison input schemas are not permitted")
    validated = validate_branch_comparisons(records)
    return LoadedBranchComparisons(
        records=validated,
        source_name=source.name,
        source_sha256=hashlib.sha256(source_payload).hexdigest(),
        source_bytes=len(source_payload),
        content_sha256=hashlib.sha256(payload).hexdigest(),
        content_bytes=len(payload),
        compression=compression,
        input_schema=next(iter(schemas), "unknown"),
    )


def _code_provenance() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())


def build_evidence_manifest(
    loaded: LoadedBranchComparisons,
    *,
    configuration: Mapping[str, Any],
) -> EvidenceManifest:
    """Build deterministic evidence provenance without leaking local paths."""

    try:
        canonical_config = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence-manifest configuration must be JSON-compatible") from exc
    normalized_config = json.loads(canonical_config)
    commit, dirty = _code_provenance()
    return EvidenceManifest(
        source={
            "logical_name": loaded.source_name,
            "sha256": loaded.source_sha256,
            "bytes": loaded.source_bytes,
            "content_sha256": loaded.content_sha256,
            "content_bytes": loaded.content_bytes,
            "compression": loaded.compression,
            "input_schema": loaded.input_schema,
            "records": len(loaded.records),
        },
        code={"ipfd_version": __version__, "git_commit": commit, "git_dirty": dirty},
        analysis={
            "configuration": normalized_config,
            "configuration_sha256": hashlib.sha256(canonical_config.encode()).hexdigest(),
            "independent_cluster_key": "cluster_id",
            "uncertainty_unit": "whole_independent_seed_cluster",
        },
    )
