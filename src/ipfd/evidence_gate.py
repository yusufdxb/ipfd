"""Fail-closed release evidence validation for IPFD headline claims.

The gate validates artifact provenance and derives its metrics from per-run data.
Aggregate-only JSON cannot pass because a percentage without the underlying runs
is not auditable evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_KEYS = frozenset({"isaaclab", "isaacsim", "torch"})
_ACTIONABILITY_RELATIONS = frozenset(
    {
        "no_alarm",
        "pre_disturbance",
        "definitely_actionable",
        "ambiguous_within_ponr_interval",
        "too_late",
        "no_ponr",
    }
)
_REQUIRED_ACTIONABILITY_RELATIONS = frozenset(
    {
        "no_alarm",
        "pre_disturbance",
        "definitely_actionable",
        "ambiguous_within_ponr_interval",
        "too_late",
    }
)


@dataclass(frozen=True)
class EvidenceCriteria:
    min_success_rate: float = 0.80
    min_seeds: int = 5
    min_actionability_accuracy: float = 1.0
    min_actionability_cases: int = 20
    min_competence_episodes: int = 32
    min_competence_sustain_steps: int = 10
    min_probe_repeats: int = 3
    min_probe_false_fraction: float = 0.8
    max_reset_boundary_delta_m: float = 1e-6
    max_ponr_error_steps: int = 10
    required_git_commit: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "min_success_rate",
            "min_actionability_accuracy",
            "min_probe_false_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.min_probe_false_fraction < 0.5:
            raise ValueError("min_probe_false_fraction must be in [0.5, 1]")
        for name in (
            "min_seeds",
            "min_actionability_cases",
            "min_competence_episodes",
            "min_competence_sustain_steps",
            "min_probe_repeats",
            "max_ponr_error_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        delta = self.max_reset_boundary_delta_m
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float, np.integer, np.floating))
            or not np.isfinite(delta)
            or delta < 0
        ):
            raise ValueError("max_reset_boundary_delta_m must be finite and >= 0")
        if (
            self.required_git_commit is not None
            and (
                not isinstance(self.required_git_commit, str)
                or _GIT_COMMIT.fullmatch(self.required_git_commit) is None
            )
        ):
            raise ValueError("required_git_commit must be a lowercase 40-character Git commit")


@dataclass(frozen=True)
class EvidenceGateResult:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]
    missing: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"missing": list(self.missing), "errors": list(self.errors)}


def _finite_number(
    data: Mapping[str, Any],
    key: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    value = data.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        errors.append(f"{key} must be a finite number")
        return None
    result = float(value)
    if minimum is not None and result < minimum:
        errors.append(f"{key} must be >= {minimum}")
        return None
    if maximum is not None and result > maximum:
        errors.append(f"{key} must be <= {maximum}")
        return None
    return result


def _positive_int(data: Mapping[str, Any], key: str, errors: list[str]) -> int | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        errors.append(f"{key} must be an integer >= 1")
        return None
    return int(value)


def _sha256(data: Mapping[str, Any], key: str, errors: list[str]) -> str | None:
    value = data.get(key)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        errors.append(f"{key} must be a lowercase SHA-256 digest")
        return None
    return value


def _runtime_fingerprint(
    data: Mapping[str, Any],
    *,
    prefix: str,
    errors: list[str],
) -> dict[str, str] | None:
    runtime = data.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append(f"{prefix}.runtime must be an object")
        return None
    normalized: dict[str, str] = {}
    for key in _RUNTIME_KEYS:
        value = runtime.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{prefix}.runtime.{key} must be a non-empty string")
        else:
            normalized[key] = value
    return normalized if len(normalized) == len(_RUNTIME_KEYS) else None


def _software_fingerprint(
    data: Mapping[str, Any],
    *,
    prefix: str,
    errors: list[str],
) -> dict[str, str | bool] | None:
    software = data.get("software")
    if not isinstance(software, Mapping):
        errors.append(f"{prefix}.software must be an object")
        return None
    version = software.get("ipfd_version")
    commit = software.get("git_commit")
    dirty = software.get("git_dirty")
    valid = True
    if not isinstance(version, str) or not version.strip():
        errors.append(f"{prefix}.software.ipfd_version must be a non-empty string")
        valid = False
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        errors.append(f"{prefix}.software.git_commit must be a lowercase 40-character Git commit")
        valid = False
    if dirty is not False:
        errors.append(f"{prefix}.software.git_dirty must be false")
        valid = False
    if not valid:
        return None
    return {
        "ipfd_version": str(version),
        "git_commit": str(commit),
        "git_dirty": False,
    }


def _require_header(
    artifact: Mapping[str, Any],
    *,
    name: str,
    schema: str,
    errors: list[str],
) -> bool:
    valid = True
    if artifact.get("schema") != schema:
        errors.append(f"{name}.schema must equal {schema!r}")
        valid = False
    if artifact.get("status") != "complete":
        errors.append(f"{name}.status must equal 'complete'")
        valid = False
    if "_error" in artifact:
        errors.append(f"{name} could not be read: {artifact['_error']}")
        valid = False
    return valid


def _validate_competence(
    artifact: Mapping[str, Any],
    criteria: EvidenceCriteria,
    errors: list[str],
    metrics: dict[str, float | int],
) -> tuple[
    bool,
    str | None,
    str | None,
    dict[str, str] | None,
    dict[str, str | bool] | None,
]:
    header_ok = _require_header(
        artifact,
        name="competence",
        schema="ipfd.competence.v1",
        errors=errors,
    )
    rate = _finite_number(
        artifact,
        "success_rate",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    episodes = _positive_int(artifact, "n_episodes", errors)
    sustain_steps = _positive_int(artifact, "sustain_steps", errors)
    success_definition_ok = (
        artifact.get("success_definition") == "sustained_final_lift_v1"
    )
    if not success_definition_ok:
        errors.append(
            "competence.success_definition must equal 'sustained_final_lift_v1'"
        )
    checkpoint = _sha256(artifact, "checkpoint_sha256", errors)
    runtime = _runtime_fingerprint(artifact, prefix="competence", errors=errors)
    software = _software_fingerprint(artifact, prefix="competence", errors=errors)
    commit_matches = (
        criteria.required_git_commit is None
        or (
            software is not None
            and software["git_commit"] == criteria.required_git_commit
        )
    )
    if not commit_matches:
        errors.append("competence.software.git_commit must match the required source commit")
    task = artifact.get("task")
    if not isinstance(task, str) or not task.strip():
        errors.append("competence.task must be a non-empty string")
        task = None
    if rate is not None:
        metrics["success_rate"] = rate
    if episodes is not None:
        metrics["competence_episodes"] = episodes
    if sustain_steps is not None:
        metrics["competence_sustain_steps"] = sustain_steps
    passed = (
        header_ok
        and rate is not None
        and rate >= criteria.min_success_rate
        and episodes is not None
        and episodes >= criteria.min_competence_episodes
        and sustain_steps is not None
        and sustain_steps >= criteria.min_competence_sustain_steps
        and success_definition_ok
        and checkpoint is not None
        and task is not None
        and runtime is not None
        and software is not None
        and commit_matches
    )
    return passed, checkpoint, task, runtime, software


def _derive_ponr_from_raw(
    value: Any,
    repeats: int,
    min_false_fraction: float,
    probe_stride: int,
) -> tuple[bool, int | None]:
    if not isinstance(value, Mapping) or not value:
        return False, None
    candidates: dict[int, bool] = {}
    for key, samples in value.items():
        if isinstance(key, bool):
            return False, None
        try:
            step = int(key)
        except (TypeError, ValueError):
            return False, None
        if step < 0 or step in candidates:
            return False, None
        if not isinstance(samples, list) or len(samples) != repeats:
            return False, None
        if any(not isinstance(sample, bool) for sample in samples):
            return False, None
        false_fraction = sum(not sample for sample in samples) / repeats
        candidates[step] = false_fraction < min_false_fraction
    ordered = sorted(candidates.items())
    steps = [step for step, _recovered in ordered]
    if steps[0] != 0 or any(
        later - earlier > probe_stride
        for earlier, later in zip(steps, steps[1:], strict=False)
    ):
        return False, None
    if ordered[-1][1]:
        return True, None
    last_recoverable = max((step for step, recovered in ordered if recovered), default=-1)
    derived = next(step for step, _recovered in ordered if step > last_recoverable)
    return True, derived


def _validate_multiseed(
    artifact: Mapping[str, Any],
    criteria: EvidenceCriteria,
    checkpoint: str | None,
    task: str | None,
    runtime: Mapping[str, str] | None,
    software: Mapping[str, str | bool] | None,
    errors: list[str],
    metrics: dict[str, float | int],
) -> bool:
    header_ok = _require_header(
        artifact,
        name="multiseed",
        schema="ipfd.multiseed.v1",
        errors=errors,
    )
    runs = artifact.get("runs")
    if not isinstance(runs, list) or not runs:
        errors.append("multiseed.runs must be a non-empty list")
        return False

    outcomes_by_seed: dict[int, set[str]] = {}
    rollout_digests: set[str] = set()
    run_keys: set[tuple[int, str]] = set()
    valid_runs = 0
    for index, run in enumerate(runs):
        prefix = f"multiseed.runs[{index}]"
        if not isinstance(run, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        run_errors: list[str] = []
        if run.get("status") != "complete":
            run_errors.append("status must equal 'complete'")
        seed = run.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            run_errors.append("seed must be an integer")
            seed = None
        expected = run.get("expected_outcome")
        if expected not in {"ponr", "no_ponr"}:
            run_errors.append("expected_outcome must be 'ponr' or 'no_ponr'")
        t_ponr = run.get("t_ponr")
        disturbance_onset = run.get("disturbance_onset")
        if (
            isinstance(disturbance_onset, bool)
            or not isinstance(disturbance_onset, (int, np.integer))
            or disturbance_onset < 0
        ):
            run_errors.append("disturbance_onset must be a non-negative integer")
        if expected == "ponr" and (
            isinstance(t_ponr, bool) or not isinstance(t_ponr, (int, np.integer)) or t_ponr < 0
        ):
            run_errors.append("t_ponr must be a non-negative integer for expected_outcome='ponr'")
        elif (
            expected == "ponr"
            and isinstance(t_ponr, (int, np.integer))
            and not isinstance(t_ponr, bool)
            and isinstance(disturbance_onset, (int, np.integer))
            and not isinstance(disturbance_onset, bool)
            and abs(int(t_ponr) - int(disturbance_onset)) > criteria.max_ponr_error_steps
        ):
            run_errors.append(
                f"t_ponr must be within {criteria.max_ponr_error_steps} steps of disturbance_onset"
            )
        if expected == "no_ponr" and t_ponr is not None:
            run_errors.append("t_ponr must be null for expected_outcome='no_ponr'")
        if run.get("fault_injection_triggered") is not True:
            run_errors.append("fault_injection_triggered must be true")
        if checkpoint is None or run.get("checkpoint_sha256") != checkpoint:
            run_errors.append("checkpoint_sha256 must match the competence artifact")
        rollout_digest = _sha256(run, "rollout_sha256", run_errors)
        if rollout_digest in rollout_digests:
            run_errors.append("rollout_sha256 duplicates another run")
        if task is None or run.get("task") != task:
            run_errors.append("task must match the competence artifact")
        run_runtime = _runtime_fingerprint(run, prefix=prefix, errors=run_errors)
        if runtime is None or run_runtime != runtime:
            run_errors.append("runtime must match the competence artifact")
        run_software = _software_fingerprint(run, prefix=prefix, errors=run_errors)
        if software is None or run_software != software:
            run_errors.append("software must match the competence artifact")
        predicate = run.get("recovery_predicate")
        if not isinstance(predicate, str) or not predicate or predicate == "height_only_legacy":
            run_errors.append("recovery_predicate must identify a non-legacy physical predicate")
        repeats = run.get("probe_repeats")
        if (
            isinstance(repeats, bool)
            or not isinstance(repeats, (int, np.integer))
            or repeats < criteria.min_probe_repeats
        ):
            run_errors.append(f"probe_repeats must be >= {criteria.min_probe_repeats}")
        probe_stride = run.get("probe_stride")
        if (
            isinstance(probe_stride, bool)
            or not isinstance(probe_stride, (int, np.integer))
            or not 1 <= probe_stride <= criteria.max_ponr_error_steps
        ):
            run_errors.append(
                f"probe_stride must be in [1, {criteria.max_ponr_error_steps}]"
            )
        probe_budget = run.get("probe_budget")
        if (
            isinstance(probe_budget, bool)
            or not isinstance(probe_budget, (int, np.integer))
            or probe_budget < 1
        ):
            run_errors.append("probe_budget must be an integer >= 1")
        min_false_fraction = run.get("probe_min_false_fraction")
        if (
            isinstance(min_false_fraction, bool)
            or not isinstance(
                min_false_fraction, (int, float, np.integer, np.floating)
            )
            or not np.isfinite(min_false_fraction)
            or not criteria.min_probe_false_fraction <= min_false_fraction <= 1.0
        ):
            run_errors.append(
                "probe_min_false_fraction must be finite and at least "
                f"{criteria.min_probe_false_fraction}"
            )
        integrity = run.get("reset_boundary_primary_pose_delta_m")
        if (
            isinstance(integrity, bool)
            or not isinstance(integrity, (int, float, np.integer, np.floating))
            or not np.isfinite(integrity)
            or integrity < 0
            or integrity > criteria.max_reset_boundary_delta_m
        ):
            run_errors.append(
                "reset_boundary_primary_pose_delta_m exceeds the configured bound or is invalid"
            )
        if (
            isinstance(repeats, (int, np.integer))
            and not isinstance(repeats, bool)
            and isinstance(probe_stride, (int, np.integer))
            and not isinstance(probe_stride, bool)
            and probe_stride >= 1
            and isinstance(min_false_fraction, (int, float, np.integer, np.floating))
            and not isinstance(min_false_fraction, bool)
            and np.isfinite(min_false_fraction)
        ):
            raw_valid, derived_ponr = _derive_ponr_from_raw(
                run.get("raw_probe_verdicts"),
                int(repeats),
                float(min_false_fraction),
                int(probe_stride),
            )
            if not raw_valid:
                run_errors.append(
                    "raw_probe_verdicts must map unique non-negative checkpoints "
                    "to one boolean list per repeat, start at zero, and respect probe_stride"
                )
            elif t_ponr != derived_ponr:
                run_errors.append(
                    f"t_ponr {t_ponr!r} does not match raw-probe-derived "
                    f"value {derived_ponr!r}"
                )
        if (
            isinstance(seed, (int, np.integer))
            and not isinstance(seed, bool)
            and expected in {"ponr", "no_ponr"}
            and (int(seed), str(expected)) in run_keys
        ):
            run_errors.append("duplicates another run for the same seed and expected_outcome")
        if run_errors:
            errors.extend(f"{prefix}.{message}" for message in run_errors)
            continue
        assert isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)
        seed_int = int(seed)
        run_keys.add((seed_int, str(expected)))
        outcomes_by_seed.setdefault(seed_int, set()).add(str(expected))
        assert rollout_digest is not None
        rollout_digests.add(rollout_digest)
        valid_runs += 1

    complete_seeds = {
        seed for seed, outcomes in outcomes_by_seed.items() if outcomes == {"ponr", "no_ponr"}
    }
    metrics["multiseed_runs"] = valid_runs
    metrics["n_seeds"] = len(complete_seeds)
    return (
        header_ok
        and len(complete_seeds) >= criteria.min_seeds
        and valid_runs == len(runs)
    )


def _validate_actionability(
    artifact: Mapping[str, Any],
    criteria: EvidenceCriteria,
    checkpoint: str | None,
    task: str | None,
    runtime: Mapping[str, str] | None,
    software: Mapping[str, str | bool] | None,
    errors: list[str],
    metrics: dict[str, float | int],
) -> bool:
    header_ok = _require_header(
        artifact,
        name="actionability",
        schema="ipfd.actionability.v1",
        errors=errors,
    )
    if artifact.get("source") != "isaac_lab":
        errors.append("actionability.source must equal 'isaac_lab'; synthetic contract cases are not evidence")
        header_ok = False
    if checkpoint is None or artifact.get("checkpoint_sha256") != checkpoint:
        errors.append("actionability.checkpoint_sha256 must match the competence artifact")
        header_ok = False
    if task is None or artifact.get("task") != task:
        errors.append("actionability.task must match the competence artifact")
        header_ok = False
    actionability_runtime = _runtime_fingerprint(
        artifact, prefix="actionability", errors=errors
    )
    if runtime is None or actionability_runtime != runtime:
        errors.append("actionability.runtime must match the competence artifact")
        header_ok = False
    actionability_software = _software_fingerprint(
        artifact, prefix="actionability", errors=errors
    )
    if software is None or actionability_software != software:
        errors.append("actionability.software must match the competence artifact")
        header_ok = False
    cases = artifact.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("actionability.cases must be a non-empty list")
        return False

    case_ids: set[str] = set()
    rollout_digests: set[str] = set()
    expected_relations: set[str] = set()
    correct = 0
    valid_cases = 0
    for index, case in enumerate(cases):
        prefix = f"actionability.cases[{index}]"
        if not isinstance(case, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        case_id = case.get("case_id")
        expected = case.get("expected_relation")
        actual = case.get("alarm_relation")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"{prefix}.case_id must be a non-empty string")
            continue
        if case_id in case_ids:
            errors.append(f"{prefix}.case_id is duplicated: {case_id!r}")
            continue
        rollout_digest = _sha256(case, "rollout_sha256", errors)
        if rollout_digest is None:
            continue
        if rollout_digest in rollout_digests:
            errors.append(f"{prefix}.rollout_sha256 duplicates another case")
            continue
        seed = case.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            errors.append(f"{prefix}.seed must be an integer")
            continue
        if expected not in _ACTIONABILITY_RELATIONS:
            errors.append(f"{prefix}.expected_relation is not a supported relation")
            continue
        if actual not in _ACTIONABILITY_RELATIONS:
            errors.append(f"{prefix}.alarm_relation is not a supported relation")
            continue
        case_ids.add(case_id)
        rollout_digests.add(rollout_digest)
        expected_relations.add(expected)
        valid_cases += 1
        correct += expected == actual

    accuracy = correct / valid_cases if valid_cases else 0.0
    metrics["actionability_cases"] = valid_cases
    metrics["actionability_accuracy"] = accuracy
    missing_relations = _REQUIRED_ACTIONABILITY_RELATIONS - expected_relations
    if missing_relations:
        errors.append(
            "actionability.cases must cover required expected relations: "
            + ", ".join(sorted(missing_relations))
        )
    return (
        header_ok
        and valid_cases == len(cases)
        and valid_cases >= criteria.min_actionability_cases
        and accuracy >= criteria.min_actionability_accuracy
        and not missing_relations
    )


def evaluate_evidence(
    competence: Mapping[str, Any] | None,
    multiseed: Mapping[str, Any] | None,
    actionability: Mapping[str, Any] | None,
    criteria: EvidenceCriteria | None = None,
) -> EvidenceGateResult:
    """Validate all evidence bundles and fail closed on any malformed field."""
    criteria = criteria or EvidenceCriteria()
    missing = [
        name
        for name, artifact in (
            ("competence", competence),
            ("multiseed", multiseed),
            ("actionability", actionability),
        )
        if artifact is None
    ]
    errors: list[str] = []
    metrics: dict[str, float | int] = {}
    checks: dict[str, bool] = {}

    checkpoint: str | None = None
    task: str | None = None
    runtime: dict[str, str] | None = None
    software: dict[str, str | bool] | None = None
    if competence is not None and not isinstance(competence, Mapping):
        errors.append("competence artifact must be a JSON object")
        checks["competence"] = False
    elif competence is not None:
        (
            checks["competence"],
            checkpoint,
            task,
            runtime,
            software,
        ) = _validate_competence(competence, criteria, errors, metrics)
    if multiseed is not None and not isinstance(multiseed, Mapping):
        errors.append("multiseed artifact must be a JSON object")
        checks["multiseed"] = False
    elif multiseed is not None:
        checks["multiseed"] = _validate_multiseed(
            multiseed,
            criteria,
            checkpoint,
            task,
            runtime,
            software,
            errors,
            metrics,
        )
    if actionability is not None and not isinstance(actionability, Mapping):
        errors.append("actionability artifact must be a JSON object")
        checks["actionability"] = False
    elif actionability is not None:
        checks["actionability"] = _validate_actionability(
            actionability,
            criteria,
            checkpoint,
            task,
            runtime,
            software,
            errors,
            metrics,
        )

    passed = (
        not missing
        and not errors
        and len(checks) == 3
        and all(checks.values())
    )
    return EvidenceGateResult(passed, checks, metrics, tuple(missing), tuple(errors))
