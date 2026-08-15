"""Seed-aware counterfactual fidelity analysis over branch decisions.

Branch comparisons are descriptive repeated measurements.  Independent seed
clusters are kept explicit and are the only units resampled for uncertainty.
Nothing in this module certifies an unseen horizon or simulator configuration.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "BootstrapInterval",
    "BranchComparison",
    "BranchErrorProfile",
    "CounterfactualAudit",
    "ErrorDirection",
    "FidelityEnvelope",
    "FidelityGateResult",
    "FidelityGateStatus",
    "FidelityPoint",
    "FrontierStatus",
    "ObservedFidelityFrontier",
    "RestoreProtocolComparison",
    "SeedSummary",
    "compare_restore_protocols",
    "audit_counterfactual_fidelity",
    "evaluate_fidelity_gate",
    "fidelity_envelope",
    "first_untrusted_horizon",
    "seed_cluster_bootstrap",
    "summarize_by_seed",
    "summarize_error_direction",
    "validate_branch_comparisons",
]

GROUP_FIELDS = ("protocol", "continuation", "disturbance", "phase", "predicate")
MANDATORY_GROUP_FIELDS = ("protocol", "continuation", "predicate")


class ErrorDirection(str, Enum):
    """Direction of a restored-branch decision error."""

    NONE = "NONE"
    FALSE_RECOVERABLE = "FALSE_RECOVERABLE"
    FALSE_UNRECOVERABLE = "FALSE_UNRECOVERABLE"


class FrontierStatus(str, Enum):
    """Shape of a descriptive threshold crossing on tested horizons."""

    BRACKETED = "BRACKETED"
    ALL_TESTED_WITHIN_TOLERANCE = "ALL_TESTED_WITHIN_TOLERANCE"
    NO_TESTED_HORIZON_WITHIN_TOLERANCE = "NO_TESTED_HORIZON_WITHIN_TOLERANCE"
    NON_MONOTONIC = "NON_MONOTONIC"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INCOMPLETE_STRATUM_COVERAGE = "INCOMPLETE_STRATUM_COVERAGE"


class FidelityGateStatus(str, Enum):
    """Fail-closed answer for one exact empirical-envelope query."""

    ACCEPT_OBSERVED_ENVELOPE = "ACCEPT_OBSERVED_ENVELOPE"
    REJECT_HIGH_DISAGREEMENT = "REJECT_HIGH_DISAGREEMENT"
    INSUFFICIENT_INDEPENDENT_SEEDS = "INSUFFICIENT_INDEPENDENT_SEEDS"
    OUTSIDE_TESTED_HORIZON = "OUTSIDE_TESTED_HORIZON"
    UNSEEN_STRATUM = "UNSEEN_STRATUM"
    INCOMPLETE_STRATUM_COVERAGE = "INCOMPLETE_STRATUM_COVERAGE"
    NON_MONOTONIC_EVIDENCE = "NON_MONOTONIC_EVIDENCE"


@dataclass(frozen=True)
class BranchComparison:
    """Canonical uninterrupted-versus-restored decision comparison."""

    cluster_id: str
    branch_id: str
    protocol: str
    continuation: str
    disturbance: str
    phase: str
    predicate: str
    horizon: int
    actual_continuation_steps: int
    reference_decision: bool
    candidate_decision: bool
    branch_step: int | None = None
    schedule_id: str | None = None
    schedule_equivalent: bool | None = None
    replicate_id: str = "0"

    @property
    def decision_match(self) -> bool:
        return self.reference_decision == self.candidate_decision

    @property
    def disagreement(self) -> bool:
        return not self.decision_match

    @property
    def error_direction(self) -> ErrorDirection:
        if self.decision_match:
            return ErrorDirection.NONE
        if not self.reference_decision and self.candidate_decision:
            return ErrorDirection.FALSE_RECOVERABLE
        return ErrorDirection.FALSE_UNRECOVERABLE

    @property
    def composite_key(self) -> tuple[str | int, ...]:
        return (
            self.cluster_id,
            self.branch_id,
            self.protocol,
            self.continuation,
            self.disturbance,
            self.phase,
            self.predicate,
            self.horizon,
            self.replicate_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "branch_id": self.branch_id,
            "branch_step": self.branch_step,
            "protocol": self.protocol,
            "continuation": self.continuation,
            "disturbance": self.disturbance,
            "phase": self.phase,
            "predicate": self.predicate,
            "horizon": self.horizon,
            "actual_continuation_steps": self.actual_continuation_steps,
            "reference_decision": self.reference_decision,
            "candidate_decision": self.candidate_decision,
            "decision_match": self.decision_match,
            "error_direction": self.error_direction.value,
            "schedule_id": self.schedule_id,
            "schedule_equivalent": self.schedule_equivalent,
            "replicate_id": self.replicate_id,
        }


@dataclass(frozen=True)
class BranchErrorProfile:
    comparisons: int
    agreements: int
    disagreements: int
    false_recoverable: int
    false_unrecoverable: int

    @property
    def disagreement_rate(self) -> float:
        return self.disagreements / self.comparisons

    def to_dict(self) -> dict[str, int | float]:
        return {
            "comparisons": self.comparisons,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "disagreement_rate": self.disagreement_rate,
            "false_recoverable": self.false_recoverable,
            "false_unrecoverable": self.false_unrecoverable,
            "reference_false_candidate_true": self.false_recoverable,
            "reference_true_candidate_false": self.false_unrecoverable,
        }


@dataclass(frozen=True)
class SeedRate:
    cluster_id: str
    comparisons: int
    disagreements: int
    false_recoverable: int
    false_unrecoverable: int

    @property
    def disagreement_rate(self) -> float:
        return self.disagreements / self.comparisons

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "cluster_id": self.cluster_id,
            "comparisons": self.comparisons,
            "disagreements": self.disagreements,
            "disagreement_rate": self.disagreement_rate,
            "false_recoverable": self.false_recoverable,
            "false_unrecoverable": self.false_unrecoverable,
        }


@dataclass(frozen=True)
class SeedSummary:
    comparisons: int
    disagreements: int
    disagreement_rate: float
    independent_seed_count: int
    seed_mean_disagreement_rate: float
    seed_rate_minimum: float
    seed_rate_maximum: float
    seeds_with_disagreement: int
    per_seed: tuple[SeedRate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparisons": self.comparisons,
            "disagreements": self.disagreements,
            "disagreement_rate": self.disagreement_rate,
            "independent_seed_count": self.independent_seed_count,
            "seed_mean_disagreement_rate": self.seed_mean_disagreement_rate,
            "seed_rate_minimum": self.seed_rate_minimum,
            "seed_rate_maximum": self.seed_rate_maximum,
            "seeds_with_disagreement": self.seeds_with_disagreement,
            "per_seed": [item.to_dict() for item in self.per_seed],
        }


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    random_seed: int
    resampling_unit: str = "independent_seed"
    interpretation: str = "DESCRIPTIVE_SEED_RESAMPLING_INTERVAL"

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "samples": self.samples,
            "random_seed": self.random_seed,
            "resampling_unit": self.resampling_unit,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True)
class FidelityPoint:
    horizon: int
    summary: SeedSummary
    error_profile: BranchErrorProfile
    seed_bootstrap: BootstrapInterval
    support_keys: tuple[tuple[str | int | None, ...], ...]

    @property
    def comparisons(self) -> int:
        return self.summary.comparisons

    @property
    def disagreements(self) -> int:
        return self.summary.disagreements

    @property
    def disagreement_rate(self) -> float:
        return self.summary.disagreement_rate

    @property
    def independent_seed_count(self) -> int:
        return self.summary.independent_seed_count

    @property
    def cluster_ids(self) -> tuple[str, ...]:
        return tuple(item.cluster_id for item in self.summary.per_seed)

    def to_dict(self) -> dict[str, Any]:
        interpretation = "no disagreement observed" if self.disagreements == 0 else "disagreement observed"
        return {
            "horizon": self.horizon,
            **self.summary.to_dict(),
            "false_recoverable": self.error_profile.false_recoverable,
            "false_unrecoverable": self.error_profile.false_unrecoverable,
            "seed_bootstrap": self.seed_bootstrap.to_dict(),
            "branch_support_count": len(self.support_keys),
            "interpretation": interpretation,
            "claim_strength": "DESCRIPTIVE_ONLY",
        }


@dataclass(frozen=True)
class FidelityEnvelope:
    scope_items: tuple[tuple[str, str], ...]
    points: tuple[FidelityPoint, ...]
    error_profile: BranchErrorProfile

    @property
    def scope(self) -> dict[str, str]:
        return dict(self.scope_items)

    @property
    def tested_horizons(self) -> tuple[int, ...]:
        return tuple(point.horizon for point in self.points)

    @property
    def first_observed_disagreement_horizon(self) -> int | None:
        return next((point.horizon for point in self.points if point.disagreements), None)

    @property
    def independent_seed_count(self) -> int:
        return len({cluster for point in self.points for cluster in point.cluster_ids})

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "tested_horizons": list(self.tested_horizons),
            "first_observed_disagreement_horizon": self.first_observed_disagreement_horizon,
            "independent_seed_count": self.independent_seed_count,
            "points": [point.to_dict() for point in self.points],
            "error_profile": self.error_profile.to_dict(),
            "claim_strength": "DESCRIPTIVE_ONLY",
        }


@dataclass(frozen=True)
class ObservedFidelityFrontier:
    status: FrontierStatus
    max_disagreement: float
    last_acceptable_tested_horizon: int | None
    first_rejected_tested_horizon: int | None
    first_observed_disagreement_horizon: int | None
    minimum_independent_seeds: int
    rate_basis: str = "EQUAL_SEED_MEAN_DISAGREEMENT_RATE"
    claim_strength: str = "DESCRIPTIVE_ONLY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "max_disagreement": self.max_disagreement,
            "last_acceptable_tested_horizon": self.last_acceptable_tested_horizon,
            "first_rejected_tested_horizon": self.first_rejected_tested_horizon,
            "first_observed_disagreement_horizon": self.first_observed_disagreement_horizon,
            "minimum_independent_seeds": self.minimum_independent_seeds,
            "rate_basis": self.rate_basis,
            "claim_strength": self.claim_strength,
            "interpolation": "NOT_PERMITTED",
            "extrapolation": "NOT_PERMITTED",
        }


@dataclass(frozen=True)
class FidelityGateResult:
    status: FidelityGateStatus
    accepted: bool
    requested_horizon: int
    scope: dict[str, str] | None
    max_disagreement: float
    minimum_independent_seeds: int
    reason: str
    rate_basis: str = "EQUAL_SEED_MEAN_DISAGREEMENT_RATE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "accepted": self.accepted,
            "requested_horizon": self.requested_horizon,
            "scope": self.scope,
            "max_disagreement": self.max_disagreement,
            "minimum_independent_seeds": self.minimum_independent_seeds,
            "reason": self.reason,
            "rate_basis": self.rate_basis,
            "claim_strength": "DESCRIPTIVE_ONLY",
        }


@dataclass(frozen=True)
class SeedProtocolDelta:
    cluster_id: str
    pairs: int
    protocol_a_disagreement_rate: float
    protocol_b_disagreement_rate: float
    difference_b_minus_a: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "cluster_id": self.cluster_id,
            "pairs": self.pairs,
            "protocol_a_disagreement_rate": self.protocol_a_disagreement_rate,
            "protocol_b_disagreement_rate": self.protocol_b_disagreement_rate,
            "difference_b_minus_a": self.difference_b_minus_a,
        }


@dataclass(frozen=True)
class RestoreProtocolComparison:
    protocol_a: str
    protocol_b: str
    pairs: int
    both_agree: int
    fixed_by_protocol_b: int
    introduced_by_protocol_b: int
    both_disagree: int
    protocol_a_profile: BranchErrorProfile
    protocol_b_profile: BranchErrorProfile
    branch_rate_difference_b_minus_a: float
    relative_disagreement_reduction: float | None
    per_seed: tuple[SeedProtocolDelta, ...]
    seed_mean_difference_b_minus_a: float
    seed_bootstrap: BootstrapInterval
    schedule_identity_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_a": self.protocol_a,
            "protocol_b": self.protocol_b,
            "pairs": self.pairs,
            "paired_outcomes": {
                "both_agree": self.both_agree,
                "fixed_by_protocol_b": self.fixed_by_protocol_b,
                "introduced_by_protocol_b": self.introduced_by_protocol_b,
                "both_disagree": self.both_disagree,
            },
            "protocol_a_profile": self.protocol_a_profile.to_dict(),
            "protocol_b_profile": self.protocol_b_profile.to_dict(),
            "branch_rate_difference_b_minus_a": self.branch_rate_difference_b_minus_a,
            "relative_disagreement_reduction": self.relative_disagreement_reduction,
            "per_seed": [item.to_dict() for item in self.per_seed],
            "seed_mean_difference_b_minus_a": self.seed_mean_difference_b_minus_a,
            "seed_bootstrap": self.seed_bootstrap.to_dict(),
            "schedule_identity_verified": self.schedule_identity_verified,
            "claim_strength": "DESCRIPTIVE_ONLY",
        }


@dataclass(frozen=True)
class CounterfactualAudit:
    """Grouped empirical envelopes under one declared analysis configuration."""

    record_count: int
    independent_seed_count: int
    group_by: tuple[str, ...]
    max_disagreement: float
    minimum_independent_seeds: int
    bootstrap_samples: int
    bootstrap_seed: int
    envelopes: tuple[FidelityEnvelope, ...]
    frontiers: tuple[ObservedFidelityFrontier, ...]

    def to_dict(self) -> dict[str, Any]:
        rendered_envelopes = []
        for envelope, frontier in zip(self.envelopes, self.frontiers, strict=True):
            rendered = envelope.to_dict()
            rendered["frontier"] = frontier.to_dict()
            rendered_envelopes.append(rendered)
        return {
            "record_count": self.record_count,
            "independent_seed_count": self.independent_seed_count,
            "configuration": {
                "group_by": list(self.group_by),
                "max_disagreement": self.max_disagreement,
                "minimum_independent_seeds": self.minimum_independent_seeds,
                "bootstrap_samples": self.bootstrap_samples,
                "bootstrap_seed": self.bootstrap_seed,
            },
            "envelopes": rendered_envelopes,
            "claim_strength": "DESCRIPTIVE_ONLY",
        }


def validate_branch_comparisons(records: Iterable[BranchComparison]) -> tuple[BranchComparison, ...]:
    """Validate canonical records and reject ambiguous repeated comparisons."""

    values = tuple(records)
    if not values:
        raise ValueError("branch comparison dataset is empty")
    seen: set[tuple[str | int, ...]] = set()
    for index, record in enumerate(values, start=1):
        if not isinstance(record, BranchComparison):
            raise TypeError(f"record {index} is not a BranchComparison")
        for field in GROUP_FIELDS:
            if not isinstance(getattr(record, field), str) or not getattr(record, field).strip():
                raise ValueError(f"record {index} has an empty {field}")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (record.cluster_id, record.branch_id, record.replicate_id)
        ):
            raise ValueError(f"record {index} has an empty identity field")
        if isinstance(record.horizon, bool) or not isinstance(record.horizon, int) or record.horizon <= 0:
            raise ValueError(f"record {index} horizon must be a positive integer")
        if (
            isinstance(record.actual_continuation_steps, bool)
            or not isinstance(record.actual_continuation_steps, int)
            or record.actual_continuation_steps <= 0
        ):
            raise ValueError(f"record {index} actual_continuation_steps must be a positive integer")
        if record.actual_continuation_steps != record.horizon:
            raise ValueError(f"record {index} actual_continuation_steps does not equal horizon")
        if record.branch_step is not None and (
            isinstance(record.branch_step, bool) or not isinstance(record.branch_step, int) or record.branch_step < 0
        ):
            raise ValueError(f"record {index} branch_step must be a nonnegative integer")
        if not isinstance(record.reference_decision, bool) or not isinstance(record.candidate_decision, bool):
            raise ValueError(f"record {index} decisions must be booleans")
        if record.schedule_id is not None and (not isinstance(record.schedule_id, str) or not record.schedule_id.strip()):
            raise ValueError(f"record {index} schedule_id must be a nonempty string")
        if record.schedule_equivalent is not None and not isinstance(record.schedule_equivalent, bool):
            raise ValueError(f"record {index} schedule_equivalent must be a boolean or None")
        if record.schedule_equivalent is False:
            raise ValueError(f"record {index} reports a nonequivalent disturbance schedule")
        if record.composite_key in seen:
            raise ValueError(f"duplicate branch comparison key at record {index}: {record.composite_key!r}")
        seen.add(record.composite_key)
    return values


def summarize_error_direction(records: Iterable[BranchComparison]) -> BranchErrorProfile:
    values = tuple(records)
    if not values:
        raise ValueError("cannot summarize an empty branch comparison set")
    false_recoverable = sum(item.error_direction is ErrorDirection.FALSE_RECOVERABLE for item in values)
    false_unrecoverable = sum(item.error_direction is ErrorDirection.FALSE_UNRECOVERABLE for item in values)
    disagreements = false_recoverable + false_unrecoverable
    return BranchErrorProfile(
        comparisons=len(values),
        agreements=len(values) - disagreements,
        disagreements=disagreements,
        false_recoverable=false_recoverable,
        false_unrecoverable=false_unrecoverable,
    )


def summarize_by_seed(records: Iterable[BranchComparison]) -> SeedSummary:
    values = tuple(records)
    if not values:
        raise ValueError("cannot summarize an empty branch comparison set")
    groups: dict[str, list[BranchComparison]] = defaultdict(list)
    for record in values:
        groups[record.cluster_id].append(record)
    per_seed: list[SeedRate] = []
    for cluster_id, group in sorted(groups.items()):
        profile = summarize_error_direction(group)
        per_seed.append(
            SeedRate(
                cluster_id=cluster_id,
                comparisons=profile.comparisons,
                disagreements=profile.disagreements,
                false_recoverable=profile.false_recoverable,
                false_unrecoverable=profile.false_unrecoverable,
            )
        )
    profile = summarize_error_direction(values)
    rates = [item.disagreement_rate for item in per_seed]
    return SeedSummary(
        comparisons=profile.comparisons,
        disagreements=profile.disagreements,
        disagreement_rate=profile.disagreement_rate,
        independent_seed_count=len(per_seed),
        seed_mean_disagreement_rate=sum(rates) / len(rates),
        seed_rate_minimum=min(rates),
        seed_rate_maximum=max(rates),
        seeds_with_disagreement=sum(item.disagreements > 0 for item in per_seed),
        per_seed=tuple(per_seed),
    )


def _validate_bootstrap(*, samples: int, random_seed: int, confidence: float) -> None:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap samples must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("bootstrap random_seed must be an integer")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be between zero and one")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_values(values: Sequence[float], *, samples: int, random_seed: int, confidence: float) -> BootstrapInterval:
    _validate_bootstrap(samples=samples, random_seed=random_seed, confidence=confidence)
    if not values:
        raise ValueError("bootstrap requires at least one independent seed")
    rng = random.Random(random_seed)
    distribution = sorted(sum(rng.choice(values) for _ in range(len(values))) / len(values) for _ in range(samples))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=sum(values) / len(values),
        lower=_percentile(distribution, alpha),
        upper=_percentile(distribution, 1.0 - alpha),
        confidence=confidence,
        samples=samples,
        random_seed=random_seed,
    )


def seed_cluster_bootstrap(
    records: Iterable[BranchComparison],
    *,
    samples: int = 10_000,
    random_seed: int = 20_260_729,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Resample whole independent seed rates with equal seed weight."""

    summary = summarize_by_seed(records)
    return _bootstrap_values(
        [item.disagreement_rate for item in summary.per_seed],
        samples=samples,
        random_seed=random_seed,
        confidence=confidence,
    )


def _support_key(record: BranchComparison) -> tuple[str | int | None, ...]:
    return (
        record.cluster_id,
        record.branch_id,
        record.replicate_id,
        record.protocol,
        record.continuation,
        record.disturbance,
        record.phase,
        record.predicate,
        record.branch_step,
        record.schedule_id,
    )


def fidelity_envelope(
    records: Iterable[BranchComparison],
    *,
    group_by: Sequence[str] = ("protocol", "continuation", "disturbance", "predicate"),
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_729,
    confidence: float = 0.95,
) -> tuple[FidelityEnvelope, ...]:
    """Build descriptive disagreement curves for exact observed strata."""

    values = validate_branch_comparisons(records)
    fields = tuple(group_by)
    if not fields:
        raise ValueError("group_by must contain at least one scope field")
    if len(fields) != len(set(fields)):
        raise ValueError("group_by fields must be unique")
    unknown = sorted(set(fields) - set(GROUP_FIELDS))
    if unknown:
        raise ValueError(f"unsupported group_by fields: {', '.join(unknown)}")
    missing_mandatory = [field for field in MANDATORY_GROUP_FIELDS if field not in fields]
    if missing_mandatory:
        raise ValueError("group_by must preserve protocol, continuation, and predicate; missing " + ", ".join(missing_mandatory))
    groups: dict[tuple[str, ...], list[BranchComparison]] = defaultdict(list)
    for record in values:
        groups[tuple(str(getattr(record, field)) for field in fields)].append(record)

    envelopes: list[FidelityEnvelope] = []
    for group_key, group in sorted(groups.items()):
        by_horizon: dict[int, list[BranchComparison]] = defaultdict(list)
        for record in group:
            by_horizon[record.horizon].append(record)
        points: list[FidelityPoint] = []
        for horizon, horizon_group in sorted(by_horizon.items()):
            points.append(
                FidelityPoint(
                    horizon=horizon,
                    summary=summarize_by_seed(horizon_group),
                    error_profile=summarize_error_direction(horizon_group),
                    seed_bootstrap=seed_cluster_bootstrap(
                        horizon_group,
                        samples=bootstrap_samples,
                        random_seed=bootstrap_seed,
                        confidence=confidence,
                    ),
                    support_keys=tuple(
                        sorted(
                            (_support_key(record) for record in horizon_group),
                            key=lambda item: tuple(map(str, item)),
                        )
                    ),
                )
            )
        envelopes.append(
            FidelityEnvelope(
                scope_items=tuple(zip(fields, group_key, strict=True)),
                points=tuple(points),
                error_profile=summarize_error_direction(group),
            )
        )
    return tuple(envelopes)


def audit_counterfactual_fidelity(
    records: Iterable[BranchComparison],
    *,
    group_by: Sequence[str] = ("protocol", "continuation", "disturbance", "predicate"),
    max_disagreement: float = 0.05,
    minimum_independent_seeds: int = 5,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_729,
    confidence: float = 0.95,
) -> CounterfactualAudit:
    """Run one grouped, seed-aware descriptive fidelity audit."""

    _validate_threshold(max_disagreement, minimum_independent_seeds)
    values = validate_branch_comparisons(records)
    envelopes = fidelity_envelope(
        values,
        group_by=group_by,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    frontiers = tuple(
        first_untrusted_horizon(
            envelope,
            max_disagreement=max_disagreement,
            minimum_independent_seeds=minimum_independent_seeds,
        )
        for envelope in envelopes
    )
    return CounterfactualAudit(
        record_count=len(values),
        independent_seed_count=len({record.cluster_id for record in values}),
        group_by=tuple(group_by),
        max_disagreement=max_disagreement,
        minimum_independent_seeds=minimum_independent_seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        envelopes=envelopes,
        frontiers=frontiers,
    )


def _validate_threshold(max_disagreement: float, minimum_independent_seeds: int) -> None:
    if isinstance(max_disagreement, bool) or not math.isfinite(max_disagreement) or not 0.0 <= max_disagreement <= 1.0:
        raise ValueError("max_disagreement must be finite and in [0, 1]")
    if (
        isinstance(minimum_independent_seeds, bool)
        or not isinstance(minimum_independent_seeds, int)
        or minimum_independent_seeds <= 0
    ):
        raise ValueError("minimum_independent_seeds must be a positive integer")


def first_untrusted_horizon(
    envelope: FidelityEnvelope,
    *,
    max_disagreement: float,
    minimum_independent_seeds: int = 1,
) -> ObservedFidelityFrontier:
    """Return a discrete observed frontier without interpolation or extrapolation."""

    _validate_threshold(max_disagreement, minimum_independent_seeds)
    if not envelope.points:
        raise ValueError("fidelity envelope contains no tested horizons")
    adequate = [point.independent_seed_count >= minimum_independent_seeds for point in envelope.points]
    accepted = [point.summary.seed_mean_disagreement_rate <= max_disagreement for point in envelope.points]
    stable_support = len({point.support_keys for point in envelope.points}) == 1
    first_observed = envelope.first_observed_disagreement_horizon
    if not stable_support:
        status = FrontierStatus.INCOMPLETE_STRATUM_COVERAGE
        last_acceptable = None
        first_rejected = None
    elif not all(adequate):
        status = FrontierStatus.INSUFFICIENT_EVIDENCE
        last_acceptable = None
        first_rejected = None
    else:
        first_false = next((index for index, value in enumerate(accepted) if not value), None)
        non_monotonic = first_false is not None and any(accepted[first_false + 1 :])
        if non_monotonic:
            status = FrontierStatus.NON_MONOTONIC
        elif all(accepted):
            status = FrontierStatus.ALL_TESTED_WITHIN_TOLERANCE
        elif not any(accepted):
            status = FrontierStatus.NO_TESTED_HORIZON_WITHIN_TOLERANCE
        else:
            status = FrontierStatus.BRACKETED
        if non_monotonic:
            last_acceptable = max(
                point.horizon for point, is_accepted in zip(envelope.points, accepted, strict=True) if is_accepted
            )
        elif first_false is not None and first_false > 0:
            last_acceptable = envelope.points[first_false - 1].horizon
        elif first_false is None:
            last_acceptable = envelope.points[-1].horizon
        else:
            last_acceptable = None
        first_rejected = envelope.points[first_false].horizon if first_false is not None else None
    return ObservedFidelityFrontier(
        status=status,
        max_disagreement=max_disagreement,
        last_acceptable_tested_horizon=last_acceptable,
        first_rejected_tested_horizon=first_rejected,
        first_observed_disagreement_horizon=first_observed,
        minimum_independent_seeds=minimum_independent_seeds,
    )


def evaluate_fidelity_gate(
    envelope: FidelityEnvelope | None,
    *,
    horizon: int,
    max_disagreement: float,
    minimum_independent_seeds: int = 5,
) -> FidelityGateResult:
    """Fail closed when an exact stratum or tested horizon lacks adequate evidence."""

    _validate_threshold(max_disagreement, minimum_independent_seeds)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("requested horizon must be a positive integer")

    def result(status: FidelityGateStatus, reason: str) -> FidelityGateResult:
        return FidelityGateResult(
            status=status,
            accepted=status is FidelityGateStatus.ACCEPT_OBSERVED_ENVELOPE,
            requested_horizon=horizon,
            scope=envelope.scope if envelope is not None else None,
            max_disagreement=max_disagreement,
            minimum_independent_seeds=minimum_independent_seeds,
            reason=reason,
        )

    if envelope is None:
        return result(FidelityGateStatus.UNSEEN_STRATUM, "no evidence matches the requested stratum")
    point_by_horizon = {point.horizon: point for point in envelope.points}
    if horizon not in point_by_horizon:
        return result(
            FidelityGateStatus.OUTSIDE_TESTED_HORIZON,
            "the requested horizon was not tested; interpolation and extrapolation are not permitted",
        )
    prefix = [point for point in envelope.points if point.horizon <= horizon]
    if any(point.independent_seed_count < minimum_independent_seeds for point in prefix):
        return result(
            FidelityGateStatus.INSUFFICIENT_INDEPENDENT_SEEDS,
            "at least one tested horizon through the request has too few independent seed groups",
        )
    support_sets = {point.support_keys for point in prefix}
    if len(support_sets) != 1:
        return result(
            FidelityGateStatus.INCOMPLETE_STRATUM_COVERAGE,
            "branch and stratum support changes across tested horizons through the request",
        )
    passed = [point.summary.seed_mean_disagreement_rate <= max_disagreement for point in prefix]
    first_failure = next((index for index, value in enumerate(passed) if not value), None)
    if first_failure is not None and any(passed[first_failure + 1 :]):
        return result(
            FidelityGateStatus.NON_MONOTONIC_EVIDENCE,
            "the empirical tolerance result reverses across tested horizons",
        )
    if not all(passed):
        return result(
            FidelityGateStatus.REJECT_HIGH_DISAGREEMENT,
            "observed disagreement exceeded the tolerance at or before the requested horizon",
        )
    return result(
        FidelityGateStatus.ACCEPT_OBSERVED_ENVELOPE,
        "the supplied sampled scope met the empirical rule through the requested tested horizon",
    )


def compare_restore_protocols(
    records: Iterable[BranchComparison],
    *,
    protocol_a: str,
    protocol_b: str,
    samples: int = 10_000,
    random_seed: int = 20_260_729,
    confidence: float = 0.95,
) -> RestoreProtocolComparison:
    """Compare two restore protocols using exact paired branch decisions."""

    if not protocol_a or not protocol_b or protocol_a == protocol_b:
        raise ValueError("protocol_a and protocol_b must be distinct nonempty names")
    values = validate_branch_comparisons(records)
    selected = [record for record in values if record.protocol in {protocol_a, protocol_b}]
    if not selected:
        raise ValueError("requested protocols are absent from the evidence")
    pairs: dict[tuple[str | int, ...], dict[str, BranchComparison]] = defaultdict(dict)
    for record in selected:
        pair_key = (
            record.cluster_id,
            record.branch_id,
            record.continuation,
            record.disturbance,
            record.phase,
            record.predicate,
            record.horizon,
            record.replicate_id,
        )
        pairs[pair_key][record.protocol] = record
    paired_values: list[tuple[BranchComparison, BranchComparison]] = []
    for identity, members in sorted(pairs.items(), key=lambda item: tuple(map(str, item[0]))):
        if set(members) != {protocol_a, protocol_b}:
            raise ValueError(f"incomplete restore-protocol pair: {identity!r}")
        item_a, item_b = members[protocol_a], members[protocol_b]
        if item_a.reference_decision != item_b.reference_decision:
            raise ValueError(f"paired reference decisions differ: {identity!r}")
        if item_a.actual_continuation_steps != item_b.actual_continuation_steps:
            raise ValueError(f"paired continuation lengths differ: {identity!r}")
        if item_a.branch_step != item_b.branch_step:
            raise ValueError(f"paired branch steps differ: {identity!r}")
        if item_a.schedule_equivalent is not True or item_b.schedule_equivalent is not True:
            raise ValueError(f"paired disturbance schedule equivalence is unavailable: {identity!r}")
        if item_a.schedule_id is None or item_b.schedule_id is None:
            raise ValueError(f"paired disturbance schedule identity is unavailable: {identity!r}")
        if item_a.schedule_id != item_b.schedule_id:
            raise ValueError(f"paired disturbance schedules differ: {identity!r}")
        paired_values.append((item_a, item_b))
    if not paired_values:
        raise ValueError("no complete restore-protocol pairs were found")

    a_records = [pair[0] for pair in paired_values]
    b_records = [pair[1] for pair in paired_values]
    a_profile = summarize_error_direction(a_records)
    b_profile = summarize_error_direction(b_records)
    both_agree = sum(not a.disagreement and not b.disagreement for a, b in paired_values)
    fixed = sum(a.disagreement and not b.disagreement for a, b in paired_values)
    introduced = sum(not a.disagreement and b.disagreement for a, b in paired_values)
    both_disagree = sum(a.disagreement and b.disagreement for a, b in paired_values)
    by_seed: dict[str, list[tuple[BranchComparison, BranchComparison]]] = defaultdict(list)
    for paired in paired_values:
        by_seed[paired[0].cluster_id].append(paired)
    seed_deltas: list[SeedProtocolDelta] = []
    for cluster_id, group in sorted(by_seed.items()):
        a_rate = sum(a.disagreement for a, _ in group) / len(group)
        b_rate = sum(b.disagreement for _, b in group) / len(group)
        seed_deltas.append(
            SeedProtocolDelta(
                cluster_id=cluster_id,
                pairs=len(group),
                protocol_a_disagreement_rate=a_rate,
                protocol_b_disagreement_rate=b_rate,
                difference_b_minus_a=b_rate - a_rate,
            )
        )
    deltas = [item.difference_b_minus_a for item in seed_deltas]
    bootstrap = _bootstrap_values(deltas, samples=samples, random_seed=random_seed, confidence=confidence)
    relative_reduction = (
        (a_profile.disagreement_rate - b_profile.disagreement_rate) / a_profile.disagreement_rate
        if a_profile.disagreement_rate > 0.0
        else None
    )
    return RestoreProtocolComparison(
        protocol_a=protocol_a,
        protocol_b=protocol_b,
        pairs=len(paired_values),
        both_agree=both_agree,
        fixed_by_protocol_b=fixed,
        introduced_by_protocol_b=introduced,
        both_disagree=both_disagree,
        protocol_a_profile=a_profile,
        protocol_b_profile=b_profile,
        branch_rate_difference_b_minus_a=b_profile.disagreement_rate - a_profile.disagreement_rate,
        relative_disagreement_reduction=relative_reduction,
        per_seed=tuple(seed_deltas),
        seed_mean_difference_b_minus_a=sum(deltas) / len(deltas),
        seed_bootstrap=bootstrap,
        schedule_identity_verified=True,
    )
