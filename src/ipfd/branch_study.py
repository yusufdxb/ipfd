"""Pure experiment contracts for the corrected branch-validity study.

The live Isaac runner imports these definitions, while the unit tests exercise
the experimental semantics without requiring a GPU runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

PHASES = (
    "free_space_pre_manipulation",
    "approach",
    "first_contact",
    "stable_grasp",
    "initial_lift",
    "sustained_lift",
    "disturbance_onset",
    "post_disturbance_recovery",
)
HORIZONS = (1, 3, 5, 10, 30, 90)
PROTOCOL_A = "scene_plus_basic_manager_state"
PROTOCOL_B = "expanded_runtime_state"
PROTOCOLS = (PROTOCOL_A, PROTOCOL_B)


@dataclass(frozen=True)
class SeedBundle:
    """Separated provenance fields for one independent seed group."""

    base_seed: int
    disturbance: str
    simulator_seed: int
    environment_seed: int
    policy_seed: int
    disturbance_seed: int
    branch_selection_seed: int

    @classmethod
    def derive(cls, base_seed: int, disturbance: str) -> SeedBundle:
        disturbance_index = {
            "object_teleport": 1,
            "gripper_open_interruption": 2,
        }.get(disturbance)
        if disturbance_index is None:
            raise ValueError(f"unsupported disturbance: {disturbance}")
        return cls(
            base_seed=base_seed,
            disturbance=disturbance,
            simulator_seed=base_seed,
            environment_seed=10_000 + 100 * base_seed + disturbance_index,
            policy_seed=20_000 + 100 * base_seed + disturbance_index,
            disturbance_seed=30_000 + 100 * base_seed + disturbance_index,
            branch_selection_seed=40_000 + 100 * base_seed + disturbance_index,
        )


def validate_seed_bundles(bundles: list[SeedBundle], *, expected_base_seeds: int = 5) -> None:
    """Reject repeated provenance fields or an undersized independent cohort."""
    if len({bundle.base_seed for bundle in bundles}) < expected_base_seeds:
        raise ValueError("fewer than the required independent base seeds")
    for field in (
        "environment_seed",
        "policy_seed",
        "disturbance_seed",
        "branch_selection_seed",
    ):
        values = [getattr(bundle, field) for bundle in bundles]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {field}")


@dataclass(frozen=True)
class DisturbanceSchedule:
    """A fully specified disturbance schedule."""

    kind: str
    start_step: int
    duration_steps: int
    magnitude: tuple[float, ...]
    target: str
    random_values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"object_teleport", "gripper_open_interruption"}:
            raise ValueError(f"unsupported disturbance kind: {self.kind}")
        if self.start_step < 0:
            raise ValueError("start_step must be nonnegative")
        if self.duration_steps <= 0:
            raise ValueError("duration_steps must be positive")
        if not self.magnitude:
            raise ValueError("magnitude must not be empty")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def active(self, step: int) -> bool:
        return self.start_step <= step < self.start_step + self.duration_steps

    def remaining(self, step: int) -> int:
        return max(0, self.start_step + self.duration_steps - max(step, self.start_step))


def assert_schedule_equivalence(
    reference: DisturbanceSchedule,
    candidate: DisturbanceSchedule,
) -> None:
    """Fail before a comparison when disturbance schedules differ."""
    if reference.canonical_dict() != candidate.canonical_dict():
        raise AssertionError(f"disturbance schedule mismatch: {reference.sha256} != {candidate.sha256}")


def validate_horizons(horizons: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Validate true positive post-branch continuation lengths."""
    values = tuple(int(value) for value in horizons)
    if not values or any(value <= 0 for value in values):
        raise ValueError("horizons must be positive control-step counts")
    if tuple(sorted(set(values))) != values:
        raise ValueError("horizons must be strictly increasing and unique")
    return values


def horizon_reached(*, branch_step: int, current_step: int, horizon: int) -> bool:
    """Return true after exactly ``horizon`` post-branch actions."""
    if branch_step < 0 or current_step < branch_step:
        raise ValueError("current_step must be at or after branch_step")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return current_step - branch_step == horizon


@dataclass(frozen=True)
class PhaseSignals:
    """Signals evaluated at one control-state boundary."""

    step: int
    object_rise_m: float
    finger_aperture_m: float
    ee_object_distance_m: float
    left_object_force_n: float
    right_object_force_n: float
    disturbance_started: bool = False
    disturbance_ended: bool = False

    @property
    def any_contact(self) -> bool:
        return max(self.left_object_force_n, self.right_object_force_n) >= 0.5

    @property
    def bilateral_contact(self) -> bool:
        return min(self.left_object_force_n, self.right_object_force_n) >= 0.5

    @property
    def grasp_geometry(self) -> bool:
        return self.bilateral_contact and self.finger_aperture_m <= 0.055 and self.ee_object_distance_m <= 0.12

    @property
    def lift_geometry(self) -> bool:
        return self.object_rise_m >= 0.04 and self.ee_object_distance_m <= 0.12


class PhaseTracker:
    """Assign first-occurrence phases from machine-derived signals."""

    def __init__(self) -> None:
        self._history: deque[PhaseSignals] = deque(maxlen=5)
        self._seen: set[str] = set()
        self._disturbance_has_ended = False

    @property
    def seen(self) -> frozenset[str]:
        return frozenset(self._seen)

    def update(self, signals: PhaseSignals) -> list[str]:
        self._history.append(signals)
        if signals.disturbance_ended:
            self._disturbance_has_ended = True

        candidates: list[str] = []
        if signals.step == 0 and not signals.any_contact and signals.object_rise_m < 0.001:
            candidates.append("free_space_pre_manipulation")

        recent_distances = [item.ee_object_distance_m for item in self._history]
        decreasing = len(recent_distances) >= 3 and recent_distances[-1] < recent_distances[-2] < recent_distances[-3]
        if not signals.any_contact and signals.ee_object_distance_m <= 0.18 and decreasing:
            candidates.append("approach")

        if signals.any_contact:
            candidates.append("first_contact")

        recent_three = list(self._history)[-3:]
        stable_grasp = len(recent_three) == 3 and all(item.grasp_geometry for item in recent_three)
        if stable_grasp:
            candidates.append("stable_grasp")
        if signals.object_rise_m >= 0.005 and signals.grasp_geometry:
            candidates.append("initial_lift")

        recent_five = list(self._history)
        sustained = len(recent_five) == 5 and all(item.lift_geometry for item in recent_five)
        if sustained:
            candidates.append("sustained_lift")
        if signals.disturbance_started:
            candidates.append("disturbance_onset")
        if self._disturbance_has_ended and sustained:
            candidates.append("post_disturbance_recovery")

        unseen = [phase for phase in candidates if phase not in self._seen]
        if not unseen:
            return []
        # One branch point is allowed per control state. If predicates become
        # true together, the earlier phase is emitted now and the later phase
        # remains eligible at the next state.
        emitted = unseen[:1]
        self._seen.update(emitted)
        return emitted


@dataclass(frozen=True)
class DecisionSignals:
    """Signals needed by the three terminal-decision predicates."""

    object_rise_m: float
    ee_object_distance_m: float
    finger_aperture_m: float
    left_object_force_n: float
    right_object_force_n: float
    terminated: bool = False

    @property
    def grasp_geometry(self) -> bool:
        return (
            min(self.left_object_force_n, self.right_object_force_n) >= 0.5
            and self.finger_aperture_m <= 0.055
            and self.ee_object_distance_m <= 0.12
        )

    @property
    def lift_geometry(self) -> bool:
        return self.object_rise_m >= 0.04 and self.ee_object_distance_m <= 0.12


def decision_predicates(history: list[DecisionSignals]) -> dict[str, bool]:
    """Evaluate terminal predicates on a state history ending at the decision."""
    if not history:
        raise ValueError("decision history must not be empty")
    final = history[-1]
    alive = not any(item.terminated for item in history)
    return {
        "final_height": alive and final.object_rise_m >= 0.06,
        "sustained_lift": (alive and len(history) >= 5 and all(item.lift_geometry for item in history[-5:])),
        "stable_grasp": (alive and len(history) >= 3 and all(item.grasp_geometry for item in history[-3:])),
    }


PROTOCOL_COMPONENTS: dict[str, frozenset[str]] = {
    PROTOCOL_A: frozenset(
        {
            "scene_root_pose_velocity",
            "scene_joint_position_velocity",
            "action_manager_current_previous",
            "command_value_time_counter",
            "episode_length",
        }
    ),
    PROTOCOL_B: frozenset(
        {
            "scene_root_pose_velocity",
            "scene_joint_position_velocity",
            "action_manager_current_previous",
            "command_value_time_counter",
            "episode_length",
            "action_term_raw_processed",
            "articulation_targets",
            "environment_outcome_buffers",
            "reward_manager_buffers",
            "termination_manager_buffers",
            "command_metrics_and_cache",
            "observation_history_if_present",
            "event_buffers_if_present",
            "disturbance_scheduler",
        }
    ),
}

UNAVAILABLE_COMPONENTS = frozenset(
    {
        "physx_warm_start_impulses",
        "physx_contact_manifolds",
        "physx_contact_cache",
        "physx_broadphase_cache",
        "physx_solver_internal_state",
    }
)


def validate_protocol_bookkeeping() -> None:
    """Assert that B is a strict exposed-state superset without hidden-state claims."""
    if not PROTOCOL_COMPONENTS[PROTOCOL_A] < PROTOCOL_COMPONENTS[PROTOCOL_B]:
        raise AssertionError("Protocol B must be a strict superset of Protocol A")
    if PROTOCOL_COMPONENTS[PROTOCOL_B] & UNAVAILABLE_COMPONENTS:
        raise AssertionError("Protocol B includes unavailable PhysX state")
