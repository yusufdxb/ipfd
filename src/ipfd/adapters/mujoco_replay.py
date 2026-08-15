"""Asset-free MuJoCo reference adapter for replay-fidelity audits.

The adapter intentionally keeps two independent :class:`mujoco.MjData`
instances. Environment 0 is the uninterrupted reference and environment 1 is
the restore target. MuJoCo is optional for IPFD, so importing this module does
not require the package; construction fails with a focused error when it is not
installed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import Any

import numpy as np

from ipfd.fidelity.contracts import (
    ArrayLike,
    ObservationRecord,
    Snapshot,
    StepRecord,
    TrajectoryRecord,
)

try:  # pragma: no cover - absence is exercised in environments without MuJoCo
    import mujoco as _mujoco
except (ImportError, OSError) as _mujoco_error:  # pragma: no cover - environment dependent
    _mujoco = None
    _MUJOCO_IMPORT_ERROR: Exception | None = _mujoco_error
else:
    _MUJOCO_IMPORT_ERROR = None


__all__ = ["MuJoCoReplayAdapter"]


_FREE_SPACE_XML = """
<mujoco model="ipfd_free_space">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="mover">
      <joint name="travel" type="slide" axis="1 0 0" damping="0.05"/>
      <geom name="payload" type="sphere" size="0.05" mass="1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive" joint="travel" gear="1"
           ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos name="travel_position" joint="travel"/>
    <jointvel name="travel_velocity" joint="travel"/>
  </sensor>
</mujoco>
"""


_INTERMITTENT_CONTACT_XML = """
<mujoco model="ipfd_intermittent_contact">
  <option timestep="0.01" gravity="0 0 0" iterations="8"/>
  <worldbody>
    <geom name="wall" type="box" pos="0.4 0 0" size="0.02 0.2 0.2"/>
    <body name="mover">
      <joint name="travel" type="slide" axis="1 0 0" damping="0.05"/>
      <geom name="probe" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive" joint="travel" gear="8"
           ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos name="travel_position" joint="travel"/>
    <jointvel name="travel_velocity" joint="travel"/>
  </sensor>
</mujoco>
"""


_SUSTAINED_CONTACT_XML = """
<mujoco model="ipfd_sustained_contact">
  <option timestep="0.01" iterations="8"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="mover" pos="0 0 0.25">
      <joint name="travel" type="slide" axis="0 0 1" damping="0.05"/>
      <geom name="press" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive" joint="travel" gear="1"
           ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos name="travel_position" joint="travel"/>
    <jointvel name="travel_velocity" joint="travel"/>
  </sensor>
</mujoco>
"""


_FILTERED_CONTACT_XML = """
<mujoco model="ipfd_filtered_contact">
  <option timestep="0.002" iterations="8"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="mover" pos="0 0 0.05">
      <joint name="travel" type="slide" axis="0 0 1" damping="0.05"/>
      <geom name="probe" type="sphere" size="0.05" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <general name="drive" joint="travel" dyntype="filterexact"
             dynprm="0.05" gainprm="2.0"
             ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos name="travel_position" joint="travel"/>
    <jointvel name="travel_velocity" joint="travel"/>
  </sensor>
</mujoco>
"""


_REGIME_XML = {
    "filtered_contact": _FILTERED_CONTACT_XML,
    "free_space": _FREE_SPACE_XML,
    "intermittent_contact": _INTERMITTENT_CONTACT_XML,
    "sustained_contact": _SUSTAINED_CONTACT_XML,
}

_SUPPORTED_PROTOCOLS = {
    "minimal_visible",
    "full_physics",
    "integration_with_warmstart",
}

_SUPPORTED_DECISIONS = {
    "within_bounds",
    "collision",
    "sustained_contact",
    "stable_contact",
    "forward_progress",
    "remains_in_contact",
}


def _enum_value(name: str) -> int:
    assert _mujoco is not None
    value = getattr(_mujoco.mjtState, name)
    return int(value.value)


def _canonical_continuation_mode(value: str) -> str:
    restored = {"restored", "preserve", "warm", "exact_action", "open_loop"}
    cold = {"cold", "cold_start", "cold_continuation"}
    if value in restored:
        return "restored"
    if value in cold:
        return "cold"
    choices = sorted(restored | cold)
    raise ValueError(f"unsupported MuJoCo continuation_mode {value!r}; expected one of {choices}")


class MuJoCoReplayAdapter:
    """Replay adapter backed by two independent MuJoCo data instances.

    Args:
        settings: Adapter settings. ``regime`` selects ``free_space``,
            ``filtered_contact``, ``intermittent_contact``, or
            ``sustained_contact``. All models are embedded MJCF and require no
            external assets.
        snapshot_protocol: One of ``minimal_visible``, ``full_physics``, or
            ``integration_with_warmstart``.
        continuation_mode: ``restored`` preserves warm-start values when the
            protocol contains them. ``cold`` deliberately zeros warm-start
            accelerations after restore. ``exact_action`` and ``open_loop`` are
            accepted aliases for restored identical-action continuation.
    """

    num_envs = 2

    def __init__(
        self,
        settings: Mapping[str, Any],
        snapshot_protocol: str,
        continuation_mode: str,
    ) -> None:
        if _mujoco is None:
            raise ImportError(
                "MuJoCoReplayAdapter requires the optional 'mujoco' package"
            ) from _MUJOCO_IMPORT_ERROR
        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping")
        self.settings = dict(settings)
        regime = self.settings.get("regime", self.settings.get("environment"))
        if not isinstance(regime, str) or regime not in _REGIME_XML:
            raise ValueError(f"settings.regime must be one of {sorted(_REGIME_XML)}")
        if snapshot_protocol not in _SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"unsupported MuJoCo snapshot_protocol {snapshot_protocol!r}; "
                f"expected one of {sorted(_SUPPORTED_PROTOCOLS)}"
            )
        if not isinstance(continuation_mode, str):
            raise TypeError("continuation_mode must be a string")

        self.regime = regime
        self.snapshot_protocol = snapshot_protocol
        self.continuation_mode = continuation_mode
        self._continuation_behavior = _canonical_continuation_mode(continuation_mode)
        capture_activation = self.settings.get("minimal_capture_actuator_activation", True)
        if not isinstance(capture_activation, bool):
            raise TypeError("settings.minimal_capture_actuator_activation must be a boolean")
        self._minimal_capture_activation = capture_activation
        self._model = _mujoco.MjModel.from_xml_string(_REGIME_XML[regime])
        self._model_xml_sha256 = hashlib.sha256(
            _REGIME_XML[regime].encode("utf-8")
        ).hexdigest()
        timestep = float(self.settings.get("timestep", self._model.opt.timestep))
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("settings.timestep must be finite and positive")
        self._model.opt.timestep = timestep
        self._frame_skip = self._positive_int("frame_skip", default=1)
        self._contact_window = self._positive_int("contact_window", default=20)
        self._activation_preload_steps = self._positive_int("activation_preload_steps", default=100)
        self._post_preload_control = float(self.settings.get("post_preload_control", 0.55))
        if not np.isfinite(self._post_preload_control) or abs(self._post_preload_control) > 1.0:
            raise ValueError("settings.post_preload_control must be finite and within [-1, 1]")
        self._bounds = float(self.settings.get("position_bound", 2.0 if regime != "sustained_contact" else 1.0))
        self._progress_threshold = float(self.settings.get("progress_threshold", 1.0e-6))
        self._stable_velocity = float(self.settings.get("stable_velocity", 1.0e-3))
        self._jitter = float(self.settings.get("initial_position_jitter", 0.005))
        for name, value in (
            ("position_bound", self._bounds),
            ("progress_threshold", self._progress_threshold),
            ("stable_velocity", self._stable_velocity),
            ("initial_position_jitter", self._jitter),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"settings.{name} must be finite and non-negative")

        self._data = [_mujoco.MjData(self._model), _mujoco.MjData(self._model)]
        self._step_counts = np.zeros(self.num_envs, dtype=np.int64)
        self._initial_qpos = np.zeros((self.num_envs, self._model.nq), dtype=np.float64)
        self._seed = 0
        self._closed = False
        self._mover_body_id = int(self._model.body("mover").id)
        self._model_configuration = self._effective_model_configuration()
        self._model_configuration_sha256 = hashlib.sha256(
            json.dumps(
                self._model_configuration,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.reset(seed=0)

    def _effective_model_configuration(self) -> dict[str, Any]:
        assert _mujoco is not None
        option = self._model.opt
        return {
            "mjcf_sha256": self._model_xml_sha256,
            "mujoco_version": _mujoco.mj_versionString(),
            "timestep": float(option.timestep),
            "integrator": int(option.integrator),
            "solver": int(option.solver),
            "iterations": int(option.iterations),
            "ls_iterations": int(option.ls_iterations),
            "tolerance": float(option.tolerance),
            "ls_tolerance": float(option.ls_tolerance),
            "gravity": np.asarray(option.gravity, dtype=np.float64).tolist(),
            "cone": int(option.cone),
            "jacobian": int(option.jacobian),
            "disableflags": int(option.disableflags),
            "enableflags": int(option.enableflags),
            "frame_skip": self._frame_skip,
            "dimensions": {
                "nq": int(self._model.nq),
                "nv": int(self._model.nv),
                "na": int(self._model.na),
                "nu": int(self._model.nu),
                "nsensordata": int(self._model.nsensordata),
                "nhistory": int(self._model.nhistory),
            },
        }

    def _positive_int(self, name: str, *, default: int) -> int:
        value = self.settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"settings.{name} must be an integer >= 1")
        return value

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MuJoCoReplayAdapter is closed")

    def _env_ids(self, env_ids: Sequence[int]) -> tuple[int, ...]:
        self._ensure_open()
        if isinstance(env_ids, (str, bytes)):
            raise TypeError("env_ids must be a sequence of integer environment ids")
        result = tuple(env_ids)
        if not result:
            raise ValueError("env_ids must not be empty")
        if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in result):
            raise TypeError("env_ids must contain integers")
        normalized = tuple(int(item) for item in result)
        if len(set(normalized)) != len(normalized):
            raise ValueError("env_ids must be unique")
        if any(item < 0 or item >= self.num_envs for item in normalized):
            raise IndexError(f"MuJoCo environment ids must be in [0, {self.num_envs})")
        return normalized

    def _signature(self) -> int:
        if self.snapshot_protocol == "minimal_visible":
            signature = (
                _enum_value("mjSTATE_TIME")
                | _enum_value("mjSTATE_QPOS")
                | _enum_value("mjSTATE_QVEL")
                | _enum_value("mjSTATE_CTRL")
                | _enum_value("mjSTATE_HISTORY")
            )
            if self._minimal_capture_activation:
                signature |= _enum_value("mjSTATE_ACT")
            return signature
        if self.snapshot_protocol == "full_physics":
            return _enum_value("mjSTATE_FULLPHYSICS")
        return _enum_value("mjSTATE_INTEGRATION")

    def _signature_components(self, signature: int | None = None) -> tuple[str, ...]:
        signature = self._signature() if signature is None else signature
        candidates = (
            ("mjSTATE_TIME", "simulation_time"),
            ("mjSTATE_QPOS", "generalized_positions"),
            ("mjSTATE_QVEL", "generalized_velocities"),
            ("mjSTATE_ACT", "actuator_activations"),
            ("mjSTATE_HISTORY", "control_and_sensor_history"),
            ("mjSTATE_WARMSTART", "solver_acceleration_warmstart"),
            ("mjSTATE_CTRL", "controls"),
            ("mjSTATE_QFRC_APPLIED", "applied_generalized_forces"),
            ("mjSTATE_XFRC_APPLIED", "applied_cartesian_forces"),
            ("mjSTATE_EQ_ACTIVE", "equality_constraint_activation"),
            ("mjSTATE_MOCAP_POS", "mocap_positions"),
            ("mjSTATE_MOCAP_QUAT", "mocap_orientations"),
            ("mjSTATE_USERDATA", "user_data"),
            ("mjSTATE_PLUGIN", "plugin_state"),
        )
        return tuple(label for enum_name, label in candidates if signature & _enum_value(enum_name))

    def _component_sizes(self) -> dict[str, int]:
        assert _mujoco is not None
        result: dict[str, int] = {}
        for enum_name, label in (
            ("mjSTATE_TIME", "simulation_time"),
            ("mjSTATE_QPOS", "generalized_positions"),
            ("mjSTATE_QVEL", "generalized_velocities"),
            ("mjSTATE_ACT", "actuator_activations"),
            ("mjSTATE_HISTORY", "control_and_sensor_history"),
            ("mjSTATE_WARMSTART", "solver_acceleration_warmstart"),
            ("mjSTATE_CTRL", "controls"),
            ("mjSTATE_QFRC_APPLIED", "applied_generalized_forces"),
            ("mjSTATE_XFRC_APPLIED", "applied_cartesian_forces"),
            ("mjSTATE_EQ_ACTIVE", "equality_constraint_activation"),
            ("mjSTATE_MOCAP_POS", "mocap_positions"),
            ("mjSTATE_MOCAP_QUAT", "mocap_orientations"),
            ("mjSTATE_USERDATA", "user_data"),
            ("mjSTATE_PLUGIN", "plugin_state"),
        ):
            result[label] = int(_mujoco.mj_stateSize(self._model, _enum_value(enum_name)))
        return result

    def _unavailable_components(self) -> tuple[str, ...]:
        captured = set(self._signature_components())
        documented = set(self._signature_components(_enum_value("mjSTATE_INTEGRATION")))
        omitted = sorted(documented - captured)
        return tuple(
            omitted
            + [
                "derived_contact_manifolds_recomputed_by_mj_forward",
                "constraint_solver_iteration_scratch",
                "renderer_state",
                "external_policy_or_controller_history",
                "adapter_random_generator_state",
            ]
        )

    def reset(self, seed: int) -> ObservationRecord:
        """Reset both instances to one identical, seed-derived initial state."""

        self._ensure_open()
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        assert _mujoco is not None
        self._seed = int(seed)
        rng = np.random.default_rng(self._seed)
        offset = float(rng.uniform(-self._jitter, self._jitter)) if self._model.nq else 0.0
        for env_id, data in enumerate(self._data):
            _mujoco.mj_resetData(self._model, data)
            if self._model.nq:
                data.qpos[:] = self._model.qpos0
                data.qpos[0] += offset
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            data.qacc_warmstart[:] = 0.0
            _mujoco.mj_forward(self._model, data)
            self._step_counts[env_id] = 0
            self._initial_qpos[env_id] = data.qpos
        return self.observe((0, 1))

    def capture(self, env_ids: Sequence[int]) -> Snapshot:
        ids = self._env_ids(env_ids)
        assert _mujoco is not None
        signature = self._signature()
        state_size = int(_mujoco.mj_stateSize(self._model, signature))
        states = np.empty((len(ids), state_size), dtype=np.float64)
        for row, env_id in enumerate(ids):
            _mujoco.mj_getState(self._model, self._data[env_id], states[row], signature)
        return Snapshot(
            protocol=self.snapshot_protocol,
            values={
                "state_vectors": states,
                "task_initial_qpos": self._initial_qpos[np.asarray(ids, dtype=int)].copy(),
                "adapter_control_steps": self._step_counts[np.asarray(ids, dtype=int)].copy(),
            },
            captured_components=self._signature_components(signature)
            + ("task_initial_qpos", "adapter_control_step_counter"),
            unavailable_components=self._unavailable_components(),
            metadata={
                "source_env_ids": list(ids),
                "state_signature": signature,
                "state_size": state_size,
                "component_sizes": self._component_sizes(),
                "environment": self.regime,
                "continuation_behavior": self._continuation_behavior,
                "model_xml_sha256": self._model_xml_sha256,
                "model_configuration_sha256": self._model_configuration_sha256,
            },
        )

    def restore(self, snapshot: Snapshot, env_ids: Sequence[int]) -> None:
        ids = self._env_ids(env_ids)
        if snapshot.protocol != self.snapshot_protocol:
            raise ValueError(
                f"snapshot protocol {snapshot.protocol!r} does not match adapter protocol "
                f"{self.snapshot_protocol!r}"
            )
        if snapshot.metadata.get("environment") != self.regime:
            raise ValueError("snapshot environment does not match this MuJoCo adapter")
        if snapshot.metadata.get("model_xml_sha256") != self._model_xml_sha256:
            raise ValueError("snapshot MuJoCo model digest does not match this adapter")
        if (
            snapshot.metadata.get("model_configuration_sha256")
            != self._model_configuration_sha256
        ):
            raise ValueError(
                "snapshot MuJoCo integration configuration does not match this adapter"
            )
        signature = snapshot.metadata.get("state_signature")
        if isinstance(signature, bool) or not isinstance(signature, (int, np.integer)):
            raise ValueError("snapshot metadata has no valid state_signature")
        signature = int(signature)
        if signature != self._signature():
            raise ValueError("snapshot state_signature does not match the selected protocol")
        states = np.asarray(snapshot.values.get("state_vectors"), dtype=np.float64)
        if states.ndim != 2:
            raise ValueError("snapshot state_vectors must be a two-dimensional array")
        if states.shape[0] not in (1, len(ids)):
            raise ValueError("snapshot must contain one state to broadcast or one state per target")
        assert _mujoco is not None
        expected_size = int(_mujoco.mj_stateSize(self._model, signature))
        if states.shape[1] != expected_size:
            raise ValueError(
                f"snapshot state vector size {states.shape[1]} does not match expected size {expected_size}"
            )
        task_initial_qpos = np.asarray(snapshot.values.get("task_initial_qpos"), dtype=np.float64)
        control_steps = np.asarray(snapshot.values.get("adapter_control_steps"))
        if task_initial_qpos.shape != (states.shape[0], self._model.nq):
            raise ValueError("snapshot task_initial_qpos shape does not match its state vectors")
        if control_steps.shape != (states.shape[0],):
            raise ValueError("snapshot adapter_control_steps shape does not match its state vectors")
        if not np.issubdtype(control_steps.dtype, np.integer):
            raise ValueError("snapshot adapter_control_steps must contain integers")
        for row, env_id in enumerate(ids):
            source_row = 0 if states.shape[0] == 1 else row
            data = self._data[env_id]
            _mujoco.mj_resetData(self._model, data)
            _mujoco.mj_setState(self._model, data, states[source_row], signature)
            if self._continuation_behavior == "cold" or not (
                signature & _enum_value("mjSTATE_WARMSTART")
            ):
                data.qacc_warmstart[:] = 0.0
            _mujoco.mj_forward(self._model, data)
            self._step_counts[env_id] = int(control_steps[source_row])
            self._initial_qpos[env_id] = task_initial_qpos[source_row]

    def _contact_measurements(self, data: Any) -> tuple[bool, int, float]:
        count = int(data.ncon)
        if count == 0:
            return False, 0, 0.0
        minimum_distance = min(float(data.contact[index].dist) for index in range(count))
        penetration = max(0.0, -minimum_distance)
        return True, count, penetration

    def observe(self, env_ids: Sequence[int]) -> ObservationRecord:
        ids = self._env_ids(env_ids)
        assert _mujoco is not None
        # MuJoCo sensors are evaluated at specific pipeline stages. A restored
        # data instance is explicitly forwarded after ``mj_setState``; forward
        # every observed instance as well so L0 compares equally refreshed
        # derived quantities rather than a post-step sensor buffer with a
        # post-forward sensor buffer.
        for item in ids:
            _mujoco.mj_forward(self._model, self._data[item])
        qpos = np.stack([np.array(self._data[item].qpos, copy=True) for item in ids])
        qvel = np.stack([np.array(self._data[item].qvel, copy=True) for item in ids])
        act = np.stack([np.array(self._data[item].act, copy=True) for item in ids])
        body_position = np.stack(
            [np.array(self._data[item].xpos[self._mover_body_id], copy=True) for item in ids]
        )
        qacc = np.stack([np.array(self._data[item].qacc, copy=True) for item in ids])
        constraint_force = np.stack(
            [np.array(self._data[item].qfrc_constraint, copy=True) for item in ids]
        )
        actuator_force = np.stack([np.array(self._data[item].qfrc_actuator, copy=True) for item in ids])
        controls = np.stack([np.array(self._data[item].ctrl, copy=True) for item in ids])
        sensors = np.stack([np.array(self._data[item].sensordata, copy=True) for item in ids])
        contacts = [self._contact_measurements(self._data[item]) for item in ids]
        within_bounds = np.max(np.abs(qpos), axis=1) <= self._bounds
        policy_vector = np.concatenate((qpos, qvel, sensors), axis=1)
        scene_state = {
            "qpos": qpos,
            "qvel": qvel,
            "mover_position": body_position,
        }
        privileged_observations = {
            "qacc": qacc,
            "constraint_force": constraint_force,
            "actuator_force": actuator_force,
        }
        unavailable = [
            "external_policy_observation_history",
            "external_task_manager_state",
            "contact_solver_cache",
        ]
        activation_unavailable = (
            self.snapshot_protocol == "minimal_visible"
            and not self._minimal_capture_activation
            and self._model.na > 0
        )
        if activation_unavailable:
            # This intentionally narrow contract models the common qpos/qvel
            # snapshot. Activation-dependent accelerations and forces are not
            # restore-boundary claims when their causal state is omitted.
            unavailable.extend(
                (
                    "actuator_activation_state",
                    "activation_dependent_acceleration_and_force",
                )
            )
            privileged_observations = {}
        else:
            scene_state["act"] = act
        return ObservationRecord(
            scene_state=scene_state,
            policy_observations={"state": policy_vector},
            privileged_observations=privileged_observations,
            task_state={
                "within_bounds": within_bounds,
                "contact_active": np.asarray([item[0] for item in contacts], dtype=bool),
                "contact_count": np.asarray([item[1] for item in contacts], dtype=np.int64),
                "initial_qpos": self._initial_qpos[np.asarray(ids, dtype=int)].copy(),
            },
            controller_targets={"ctrl": controls},
            sensor_state={"sensordata": sensors},
            counters={
                "simulation_time": np.asarray([self._data[item].time for item in ids], dtype=np.float64),
                "control_steps": self._step_counts[np.asarray(ids, dtype=int)].copy(),
            },
            unavailable=tuple(unavailable),
        )

    def _default_action(self, step: int) -> np.ndarray:
        if self.regime == "free_space":
            return np.asarray([0.25], dtype=np.float64)
        if self.regime == "filtered_contact":
            value = -1.0 if step < self._activation_preload_steps else self._post_preload_control
            return np.asarray([value], dtype=np.float64)
        if self.regime == "intermittent_contact":
            push_steps = self._positive_int("contact_push_steps", default=40)
            return np.asarray([1.0 if step < push_steps else -1.0], dtype=np.float64)
        return np.asarray([-0.1], dtype=np.float64)

    def action(self, step: int, source: Any, env_ids: Sequence[int]) -> np.ndarray:
        """Return one deterministic or recorded action for each requested instance."""

        ids = self._env_ids(env_ids)
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("step must be an integer >= 0")
        if isinstance(source, str):
            if source == "zero":
                value = np.zeros(self._model.nu, dtype=np.float64)
            elif source in {
                "adapter",
                "deterministic",
                "recorded",
                "uninterrupted",
                "restored",
                "exact_action",
                "open_loop",
            }:
                value = self._default_action(int(step))
            else:
                raise ValueError(f"unknown MuJoCo action source {source!r}")
            return np.repeat(value.reshape(1, -1), len(ids), axis=0)
        if isinstance(source, Mapping):
            rows = [np.asarray(source[item], dtype=np.float64).reshape(-1) for item in ids]
            action = np.stack(rows)
        else:
            trace = np.asarray(source, dtype=np.float64)
            if trace.ndim == 1:
                value = trace
            elif trace.ndim >= 2:
                if int(step) >= trace.shape[0]:
                    raise IndexError("recorded action source is shorter than the requested step")
                value = np.asarray(trace[int(step)], dtype=np.float64).reshape(-1)
            else:
                raise ValueError("action source must provide at least one dimension")
            action = np.repeat(value.reshape(1, -1), len(ids), axis=0)
        if action.shape != (len(ids), self._model.nu):
            raise ValueError(
                f"actions must have shape {(len(ids), self._model.nu)}, got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError("actions must contain only finite values")
        return action

    def _coerce_actions(self, actions: ArrayLike) -> np.ndarray:
        if isinstance(actions, Mapping):
            try:
                array = np.stack(
                    [np.asarray(actions[item], dtype=np.float64).reshape(-1) for item in range(self.num_envs)]
                )
            except KeyError as exc:
                raise ValueError("action mapping must provide environment ids 0 and 1") from exc
        else:
            array = np.asarray(actions, dtype=np.float64)
            if array.ndim == 1:
                array = np.repeat(array.reshape(1, -1), self.num_envs, axis=0)
        expected = (self.num_envs, self._model.nu)
        if array.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("actions must contain only finite values")
        return array

    def step(self, actions: ArrayLike) -> StepRecord:
        self._ensure_open()
        assert _mujoco is not None
        action = self._coerce_actions(actions)
        for env_id, data in enumerate(self._data):
            data.ctrl[:] = action[env_id]
            for _ in range(self._frame_skip):
                _mujoco.mj_step(self._model, data)
            self._step_counts[env_id] += 1
        observation = self.observe((0, 1))
        contacts = [self._contact_measurements(data) for data in self._data]
        qpos = np.stack([np.array(data.qpos, copy=True) for data in self._data])
        qvel = np.stack([np.array(data.qvel, copy=True) for data in self._data])
        within_bounds = np.max(np.abs(qpos), axis=1) <= self._bounds
        progress = qpos[:, 0] - self._initial_qpos[:, 0]
        terminated = ~within_bounds
        reward = qvel[:, 0] - 0.01 * np.square(action[:, 0])
        return StepRecord(
            observation=observation,
            contact_state={
                "active": np.asarray([item[0] for item in contacts], dtype=bool),
                "count": np.asarray([item[1] for item in contacts], dtype=np.int64),
                "maximum_penetration": np.asarray([item[2] for item in contacts], dtype=np.float64),
            },
            task_outputs={
                "within_bounds": within_bounds,
                "forward_progress": progress,
                "speed": np.linalg.norm(qvel, axis=1),
            },
            terminated={"terminated": terminated, "out_of_bounds": terminated.copy()},
            reward={"value": reward},
            semantic={"regime": self.regime},
            applied_actions=action.copy(),
        )

    @staticmethod
    def _row(value: Any, env_id: int) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 0:
            return array
        if 0 <= env_id < array.shape[0]:
            return np.asarray(array[env_id])
        if array.shape[0] == 1:
            return np.asarray(array[0])
        raise IndexError(f"trajectory env_id {env_id} is outside recorded shape {array.shape}")

    def decision(self, record: TrajectoryRecord, name: str) -> bool:
        if name not in _SUPPORTED_DECISIONS:
            raise ValueError(
                f"unknown MuJoCo decision {name!r}; expected one of {sorted(_SUPPORTED_DECISIONS)}"
            )
        if not record.steps:
            raise ValueError("trajectory record has no steps")
        env_id = int(record.env_id)
        active = [
            bool(self._row(step.contact_state["active"], env_id).item()) for step in record.steps
        ]
        if name == "within_bounds":
            return all(
                bool(self._row(step.task_outputs["within_bounds"], env_id).item())
                for step in record.steps
            )
        if name == "remains_in_contact":
            return all(active)
        if name == "collision":
            return any(active)
        if name == "sustained_contact":
            run = maximum = 0
            for value in active:
                run = run + 1 if value else 0
                maximum = max(maximum, run)
            return maximum >= self._contact_window
        if name == "stable_contact":
            if len(active) < self._contact_window or not all(active[-self._contact_window :]):
                return False
            speeds = [
                float(self._row(step.task_outputs["speed"], env_id).item())
                for step in record.steps[-self._contact_window :]
            ]
            return max(speeds) <= self._stable_velocity
        first = self._row(record.steps[0].observation.scene_state["qpos"], env_id).reshape(-1)
        final = self._row(record.steps[-1].observation.scene_state["qpos"], env_id).reshape(-1)
        if first.size == 0 or final.size == 0:
            raise ValueError("forward_progress requires at least one generalized position")
        return bool(final[0] - first[0] > self._progress_threshold)

    def provenance(self) -> Mapping[str, object]:
        self._ensure_open()
        assert _mujoco is not None
        try:
            package_version = metadata.version("mujoco")
        except metadata.PackageNotFoundError:  # pragma: no cover - unusual source install
            package_version = _mujoco.mj_versionString()
        warmstart_in_protocol = bool(self._signature() & _enum_value("mjSTATE_WARMSTART"))
        return {
            "adapter": "MuJoCoReplayAdapter",
            "simulator": "MuJoCo",
            "simulator_version": _mujoco.mj_versionString(),
            "package_version": package_version,
            "environment": self.regime,
            "model_source": "embedded_asset_free_mjcf",
            "model_xml_sha256": self._model_xml_sha256,
            "model_configuration_sha256": self._model_configuration_sha256,
            "model_configuration": self._model_configuration,
            "snapshot_protocol": self.snapshot_protocol,
            "minimal_capture_actuator_activation": self._minimal_capture_activation,
            "continuation_mode": self.continuation_mode,
            "continuation_behavior": self._continuation_behavior,
            "state_components_captured": list(self._signature_components())
            + ["task_initial_qpos", "adapter_control_step_counter"],
            "state_components_unavailable": list(self._unavailable_components()),
            "task_state_captured": [
                "initial_qpos",
            ],
            "task_state_recomputed_or_measured": [
                "within_bounds",
                "contact_active",
                "contact_count",
            ],
            "controller_or_policy_history_captured": False,
            "random_state_handling": {
                "simulator_random_state": "none_used_by_embedded_models",
                "reset_seed": self._seed,
                "adapter_rng_state_captured": False,
                "post_reset_stochastic_events": False,
            },
            "solver_state_availability": {
                "warm_start_in_snapshot_protocol": warmstart_in_protocol,
                "warm_start_preserved_for_continuation": (
                    warmstart_in_protocol and self._continuation_behavior == "restored"
                ),
                "contact_manifolds": "recomputed_not_serialized",
                "constraint_solver_iteration_scratch": "unavailable",
            },
            "sensor_refresh_behavior": (
                "mj_forward is called after mj_setState; sensors and derived contact state are recomputed"
            ),
            "callback_identity": "no MuJoCo callbacks registered by this adapter",
            "managed_external_state": {
                "policy_or_controller": "none",
                "renderer": "not captured",
                "plugins": "none in bundled models",
            },
            "unsupported_restoration_claims": [
                "No claim that derived contact manifolds or solver scratch are serialized",
                "No claim that external policy, controller, renderer, or process state is restored",
                "No universal MuJoCo fidelity verdict",
            ],
            "model_dimensions": {
                "nq": int(self._model.nq),
                "nv": int(self._model.nv),
                "na": int(self._model.na),
                "nu": int(self._model.nu),
                "nsensordata": int(self._model.nsensordata),
                "nhistory": int(self._model.nhistory),
            },
            "timestep": float(self._model.opt.timestep),
            "frame_skip": self._frame_skip,
            "deterministic_action_schedule": (
                {
                    "activation_preload_steps": self._activation_preload_steps,
                    "preload_control": -1.0,
                    "post_preload_control": self._post_preload_control,
                }
                if self.regime == "filtered_contact"
                else None
            ),
            "decision_contracts": {
                "within_bounds": {
                    "definition": "all generalized positions remain within the configured absolute bound",
                    "position_bound": self._bounds,
                },
                "collision": {"definition": "at least one contact is active in the continuation"},
                "remains_in_contact": {
                    "definition": "contact is active at every continuation step"
                },
                "sustained_contact": {
                    "definition": "the continuation contains a consecutive contact run of the configured length",
                    "contact_window": self._contact_window,
                },
                "stable_contact": {
                    "definition": "contact remains active and speed stays below the configured threshold",
                    "contact_window": self._contact_window,
                    "stable_velocity": self._stable_velocity,
                },
                "forward_progress": {
                    "definition": "terminal generalized position exceeds the first continuation position",
                    "progress_threshold": self._progress_threshold,
                },
            },
            "instances": self.num_envs,
        }

    def close(self) -> None:
        """Release references to the model and both data instances."""

        if self._closed:
            return
        self._data.clear()
        self._model = None
        self._closed = True
