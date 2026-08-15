"""Isaac Lab implementation of the replay-fidelity adapter contract.

This adapter supports a built-in manager-based environment path and an optional
factory hook for custom environments. Missing runtime and PhysX state is exposed
in provenance and every snapshot instead of being hidden by the common interface.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..fidelity.contracts import ObservationRecord, Snapshot, StepRecord, TrajectoryRecord

__all__ = ["IsaacLabReplayAdapter"]

_SCENE_ONLY = "scene_only"
_EXPANDED = "expanded_runtime_state"
_PROTOCOLS = {_SCENE_ONLY, _EXPANDED}


def _clone(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {str(key): _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value


def _take(value: Any, indices: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _take(item, indices) for key, item in value.items()}
    if isinstance(value, list):
        return [_take(item, indices) for item in value]
    if isinstance(value, tuple):
        return tuple(_take(item, indices) for item in value)
    if hasattr(value, "__getitem__") and hasattr(value, "shape"):
        try:
            return _clone(value[indices])
        except (IndexError, RuntimeError, TypeError):
            return _clone(value)
    return value


def _assign(target: Any, indices: Any, value: Any) -> None:
    target[indices] = value


def _load_callable(path: str) -> Callable[..., Any]:
    module_name, separator, attribute = path.rpartition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory paths must use 'module:callable' syntax")
    function = getattr(importlib.import_module(module_name), attribute)
    if not callable(function):
        raise TypeError(f"configured factory is not callable: {path}")
    return function


class IsaacLabReplayAdapter:
    """Audit a paired Isaac Lab manager-based environment."""

    def __init__(
        self,
        env: Any,
        *,
        settings: Mapping[str, Any],
        snapshot_protocol: str,
        continuation_mode: str,
        action_provider: Callable[..., Any] | None = None,
        decision_functions: Mapping[str, Callable[[TrajectoryRecord], bool]] | None = None,
        app: Any = None,
        owns_env: bool = False,
    ) -> None:
        if snapshot_protocol not in _PROTOCOLS:
            raise ValueError(f"unknown Isaac Lab snapshot protocol: {snapshot_protocol}")
        self.env = env
        self.base = env.unwrapped
        self.settings = dict(settings)
        self.snapshot_protocol = snapshot_protocol
        self.continuation_mode = continuation_mode
        self.action_provider = action_provider
        self.decision_functions = dict(decision_functions or {})
        self.app = app
        self.owns_env = owns_env
        self._last_observation: Any = None
        self._last_reward: Any = None
        self._last_terminated: Any = None
        self._last_truncated: Any = None
        if int(getattr(self.base, "num_envs", 0)) < 2:
            raise ValueError("Isaac Lab replay fidelity requires at least two environments")

    @classmethod
    def from_config(
        cls,
        settings: Mapping[str, Any],
        *,
        snapshot_protocol: str,
        continuation_mode: str,
    ) -> IsaacLabReplayAdapter:
        """Construct a built-in task or consume a configured environment factory."""

        factory_path = settings.get("environment_factory")
        if isinstance(factory_path, str) and factory_path:
            built = _load_callable(factory_path)(dict(settings))
            if not isinstance(built, Mapping) or "env" not in built:
                raise TypeError("environment_factory must return a mapping containing env")
            return cls(
                built["env"],
                settings=settings,
                snapshot_protocol=snapshot_protocol,
                continuation_mode=continuation_mode,
                action_provider=built.get("action_provider"),
                decision_functions=built.get("decision_functions"),
                app=built.get("app"),
                owns_env=bool(built.get("owns_env", False)),
            )

        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=bool(settings.get("headless", True)))
        app = launcher.app
        try:
            asset_root = settings.get("asset_root")
            if asset_root is not None:
                if not isinstance(asset_root, str) or not asset_root.strip():
                    raise ValueError("adapter.asset_root must be a non-empty string when provided")
                import isaaclab.utils.assets as assets

                root = asset_root.rstrip("/")
                assets.NUCLEUS_ASSET_ROOT_DIR = root
                assets.NVIDIA_NUCLEUS_DIR = f"{root}/NVIDIA"
                assets.ISAAC_NUCLEUS_DIR = f"{root}/Isaac"
                assets.ISAACLAB_NUCLEUS_DIR = f"{root}/Isaac/IsaacLab"
                import isaaclab_assets.robots.franka as franka

                panda_path = f"{assets.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
                franka.FRANKA_PANDA_CFG.spawn.usd_path = panda_path
                franka.FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path = panda_path
            import gymnasium as gym
            import isaaclab_tasks  # noqa: F401
            from isaaclab_tasks.utils import parse_env_cfg

            task = str(settings.get("task", "Isaac-Lift-Cube-Franka-v0"))
            device = str(settings.get("device", "cuda:0"))
            num_envs = int(settings.get("num_envs", 2))
            if num_envs < 2:
                raise ValueError("adapter.num_envs must be at least 2")
            env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
            if hasattr(env_cfg, "seed"):
                env_cfg.seed = int(settings.get("initial_seed", 0))
            policy_group = getattr(getattr(env_cfg, "observations", None), "policy", None)
            if policy_group is not None and hasattr(policy_group, "enable_corruption"):
                policy_group.enable_corruption = False
            env = gym.make(task, cfg=env_cfg)
            action_provider = None
            provider_path = settings.get("action_provider_factory")
            if isinstance(provider_path, str) and provider_path:
                action_provider = _load_callable(provider_path)(env, dict(settings))
        except BaseException:
            app.close()
            raise
        return cls(
            env,
            settings=settings,
            snapshot_protocol=snapshot_protocol,
            continuation_mode=continuation_mode,
            action_provider=action_provider,
            app=app,
            owns_env=True,
        )

    def _ids(self, env_ids: Sequence[int]) -> Any:
        import torch

        return torch.as_tensor(tuple(env_ids), device=self.base.device, dtype=torch.long)

    def _manager_state(self, ids: Any) -> dict[str, Any]:
        state: dict[str, Any] = {
            "action": _take(self.base.action_manager.action, ids),
            "previous_action": _take(self.base.action_manager.prev_action, ids),
            "episode_length": _take(self.base.episode_length_buf, ids),
        }
        commands: dict[str, Any] = {}
        for name in getattr(self.base.command_manager, "active_terms", []):
            term = self.base.command_manager.get_term(name)
            values = {}
            for attribute in ("command", "time_left", "command_counter", "pose_command_w"):
                if hasattr(term, attribute):
                    values[attribute] = _take(getattr(term, attribute), ids)
            if hasattr(term, "metrics"):
                values["metrics"] = {key: _take(value, ids) for key, value in term.metrics.items()}
            commands[name] = values
        state["commands"] = commands
        return state

    def _controller_targets(self, ids: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        articulations = getattr(self.base.scene, "articulations", {})
        for name, articulation in articulations.items():
            values = {}
            for attribute in ("joint_pos_target", "joint_vel_target", "joint_effort_target"):
                if hasattr(articulation.data, attribute):
                    values[attribute] = _take(getattr(articulation.data, attribute), ids)
            result[str(name)] = values
        return result

    def _sensor_state(self, ids: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        sensors = getattr(self.base.scene, "sensors", {})
        for name, sensor in sensors.items():
            values = {}
            data = getattr(sensor, "data", None)
            if data is None:
                continue
            for attribute in (
                "net_forces_w",
                "force_matrix_w",
                "pos_w",
                "quat_w_world",
                "target_pos_w",
                "target_quat_w",
            ):
                value = getattr(data, attribute, None)
                if value is not None:
                    values[attribute] = _take(value, ids)
            if values:
                result[str(name)] = values
        return result

    def capture(self, env_ids: Sequence[int]) -> Snapshot:
        ids = self._ids(env_ids)
        values: dict[str, Any] = {
            "scene_state": _take(self.base.scene.get_state(is_relative=True), ids),
        }
        captured = [
            "articulation root pose and velocity",
            "articulation joint position and velocity",
            "rigid-object root pose and velocity",
        ]
        unavailable = [
            "PhysX warm-start impulses",
            "PhysX solver contact manifolds and persistent contact caches",
            "PhysX broadphase pair caches",
            "per-contact friction solver state",
            "internal GPU solver scheduling state",
            "per-environment Python, NumPy, torch, and CUDA RNG state",
            "restorable per-environment sensor internal state",
        ]
        if self.snapshot_protocol == _EXPANDED:
            values["manager_state"] = self._manager_state(ids)
            values["controller_targets"] = self._controller_targets(ids)
            captured.extend(
                [
                    "action and previous-action buffers",
                    "command state and metrics",
                    "episode-length buffer",
                    "articulation position, velocity, and effort targets",
                ]
            )
            provider_capture = getattr(self.action_provider, "capture_state", None)
            if callable(provider_capture):
                values["policy_history"] = provider_capture(tuple(env_ids))
                captured.append("policy or controller history supplied by action provider")
            else:
                unavailable.append("policy or controller recurrent history")
            unavailable.extend(
                [
                    "observation history without an adapter-specific copy hook",
                    "event-manager interval history without an adapter-specific copy hook",
                    "reward-manager buffers without an adapter-specific copy hook",
                    "termination-manager buffers without an adapter-specific copy hook",
                    "disturbance schedule without an adapter-specific state hook",
                ]
            )
        else:
            unavailable.extend(
                [
                    "task-manager buffers",
                    "controller targets",
                    "policy or controller history",
                    "observation and event history",
                    "disturbance schedule",
                ]
            )
        return Snapshot(
            protocol=self.snapshot_protocol,
            values=values,
            captured_components=captured,
            unavailable_components=unavailable,
            metadata={
                "source_env_ids": list(env_ids),
                "sensor_refresh_behavior": (
                    "sensor values are observed after the environment step; "
                    "sensor internals are not restored"
                ),
            },
        )

    def restore(self, snapshot: Snapshot, env_ids: Sequence[int]) -> None:
        if snapshot.protocol != self.snapshot_protocol:
            raise ValueError(
                f"snapshot protocol {snapshot.protocol!r} does not match adapter protocol {self.snapshot_protocol!r}"
            )
        ids = self._ids(env_ids)
        if hasattr(self.base, "_reset_idx"):
            self.base._reset_idx(ids)
        self.base.scene.reset_to(snapshot.values["scene_state"], env_ids=ids, is_relative=True)
        if self.snapshot_protocol == _EXPANDED:
            manager = snapshot.values["manager_state"]
            _assign(self.base.action_manager._action, ids, manager["action"])
            _assign(self.base.action_manager._prev_action, ids, manager["previous_action"])
            _assign(self.base.episode_length_buf, ids, manager["episode_length"])
            for name, values in manager["commands"].items():
                term = self.base.command_manager.get_term(name)
                for attribute, value in values.items():
                    if attribute == "metrics":
                        for metric_name, metric_value in value.items():
                            _assign(term.metrics[metric_name], ids, metric_value)
                    else:
                        _assign(getattr(term, attribute), ids, value)
            for name, values in snapshot.values["controller_targets"].items():
                articulation = self.base.scene.articulations[name]
                if "joint_pos_target" in values:
                    articulation.set_joint_position_target_index(values["joint_pos_target"], env_ids=ids)
                if "joint_vel_target" in values:
                    articulation.set_joint_velocity_target_index(values["joint_vel_target"], env_ids=ids)
                if "joint_effort_target" in values:
                    articulation.set_joint_effort_target_index(values["joint_effort_target"], env_ids=ids)
            provider_restore = getattr(self.action_provider, "restore_state", None)
            if "policy_history" in snapshot.values and callable(provider_restore):
                provider_restore(snapshot.values["policy_history"], tuple(env_ids))
        self.base.scene.write_data_to_sim()
        self._last_observation = self.env.get_observations()

    def observe(self, env_ids: Sequence[int]) -> ObservationRecord:
        ids = self._ids(env_ids)
        observation = self._last_observation
        if observation is None:
            observation = self.env.get_observations()
        if isinstance(observation, tuple):
            observation = observation[0]
        if isinstance(observation, Mapping):
            policy = observation.get("policy", observation)
            privileged = {
                str(key): value
                for key, value in observation.items()
                if key != "policy"
            }
        else:
            policy = observation
            privileged = {}
        counters = {
            "episode_length": _take(self.base.episode_length_buf, ids),
            "sim_step_counter": np.repeat(int(getattr(self.base, "_sim_step_counter", -1)), len(env_ids)),
            "common_step_counter": np.repeat(int(getattr(self.base, "common_step_counter", -1)), len(env_ids)),
        }
        return ObservationRecord(
            scene_state=_take(self.base.scene.get_state(is_relative=True), ids),
            policy_observations={"policy": _take(policy, ids)},
            privileged_observations=_take(privileged, ids),
            task_state=self._manager_state(ids),
            controller_targets=self._controller_targets(ids),
            sensor_state=self._sensor_state(ids),
            counters=counters,
            unavailable=(
                "unexposed PhysX solver state",
                "sensor internal refresh history",
            ),
        )

    def reset(self, seed: int) -> None:
        if hasattr(self.base, "seed"):
            self.base.seed(int(seed))
        reset_result = self.env.reset(seed=int(seed))
        self._last_observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        source = self.capture((0,))
        self.restore(source, (1,))

    def action(self, step: int, source: str, env_ids: Sequence[int]) -> Any:
        import torch

        current = self.base.action_manager.action
        actions = torch.zeros_like(current)
        if self.action_provider is not None:
            proposed = self.action_provider(self._last_observation, step=step, source=source)
            proposed = torch.as_tensor(proposed, device=actions.device, dtype=actions.dtype)
            if proposed.shape == actions.shape:
                actions[:] = proposed
            elif proposed.shape == actions[0].shape:
                actions[:] = proposed
            else:
                raise ValueError(
                    f"action provider returned {tuple(proposed.shape)}, "
                    f"expected {tuple(actions.shape)} or {tuple(actions[0].shape)}"
                )
        if source in {"recorded", "exact_action", "open_loop", "zero"}:
            for target_id in env_ids[1:]:
                actions[target_id] = actions[env_ids[0]]
        return actions

    def step(self, actions: Any) -> StepRecord:
        result = self.env.step(actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("Isaac Lab environment step must return observation, reward, terminated, truncated, info")
        observation, reward, terminated, truncated, info = result
        self._last_observation = observation
        self._last_reward = reward
        self._last_terminated = terminated
        self._last_truncated = truncated
        observed = self.observe((0, 1))
        contact: dict[str, Any] = {}
        for name, values in observed.sensor_state.items():
            for field, value in values.items():
                array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
                if "force" in field and array.ndim >= 2:
                    contact[f"{name}.{field}.active"] = np.max(np.abs(array).reshape(array.shape[0], -1), axis=1) > 1e-6
        task_outputs = {"info_present": np.repeat(bool(info), 2)}
        return StepRecord(
            observation=observed,
            contact_state=contact,
            task_outputs=task_outputs,
            terminated={
                "terminated": _take(terminated, self._ids((0, 1))),
                "truncated": _take(truncated, self._ids((0, 1))),
            },
            reward={"reward": _take(reward, self._ids((0, 1)))},
        )

    def decision(self, record: TrajectoryRecord, name: str) -> bool:
        custom = self.decision_functions.get(name)
        if custom is not None:
            return bool(custom(record))
        if not record.steps:
            raise ValueError("decision requires a non-empty trajectory")

        def scalar_bool(value: Any) -> bool:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            return bool(np.asarray(value)[record.env_id])

        if name == "task_failure":
            return any(
                scalar_bool(step.terminated["terminated"])
                or scalar_bool(step.terminated["truncated"])
                for step in record.steps
            )
        if name == "task_success":
            return not self.decision(record, "task_failure")
        if name in {"collision", "sustained_grasp"}:
            contacts = []
            for step in record.steps:
                values = [
                    scalar_bool(value)
                    for value in step.contact_state.values()
                ]
                contacts.append(any(values))
            if name == "collision":
                return any(contacts)
            window = int(self.settings.get("sustained_contact_steps", 5))
            return len(contacts) >= window and all(contacts[-window:])
        raise KeyError(
            f"unknown Isaac Lab decision {name!r}; provide it through environment_factory decision_functions"
        )

    def provenance(self) -> Mapping[str, object]:
        try:
            version = importlib.metadata.version("isaaclab")
        except importlib.metadata.PackageNotFoundError:
            version = None
        captured = (
            ["scene entity state"]
            if self.snapshot_protocol == _SCENE_ONLY
            else [
                "scene entity state",
                "action buffers",
                "command state",
                "episode counter",
                "articulation targets",
                "optional policy history hook",
            ]
        )
        return {
            "adapter": "IsaacLabReplayAdapter",
            "simulator": "Isaac Lab",
            "simulator_version": version,
            "environment": self.settings.get("task", type(self.env).__name__),
            "asset_root": self.settings.get("asset_root"),
            "snapshot_protocol": self.snapshot_protocol,
            "state_components_captured": captured,
            "state_components_unavailable": [
                "PhysX warm-start and solver caches",
                "contact manifolds and broadphase caches",
                "per-environment RNG state",
                "sensor internal history",
                "observation and event history without custom hooks",
            ],
            "task_state_captured": (
                [
                    "action and previous-action buffers",
                    "command values, timing, counters, and metrics where exposed",
                    "episode-length buffer",
                ]
                if self.snapshot_protocol == _EXPANDED
                else []
            ),
            "controller_or_policy_history_captured": bool(
                self.snapshot_protocol == _EXPANDED and hasattr(self.action_provider, "capture_state")
            ),
            "random_state_handling": "seeded reset only; no supported per-environment RNG restore",
            "solver_state_availability": "unavailable through supported Isaac Lab scene APIs",
            "sensor_refresh_behavior": "read after step; sensor internals not restored",
            "unsupported_restoration_claims": [
                "complete PhysX snapshot",
                "universal deterministic replay",
                "decision fidelity outside the audited scope",
            ],
        }

    def close(self) -> None:
        if self.owns_env and hasattr(self.env, "close"):
            self.env.close()
        if self.app is not None and hasattr(self.app, "close"):
            self.app.close()
