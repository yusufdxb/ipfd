"""Isaac Lab adapter: collect a Franka pick-and-place rollout with a recovery probe.

STATUS: **runtime-exercised with a local ``isaaclab`` 4.5.22 distribution**.
Learned-policy recovery semantics are under revalidation. Historical experiments
established these narrow observations:

  * Exposed scene state round-tripped exactly in a one-step check, while a
    trajectory replay after evolved, contact-rich state diverged. The missing
    simulator or task state was not isolated.
  * In-place probing changed the continuation of the primary rollout.
  * In the measured two-environment setup, ``reset_to(..., env_ids=[1])`` did not
    change env 0's object pose at the reset boundary.

The recovery probe therefore **requires ``num_envs >= 2``**: env 0 is recorded
before probing and is never restored; env 1 receives origin-shifted snapshots.
The reset-boundary check only measures whether resetting env 1 immediately
changes env 0's object pose. It does not claim that env 0 stays static while the
vectorized simulator advances all environments during a probe.

The pure, simulator-free pieces of that mechanic (``slice_state``,
``offset_root_positions``, ``forward_fill_recovery``) live here and are unit-tested
in CI. Only the functions marked ``requires live sim`` touch Isaac Lab, and the
whole module is import-gated so the analysis layer (detectors, PoNR, metrics,
report, viz) runs without Isaac Lab or a GPU.

Reference pattern (Isaac Lab manager-based env):
    env.reset() -> (obs_dict, info)
    env.step(action) -> (obs_dict, reward, terminated, truncated, info)
    env.unwrapped.scene.get_state() / .reset_to(state, env_ids)
"""

from __future__ import annotations

import os
import warnings
from importlib import metadata
from typing import Any, Protocol

import numpy as np

from ..types import Rollout

# This fingerprints the locally validated distribution. It is not a claim that
# the same version is available from every public Isaac Lab installation path.
TESTED_ISAAC_LAB_VERSION = "4.5.22"

__all__ = [
    "TESTED_ISAAC_LAB_VERSION",
    "configure_asset_root",
    "Policy",
    "collect_rollout",
    "slice_state",
    "offset_root_positions",
    "forward_fill_recovery",
    "probe_recovery_isolated",
    "evaluate_recovery_isolated",
    "PhysicalRecoveryCheck",
    "make_pick_lift_recovery_check",
    "RECORD_CAMERA_NAME",
    "attach_record_camera",
    "disable_debug_visualizers",
    "FrameRecorder",
]

_isaac_lab_version_checked = False


def configure_asset_root(root: str | None) -> None:
    """Override Isaac asset URLs before task registration.

    Isaac Lab and Isaac Sim can be installed at different release levels. The
    public IPFD validation target uses the Isaac 4.5 asset tree, so callers can
    set this before importing ``isaaclab_tasks`` when their runtime defaults to
    another asset channel.
    """
    if not root:
        return
    import isaaclab.utils.assets as assets

    root = root.rstrip("/")
    assets.NUCLEUS_ASSET_ROOT_DIR = root
    assets.NVIDIA_NUCLEUS_DIR = f"{root}/NVIDIA"
    assets.ISAAC_NUCLEUS_DIR = f"{root}/Isaac"
    assets.ISAACLAB_NUCLEUS_DIR = f"{root}/Isaac/IsaacLab"

    # Robot configs may have captured the module-level URL before task
    # registration. Update the Franka configs used by the shipped Lift task.
    import isaaclab_assets.robots.franka as franka

    panda_path = f"{assets.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    franka.FRANKA_PANDA_CFG.spawn.usd_path = panda_path
    franka.FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path = panda_path


class Policy(Protocol):
    """Batched policy interface the adapter drives.

    Called on the wrapped env's observation and returns a batched action tensor
    (one row per env). It may optionally expose per-env ``last_entropy`` and
    ``last_embedding`` attributes (the side channels IPFD's detectors instrument;
    absent means that detector is disabled) and a ``reset(dones)`` method.
    """

    def __call__(self, obs: Any) -> Any:
        ...


def _require_isaac_lab() -> None:
    try:
        import isaaclab
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The Isaac Lab adapter requires a working Isaac Lab install and a GPU. "
            "Install Isaac Lab and run this outside CI. The analysis layer "
            "(ipfd.build_report / plot_timeline) does not need it."
        ) from exc
    _warn_if_untested_isaac_lab_version(isaaclab)


def _warn_if_untested_isaac_lab_version(isaaclab_module: Any) -> None:
    """Warn once when the installed Isaac Lab version is outside the tested target."""
    global _isaac_lab_version_checked
    if _isaac_lab_version_checked:
        return

    installed_version: str | None
    try:
        installed_version = metadata.version("isaaclab")
    except metadata.PackageNotFoundError:
        installed_version = getattr(isaaclab_module, "__version__", None)

    if installed_version is None:
        return

    _isaac_lab_version_checked = True
    expected = os.getenv("IPFD_EXPECTED_ISAAC_LAB_VERSION", TESTED_ISAAC_LAB_VERSION)
    if installed_version != expected:
        warnings.warn(
            f"IPFD expected Isaac Lab {expected}, "
            f"but {installed_version} is installed. Continuing without blocking; "
            "recovery-probe results are unverified on this version.",
            RuntimeWarning,
            stacklevel=2,
        )


# --- Pure state helpers (no simulator; unit-tested in CI) ---------------------


def _deep_clone(x: Any) -> Any:
    """Clone a nested sim-state container without importing torch.

    Handles torch tensors (duck-typed ``.clone()``), NumPy arrays (``.copy()``),
    and nested dict/list/tuple. Everything else is returned as-is.
    """
    if hasattr(x, "clone"):  # torch.Tensor
        return x.clone()
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, dict):
        return {k: _deep_clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_deep_clone(v) for v in x)
    return x


def slice_state(state: Any, idx: Any) -> Any:
    """Index every leaf tensor of a nested sim state by ``idx`` and clone.

    ``idx`` may be an int, a slice, or a tensor of env ids. Non-indexable leaves
    (scalars, strings) pass through unchanged.
    """
    is_tensor = hasattr(state, "clone") and hasattr(state, "__getitem__") and not isinstance(
        state, (dict, list, tuple)
    )
    if is_tensor:
        return state[idx].clone()
    if isinstance(state, dict):
        return {k: slice_state(v, idx) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(slice_state(v, idx) for v in state)
    return state


def offset_root_positions(state: Any, delta: Any) -> Any:
    """Shift every articulation/rigid-object root position by ``delta``.

    ``scene.get_state()`` poses are ABSOLUTE world coordinates, so a checkpoint
    captured in env 0 must be translated into the probe env's own cell (by the
    difference of ``env_origins``) before ``reset_to``, or the two arms collide.
    Joint states and velocities are origin-independent and left untouched.
    """
    s = _deep_clone(state)
    if not isinstance(s, dict):
        return s
    for grp in ("articulation", "rigid_object"):
        for _name, fields in s.get(grp, {}).items():
            if isinstance(fields, dict) and "root_pose" in fields:
                fields["root_pose"][:, :3] += delta
    return s


def forward_fill_recovery(verdicts: dict[int, bool], length: int) -> np.ndarray:
    """Expand strided recovery verdicts to a length-``length`` boolean array.

    ``verdicts`` maps a probed step index to whether recovery succeeded. Steps
    between probes inherit the most recent verdict (recovery status is assumed to
    persist until the next probe contradicts it). Steps before the first probe
    default to ``True`` (recoverable until proven otherwise), matching the
    ``point_of_no_return`` convention that PoNR is the last True->False flip.
    """
    rec = np.zeros(length, dtype=bool)
    last = True
    for t in range(length):
        if t in verdicts:
            last = bool(verdicts[t])
        rec[t] = last
    return rec


class PhysicalRecoveryCheck:
    """Conservative pick/lift recovery predicate.

    Height alone is insufficient: an airborne or teleported object can cross a
    lift threshold without being grasped.  This predicate requires the object to
    be above the table, inside a reachable workspace, and (when available) close
    to the end effector with a closed gripper.  The condition must hold for
    ``sustain_steps`` consecutive simulator steps.  Missing required signals are
    treated as unknown and therefore return ``False`` rather than claiming
    recovery.
    """

    def __init__(
        self,
        *,
        object_name: str = "object",
        robot_name: str = "robot",
        ee_body_name: str = "panda_hand",
        ee_body_index: int | None = None,
        workspace_radius: float = 0.85,
        max_ee_distance: float = 0.15,
        gripper_joint_indices: tuple[int, int] = (-2, -1),
        gripper_close_threshold: float = 0.07,
        sustain_steps: int = 8,
    ):
        if not object_name or not robot_name or not ee_body_name:
            raise ValueError("object_name, robot_name, and ee_body_name must be non-empty")
        if isinstance(sustain_steps, bool) or not isinstance(sustain_steps, int) or sustain_steps < 1:
            raise ValueError("sustain_steps must be an integer >= 1")
        if ee_body_index is not None and (
            isinstance(ee_body_index, bool) or not isinstance(ee_body_index, int) or ee_body_index < 0
        ):
            raise ValueError("ee_body_index must be a non-negative integer or None")
        if len(gripper_joint_indices) != 2 or not all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in gripper_joint_indices
        ):
            raise ValueError("gripper_joint_indices must contain exactly two integer indices")
        numeric = {
            "workspace_radius": workspace_radius,
            "max_ee_distance": max_ee_distance,
            "gripper_close_threshold": gripper_close_threshold,
        }
        for name, value in numeric.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite and > 0") from exc
            if isinstance(value, bool) or not np.isfinite(numeric_value) or numeric_value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        self.object_name = object_name
        self.robot_name = robot_name
        self.ee_body_name = ee_body_name
        self.ee_body_index = ee_body_index
        self.workspace_radius = float(workspace_radius)
        self.max_ee_distance = float(max_ee_distance)
        self.gripper_joint_indices = tuple(gripper_joint_indices)
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.sustain_steps = int(sustain_steps)
        self._counts: dict[int, int] = {}
        self._resolved_ee_body_index: int | None = ee_body_index

    def reset(self, env_idx: int) -> None:
        self._counts.pop(int(env_idx), None)

    @staticmethod
    def _tensor(value: Any) -> Any:
        if hasattr(value, "detach"):
            return value.detach()
        try:
            import warp as wp

            return wp.to_torch(value).detach()
        except (ImportError, RuntimeError, TypeError):
            return np.asarray(value)

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value, dtype=float)

    def _resolve_ee_index(self, robot: Any) -> int | None:
        if self._resolved_ee_body_index is not None:
            return self._resolved_ee_body_index
        if not hasattr(robot, "find_bodies"):
            return None
        indices, _names = robot.find_bodies(self.ee_body_name, preserve_order=True)
        if len(indices) != 1:
            return None
        self._resolved_ee_body_index = int(indices[0])
        return self._resolved_ee_body_index

    def __call__(self, env: Any, env_idx: int, rest_height: float,
                 lift_threshold: float) -> bool:
        try:
            scene = env.unwrapped.scene
            object_entity = scene[self.object_name]
            robot_entity = scene[self.robot_name]
            obj_pos = self._tensor(object_entity.data.root_pos_w)[env_idx, :3]
            origins = getattr(scene, "env_origins", None)
            if origins is not None:
                origin = self._tensor(origins)[env_idx]
                obj_pos = obj_pos - origin
            z = float(obj_pos[2].item() if hasattr(obj_pos[2], "item") else obj_pos[2])
            if z - float(rest_height) <= float(lift_threshold):
                self.reset(env_idx)
                return False
            # Reject obvious teleport/debris states outside the robot workspace.
            if float(np.linalg.norm(self._numpy(obj_pos[:2]))) > self.workspace_radius:
                self.reset(env_idx)
                return False
            robot_data = robot_entity.data
            ee_index = self._resolve_ee_index(robot_entity)
            if ee_index is None or not hasattr(robot_data, "body_pos_w"):
                self.reset(env_idx)
                return False
            ee = self._tensor(robot_data.body_pos_w)[env_idx, ee_index, :3]
            if origins is not None:
                ee = ee - origin
            if float(np.linalg.norm(self._numpy(ee - obj_pos))) > self.max_ee_distance:
                self.reset(env_idx)
                return False
            if not hasattr(robot_data, "joint_pos"):
                self.reset(env_idx)
                return False
            joints = self._numpy(self._tensor(robot_data.joint_pos)[env_idx])
            fingers = joints[list(self.gripper_joint_indices)]
            width = float(np.abs(fingers).sum())
            if width >= self.gripper_close_threshold:
                self.reset(env_idx)
                return False
        except (AttributeError, KeyError, IndexError, RuntimeError, TypeError, ValueError):
            self.reset(env_idx)
            return False
        self._counts[env_idx] = self._counts.get(env_idx, 0) + 1
        return self._counts[env_idx] >= self.sustain_steps


def make_pick_lift_recovery_check(**kwargs: Any) -> PhysicalRecoveryCheck:
    """Return the conservative default recovery contract for pick-and-lift."""
    return PhysicalRecoveryCheck(**kwargs)


# --- Simulator touchpoints (thin; require a live Isaac Lab GPU env) -----------


def _extract_obs(obs: Any, obs_key: str, env_idx: int = 0) -> np.ndarray:
    """Flatten one env's observation group to a float64 vector (runtime API check)."""
    group = obs[obs_key] if isinstance(obs, dict) else obs
    arr = group.detach().cpu().numpy() if hasattr(group, "detach") else np.asarray(group)
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr[env_idx]
    return arr.reshape(-1).astype(np.float64)


def probe_recovery_isolated(
    env: Any,
    saved_state: Any,
    recovery_policy: Any,
    *,
    recovered: Any,
    primary_env: int = 0,
    probe_env: int = 1,
    budget: int = 90,
    locality: dict[str, Any] | None = None,
    reset_recovered: Any | None = None,
) -> bool:  # pragma: no cover - requires live sim
    """One env-isolated recovery probe.

    Origin-shift ``saved_state`` into ``probe_env``, ``reset_to`` **only** that env,
    run the batched ``recovery_policy`` there for ``budget`` steps, and report
    whether ``recovered(env, probe_env)`` ever became true. Primary rollout data
    was recorded before this pass; the primary environment itself still advances
    during the batched recovery rollout.

    Args:
        recovery_policy: Batched callable ``policy(obs) -> actions`` (all envs).
        recovered: ``callable(env, probe_env) -> bool`` success test (e.g. a lift).
        locality: If given (a ``{"max", "n"}`` dict), the probe records the primary
            object pose immediately before and after ``reset_to`` and accumulates
            the max reset-boundary delta. Requires ``num_envs >= 2``.
    """
    import torch
    import warp as wp

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"budget must be an integer >= 1, got {budget!r}")
    scene = env.unwrapped.scene
    delta = (scene.env_origins[probe_env] - scene.env_origins[primary_env]).detach()
    state_probe = offset_root_positions(saved_state, delta)

    pose_before = (
        wp.to_torch(scene["object"].data.root_pose_w)[primary_env].detach().clone()
        if locality is not None else None
    )
    env_ids = torch.tensor([probe_env], device=env.unwrapped.device, dtype=torch.long)
    scene.reset_to(state_probe, env_ids)
    if reset_recovered is not None:
        reset_recovered(probe_env)
    if locality is not None:
        pose_after = wp.to_torch(scene["object"].data.root_pose_w)[primary_env].detach()
        locality["max"] = max(locality["max"], float((pose_after - pose_before).abs().max().item()))
        locality["n"] = locality.get("n", 0) + 1

    if hasattr(env.unwrapped, "episode_length_buf"):
        env.unwrapped.episode_length_buf[probe_env] = 0  # avoid a timeout auto-reset mid-probe

    if hasattr(recovery_policy, "reset"):
        reset_mask = torch.ones(
            int(env.unwrapped.num_envs),
            device=env.unwrapped.device,
            dtype=torch.bool,
        )
        recovery_policy.reset(reset_mask)

    obs = env.get_observations()
    for _ in range(budget):
        actions = recovery_policy(obs)
        obs, _reward, dones, _info = env.step(actions)
        if recovered(env, probe_env):
            return True
        if bool(dones[probe_env].item()):
            return False
        if hasattr(recovery_policy, "reset"):
            recovery_policy.reset(dones)
    return False


def evaluate_recovery_isolated(
    env: Any,
    states: dict[int, Any],
    recovery_policy: Any,
    *,
    recovered: Any,
    primary_env: int = 0,
    probe_env: int = 1,
    budget: int = 90,
    repeats: int = 1,
    min_false_fraction: float = 0.8,
    reset_recovered: Any | None = None,
) -> tuple[dict[int, bool], dict[str, Any]]:  # pragma: no cover - requires live sim
    """Evaluate recovery from each saved checkpoint via env isolation (Pass 2).

    Returns ``(verdicts, locality)``: ``verdicts[step]`` is whether recovery
    succeeded from the state saved at ``step``, and ``locality`` reports the live
    primary-integrity assertion (``max`` env-0 pose delta, ``n`` resets).
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be an integer >= 1")
    if isinstance(min_false_fraction, bool):
        raise ValueError("min_false_fraction must be in [0.5, 1.0]")
    try:
        min_false_fraction = float(min_false_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_false_fraction must be in [0.5, 1.0]") from exc
    if not np.isfinite(min_false_fraction) or not (
        0.5 <= min_false_fraction <= 1.0
    ):
        raise ValueError("min_false_fraction must be in [0.5, 1.0]")
    locality: dict[str, Any] = {"max": 0.0, "n": 0}
    verdicts: dict[int, bool] = {}
    false_fractions: list[float] = []
    raw_samples: dict[str, list[bool]] = {}
    false_fraction_by_step: dict[str, float] = {}
    for step, state in sorted(states.items()):
        samples = [
            probe_recovery_isolated(
                env, state, recovery_policy, recovered=recovered,
                primary_env=primary_env, probe_env=probe_env, budget=budget, locality=locality,
                reset_recovered=reset_recovered,
            )
            for _ in range(repeats)
        ]
        false_fraction = sum(not sample for sample in samples) / repeats
        false_fractions.append(false_fraction)
        raw_samples[str(step)] = samples
        false_fraction_by_step[str(step)] = false_fraction
        verdicts[step] = false_fraction < min_false_fraction
    locality["probe_repeats"] = repeats
    locality["max_false_fraction"] = max(false_fractions, default=0.0)
    locality["raw_probe_verdicts"] = raw_samples
    locality["false_fraction_by_step"] = false_fraction_by_step
    return verdicts, locality


def collect_rollout(
    env: Any,
    policy: Any,
    *,
    object_height: Any,
    rest_height: float,
    lift_threshold: float = 0.06,
    recovery_check: Any | None = None,
    recovery_policy: Any | None = None,
    primary_env: int = 0,
    probe_env: int = 1,
    max_steps: int = 220,
    probe_stride: int = 8,
    probe_budget: int = 90,
    probe_repeats: int = 3,
    probe_min_false_fraction: float = 0.8,
    on_step: Any | None = None,
    obs_key: str = "policy",
    seed: int | None = None,
    meta: dict | None = None,
) -> Rollout:  # pragma: no cover - requires live sim
    """Roll out a batched ``policy`` in ``primary_env`` and build a :class:`Rollout`.

    The packaged implementation of the env-isolated recovery-probe design exercised
    end-to-end on a trained rsl_rl policy in ``scripts/verify_learned_policy.py``
    (local ``isaaclab`` distribution 4.5.22). It is a **decoupled two-pass**: record the primary rollout in
    ``primary_env`` before probing (and never ``reset_to`` it), then, if
    ``recovery_policy`` is given,
    evaluate recovery from the saved checkpoints in ``probe_env`` via
    :func:`evaluate_recovery_isolated`. The single-env interleaved design that
    changed the historical continuation is not used.

    Args:
        env: Isaac Lab manager-based RL env, wrapped so ``get_observations()`` and
            ``step(actions)`` take/return batched tensors. ``num_envs >= 2`` when
            ``recovery_policy`` is set.
        policy: Batched callable ``policy(obs) -> actions``. May expose per-env
            ``last_entropy`` / ``last_embedding`` (to feed the detectors) and
            ``reset(dones)``; a missing signal simply disables its detector.
        object_height: ``callable(env, env_idx) -> float`` for the tracked object.
        rest_height: Settled resting height of the object [m] (caller-measured).
        lift_threshold: Rise above ``rest_height`` [m] counted as a lift / recovery.
        recovery_check: Optional ``callable(env, env_idx, rest_height, lift_threshold)``
            returning a physically meaningful recovery verdict. If omitted, the
            legacy height-only check is used for backwards compatibility.
        recovery_policy: Batched recovery oracle for the probe; ``None`` skips PoNR.
        on_step: Optional ``callable(step, env, actions)`` invoked after the policy
            acts and before ``env.step``, the hook a caller uses to inject a fault
            (edit ``actions`` in place, or write the sim). The recorded action is the
            policy's *pre-injection* action, so the detectors see the true policy.

    Returns:
        A fully populated :class:`Rollout` (with ``recovery_success`` when probed).
    """
    unwrapped = getattr(env, "unwrapped", env)
    n_envs = int(getattr(unwrapped, "num_envs", 1))
    for name, int_value in (("max_steps", max_steps), ("probe_budget", probe_budget)):
        if isinstance(int_value, bool) or not isinstance(int_value, int) or int_value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {int_value!r}.")
    for name, float_value in (("rest_height", rest_height), ("lift_threshold", lift_threshold)):
        try:
            numeric_value = float(float_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite, got {float_value!r}.") from exc
        if isinstance(float_value, bool) or not np.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite, got {float_value!r}.")
    rest_height = float(rest_height)
    lift_threshold = float(lift_threshold)
    if lift_threshold <= 0:
        raise ValueError(f"lift_threshold must be > 0, got {lift_threshold!r}.")
    if recovery_policy is not None:
        if n_envs < 2:
            raise ValueError(
                "The recovery probe requires num_envs >= 2 (environment separation): "
                "env 0 is recorded before probing and env 1 is the probe cell. Historical "
                "single-env continuations diverged after reset_to. Create the "
                "env with num_envs>=2, or omit recovery_policy to skip PoNR."
            )
        if primary_env == probe_env:
            raise ValueError(
                f"primary_env and probe_env must be distinct for environment isolation, got {primary_env}."
            )
        if not (0 <= primary_env < n_envs):
            raise ValueError(f"primary_env {primary_env} out of range [0, {n_envs}).")
        if not (0 <= probe_env < n_envs):
            raise ValueError(f"probe_env {probe_env} out of range [0, {n_envs}).")
        if probe_stride < 1:
            raise ValueError(f"probe_stride must be >= 1, got {probe_stride!r}.")
        if (
            isinstance(probe_repeats, bool)
            or not isinstance(probe_repeats, int)
            or probe_repeats < 1
        ):
            raise ValueError(f"probe_repeats must be an integer >= 1, got {probe_repeats!r}.")
        if isinstance(probe_min_false_fraction, bool):
            raise ValueError("probe_min_false_fraction must be in [0.5, 1.0].")
        try:
            probe_min_false_fraction = float(probe_min_false_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe_min_false_fraction must be in [0.5, 1.0].") from exc
        if not np.isfinite(probe_min_false_fraction) or not (
            0.5 <= probe_min_false_fraction <= 1.0
        ):
            raise ValueError("probe_min_false_fraction must be in [0.5, 1.0].")

    _require_isaac_lab()

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    ent_list: list[float] = []
    emb_list: list[np.ndarray] = []

    obs = env.get_observations()
    states: dict[int, Any] = {}
    t_failure: int | None = None

    for step in range(max_steps):
        if recovery_policy is not None and step % probe_stride == 0:
            states[step] = slice_state(
                unwrapped.scene.get_state(), slice(primary_env, primary_env + 1)
            )

        actions = policy(obs)
        obs_list.append(np.asarray(obs[obs_key][primary_env].detach().cpu(), dtype=np.float64).reshape(-1))
        act_list.append(np.asarray(actions[primary_env].detach().cpu(), dtype=np.float64).reshape(-1))
        ent = getattr(policy, "last_entropy", None)
        emb = getattr(policy, "last_embedding", None)
        if ent is not None:
            ent_list.append(float(ent[primary_env]))
        if emb is not None:
            emb_list.append(np.asarray(emb[primary_env], dtype=np.float64).reshape(-1))

        if on_step is not None:
            on_step(step, env, actions)

        obs, _reward, dones, _info = env.step(actions)
        if hasattr(policy, "reset"):
            policy.reset(dones)
        if bool(dones[primary_env].item()):
            t_failure = step
            break

    T = len(obs_list)
    success = False
    if t_failure is None:
        success = (object_height(env, primary_env) - rest_height) > lift_threshold
        if not success:
            t_failure = T - 1

    recovery = None
    locality: dict[str, Any] = {"max": 0.0, "n": 0}
    if recovery_policy is not None and states:
        env.reset()  # clean base before probing; probe reset_to overrides probe_env

        def recovered(e: Any, i: int) -> bool:
            if recovery_check is not None:
                return bool(recovery_check(e, i, rest_height, lift_threshold))
            return (object_height(e, i) - rest_height) > lift_threshold

        verdicts, locality = evaluate_recovery_isolated(
            env, states, recovery_policy, recovered=recovered,
            primary_env=primary_env, probe_env=probe_env, budget=probe_budget,
            repeats=probe_repeats, min_false_fraction=probe_min_false_fraction,
            reset_recovered=getattr(recovery_check, "reset", None),
        )
        recovery = forward_fill_recovery(verdicts, T)

    full_meta: dict[str, Any] = {
        "source": "isaac_lab",
        "robot": "franka",
        "task": "pick_place",
        "probe": "env_isolated" if recovery_policy is not None else "none",
        "reset_boundary_primary_pose_delta_m": locality["max"],
        # Historical fixtures use this name. It is an alias for the narrower
        # reset-boundary measurement, not an end-to-end integrity proof.
        "primary_integrity_max_delta": locality["max"],
        "probe_resets": locality["n"],
        "probe_stride": probe_stride,
        "probe_budget": probe_budget,
        "probe_repeats": probe_repeats,
        "probe_min_false_fraction": probe_min_false_fraction,
        "probe_max_false_fraction": locality.get("max_false_fraction", 0.0),
        "raw_probe_verdicts": locality.get("raw_probe_verdicts", {}),
        "probe_false_fraction_by_step": locality.get("false_fraction_by_step", {}),
        "rest_height": rest_height,
        "lift_threshold": lift_threshold,
        "recovery_predicate": "custom" if recovery_check is not None else "height_only_legacy",
    }
    if meta:
        full_meta.update(meta)

    return Rollout(
        observations=np.asarray(obs_list),
        actions=np.asarray(act_list),
        entropy=np.asarray(ent_list) if len(ent_list) == T and ent_list else None,
        embeddings=np.asarray(emb_list) if len(emb_list) == T and emb_list else None,
        success=success,
        t_failure=None if success else t_failure,
        recovery_success=recovery,
        dt=float(getattr(unwrapped, "step_dt", 1.0 / 60.0)),
        seed=seed,
        meta=full_meta,
    )


# ---------------------------------------------------------------------------
# Frame recording (requires live sim)
#
# The viewport render product returns all-zero frames under headless offscreen
# rendering on the validated runtime: env.render() yields a correctly shaped
# uint8 array whose every pixel is 0, so gymnasium's RecordVideo wrapper writes
# a black video. An explicit Camera sensor owns its own render product and does
# return real pixels, so recording goes through the scene instead of the
# viewport. Recording still requires the app to be launched with
# ``enable_cameras=True``.
# ---------------------------------------------------------------------------

RECORD_CAMERA_NAME = "ipfd_record_cam"


def disable_debug_visualizers(env_cfg: Any) -> None:
    """Turn off marker overlays (goal-pose arrows, frame axes) before ``gym.make``.

    The Lift task draws command and frame visualizers into the scene. They are
    useful when debugging interactively and are clutter in a recording.
    """
    for group_name in ("commands", "scene"):
        group = getattr(env_cfg, group_name, None)
        if group is None:
            continue
        for attr in dir(group):
            if attr.startswith("_"):
                continue
            term: Any = getattr(group, attr, None)
            if term is not None and hasattr(term, "debug_vis"):
                term.debug_vis = False


def attach_record_camera(
    env_cfg: Any,
    *,
    width: int = 1280,
    height: int = 720,
    name: str = RECORD_CAMERA_NAME,
) -> None:
    """Add an RGB Camera sensor to ``env_cfg``'s scene. Call before ``gym.make``.

    requires live sim (imports Isaac Lab).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    setattr(
        env_cfg.scene,
        name,
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            update_period=0.0,
            width=width,
            height=height,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 20.0),
            ),
            # Aimed explicitly after reset via set_world_poses_from_view.
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), convention="world"),
        ),
    )


class FrameRecorder:
    """Stream RGB frames from the record camera to a PNG sequence.

    Frames are written as they are captured rather than accumulated, because a
    few hundred 720p frames held in memory is hundreds of megabytes.

    requires live sim.
    """

    def __init__(
        self,
        outdir: str,
        *,
        env_index: int = 0,
        name: str = RECORD_CAMERA_NAME,
        eye: tuple[float, float, float] = (1.3, 0.9, 0.75),
        target: tuple[float, float, float] = (0.45, 0.0, 0.15),
    ) -> None:
        self.outdir = outdir
        self.env_index = env_index
        self.name = name
        self.eye = eye
        self.target = target
        self.count = 0
        os.makedirs(outdir, exist_ok=True)

    def aim(self, env: Any) -> None:
        """Point the camera using eye/target, which avoids hand-built quaternions."""
        import torch

        scene = env.unwrapped.scene
        cam = scene[self.name]
        origins = scene.env_origins
        eyes = origins + torch.tensor(self.eye, device=origins.device)
        targets = origins + torch.tensor(self.target, device=origins.device)
        cam.set_world_poses_from_view(eyes, targets)

    def capture(self, env: Any) -> bool:
        """Write one frame. Returns False when the camera produced no pixels yet."""
        import imageio.v3 as iio

        cam = env.unwrapped.scene[self.name]
        rgb = cam.data.output.get("rgb") if hasattr(cam.data.output, "get") else None
        if rgb is None:
            return False
        arr = rgb[self.env_index, ..., :3].detach().cpu().numpy().astype(np.uint8)
        if not arr.size:
            return False
        iio.imwrite(os.path.join(self.outdir, f"frame_{self.count:05d}.png"), arr)
        self.count += 1
        return True
