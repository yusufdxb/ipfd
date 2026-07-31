"""Validate the learned-policy recovery oracle against uninterrupted execution.

The primary environment is never restored. Checkpoints from its deterministic
Franka Lift rollouts are origin-shifted into two probe cells. One probe receives
the uninterrupted action suffix and the other re-runs the deterministic policy.

Run a short control check before the decisive experiment:

    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src \
      ~/Sim/isaac-sim-venv/bin/python \
      scripts/validate_recovery_oracle.py --headless --use-pretrained \
      --repeats 2 --output-dir /tmp/ipfd-oracle-smoke
"""

from __future__ import annotations

# ruff: noqa: E402, I001
import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--checkpoint", default="")
parser.add_argument("--use-pretrained", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--seeds",
    default="",
    help="Comma-separated seeds. When omitted, --seed preserves the legacy single-seed run.",
)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--probe-budget", type=int, default=90)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--lift-threshold", type=float, default=0.06)
parser.add_argument("--reach-push", type=float, default=1.0)
parser.add_argument(
    "--horizons",
    default="1,3,5,10",
    help="Comma-separated checkpoint offsets before the nominal lift boundary.",
)
parser.add_argument(
    "--disturbances",
    default="teleport,gripper_interrupt",
    help="Comma-separated subset of teleport and gripper_interrupt.",
)
parser.add_argument("--gripper-open", type=float, default=1.0)
parser.add_argument("--gripper-interrupt-steps", type=int, default=8)
parser.add_argument(
    "--restoration-ablations",
    default="no_action_history,no_manager_state",
    help="Comma-separated one-factor ablations, or empty to disable.",
)
parser.add_argument(
    "--ablation-checkpoints",
    default="horizon_1,horizon_5",
    help="Comma-separated checkpoint names ablated on the first seed and repeat.",
)
parser.add_argument(
    "--asset-root",
    default="",
    help="Optional Isaac asset root override; empty uses Isaac Lab's native resolver.",
)
parser.add_argument(
    "--checkpoint-classes",
    default=(
        "horizon_1,horizon_3,horizon_5,horizon_10,pre_manipulation,"
        "contact_onset,mid_contact,post_contact_onset,gripper_onset,"
        "gripper_mid,gripper_post,teleport_post"
    ),
    help="Comma-separated checkpoint names; unavailable names are reported per case.",
)
parser.add_argument("--output-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.num_envs < 4:
    parser.error("--num-envs must be at least 4")

app = AppLauncher(args).app

import importlib.metadata as metadata

import gymnasium as gym
import numpy as np
import torch
import warp as wp

if args.asset_root:
    import isaaclab.utils.assets as assets
    import isaaclab_assets.robots.franka as franka

    asset_root = args.asset_root.rstrip("/")
    assets.NUCLEUS_ASSET_ROOT_DIR = asset_root
    assets.NVIDIA_NUCLEUS_DIR = f"{asset_root}/NVIDIA"
    assets.ISAAC_NUCLEUS_DIR = f"{asset_root}/Isaac"
    assets.ISAACLAB_NUCLEUS_DIR = f"{asset_root}/Isaac/IsaacLab"
    panda_path = f"{assets.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    franka.FRANKA_PANDA_CFG.spawn.usd_path = panda_path
    franka.FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path = panda_path

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))
from ipfd.adapters.isaac_lab import offset_root_positions, slice_state
from ipfd.oracle_equivalence import compare_traces, terminal_outcome
from ipfd.oracles.rsl_rl_policy import LearnedPolicy

PRIMARY = 0
EXACT = 1
POLICY = 2
DISTURBANCE_NAMES = {"teleport", "gripper_interrupt"}
RESTORATION_MODES = {"full", "no_action_history", "no_manager_state"}


def log(message: str) -> None:
    print(f"[oracle-equivalence] {message}", flush=True)


def clone_tree(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_tree(item) for item in value)
    return value


@dataclass
class Checkpoint:
    name: str
    state: Any
    observation: np.ndarray
    action: Any
    prev_action: Any
    commands: dict[str, dict[str, Any]]
    episode_length: int
    point_index: int
    rest_height: float
    rise: float
    finger_aperture: float
    task_phase: str
    disturbance: str
    disturbance_remaining: int
    horizon_to_boundary: int | None = None


def task_phase(rise: float, finger_aperture: float, lift_threshold: float) -> str:
    if rise <= 0.001 and finger_aperture >= 0.055:
        return "pre_manipulation"
    if rise <= 0.005:
        return "mid_contact"
    if rise < lift_threshold:
        return "post_contact"
    return "lifted"


def object_height(env: Any, env_id: int) -> float:
    scene = env.unwrapped.scene
    z_world = float(wp.to_torch(scene["object"].data.root_pos_w)[env_id, 2].item())
    return z_world - float(scene.env_origins[env_id, 2].item())


def live_sample(env: Any, observation: Any, env_id: int) -> dict[str, np.ndarray | float]:
    scene = env.unwrapped.scene
    robot = scene["robot"].data
    obj = scene["object"].data
    object_pose = wp.to_torch(obj.root_pose_w)[env_id].detach().clone()
    object_pose[:3] -= scene.env_origins[env_id]
    return {
        "observations": observation["policy"][env_id].detach().cpu().numpy().astype(np.float64),
        "joint_pos": wp.to_torch(robot.joint_pos)[env_id].detach().cpu().numpy().astype(np.float64),
        "joint_vel": wp.to_torch(robot.joint_vel)[env_id].detach().cpu().numpy().astype(np.float64),
        "object_pose": object_pose.detach().cpu().numpy().astype(np.float64),
        "object_vel": wp.to_torch(obj.root_vel_w)[env_id].detach().cpu().numpy().astype(np.float64),
        "height": object_height(env, env_id),
    }


def capture_checkpoint(
    env: Any,
    observation: Any,
    *,
    name: str,
    point_index: int,
    rest_height: float,
    lift_threshold: float,
    disturbance: str = "none",
    disturbance_remaining: int = 0,
    horizon_to_boundary: int | None = None,
) -> Checkpoint:
    base = env.unwrapped
    commands = {}
    for term_name in base.command_manager.active_terms:
        term = base.command_manager.get_term(term_name)
        commands[term_name] = {
            "command": term.command[PRIMARY : PRIMARY + 1].detach().clone(),
            "time_left": term.time_left[PRIMARY : PRIMARY + 1].detach().clone(),
            "command_counter": term.command_counter[PRIMARY : PRIMARY + 1].detach().clone(),
        }
    joints = wp.to_torch(base.scene["robot"].data.joint_pos)[PRIMARY]
    rise = object_height(env, PRIMARY) - rest_height
    aperture = float(joints[-2:].sum().item())
    return Checkpoint(
        name=name,
        state=slice_state(base.scene.get_state(), slice(PRIMARY, PRIMARY + 1)),
        observation=observation["policy"][PRIMARY].detach().cpu().numpy().astype(np.float64),
        action=base.action_manager.action[PRIMARY : PRIMARY + 1].detach().clone(),
        prev_action=base.action_manager.prev_action[PRIMARY : PRIMARY + 1].detach().clone(),
        commands=commands,
        episode_length=int(base.episode_length_buf[PRIMARY].item()),
        point_index=point_index,
        rest_height=rest_height,
        rise=rise,
        finger_aperture=aperture,
        task_phase=task_phase(rise, aperture, lift_threshold),
        disturbance=disturbance,
        disturbance_remaining=disturbance_remaining,
        horizon_to_boundary=horizon_to_boundary,
    )


def empty_records() -> dict[str, list[Any]]:
    return {
        "observations": [], "joint_pos": [], "joint_vel": [], "object_pose": [],
        "object_vel": [], "height": [], "actions": [], "rewards": [], "dones": [],
    }


def append_sample(records: dict[str, list[Any]], sample: dict[str, Any]) -> None:
    for key in ("observations", "joint_pos", "joint_vel", "object_pose", "object_vel", "height"):
        records[key].append(sample[key])


def clipped_actions(env: Any, actions: Any) -> Any:
    limit = getattr(env, "clip_actions", None)
    return torch.clamp(actions, -limit, limit) if limit is not None else actions


def reset_seeded(env: Any, seed: int) -> Any:
    env.seed(seed)
    observation, _ = env.reset()
    return observation


def make_trace(
    records: dict[str, list[Any]],
    *,
    start: int,
    budget: int,
    rest_height: float,
    lift_threshold: float,
) -> dict[str, Any]:
    end = min(start + budget, len(records["actions"]))
    trace = {key: np.asarray(values[start:end]) for key, values in records.items()}
    terminated = bool(np.asarray(trace["dones"], dtype=bool).any())
    success, outcome_step = terminal_outcome(
        trace["height"], rest_height=rest_height, lift_threshold=lift_threshold, terminated=terminated
    )
    trace.update({"success": success, "outcome_step": outcome_step, "horizon": len(trace["actions"])})
    return trace


def collect_primary(
    env: Any,
    policy: Any,
    *,
    disturbance: str,
    seed: int,
    budget: int,
    lift_threshold: float,
    reach_push: float,
    horizons: tuple[int, ...],
    gripper_open: float,
    gripper_interrupt_steps: int,
) -> tuple[dict[str, Any], dict[str, Checkpoint], int | None]:
    observation = reset_seeded(env, seed)
    rest_height = object_height(env, PRIMARY)
    records = empty_records()
    checkpoints: dict[str, Checkpoint] = {}
    history: dict[int, Checkpoint] = {}
    boundary_step = None
    target_steps = budget
    max_steps = max(220, budget + 100)
    interrupt_remaining = 0

    def capture(
        name: str,
        point_index: int,
        current_observation: Any,
        *,
        remaining: int = 0,
        horizon: int | None = None,
    ) -> Checkpoint:
        return capture_checkpoint(
            env,
            current_observation,
            name=name,
            point_index=point_index,
            rest_height=rest_height,
            lift_threshold=lift_threshold,
            disturbance=disturbance,
            disturbance_remaining=remaining,
            horizon_to_boundary=horizon,
        )

    for _ in range(max_steps):
        point_index = len(records["actions"])
        history[point_index] = capture(
            f"step_{point_index}",
            point_index,
            observation,
            remaining=interrupt_remaining,
        )
        append_sample(records, live_sample(env, observation, PRIMARY))
        actions = clipped_actions(env, policy(observation))
        interrupt_applied = disturbance == "gripper_interrupt" and interrupt_remaining > 0
        if interrupt_applied:
            actions[PRIMARY, -1] = gripper_open
            interrupt_remaining -= 1
        records["actions"].append(actions[PRIMARY].detach().cpu().numpy().astype(np.float64))
        next_observation, reward, dones, _ = env.step(actions)
        records["rewards"].append(float(reward[PRIMARY].item()))
        done = bool(dones[PRIMARY].item())
        records["dones"].append(done)
        if done:
            break

        point_index = len(records["actions"])
        lifted = object_height(env, PRIMARY) > rest_height + lift_threshold
        if boundary_step is None and lifted:
            boundary_step = point_index
            if disturbance == "teleport":
                obj = env.unwrapped.scene["object"]
                pose = wp.to_torch(obj.data.root_pose_w)[PRIMARY : PRIMARY + 1].detach().clone()
                pose[0, 0] += reach_push
                obj.write_root_pose_to_sim(pose, env_ids=torch.tensor([PRIMARY], device=env.unwrapped.device))
                next_observation = env.get_observations()
                checkpoints["teleport_post"] = capture(
                    "teleport_post", point_index, next_observation, horizon=0
                )
            elif disturbance == "gripper_interrupt":
                interrupt_remaining = gripper_interrupt_steps
                checkpoints["gripper_onset"] = capture(
                    "gripper_onset",
                    point_index,
                    next_observation,
                    remaining=interrupt_remaining,
                    horizon=0,
                )
            else:
                checkpoints["post_contact"] = capture(
                    "post_contact", point_index, next_observation, horizon=0
                )
            target_steps = point_index + budget

        if interrupt_applied:
            executed = gripper_interrupt_steps - interrupt_remaining
            if executed == max(1, gripper_interrupt_steps // 2):
                checkpoints["gripper_mid"] = capture(
                    "gripper_mid",
                    point_index,
                    next_observation,
                    remaining=interrupt_remaining,
                    horizon=boundary_step - point_index if boundary_step is not None else None,
                )
            if interrupt_remaining == 0:
                checkpoints["gripper_post"] = capture(
                    "gripper_post",
                    point_index,
                    next_observation,
                    remaining=0,
                    horizon=boundary_step - point_index if boundary_step is not None else None,
                )

        observation = next_observation
        if boundary_step is not None and len(records["actions"]) >= target_steps:
            break

    if boundary_step is not None and disturbance == "none":
        for horizon in horizons:
            point_index = boundary_step - horizon
            if point_index in history:
                checkpoints[f"horizon_{horizon}"] = replace(
                    history[point_index],
                    name=f"horizon_{horizon}",
                    horizon_to_boundary=horizon,
                )
        if 0 in history:
            checkpoints["pre_manipulation"] = replace(
                history[0],
                name="pre_manipulation",
                horizon_to_boundary=boundary_step,
            )
        before_boundary = [history[index] for index in sorted(history) if index < boundary_step]
        contact = [item for item in before_boundary if item.task_phase == "mid_contact"]
        if contact:
            checkpoints["contact_onset"] = replace(
                contact[0],
                name="contact_onset",
                horizon_to_boundary=boundary_step - contact[0].point_index,
            )
            middle = contact[len(contact) // 2]
            checkpoints["mid_contact"] = replace(
                middle,
                name="mid_contact",
                horizon_to_boundary=boundary_step - middle.point_index,
            )
        post_contact = [item for item in before_boundary if item.task_phase == "post_contact"]
        if post_contact:
            checkpoints["post_contact_onset"] = replace(
                post_contact[0],
                name="post_contact_onset",
                horizon_to_boundary=boundary_step - post_contact[0].point_index,
            )

    if records["height"]:
        gripper = np.asarray(records["actions"])[:, -1]
        log(
            f"primary disturbance={disturbance} seed={seed} steps={len(records['actions'])} "
            f"rest={rest_height:.4f} max_height={max(records['height']):.4f} "
            f"gripper_range=[{gripper.min():.3f},{gripper.max():.3f}] boundary={boundary_step}"
        )
    return records, checkpoints, boundary_step


def restore_probe(
    env: Any,
    checkpoint: Checkpoint,
    env_id: int,
    restoration_mode: str = "full",
) -> float:
    base = env.unwrapped
    scene = base.scene
    delta = (scene.env_origins[env_id] - scene.env_origins[PRIMARY]).detach()
    shifted = offset_root_positions(checkpoint.state, delta)
    primary_before = wp.to_torch(scene["object"].data.root_pose_w)[PRIMARY].detach().clone()
    ids = torch.tensor([env_id], device=base.device, dtype=torch.long)
    scene.reset_to(shifted, ids)
    primary_after = wp.to_torch(scene["object"].data.root_pose_w)[PRIMARY].detach()

    if restoration_mode not in RESTORATION_MODES:
        raise ValueError(f"unknown restoration mode: {restoration_mode}")
    if restoration_mode != "no_action_history":
        base.action_manager._action[env_id : env_id + 1] = checkpoint.action
        base.action_manager._prev_action[env_id : env_id + 1] = checkpoint.prev_action
    base.episode_length_buf[env_id] = checkpoint.episode_length
    if restoration_mode != "no_manager_state":
        for term_name, values in checkpoint.commands.items():
            term = base.command_manager.get_term(term_name)
            term.command[env_id : env_id + 1] = values["command"]
            term.time_left[env_id : env_id + 1] = values["time_left"]
            term.command_counter[env_id : env_id + 1] = values["command_counter"]
    return float((primary_after - primary_before).abs().max().item())


def run_restored_pair(
    env: Any,
    policy: Any,
    checkpoint: Checkpoint,
    reference: dict[str, Any],
    *,
    seed: int,
    budget: int,
    lift_threshold: float,
    restoration_mode: str = "full",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], float, dict[str, dict[str, np.ndarray]]]:
    reset_seeded(env, seed)
    locality = max(
        restore_probe(env, checkpoint, EXACT, restoration_mode),
        restore_probe(env, checkpoint, POLICY, restoration_mode),
    )
    observation = env.get_observations()
    immediate = {
        "exact_obs_max_abs": float(np.max(np.abs(checkpoint.observation - observation["policy"][EXACT].cpu().numpy()))),
        "policy_obs_max_abs": float(np.max(np.abs(checkpoint.observation - observation["policy"][POLICY].cpu().numpy()))),
    }
    records = {"exact_action": empty_records(), "policy": empty_records()}
    active = {"exact_action": True, "policy": True}

    for step in range(budget):
        for mode, env_id in (("exact_action", EXACT), ("policy", POLICY)):
            if active[mode]:
                append_sample(records[mode], live_sample(env, observation, env_id))

        actions = clipped_actions(env, policy(observation))
        if step < len(reference["actions"]):
            actions[EXACT] = torch.as_tensor(reference["actions"][step], device=actions.device, dtype=actions.dtype)
        for mode, env_id in (("exact_action", EXACT), ("policy", POLICY)):
            if active[mode]:
                records[mode]["actions"].append(actions[env_id].detach().cpu().numpy().astype(np.float64))
        next_observation, reward, dones, _ = env.step(actions)
        for mode, env_id in (("exact_action", EXACT), ("policy", POLICY)):
            if not active[mode]:
                continue
            records[mode]["rewards"].append(float(reward[env_id].item()))
            done = bool(dones[env_id].item())
            records[mode]["dones"].append(done)
            if done:
                active[mode] = False
        observation = next_observation
        if not any(active.values()):
            break

    traces = {}
    comparisons = {}
    trace_arrays = {}
    for mode in ("exact_action", "policy"):
        trace = make_trace(
            records[mode], start=0, budget=budget, rest_height=checkpoint.rest_height,
            lift_threshold=lift_threshold,
        )
        traces[mode] = trace
        comparisons[mode] = compare_traces(reference, trace)
        trace_arrays[mode] = {key: np.asarray(value) for key, value in trace.items() if isinstance(value, np.ndarray)}
    return comparisons, immediate, locality, trace_arrays


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_compatible(env: Any, agent_cfg: dict[str, Any], checkpoint: Path) -> LearnedPolicy:
    """Load current or legacy published checkpoints into the current RSL-RL actor."""
    from rsl_rl.runners import OnPolicyRunner

    device = str(env.unwrapped.device)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=device)
    try:
        payload = torch.load(checkpoint, weights_only=True, map_location=device)
    except TypeError:
        raise RuntimeError("This experiment requires torch with weights_only loading.") from None
    except Exception as exc:
        raise RuntimeError("Checkpoint was rejected by safe tensor-only loading.") from exc
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")

    actor = runner.alg.get_policy()
    current = payload.get("actor_state_dict")
    if current is not None:
        if not isinstance(current, dict):
            raise TypeError("checkpoint actor_state_dict must be a dictionary")
        actor.load_state_dict(current, strict=True)
    else:
        legacy = payload.get("model_state_dict")
        if not isinstance(legacy, dict):
            raise KeyError("checkpoint has neither actor_state_dict nor model_state_dict")
        mapped = {}
        for target_key, target_value in actor.state_dict().items():
            if target_key.startswith("mlp."):
                source_key = f"actor.{target_key.removeprefix('mlp.')}"
            elif target_key == "distribution.std_param":
                source_key = "std"
            else:
                raise KeyError(f"no legacy mapping for actor key {target_key}")
            source_value = legacy[source_key]
            if source_value.shape != target_value.shape:
                raise ValueError(f"shape mismatch for {source_key}: {source_value.shape} != {target_value.shape}")
            mapped[target_key] = source_value
        actor.load_state_dict(mapped, strict=True)
    return LearnedPolicy(runner.get_inference_policy(device=device), runner.alg.get_policy())


def csv_values(value: str, cast: Any) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def reference_for_checkpoint(
    checkpoint: Checkpoint,
    nominal_records: dict[str, Any],
    disturbed_records: dict[str, Any],
    budget: int,
    lift_threshold: float,
) -> dict[str, Any]:
    records = nominal_records if checkpoint.disturbance == "none" else disturbed_records
    return make_trace(
        records,
        start=checkpoint.point_index,
        budget=budget,
        rest_height=checkpoint.rest_height,
        lift_threshold=lift_threshold,
    )


def aggregate_validity(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for name, item in run.get("checkpoints", {}).items():
            key = f"{item['disturbance']}:{name}"
            grouped.setdefault(key, []).append(item)
    result = {}
    for key, items in sorted(grouped.items()):
        exact = [item["comparisons"]["exact_action"] for item in items]
        policy = [item["comparisons"]["policy"] for item in items]
        result[key] = {
            "count": len(items),
            "disturbance": items[0]["disturbance"],
            "task_phase": items[0]["task_phase"],
            "horizon_to_boundary": items[0]["horizon_to_boundary"],
            "point_indices": sorted({int(item["point_index"]) for item in items}),
            "reference_success": sorted({bool(item["reference_success"]) for item in items}),
            "exact_outcome_agreement": sum(bool(item["success_match"]) for item in exact) / len(exact),
            "policy_outcome_agreement": sum(bool(item["success_match"]) for item in policy) / len(policy),
            "exact_recovery_verdict_agreement": sum(bool(item["recovery_verdict_match"]) for item in exact) / len(exact),
            "policy_recovery_verdict_agreement": sum(bool(item["recovery_verdict_match"]) for item in policy) / len(policy),
            "max_observation_divergence": max(float(item["obs_max_abs_30"]) for item in exact),
            "max_object_pose_divergence": max(float(item["object_pose_max_abs_30"]) for item in exact),
            "max_joint_position_divergence": max(float(item["joint_pos_max_abs_30"]) for item in exact),
            "max_joint_velocity_divergence": max(float(item["joint_vel_max_abs_30"]) for item in exact),
            "first_material_divergence_steps": sorted(
                {
                    int(item["first_material_divergence_step"])
                    for item in exact
                    if item["first_material_divergence_step"] is not None
                }
            ),
            "success_time_differences": sorted({int(item["success_time_difference"]) for item in exact}),
            "termination_differences": sum(bool(item["termination_difference"]) for item in exact),
        }
    return result


def classify_expanded(runs: list[dict[str, Any]], validity: dict[str, Any]) -> tuple[str, list[str]]:
    usable = [run for run in runs if run.get("controls_non_degenerate") and run.get("checkpoints")]
    if not usable:
        return "UNRESOLVED", ["no seed/disturbance case produced a non-degenerate recovery interval"]
    comparisons = [
        comparison
        for run in usable
        for item in run["checkpoints"].values()
        for comparison in item["comparisons"].values()
    ]
    mismatches = [item for item in comparisons if not bool(item["success_match"])]
    if mismatches:
        return "GENERALLY_INVALID", [f"outcome disagreement occurred in {len(mismatches)} replay comparisons"]
    disturbances = {run["disturbance"] for run in usable}
    divergence = any(
        item["max_observation_divergence"] > 1e-3 or item["max_object_pose_divergence"] > 1e-4
        for item in validity.values()
    )
    if len(disturbances) >= 2 and divergence:
        return "DISTURBANCE_CONDITIONAL_ORACLE", [
            "outcome labels remained stable across two disturbance families while contact trajectories diverged"
        ]
    if divergence:
        return "PHASE_CONDITIONAL_ORACLE", [
            "outcome labels remained stable but restored trajectories exceeded contact-state tolerances"
        ]
    return "SHORT_HORIZON_OUTCOME_ORACLE", [
        "outcome and trajectory comparisons remained stable over the tested bounded slices"
    ]


def main() -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {item.strip() for item in args.checkpoint_classes.split(",") if item.strip()}
    horizons = tuple(sorted(set(csv_values(args.horizons, int)), reverse=False))
    seeds = csv_values(args.seeds, int) if args.seeds else [args.seed]
    disturbances = csv_values(args.disturbances, str)
    unknown_disturbances = set(disturbances) - DISTURBANCE_NAMES
    if unknown_disturbances or not horizons or not seeds or not disturbances:
        raise SystemExit(
            f"invalid horizons, seeds, or disturbances: horizons={horizons} seeds={seeds} "
            f"disturbances={disturbances} unknown={sorted(unknown_disturbances)}"
        )
    restoration_ablations = csv_values(args.restoration_ablations, str)
    unknown_ablations = set(restoration_ablations) - RESTORATION_MODES
    if unknown_ablations:
        raise SystemExit(f"unknown restoration ablations: {sorted(unknown_ablations)}")
    ablation_checkpoints = {
        item.strip() for item in args.ablation_checkpoints.split(",") if item.strip()
    }

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    try:
        checkpoint_path = args.checkpoint
        if args.use_pretrained or not checkpoint_path:
            from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

            checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", args.task)
        if not checkpoint_path:
            raise SystemExit("No checkpoint resolved")
        checkpoint_file = Path(checkpoint_path).resolve()
        policy = load_policy_compatible(env, agent_cfg.to_dict(), checkpoint_file)
        log(
            f"task={args.task} checkpoint={checkpoint_file} seeds={seeds} "
            f"repeats={args.repeats} horizons={horizons} disturbances={disturbances}"
        )

        runs: list[dict[str, Any]] = []
        trace_archive: dict[str, np.ndarray] = {}
        ablation_results: list[dict[str, Any]] = []
        for seed_index, run_seed in enumerate(seeds):
            for repeat in range(args.repeats):
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)
                nominal_records, nominal_checkpoints, nominal_boundary = collect_primary(
                    env,
                    policy,
                    disturbance="none",
                    seed=run_seed,
                    budget=args.probe_budget,
                    lift_threshold=args.lift_threshold,
                    reach_push=args.reach_push,
                    horizons=horizons,
                    gripper_open=args.gripper_open,
                    gripper_interrupt_steps=args.gripper_interrupt_steps,
                )
                for disturbance in disturbances:
                    disturbed_records, disturbed_checkpoints, disturbed_boundary = collect_primary(
                        env,
                        policy,
                        disturbance=disturbance,
                        seed=run_seed,
                        budget=args.probe_budget,
                        lift_threshold=args.lift_threshold,
                        reach_push=args.reach_push,
                        horizons=horizons,
                        gripper_open=args.gripper_open,
                        gripper_interrupt_steps=args.gripper_interrupt_steps,
                    )
                    available = {**nominal_checkpoints, **disturbed_checkpoints}
                    case_names = sorted(selected & available.keys())
                    missing = sorted(selected - available.keys())
                    controls_non_degenerate = bool(
                        nominal_boundary is not None
                        and disturbed_boundary is not None
                        and abs(nominal_boundary - disturbed_boundary) <= 1
                        and case_names
                    )
                    run: dict[str, Any] = {
                        "seed": run_seed,
                        "seed_index": seed_index,
                        "repeat": repeat,
                        "disturbance": disturbance,
                        "controls_non_degenerate": controls_non_degenerate,
                        "missing_checkpoints": missing,
                        "nominal_boundary_step": nominal_boundary,
                        "disturbed_boundary_step": disturbed_boundary,
                        "boundary_shift": (
                            None
                            if nominal_boundary is None or disturbed_boundary is None
                            else int(disturbed_boundary - nominal_boundary)
                        ),
                        "primary_locality_max_abs": 0.0,
                        "checkpoints": {},
                    }
                    for name in case_names:
                        checkpoint = available[name]
                        reference = reference_for_checkpoint(
                            checkpoint,
                            nominal_records,
                            disturbed_records,
                            args.probe_budget,
                            args.lift_threshold,
                        )
                        comparisons, immediate, locality, restored_arrays = run_restored_pair(
                            env,
                            policy,
                            checkpoint,
                            reference,
                            seed=run_seed,
                            budget=args.probe_budget,
                            lift_threshold=args.lift_threshold,
                        )
                        run["primary_locality_max_abs"] = max(
                            run["primary_locality_max_abs"], locality
                        )
                        run["checkpoints"][name] = {
                            "point_index": checkpoint.point_index,
                            "rise_m": checkpoint.rise,
                            "finger_aperture_m": checkpoint.finger_aperture,
                            "task_phase": checkpoint.task_phase,
                            "disturbance": checkpoint.disturbance,
                            "horizon_to_boundary": checkpoint.horizon_to_boundary,
                            "reference_success": reference["success"],
                            "reference_outcome_step": reference["outcome_step"],
                            "reference_horizon": reference["horizon"],
                            "immediate_roundtrip": immediate,
                            "comparisons": comparisons,
                        }
                        prefix = f"s{run_seed}_r{repeat}_{disturbance}_{name}"
                        for key, array in reference.items():
                            if isinstance(array, np.ndarray):
                                trace_archive[f"{prefix}_uninterrupted_{key}"] = array
                        for mode, arrays in restored_arrays.items():
                            for key, array in arrays.items():
                                trace_archive[f"{prefix}_{mode}_{key}"] = array
                        log(
                            f"seed={run_seed} repeat={repeat} disturbance={disturbance} "
                            f"{name} baseline={reference['success']} "
                            f"exact={comparisons['exact_action']['candidate_success']} "
                            f"policy={comparisons['policy']['candidate_success']} "
                            f"first_div={comparisons['exact_action']['first_material_divergence_step']}"
                        )
                    runs.append(run)

                    if seed_index == 0 and repeat == 0 and disturbance == disturbances[0]:
                        for name in sorted(ablation_checkpoints & available.keys()):
                            checkpoint = available[name]
                            reference = reference_for_checkpoint(
                                checkpoint,
                                nominal_records,
                                disturbed_records,
                                args.probe_budget,
                                args.lift_threshold,
                            )
                            for mode in restoration_ablations:
                                if mode == "full":
                                    continue
                                comparisons, immediate, locality, _ = run_restored_pair(
                                    env,
                                    policy,
                                    checkpoint,
                                    reference,
                                    seed=run_seed,
                                    budget=args.probe_budget,
                                    lift_threshold=args.lift_threshold,
                                    restoration_mode=mode,
                                )
                                ablation_results.append(
                                    {
                                        "seed": run_seed,
                                        "repeat": repeat,
                                        "disturbance": disturbance,
                                        "checkpoint": name,
                                        "restoration_mode": mode,
                                        "immediate_roundtrip": immediate,
                                        "locality": locality,
                                        "comparisons": comparisons,
                                    }
                                )

        validity = aggregate_validity(runs)
        classification, reasons = classify_expanded(runs, validity)
        report = {
            "schema_version": 2,
            "classification": classification,
            "reasons": reasons,
            "contract": {
                "task": args.task,
                "checkpoint_id": checkpoint_file.name,
                "checkpoint_sha256": sha256(checkpoint_file),
                "generator_sha256": sha256(Path(__file__).resolve()),
                "seeds": seeds,
                "repeats_per_seed": args.repeats,
                "probe_budget": args.probe_budget,
                "horizons": horizons,
                "disturbances": disturbances,
                "lift_threshold_m": args.lift_threshold,
                "reach_push_m": args.reach_push,
                "continuations": ["uninterrupted", "exact_action", "policy"],
                "restoration_ablations": restoration_ablations,
            },
            "runs": runs,
            "oracle_validity_map": validity,
            "restoration_ablation_results": ablation_results,
            "baseline_comparison": None,
            "intervention_result": None,
        }
        report_path = output_dir / "oracle_equivalence.json"
        report_path.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(output_dir / "traces.npz", **trace_archive)
        manifest = {
            "generator_sha256": sha256(Path(__file__).resolve()),
            "checkpoint_id": checkpoint_file.name,
            "checkpoint_sha256": sha256(checkpoint_file),
            "artifacts": {},
        }
        for artifact in (report_path, output_dir / "traces.npz"):
            manifest["artifacts"][artifact.name] = {
                "sha256": sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        manifest_path = output_dir / "checkpoint_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        print("\nORACLE_EQUIVALENCE_STATUS")
        print(f"classification: {classification}")
        for reason in reasons:
            print(f"reason: {reason}")
        print(f"report: {report_path}")
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
