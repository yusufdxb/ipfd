"""Run the corrected Protocol A versus Protocol B branch-validity study.

Primary command:

    python scripts/run_snapshot_protocol_study.py \
      --checkpoint /path/to/checkpoint.pt \
      --asset-root https://.../Assets/Isaac/4.5 \
      --output-dir results/branch_validity/corrected_five_seed

The orchestrator validates assets, starts one isolated Isaac process per
seed-disturbance trajectory, and aggregates the paired records. Worker failures
always propagate as a nonzero orchestrator exit status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
import urllib.request
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from ipfd.branch_study import (
    HORIZONS,
    PHASES,
    PROTOCOL_A,
    PROTOCOL_B,
    DecisionSignals,
    DisturbanceSchedule,
    PhaseSignals,
    PhaseTracker,
    SeedBundle,
    assert_schedule_equivalence,
    decision_predicates,
    validate_horizons,
    validate_protocol_bookkeeping,
    validate_seed_bundles,
)

BASE_SEEDS = (101, 211, 307, 401, 503)
DISTURBANCES = ("object_teleport", "gripper_open_interruption")
CONTINUATIONS = ("exact_action", "closed_loop_policy")
PROTOCOLS = (PROTOCOL_A, PROTOCOL_B)
PROBES_PER_PHASE = len(PROTOCOLS) * len(CONTINUATIONS)
DEFAULT_NUM_ENVS = 1 + len(PHASES) * PROBES_PER_PHASE
SOURCE_ENV = 0
DISTURBANCE_START_STEP = 70
MAX_CONTROL_STEPS = 220


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def schedule_for(disturbance: str) -> DisturbanceSchedule:
    if disturbance == "object_teleport":
        return DisturbanceSchedule(
            kind=disturbance,
            start_step=DISTURBANCE_START_STEP,
            duration_steps=1,
            magnitude=(1.0, 0.0, 0.0),
            target="object_root_pose",
        )
    if disturbance == "gripper_open_interruption":
        return DisturbanceSchedule(
            kind=disturbance,
            start_step=DISTURBANCE_START_STEP,
            duration_steps=8,
            magnitude=(1.0,),
            target="gripper_action",
        )
    raise ValueError(f"unsupported disturbance: {disturbance}")


def asset_urls(asset_root: str) -> dict[str, str]:
    root = asset_root.rstrip("/")
    return {
        "franka_entry_usd": (f"{root}/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"),
        "cube_entry_usd": f"{root}/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        "table_entry_usd": (f"{root}/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    }


def validate_asset_entries(asset_root: str) -> dict[str, dict[str, Any]]:
    """Fetch and hash the three task entry USDs before launching workers."""
    result: dict[str, dict[str, Any]] = {}
    for name, url in asset_urls(asset_root).items():
        request = urllib.request.Request(url, headers={"User-Agent": "ipfd-branch-study/1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
            if not content:
                raise RuntimeError(f"empty asset response: {url}")
            result[name] = {
                "url": url,
                "status": int(response.status),
                "bytes": len(content),
                "etag": response.headers.get("ETag", "").strip('"'),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
    return result


def git_provenance(path: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=path,
            text=True,
        ).strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short").splitlines(),
    }


def worker_command(args: argparse.Namespace, bundle: SeedBundle, output: Path) -> list[str]:
    return [
        sys.executable,
        os.fspath(Path(__file__).resolve()),
        "--worker",
        "--task",
        args.task,
        "--checkpoint",
        os.fspath(args.checkpoint),
        "--asset-root",
        args.asset_root,
        "--base-seed",
        str(bundle.base_seed),
        "--disturbance",
        bundle.disturbance,
        "--num-envs",
        str(args.num_envs),
        "--output",
        os.fspath(output),
        "--device",
        args.device,
    ]


def grouped_rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    disagreements = sum(not bool(record["decision_match"]) for record in records)
    false_recoverable = sum(not bool(record["reference_decision"]) and bool(record["candidate_decision"]) for record in records)
    false_unrecoverable = sum(bool(record["reference_decision"]) and not bool(record["candidate_decision"]) for record in records)
    return {
        "records": total,
        "agreements": total - disagreements,
        "disagreements": disagreements,
        "disagreement_rate": disagreements / total if total else None,
        "false_recoverable": false_recoverable,
        "false_unrecoverable": false_unrecoverable,
    }


def group_summaries(
    records: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(str(record[field]) for field in fields)
        groups[key].append(record)
    return {"|".join(key): grouped_rate(group) for key, group in sorted(groups.items())}


def seed_cluster_bootstrap(
    records: list[dict[str, Any]],
    *,
    samples: int = 10_000,
) -> dict[str, Any]:
    """Bootstrap the paired B minus A disagreement-rate difference by base seed."""
    seeds = sorted({int(record["base_seed"]) for record in records})
    rng = np.random.default_rng(20_260_729)
    by_seed_protocol: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed_protocol[(int(record["base_seed"]), record["protocol"])].append(record)

    per_seed = []
    for seed in seeds:
        rate_a = grouped_rate(by_seed_protocol[(seed, PROTOCOL_A)])["disagreement_rate"]
        rate_b = grouped_rate(by_seed_protocol[(seed, PROTOCOL_B)])["disagreement_rate"]
        if rate_a is None or rate_b is None:
            continue
        per_seed.append(
            {
                "base_seed": seed,
                "protocol_a_rate": rate_a,
                "protocol_b_rate": rate_b,
                "b_minus_a": rate_b - rate_a,
            }
        )
    differences = np.asarray([item["b_minus_a"] for item in per_seed], dtype=np.float64)
    if not len(differences):
        return {"per_seed": [], "paired_mean_difference": None, "bootstrap_95": None}
    draws = rng.choice(differences, size=(samples, len(differences)), replace=True).mean(axis=1)
    return {
        "per_seed": per_seed,
        "paired_mean_difference": float(differences.mean()),
        "bootstrap_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "independent_units": len(differences),
        "bootstrap_samples": samples,
    }


def aggregate_worker_results(
    worker_results: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    records = [record for result in worker_results for record in result.get("records", [])]
    primary = [
        record for record in records if record["continuation"] == "exact_action" and record["predicate"] == "sustained_lift"
    ]
    protocol_a = [record for record in primary if record["protocol"] == PROTOCOL_A]
    protocol_b = [record for record in primary if record["protocol"] == PROTOCOL_B]
    rate_a = grouped_rate(protocol_a)
    rate_b = grouped_rate(protocol_b)
    if rate_a["disagreement_rate"] in (None, 0.0):
        relative_reduction = None
    else:
        relative_reduction = (rate_a["disagreement_rate"] - rate_b["disagreement_rate"]) / rate_a["disagreement_rate"]
    cluster = seed_cluster_bootstrap(primary)
    improved_seeds = sum(item["protocol_b_rate"] < item["protocol_a_rate"] for item in cluster["per_seed"])
    no_stratum_dropped = len(protocol_a) == len(protocol_b) and len(protocol_a) > 0
    threshold_passed = bool(
        relative_reduction is not None and relative_reduction >= 0.5 and improved_seeds >= 2 and no_stratum_dropped
    )
    controls = [result["controls"] for result in worker_results]
    all_controls_passed = all(control["passed"] for control in controls)
    if not all_controls_passed:
        threshold_passed = False
    return {
        "schema_version": 1,
        "scientific_object": "counterfactual_branch_decision_fidelity",
        "preregistered_primary_comparison": {
            "continuation": "exact_action",
            "predicate": "sustained_lift",
            "protocol_a": rate_a,
            "protocol_b": rate_b,
            "relative_disagreement_reduction": relative_reduction,
            "seed_cluster_uncertainty": cluster,
            "seeds_improved": improved_seeds,
            "required_seeds_improved": 2,
            "required_relative_reduction": 0.5,
            "paired_record_counts_equal": no_stratum_dropped,
            "threshold_passed": threshold_passed,
        },
        "stopping_rule": {
            "positive_control_meaningfully_improved": threshold_passed,
            "decision": ("CONTINUE_TO_VALIDITY_GATE" if threshold_passed else "STOP_BRANCH_VALIDITY_DIRECTION"),
            "gate_eligible": threshold_passed,
        },
        "controls": {
            "all_passed": all_controls_passed,
            "workers": controls,
        },
        "overall_by_protocol_and_continuation": group_summaries(records, ("protocol", "continuation")),
        "by_phase": group_summaries(records, ("protocol", "continuation", "phase")),
        "by_horizon": group_summaries(records, ("protocol", "continuation", "horizon")),
        "by_disturbance": group_summaries(records, ("protocol", "continuation", "disturbance")),
        "by_predicate": group_summaries(records, ("protocol", "continuation", "predicate")),
        "record_count": len(records),
        "worker_summaries": [
            {
                "base_seed": result["seed_bundle"]["base_seed"],
                "disturbance": result["seed_bundle"]["disturbance"],
                "phases_observed": result["controls"]["phases_observed"],
                "branch_steps": result["controls"]["branch_steps"],
                "records": len(result["records"]),
            }
            for result in worker_results
        ],
        "provenance": provenance,
        "limitations": [
            "Five base seeds are the independent uncertainty units.",
            "Multiple phases, horizons, predicates, and continuations within a seed are correlated.",
            "Protocol B does not restore unexposed PhysX solver or contact-cache state.",
            "Entry-USD hashes do not cover transitive asset dependencies.",
        ],
    }


def orchestrator_main(args: argparse.Namespace) -> int:
    validate_protocol_bookkeeping()
    validate_horizons(HORIZONS)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = [SeedBundle.derive(base_seed, disturbance) for base_seed in BASE_SEEDS for disturbance in DISTURBANCES]
    validate_seed_bundles(bundles)
    assets = validate_asset_entries(args.asset_root)
    study_config = {
        "task": args.task,
        "base_seeds": list(BASE_SEEDS),
        "disturbances": list(DISTURBANCES),
        "continuations": list(CONTINUATIONS),
        "protocols": list(PROTOCOLS),
        "phases": list(PHASES),
        "horizons": list(HORIZONS),
        "disturbance_start_step": DISTURBANCE_START_STEP,
        "max_control_steps": MAX_CONTROL_STEPS,
        "num_envs": args.num_envs,
    }
    provenance = {
        "ipfd_git": git_provenance(_REPO),
        "isaac_lab_git": git_provenance(resolve_isaac_lab_root(args.isaac_lab_root)),
        "checkpoint": {
            "path": os.fspath(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "generator": {
            "path": os.fspath(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "configuration": {
            "value": study_config,
            "sha256": sha256_json(study_config),
        },
        "asset_entries": assets,
    }
    expected_checkpoint = "fb658f989bf5ebf35b20347813275979a6778ade8d3823d12eb3190612f9e36d"
    if provenance["checkpoint"]["sha256"] != expected_checkpoint:
        raise RuntimeError("checkpoint hash does not match the preregistered learned checkpoint")
    write_json(output_dir / "study_provenance.json", provenance)

    worker_results = []
    worker_dir = output_dir / "workers"
    worker_dir.mkdir()
    for bundle in bundles:
        output = worker_dir / f"seed_{bundle.base_seed}_{bundle.disturbance}.json"
        log_path = worker_dir / f"seed_{bundle.base_seed}_{bundle.disturbance}.log"
        command = worker_command(args, bundle, output)
        with log_path.open("w", encoding="utf-8") as log_stream:
            completed = subprocess.run(
                command,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
            )
        if completed.returncode != 0:
            raise RuntimeError(f"worker failed with exit {completed.returncode}; see {log_path}")
        if not output.is_file():
            raise RuntimeError(f"worker returned zero without output: {log_path}")
        with output.open(encoding="utf-8") as stream:
            result = json.load(stream)
        if not result["controls"]["passed"]:
            raise RuntimeError(f"worker controls failed; see {output} and {log_path}")
        worker_results.append(result)

    summary = aggregate_worker_results(worker_results, provenance)
    records_path = output_dir / "per_branch_records.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for result in worker_results:
            for record in result["records"]:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path = output_dir / "protocol_comparison.json"
    write_json(summary_path, summary)
    manifest = {
        "artifacts": {},
        "source_generator_sha256": provenance["generator"]["sha256"],
        "checkpoint_sha256": provenance["checkpoint"]["sha256"],
        "configuration_sha256": provenance["configuration"]["sha256"],
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest["artifacts"][os.fspath(path.relative_to(output_dir))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    write_json(output_dir / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": summary["stopping_rule"]["decision"],
                "threshold_passed": summary["preregistered_primary_comparison"]["threshold_passed"],
                "summary": os.fspath(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


#: Environment variable naming the Isaac Lab checkout. This is the same variable
#: the reproduction commands in CORRECTED_EXPERIMENT_PROTOCOL.md already export.
IPFD_ISAACLAB_ROOT_ENV = "IPFD_ISAACLAB_ROOT"


def _default_isaac_lab_root() -> Path | None:
    """Resolve the Isaac Lab root from the environment, or return None.

    Returning None makes ``--isaac-lab-root`` a required flag, so a run on a
    machine that has neither the environment variable nor the flag fails with an
    explicit argparse error instead of silently recording provenance for a path
    that does not exist.
    """
    value = os.environ.get(IPFD_ISAACLAB_ROOT_ENV, "").strip()
    return Path(value) if value else None


def resolve_isaac_lab_root(value: Path | None) -> Path:
    """Validate the Isaac Lab root and fail loudly when it is unusable."""
    if value is None:
        raise SystemExit(
            "Isaac Lab root is not set. Pass --isaac-lab-root </path/to/IsaacLab> "
            f"or export {IPFD_ISAACLAB_ROOT_ENV}."
        )
    resolved = Path(value).expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(
            f"Isaac Lab root does not exist or is not a directory: {resolved}. "
            f"Pass --isaac-lab-root </path/to/IsaacLab> or export {IPFD_ISAACLAB_ROOT_ENV}."
        )
    return resolved


def build_orchestrator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-v0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--isaac-lab-root",
        type=Path,
        default=_default_isaac_lab_root(),
        required=IPFD_ISAACLAB_ROOT_ENV not in os.environ,
        help=(
            "Path to the Isaac Lab checkout, used to record simulator provenance. "
            f"Defaults to ${IPFD_ISAACLAB_ROOT_ENV} when that variable is set; "
            "otherwise this flag is required."
        ),
    )
    return parser


_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--worker", action="store_true")
_bootstrap_args, _ = _bootstrap.parse_known_args()


if _bootstrap_args.worker:
    from isaaclab.app import AppLauncher

    _worker_parser = argparse.ArgumentParser(description=__doc__)
    _worker_parser.add_argument("--worker", action="store_true", required=True)
    _worker_parser.add_argument("--task", required=True)
    _worker_parser.add_argument("--checkpoint", type=Path, required=True)
    _worker_parser.add_argument("--asset-root", required=True)
    _worker_parser.add_argument("--base-seed", type=int, required=True)
    _worker_parser.add_argument("--disturbance", choices=DISTURBANCES, required=True)
    _worker_parser.add_argument("--num-envs", type=int, required=True)
    _worker_parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(_worker_parser)
    _worker_args = _worker_parser.parse_args()
    _worker_args.headless = True
    _app = AppLauncher(_worker_args).app

    import importlib.metadata as metadata

    import gymnasium as gym
    import torch
    import warp as wp

    if _worker_args.asset_root:
        import isaaclab.utils.assets as assets
        import isaaclab_assets.robots.franka as franka

        _asset_root = _worker_args.asset_root.rstrip("/")
        assets.NUCLEUS_ASSET_ROOT_DIR = _asset_root
        assets.NVIDIA_NUCLEUS_DIR = f"{_asset_root}/NVIDIA"
        assets.ISAAC_NUCLEUS_DIR = f"{_asset_root}/Isaac"
        assets.ISAACLAB_NUCLEUS_DIR = f"{_asset_root}/Isaac/IsaacLab"
        _panda_path = f"{assets.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
        franka.FRANKA_PANDA_CFG.spawn.usd_path = _panda_path
        franka.FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path = _panda_path

    import isaaclab_tasks  # noqa: F401
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    from ipfd.adapters.isaac_lab import offset_root_positions, slice_state
    from ipfd.oracles.rsl_rl_policy import LearnedPolicy

    def clone_tree(value: Any) -> Any:
        if hasattr(value, "clone"):
            return value.clone()
        if isinstance(value, dict):
            return {key: clone_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone_tree(item) for item in value)
        return value

    def torch_view(value: Any) -> torch.Tensor:
        return value if isinstance(value, torch.Tensor) else wp.to_torch(value)

    def one(value: torch.Tensor, env_id: int) -> torch.Tensor:
        return value[env_id : env_id + 1].detach().clone()

    @dataclass
    class RuntimeSnapshot:
        scene_state: Any
        action: torch.Tensor
        prev_action: torch.Tensor
        action_terms: dict[str, dict[str, torch.Tensor]]
        articulation_targets: dict[str, torch.Tensor]
        commands: dict[str, dict[str, Any]]
        environment_buffers: dict[str, torch.Tensor]
        reward_buffers: dict[str, Any]
        termination_buffers: dict[str, torch.Tensor]
        observation_history_present: bool
        event_interval_buffers_present: bool
        source_step: int
        source_episode_length: int
        sim_step_counter: int
        common_step_counter: int
        decision_history: list[DecisionSignals]

    @dataclass
    class CandidateTracker:
        env_id: int
        protocol: str
        continuation: str
        history: deque[DecisionSignals]
        immediate: dict[str, float]
        first_step: dict[str, float] | None = None
        first_numerical_divergence_step: int | None = None
        first_observation_divergence_step: int | None = None
        first_contact_divergence_step: int | None = None
        first_action_divergence_step: int | None = None
        first_predicate_disagreement_step: int | None = None
        maximum_trajectory_error: float = 0.0
        terminal_trajectory_error: float = 0.0

    @dataclass
    class BranchTracker:
        phase: str
        branch_step: int
        source_history: deque[DecisionSignals]
        candidates: dict[tuple[str, str], CandidateTracker] = field(default_factory=dict)

    def load_policy_compatible(
        env: Any,
        agent_cfg: dict[str, Any],
        checkpoint: Path,
    ) -> LearnedPolicy:
        from rsl_rl.runners import OnPolicyRunner

        device = str(env.unwrapped.device)
        runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=device)
        payload = torch.load(
            checkpoint,
            weights_only=True,
            map_location=device,
        )
        if not isinstance(payload, dict):
            raise TypeError("checkpoint payload must be a dictionary")
        actor = runner.alg.get_policy()
        current = payload.get("actor_state_dict")
        if not isinstance(current, dict):
            raise KeyError("checkpoint has no actor_state_dict")
        actor.load_state_dict(current, strict=True)
        return LearnedPolicy(
            runner.get_inference_policy(device=device),
            runner.alg.get_policy(),
        )

    def capture_runtime(
        base: Any,
        *,
        env_id: int,
        step: int,
        decision_history: deque[DecisionSignals],
    ) -> RuntimeSnapshot:
        action_terms = {}
        for term_name in base.action_manager.active_terms:
            term = base.action_manager.get_term(term_name)
            action_terms[term_name] = {
                "raw": one(term.raw_actions, env_id),
                "processed": one(term.processed_actions, env_id),
            }
        robot = base.scene["robot"]
        articulation_targets = {
            "position": one(torch_view(robot.data.joint_pos_target), env_id),
            "velocity": one(torch_view(robot.data.joint_vel_target), env_id),
            "effort": one(torch_view(robot.data.joint_effort_target), env_id),
        }
        commands = {}
        for term_name in base.command_manager.active_terms:
            term = base.command_manager.get_term(term_name)
            values: dict[str, Any] = {
                "command": one(term.command, env_id),
                "time_left": one(term.time_left, env_id),
                "command_counter": one(term.command_counter, env_id),
                "metrics": {name: one(value, env_id) for name, value in term.metrics.items()},
            }
            if hasattr(term, "pose_command_w"):
                values["pose_command_w"] = one(term.pose_command_w, env_id)
            commands[term_name] = values
        environment_buffers = {
            name: one(getattr(base, name), env_id)
            for name in (
                "reward_buf",
                "reset_buf",
                "reset_terminated",
                "reset_time_outs",
            )
            if hasattr(base, name)
        }
        reward = base.reward_manager
        reward_buffers = {
            "episode_sums": {name: one(value, env_id) for name, value in reward._episode_sums.items()},
            "reward_buf": one(reward._reward_buf, env_id),
            "step_reward": one(reward._step_reward, env_id),
        }
        termination = base.termination_manager
        termination_buffers = {
            name: one(getattr(termination, name), env_id)
            for name in (
                "_term_dones",
                "_last_episode_dones",
                "_truncated_buf",
                "_terminated_buf",
            )
        }
        observation_history_present = any(
            bool(terms) for terms in base.observation_manager._group_obs_term_history_buffer.values()
        )
        event_interval_buffers_present = bool(getattr(base.event_manager, "_interval_term_time_left", []))
        return RuntimeSnapshot(
            scene_state=slice_state(
                base.scene.get_state(),
                slice(env_id, env_id + 1),
            ),
            action=one(base.action_manager.action, env_id),
            prev_action=one(base.action_manager.prev_action, env_id),
            action_terms=action_terms,
            articulation_targets=articulation_targets,
            commands=commands,
            environment_buffers=environment_buffers,
            reward_buffers=reward_buffers,
            termination_buffers=termination_buffers,
            observation_history_present=observation_history_present,
            event_interval_buffers_present=event_interval_buffers_present,
            source_step=step,
            source_episode_length=int(base.episode_length_buf[env_id].item()),
            sim_step_counter=int(base._sim_step_counter),
            common_step_counter=int(base.common_step_counter),
            decision_history=list(decision_history),
        )

    def restore_runtime(
        base: Any,
        snapshot: RuntimeSnapshot,
        *,
        source_env: int,
        target_env: int,
        protocol: str,
    ) -> None:
        ids = torch.tensor([target_env], device=base.device, dtype=torch.long)
        base._reset_idx(ids)
        delta = base.scene.env_origins[target_env] - base.scene.env_origins[source_env]
        state = offset_root_positions(snapshot.scene_state, delta)
        base.scene.reset_to(state, ids)
        base.action_manager._action[target_env : target_env + 1] = snapshot.action
        base.action_manager._prev_action[target_env : target_env + 1] = snapshot.prev_action
        base.episode_length_buf[target_env] = snapshot.source_episode_length
        for term_name, values in snapshot.commands.items():
            term = base.command_manager.get_term(term_name)
            term.command[target_env : target_env + 1] = values["command"]
            term.time_left[target_env : target_env + 1] = values["time_left"]
            term.command_counter[target_env : target_env + 1] = values["command_counter"]
        if protocol == PROTOCOL_A:
            return
        if protocol != PROTOCOL_B:
            raise ValueError(f"unknown protocol: {protocol}")

        for term_name, values in snapshot.action_terms.items():
            term = base.action_manager.get_term(term_name)
            term._raw_actions[target_env : target_env + 1] = values["raw"]
            term._processed_actions[target_env : target_env + 1] = values["processed"]
        robot = base.scene["robot"]
        robot.set_joint_position_target_index(
            target=snapshot.articulation_targets["position"],
            env_ids=ids,
        )
        robot.set_joint_velocity_target_index(
            target=snapshot.articulation_targets["velocity"],
            env_ids=ids,
        )
        robot.set_joint_effort_target_index(
            target=snapshot.articulation_targets["effort"],
            env_ids=ids,
        )
        for name, value in snapshot.environment_buffers.items():
            getattr(base, name)[target_env : target_env + 1] = value
        for name, value in snapshot.reward_buffers["episode_sums"].items():
            base.reward_manager._episode_sums[name][target_env : target_env + 1] = value
        base.reward_manager._reward_buf[target_env : target_env + 1] = snapshot.reward_buffers["reward_buf"]
        base.reward_manager._step_reward[target_env : target_env + 1] = snapshot.reward_buffers["step_reward"]
        for name, value in snapshot.termination_buffers.items():
            getattr(base.termination_manager, name)[target_env : target_env + 1] = value
        for term_name, values in snapshot.commands.items():
            term = base.command_manager.get_term(term_name)
            for metric_name, metric_value in values["metrics"].items():
                term.metrics[metric_name][target_env : target_env + 1] = metric_value
            if "pose_command_w" in values:
                term.pose_command_w[target_env : target_env + 1] = values["pose_command_w"]
        if snapshot.observation_history_present:
            raise RuntimeError("observation history is present but no supported per-environment copy path was configured")
        if snapshot.event_interval_buffers_present:
            raise RuntimeError("event interval state is present but no supported per-environment copy path was configured")
        base.scene.write_data_to_sim()

    def contact_force(base: Any, sensor_name: str, env_id: int) -> float:
        sensor = base.scene[sensor_name]
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            raise RuntimeError(f"{sensor_name} has no object-filtered force matrix")
        values = torch_view(matrix)[env_id]
        return float(torch.linalg.vector_norm(values.reshape(-1, 3), dim=-1).max().item())

    def state_sample(
        base: Any,
        observation: Any,
        env_id: int,
        rest_height: float,
    ) -> dict[str, Any]:
        origin = base.scene.env_origins[env_id]
        robot = base.scene["robot"]
        obj = base.scene["object"]
        robot_root = torch_view(robot.data.root_pose_w)[env_id].detach().clone()
        robot_root[:3] -= origin
        object_pose = torch_view(obj.data.root_pose_w)[env_id].detach().clone()
        object_pose[:3] -= origin
        ee_position = torch_view(base.scene["ee_frame"].data.target_pos_w)[env_id, 0].detach().clone()
        ee_position -= origin
        object_position = object_pose[:3]
        joint_position = torch_view(robot.data.joint_pos)[env_id].detach().clone()
        left_force = contact_force(base, "left_object_contact", env_id)
        right_force = contact_force(base, "right_object_contact", env_id)
        object_height = float(object_position[2].item())
        command = base.command_manager.get_command("object_pose")[env_id].detach().clone()
        return {
            "observation": observation["policy"][env_id].detach().clone(),
            "robot_root_pose": robot_root,
            "robot_root_velocity": torch_view(robot.data.root_vel_w)[env_id].detach().clone(),
            "joint_position": joint_position,
            "joint_velocity": torch_view(robot.data.joint_vel)[env_id].detach().clone(),
            "object_pose": object_pose,
            "object_velocity": torch_view(obj.data.root_vel_w)[env_id].detach().clone(),
            "targets": torch.cat(
                (
                    torch_view(robot.data.joint_pos_target)[env_id],
                    torch_view(robot.data.joint_vel_target)[env_id],
                    torch_view(robot.data.joint_effort_target)[env_id],
                )
            )
            .detach()
            .clone(),
            "command": command,
            "left_force": left_force,
            "right_force": right_force,
            "object_rise": object_height - rest_height,
            "aperture": float(joint_position[-2:].sum().item()),
            "ee_object_distance": float(torch.linalg.vector_norm(ee_position - object_position).item()),
            "episode_length": int(base.episode_length_buf[env_id].item()),
            "sim_step_counter": int(base._sim_step_counter),
            "common_step_counter": int(base.common_step_counter),
        }

    def decision_signal(sample: dict[str, Any], *, terminated: bool) -> DecisionSignals:
        return DecisionSignals(
            object_rise_m=float(sample["object_rise"]),
            ee_object_distance_m=float(sample["ee_object_distance"]),
            finger_aperture_m=float(sample["aperture"]),
            left_object_force_n=float(sample["left_force"]),
            right_object_force_n=float(sample["right_force"]),
            terminated=terminated,
        )

    def phase_signal(
        sample: dict[str, Any],
        *,
        step: int,
        schedule: DisturbanceSchedule,
    ) -> PhaseSignals:
        return PhaseSignals(
            step=step,
            object_rise_m=float(sample["object_rise"]),
            finger_aperture_m=float(sample["aperture"]),
            ee_object_distance_m=float(sample["ee_object_distance"]),
            left_object_force_n=float(sample["left_force"]),
            right_object_force_n=float(sample["right_force"]),
            disturbance_started=step == schedule.start_step,
            disturbance_ended=step == schedule.start_step + schedule.duration_steps,
        )

    def numeric_difference(source: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
        tensor_fields = (
            "observation",
            "robot_root_pose",
            "robot_root_velocity",
            "joint_position",
            "joint_velocity",
            "object_pose",
            "object_velocity",
            "targets",
            "command",
        )
        differences = {name: float(torch.max(torch.abs(source[name] - candidate[name])).item()) for name in tensor_fields}
        differences.update(
            {
                "contact_force": max(
                    abs(source["left_force"] - candidate["left_force"]),
                    abs(source["right_force"] - candidate["right_force"]),
                ),
                "episode_length": float(abs(source["episode_length"] - candidate["episode_length"])),
                "sim_step_counter": float(abs(source["sim_step_counter"] - candidate["sim_step_counter"])),
                "common_step_counter": float(abs(source["common_step_counter"] - candidate["common_step_counter"])),
            }
        )
        differences["trajectory"] = max(
            differences["joint_position"],
            differences["joint_velocity"],
            differences["object_pose"],
            differences["object_velocity"],
        )
        return differences

    def apply_teleport(
        base: Any,
        env_ids: list[int],
        schedule: DisturbanceSchedule,
    ) -> None:
        if not env_ids:
            return
        ids = torch.tensor(env_ids, device=base.device, dtype=torch.long)
        obj = base.scene["object"]
        poses = torch_view(obj.data.root_pose_w)[ids].detach().clone()
        offset = torch.tensor(
            schedule.magnitude,
            device=base.device,
            dtype=poses.dtype,
        )
        poses[:, :3] += offset
        obj.write_root_pose_to_sim_index(root_pose=poses, env_ids=ids)

    def worker_main(args: argparse.Namespace) -> int:
        if args.num_envs < DEFAULT_NUM_ENVS:
            raise ValueError(f"num-envs must be at least {DEFAULT_NUM_ENVS} for paired phase slots")
        bundle = SeedBundle.derive(args.base_seed, args.disturbance)
        schedule = schedule_for(args.disturbance)
        random.seed(bundle.branch_selection_seed)
        np.random.seed(bundle.disturbance_seed)
        torch.manual_seed(bundle.policy_seed)
        torch.cuda.manual_seed_all(bundle.policy_seed)

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.commands.object_pose.debug_vis = False
        env_cfg.terminations.object_dropping = None
        env_cfg.terminations.time_out = None
        env_cfg.seed = bundle.environment_seed
        env_cfg.scene.robot.spawn.activate_contact_sensors = True
        env_cfg.scene.left_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        env_cfg.scene.right_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg,
            metadata.version("rsl-rl-lib"),
        )
        raw_env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env)
        base = env.unwrapped
        try:
            policy = load_policy_compatible(
                env,
                agent_cfg.to_dict(),
                args.checkpoint.resolve(),
            )
            base.seed(bundle.environment_seed)
            observation, _ = env.reset()
            source_initial = state_sample(base, observation, SOURCE_ENV, 0.0)
            rest_height = float(source_initial["object_rise"])
            # The initial call used zero as the reference; use its actual relative
            # height as the rest height for all subsequent samples.
            rest_height = float(
                torch_view(base.scene["object"].data.root_pos_w)[SOURCE_ENV, 2].item()
                - base.scene.env_origins[SOURCE_ENV, 2].item()
            )
            observation = env.get_observations()
            phase_tracker = PhaseTracker()
            source_history: deque[DecisionSignals] = deque(maxlen=5)
            source_logical_failure = False
            branches: dict[str, BranchTracker] = {}
            records: list[dict[str, Any]] = []
            target_map: dict[str, dict[tuple[str, str], int]] = {}
            next_env = 1
            for phase in PHASES:
                target_map[phase] = {}
                for protocol in PROTOCOLS:
                    for continuation in CONTINUATIONS:
                        target_map[phase][(protocol, continuation)] = next_env
                        next_env += 1

            disturbance_applications: dict[int, dict[str, Any]] = {}
            candidate_logical_failure: dict[int, bool] = defaultdict(bool)
            for step in range(MAX_CONTROL_STEPS):
                active_ids = [candidate.env_id for branch in branches.values() for candidate in branch.candidates.values()]
                all_evaluated_ids = [SOURCE_ENV, *active_ids]
                if schedule.kind == "object_teleport" and step == schedule.start_step:
                    apply_teleport(base, all_evaluated_ids, schedule)
                    disturbance_applications[step] = {
                        "env_ids": all_evaluated_ids,
                        "schedule_sha256": schedule.sha256,
                    }
                    observation = env.get_observations()

                source_sample = state_sample(
                    base,
                    observation,
                    SOURCE_ENV,
                    rest_height,
                )
                source_logical_failure |= source_sample["object_rise"] < -0.105
                source_history.append(
                    decision_signal(
                        source_sample,
                        terminated=source_logical_failure,
                    )
                )
                emitted = phase_tracker.update(
                    phase_signal(
                        source_sample,
                        step=step,
                        schedule=schedule,
                    )
                )
                for phase in emitted:
                    if phase in branches:
                        raise AssertionError(f"duplicated phase branch: {phase}")
                    snapshot = capture_runtime(
                        base,
                        env_id=SOURCE_ENV,
                        step=step,
                        decision_history=source_history,
                    )
                    branch = BranchTracker(
                        phase=phase,
                        branch_step=step,
                        source_history=deque(source_history, maxlen=5),
                    )
                    source_before = state_sample(
                        base,
                        observation,
                        SOURCE_ENV,
                        rest_height,
                    )
                    for key, target_env in target_map[phase].items():
                        protocol, continuation = key
                        assert_schedule_equivalence(schedule, schedule)
                        restore_runtime(
                            base,
                            snapshot,
                            source_env=SOURCE_ENV,
                            target_env=target_env,
                            protocol=protocol,
                        )
                    observation = env.get_observations()
                    source_after = state_sample(
                        base,
                        observation,
                        SOURCE_ENV,
                        rest_height,
                    )
                    if numeric_difference(source_before, source_after)["trajectory"] != 0.0:
                        raise RuntimeError("probe restoration changed the source environment")
                    for key, target_env in target_map[phase].items():
                        protocol, continuation = key
                        candidate_sample = state_sample(
                            base,
                            observation,
                            target_env,
                            rest_height,
                        )
                        branch.candidates[key] = CandidateTracker(
                            env_id=target_env,
                            protocol=protocol,
                            continuation=continuation,
                            history=deque(snapshot.decision_history, maxlen=5),
                            immediate=numeric_difference(
                                source_after,
                                candidate_sample,
                            ),
                        )
                    branches[phase] = branch

                active_ids = [
                    candidate.env_id
                    for branch in branches.values()
                    for candidate in branch.candidates.values()
                ]
                all_evaluated_ids = [SOURCE_ENV, *active_ids]
                actions = policy(observation).detach().clone()
                actions.zero_()
                source_policy_action = policy(observation)[SOURCE_ENV].detach().clone()
                actions[SOURCE_ENV] = source_policy_action
                action_by_env = {SOURCE_ENV: source_policy_action.detach().clone()}
                for branch in branches.values():
                    for candidate in branch.candidates.values():
                        if candidate.continuation == "exact_action":
                            candidate_action = source_policy_action.detach().clone()
                        else:
                            candidate_action = policy(observation)[candidate.env_id].detach().clone()
                        actions[candidate.env_id] = candidate_action
                        action_by_env[candidate.env_id] = candidate_action.detach().clone()
                if schedule.kind == "gripper_open_interruption" and schedule.active(step):
                    for env_id in all_evaluated_ids:
                        actions[env_id, -1] = schedule.magnitude[0]
                        action_by_env[env_id] = actions[env_id].detach().clone()
                    disturbance_applications[step] = {
                        "env_ids": all_evaluated_ids,
                        "schedule_sha256": schedule.sha256,
                    }

                next_observation, _, _, _ = env.step(actions)
                next_step = step + 1
                source_next = state_sample(
                    base,
                    next_observation,
                    SOURCE_ENV,
                    rest_height,
                )
                source_logical_failure |= source_next["object_rise"] < -0.105
                source_next_signal = decision_signal(
                    source_next,
                    terminated=source_logical_failure,
                )
                for branch in branches.values():
                    branch.source_history.append(source_next_signal)
                    elapsed = next_step - branch.branch_step
                    for candidate in branch.candidates.values():
                        candidate_sample = state_sample(
                            base,
                            next_observation,
                            candidate.env_id,
                            rest_height,
                        )
                        candidate_logical_failure[candidate.env_id] |= candidate_sample["object_rise"] < -0.105
                        candidate_signal = decision_signal(
                            candidate_sample,
                            terminated=candidate_logical_failure[candidate.env_id],
                        )
                        candidate.history.append(candidate_signal)
                        differences = numeric_difference(
                            source_next,
                            candidate_sample,
                        )
                        candidate.maximum_trajectory_error = max(
                            candidate.maximum_trajectory_error,
                            differences["trajectory"],
                        )
                        candidate.terminal_trajectory_error = differences["trajectory"]
                        if candidate.first_step is None:
                            candidate.first_step = differences
                        if candidate.first_numerical_divergence_step is None and differences["trajectory"] > 1e-6:
                            candidate.first_numerical_divergence_step = elapsed
                        if candidate.first_observation_divergence_step is None and differences["observation"] > 1e-6:
                            candidate.first_observation_divergence_step = elapsed
                        if candidate.first_contact_divergence_step is None and differences["contact_force"] > 0.1:
                            candidate.first_contact_divergence_step = elapsed
                        source_action = action_by_env[SOURCE_ENV]
                        candidate_action = action_by_env[candidate.env_id]
                        action_difference = float(torch.max(torch.abs(source_action - candidate_action)).item())
                        if candidate.continuation == "exact_action" and action_difference != 0.0:
                            raise RuntimeError(
                                "recorded identical-action continuation received "
                                f"different actions at phase={branch.phase} elapsed={elapsed}"
                            )
                        if candidate.first_action_divergence_step is None and action_difference > 1e-6:
                            candidate.first_action_divergence_step = elapsed
                        reference_decisions = decision_predicates(list(branch.source_history))
                        candidate_decisions = decision_predicates(list(candidate.history))
                        if candidate.first_predicate_disagreement_step is None and reference_decisions != candidate_decisions:
                            candidate.first_predicate_disagreement_step = elapsed
                        if elapsed in HORIZONS:
                            for predicate in sorted(reference_decisions):
                                records.append(
                                    {
                                        **asdict(bundle),
                                        "branch_id": (
                                            f"{bundle.base_seed}:{bundle.disturbance}:{branch.phase}:{branch.branch_step}"
                                        ),
                                        "phase": branch.phase,
                                        "branch_step": branch.branch_step,
                                        "horizon": elapsed,
                                        "actual_continuation_steps": elapsed,
                                        "disturbance": bundle.disturbance,
                                        "schedule": schedule.canonical_dict(),
                                        "schedule_sha256": schedule.sha256,
                                        "schedule_equivalent": True,
                                        "protocol": candidate.protocol,
                                        "continuation": candidate.continuation,
                                        "predicate": predicate,
                                        "reference_decision": reference_decisions[predicate],
                                        "candidate_decision": candidate_decisions[predicate],
                                        "decision_match": (reference_decisions[predicate] == candidate_decisions[predicate]),
                                        "immediate_mismatch": candidate.immediate,
                                        "first_step_mismatch": candidate.first_step,
                                        "first_numerical_divergence_step": (candidate.first_numerical_divergence_step),
                                        "first_observation_divergence_step": (candidate.first_observation_divergence_step),
                                        "first_contact_divergence_step": (candidate.first_contact_divergence_step),
                                        "first_action_divergence_step": (candidate.first_action_divergence_step),
                                        "first_predicate_disagreement_step": (candidate.first_predicate_disagreement_step),
                                        "maximum_trajectory_error": (candidate.maximum_trajectory_error),
                                        "terminal_trajectory_error": (candidate.terminal_trajectory_error),
                                    }
                                )
                observation = next_observation
                complete = branches and all(next_step - branch.branch_step >= max(HORIZONS) for branch in branches.values())
                if next_step >= schedule.start_step + max(HORIZONS) and complete:
                    break

            branch_steps = {phase: branch.branch_step for phase, branch in branches.items()}
            controls = {
                "contact_measurement_available": bool(phase_tracker.seen & {"first_contact", "stable_grasp", "initial_lift"}),
                "nominal_sustained_lift_before_disturbance": (
                    "sustained_lift" in branch_steps and branch_steps["sustained_lift"] < schedule.start_step
                ),
                "disturbance_onset_observed": ("disturbance_onset" in branch_steps),
                "branch_steps_unique": (len(branch_steps.values()) == len(set(branch_steps.values()))),
                "horizon_semantics_exact": all(record["horizon"] == record["actual_continuation_steps"] for record in records),
                "schedule_equivalence_asserted": all(record["schedule_equivalent"] for record in records),
                "exact_actions_identical": all(
                    record["first_action_divergence_step"] is None
                    for record in records
                    if record["continuation"] == "exact_action"
                ),
                "required_protocols_present": set(record["protocol"] for record in records) == set(PROTOCOLS),
                "required_continuations_present": set(record["continuation"] for record in records) == set(CONTINUATIONS),
                "required_horizons_present": set(int(record["horizon"]) for record in records) == set(HORIZONS),
                "phases_observed": sorted(branch_steps),
                "phases_missing": sorted(set(PHASES) - set(branch_steps)),
                "branch_steps": branch_steps,
            }
            controls["passed"] = all(
                bool(controls[name])
                for name in (
                    "contact_measurement_available",
                    "nominal_sustained_lift_before_disturbance",
                    "disturbance_onset_observed",
                    "branch_steps_unique",
                    "horizon_semantics_exact",
                    "schedule_equivalence_asserted",
                    "exact_actions_identical",
                    "required_protocols_present",
                    "required_continuations_present",
                    "required_horizons_present",
                )
            )
            result = {
                "schema_version": 1,
                "seed_bundle": asdict(bundle),
                "schedule": schedule.canonical_dict(),
                "schedule_sha256": schedule.sha256,
                "protocol_inventory": {
                    PROTOCOL_A: "scene plus basic manager state",
                    PROTOCOL_B: "expanded exposed runtime state",
                    "unavailable": [
                        "PhysX warm-start impulses",
                        "PhysX contact manifolds and caches",
                        "PhysX broadphase and solver-internal state",
                    ],
                    "policy_recurrent_state": "absent_feed_forward_actor",
                    "observation_history": "absent_in_task_configuration",
                    "interval_events": "absent_in_task_configuration",
                },
                "controls": controls,
                "disturbance_applications": disturbance_applications,
                "records": records,
            }
            write_json(args.output.resolve(), result)
            if not controls["passed"]:
                raise RuntimeError(f"worker evidence controls failed: {controls}")
            return 0
        finally:
            env.close()


def _run_worker_and_exit() -> None:
    exit_code = 1
    try:
        exit_code = worker_main(_worker_args)
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            _app.close()
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    if _bootstrap_args.worker:
        _run_worker_and_exit()
    else:
        raise SystemExit(orchestrator_main(build_orchestrator_parser().parse_args()))
