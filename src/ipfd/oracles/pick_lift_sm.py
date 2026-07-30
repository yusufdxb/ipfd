"""Vendored Isaac Lab scripted pick-and-lift state machine (installable).

Copyright (c) 2022-2026, The Isaac Lab Project Developers. BSD-3-Clause.
Reproduced verbatim from ``scripts/environments/state_machine/lift_cube_sm.py`` in
Isaac Lab so the scripted policy matches the upstream reference implementation. This is the recovery *oracle* whose
success/failure defines the Point of No Return (see :mod:`ipfd.ponr`).

IMPORTANT: this module imports ``torch`` and ``warp`` at module scope and compiles
its ``@wp.kernel`` / ``wp.constant`` against the live warp runtime. Import it ONLY
AFTER Isaac Lab's ``AppLauncher`` has launched the sim, exactly as in the
reference script. It is therefore deliberately *not* re-exported from
``ipfd.oracles.__init__``, so the pure analysis layer stays GPU-free.
"""

from __future__ import annotations

import torch
import warp as wp


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    def __init__(self, dt, num_envs, device="cpu", position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def snapshot(self):
        """Save the SM's internal progress so two branches can drive identically."""
        return (self.sm_state.clone(), self.sm_wait_time.clone())

    def restore(self, snap):
        """Restore SM internal progress (warp views share memory, so no re-bind)."""
        self.sm_state[:] = snap[0]
        self.sm_wait_time[:] = snap[1]

    def compute(self, ee_pose, object_pose, des_object_pose):
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp, self.sm_state_wp, self.sm_wait_time_wp,
                ee_pose_wp, object_pose_wp, des_object_pose_wp,
                self.des_ee_pose_wp, self.des_gripper_state_wp, self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )
        return torch.cat([self.des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


# --- env-facing helpers (shared by the validation + diagnostic scripts) --------

def sm_action(env, sm: PickAndLiftSm, desired_orientation: torch.Tensor) -> torch.Tensor:
    """Compute the scripted action from live env tensors (mirrors the reference)."""
    ee = env.unwrapped.scene["ee_frame"]
    origins = env.unwrapped.scene.env_origins
    tcp_pos = wp.to_torch(ee.data.target_pos_w)[..., 0, :].clone() - origins
    tcp_quat = wp.to_torch(ee.data.target_quat_w)[..., 0, :].clone()
    obj = env.unwrapped.scene["object"].data
    obj_pos = wp.to_torch(obj.root_pos_w) - origins
    des_pos = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
    return sm.compute(
        torch.cat([tcp_pos, tcp_quat], dim=-1),
        torch.cat([obj_pos, desired_orientation], dim=-1),
        torch.cat([des_pos, desired_orientation], dim=-1),
    )


def object_z(env) -> float:
    return float(wp.to_torch(env.unwrapped.scene["object"].data.root_pos_w)[0, 2].item())


def identity_action(dev) -> torch.Tensor:
    a = torch.zeros((1, 8), dtype=torch.float32, device=dev)
    a[:, 3] = 1.0  # matches reference initial action
    return a
