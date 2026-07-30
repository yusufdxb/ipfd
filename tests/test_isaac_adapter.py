"""CI tests for the simulator-free primitives of the Isaac Lab adapter.

The adapter's *sim touchpoints* need a GPU and Isaac Lab, but the pieces that
encode the env-isolation recovery-probe logic -- state cloning, origin-shifting a
checkpoint into the probe cell, and expanding strided probe verdicts into a
per-step recovery array -- are pure NumPy and must be correct. These are the parts
that used to live, untested, in scripts/. They run here with no GPU.
"""

from __future__ import annotations

import sys
import warnings
from types import ModuleType

import numpy as np
import pytest

import ipfd.adapters.isaac_lab as isaac_adapter
from ipfd.adapters.isaac_lab import (
    TESTED_ISAAC_LAB_VERSION,
    PhysicalRecoveryCheck,
    _deep_clone,
    _require_isaac_lab,
    collect_rollout,
    forward_fill_recovery,
    offset_root_positions,
    probe_recovery_isolated,
    slice_state,
)


def _mock_isaac_lab(monkeypatch, version):
    module = ModuleType("isaaclab")
    monkeypatch.setitem(sys.modules, "isaaclab", module)
    monkeypatch.setattr(isaac_adapter, "_isaac_lab_version_checked", False)
    if version is None:
        def missing_version(_name):
            raise isaac_adapter.metadata.PackageNotFoundError

        monkeypatch.setattr(isaac_adapter.metadata, "version", missing_version)
    else:
        monkeypatch.setattr(isaac_adapter.metadata, "version", lambda _name: version)
    return module


def test_matching_isaac_lab_version_does_not_warn(monkeypatch):
    _mock_isaac_lab(monkeypatch, TESTED_ISAAC_LAB_VERSION)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _require_isaac_lab()

    assert caught == []


def test_mismatched_isaac_lab_version_warns_exactly_once(monkeypatch):
    installed = "5.0.0"
    _mock_isaac_lab(monkeypatch, installed)
    monkeypatch.setenv("IPFD_EXPECTED_ISAAC_LAB_VERSION", TESTED_ISAAC_LAB_VERSION)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _require_isaac_lab()
        _require_isaac_lab()

    assert len(caught) == 1
    assert TESTED_ISAAC_LAB_VERSION in str(caught[0].message)
    assert installed in str(caught[0].message)


def test_missing_isaac_lab_version_metadata_is_import_safe(monkeypatch):
    _mock_isaac_lab(monkeypatch, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _require_isaac_lab()

    assert caught == []


def _fake_state() -> dict:
    """A minimal get_state()-shaped nested dict using NumPy arrays as tensors."""
    return {
        "articulation": {
            "robot": {
                "root_pose": np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
                "joint_pos": np.array([[0.1, 0.2]]),
            }
        },
        "rigid_object": {
            "object": {"root_pose": np.array([[0.5, 0.5, 0.05, 0.0, 0.0, 0.0, 1.0]])}
        },
    }


def test_deep_clone_is_independent():
    s = _fake_state()
    c = _deep_clone(s)
    c["articulation"]["robot"]["root_pose"][0, 0] = 999.0
    assert s["articulation"]["robot"]["root_pose"][0, 0] == 1.0  # original untouched


def test_offset_shifts_only_positions():
    s = _fake_state()
    delta = np.array([10.0, 0.0, 0.0])
    out = offset_root_positions(s, delta)
    # x shifted by 10, orientation quaternion (indices 3:7) untouched.
    np.testing.assert_allclose(out["articulation"]["robot"]["root_pose"][0, :3], [11.0, 2.0, 3.0])
    np.testing.assert_allclose(out["articulation"]["robot"]["root_pose"][0, 3:], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(out["rigid_object"]["object"]["root_pose"][0, :3], [10.5, 0.5, 0.05])
    # joint state is origin-independent and must be untouched.
    np.testing.assert_allclose(out["articulation"]["robot"]["joint_pos"], [[0.1, 0.2]])
    # source is not mutated.
    np.testing.assert_allclose(s["articulation"]["robot"]["root_pose"][0, :3], [1.0, 2.0, 3.0])


class _FakeTensor:
    """Stand-in for a torch tensor: indexable, cloneable, not a dict/list/tuple.

    ``get_state()`` leaves are torch tensors; ``slice_state`` slices those (via the
    duck-typed ``.clone``) and leaves everything else untouched.
    """

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def __getitem__(self, idx):
        return _FakeTensor(self.data[idx])

    def clone(self):
        return _FakeTensor(self.data.copy())


def test_slice_state_indexes_tensor_leaves_only():
    state = {"a": _FakeTensor([[1.0], [2.0], [3.0]]), "b": ("tag", _FakeTensor([[4.0], [5.0], [6.0]]))}
    out = slice_state(state, slice(1, 2))
    np.testing.assert_allclose(out["a"].data, [[2.0]])
    assert out["b"][0] == "tag"  # non-tensor leaf passes through
    np.testing.assert_allclose(out["b"][1].data, [[5.0]])


def test_forward_fill_recovery_flips_and_holds():
    # Recoverable through step 30, then lost from step 31 onward.
    verdicts = {0: True, 10: True, 20: True, 30: True, 40: False, 50: False}
    rec = forward_fill_recovery(verdicts, length=60)
    assert rec[:40].all()  # holds True up to the last True probe's reach
    assert not rec[40:].any()  # holds False after the flip


def test_forward_fill_defaults_true_before_first_probe():
    rec = forward_fill_recovery({5: False}, length=10)
    assert rec[:5].all()  # unprobed prefix defaults recoverable
    assert not rec[5:].any()


def test_forward_fill_noisy_prefix_last_flip_wins():
    # A stray early True among False (the known cold-contact pre-grasp noise) must
    # not move the PoNR flip, which is defined by the LAST True->False transition.
    verdicts = {0: False, 10: True, 20: False, 90: True, 100: True, 110: False}
    rec = forward_fill_recovery(verdicts, length=120)
    # last True probe is 100 (reaches to 109); flip to False at 110.
    assert rec[100] and rec[109]
    assert not rec[110]


def test_collect_rollout_rejects_single_env_probe():
    """The unsupported single-env probe is gone: num_envs==1 must error."""

    class _Env:
        class unwrapped:
            num_envs = 1

    def _policy(obs):
        return np.zeros((1, 8))

    import pytest

    with pytest.raises(ValueError, match="num_envs >= 2"):
        collect_rollout(
            _Env(), _policy,
            object_height=lambda e, i: 0.0, rest_height=0.0,
            recovery_policy=_policy,
        )


def test_collect_rollout_rejects_same_primary_and_probe_env():
    class _Env:
        class unwrapped:
            num_envs = 2

    def _policy(obs):
        return np.zeros((2, 8))

    import pytest

    with pytest.raises(ValueError, match="primary_env.*probe_env.*distinct"):
        collect_rollout(
            _Env(), _policy,
            object_height=lambda e, i: 0.0, rest_height=0.0,
            recovery_policy=_policy,
            primary_env=0, probe_env=0,
        )


def test_collect_rollout_rejects_probe_env_out_of_range():
    class _Env:
        class unwrapped:
            num_envs = 2

    def _policy(obs):
        return np.zeros((2, 8))

    import pytest

    with pytest.raises(ValueError, match="probe_env.*range"):
        collect_rollout(
            _Env(), _policy,
            object_height=lambda e, i: 0.0, rest_height=0.0,
            recovery_policy=_policy,
            primary_env=0, probe_env=2,
        )


def test_collect_rollout_rejects_zero_probe_stride():
    class _Env:
        class unwrapped:
            num_envs = 2

    def _policy(obs):
        return np.zeros((2, 8))

    import pytest

    with pytest.raises(ValueError, match="probe_stride.*>= 1"):
        collect_rollout(
            _Env(), _policy,
            object_height=lambda e, i: 0.0, rest_height=0.0,
            recovery_policy=_policy,
            probe_stride=0,
        )


class _Entity:
    def __init__(self, data):
        self.data = data

    def find_bodies(self, name, preserve_order=True):
        assert name == "panda_hand"
        assert preserve_order
        return [0], ["panda_hand"]


class _Data:
    pass


def _physical_env(*, object_position=(0.5, 0.0, 0.2), ee_position=None, finger_width=0.05):
    if ee_position is None:
        ee_position = object_position
    obj = _Data()
    obj.root_pos_w = np.array([object_position], dtype=float)
    robot = _Data()
    robot.body_pos_w = np.array([[ee_position]], dtype=float)
    robot.joint_pos = np.array([[0.0, 0.0, finger_width / 2, finger_width / 2]], dtype=float)

    class _Scene(dict):
        env_origins = np.zeros((1, 3), dtype=float)

    class _Unwrapped:
        scene = _Scene(object=_Entity(obj), robot=_Entity(robot))

    class _Env:
        unwrapped = _Unwrapped()

    return _Env()


def test_physical_recovery_check_requires_sustained_grasp_geometry():
    check = PhysicalRecoveryCheck(sustain_steps=2)
    env = _physical_env()
    assert not check(env, 0, rest_height=0.05, lift_threshold=0.06)
    assert check(env, 0, rest_height=0.05, lift_threshold=0.06)
    check.reset(0)
    assert not check(env, 0, rest_height=0.05, lift_threshold=0.06)


def test_physical_recovery_check_fails_closed_outside_grasp():
    check = PhysicalRecoveryCheck(sustain_steps=1)
    assert not check(
        _physical_env(ee_position=(0.0, 0.0, 0.2)),
        0,
        rest_height=0.05,
        lift_threshold=0.06,
    )
    assert not check(
        _physical_env(finger_width=0.08),
        0,
        rest_height=0.05,
        lift_threshold=0.06,
    )


def test_probe_resets_policy_and_recovery_state(monkeypatch):
    # probe_recovery_isolated imports torch internally, so this one test cannot
    # run on the torch-free analysis-layer install that CI uses.
    torch = pytest.importorskip("torch")

    class _Scene(dict):
        env_origins = torch.zeros((2, 3))

        def reset_to(self, state, env_ids):
            self.last_reset = (state, env_ids)

    object_data = _Data()
    object_data.root_pose_w = torch.zeros((2, 7))
    scene = _Scene(object=_Entity(object_data))

    class _Unwrapped:
        num_envs = 2
        device = "cpu"
        episode_length_buf = torch.ones(2, dtype=torch.long)

        def __init__(self):
            self.scene = scene

    class _Env:
        def __init__(self):
            self.unwrapped = _Unwrapped()

        def get_observations(self):
            return {"policy": torch.zeros((2, 1))}

        def step(self, actions):
            return self.get_observations(), torch.zeros(2), torch.zeros(2, dtype=torch.bool), {}

    class _Policy:
        def __init__(self):
            self.resets = []

        def __call__(self, obs):
            return torch.zeros((2, 1))

        def reset(self, dones):
            self.resets.append(dones.clone())

    warp = ModuleType("warp")
    warp.to_torch = lambda value: value
    monkeypatch.setitem(sys.modules, "warp", warp)
    recovery_resets = []
    policy = _Policy()
    assert probe_recovery_isolated(
        _Env(),
        {},
        policy,
        recovered=lambda _env, _index: True,
        budget=2,
        reset_recovered=recovery_resets.append,
    )
    assert recovery_resets == [1]
    assert len(policy.resets) == 1
    assert policy.resets[0].tolist() == [True, True]
