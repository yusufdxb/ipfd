"""CI tests for the simulator-free primitives of the Isaac Lab adapter.

The adapter's *sim touchpoints* need a GPU and Isaac Lab, but the pieces that
encode the env-isolation recovery-probe logic -- state cloning, origin-shifting a
checkpoint into the probe cell, and expanding strided probe verdicts into a
per-step recovery array -- are pure NumPy and must be correct. These are the parts
that used to live, untested, in scripts/. They run here with no GPU.
"""

from __future__ import annotations

import numpy as np

from ipfd.adapters.isaac_lab import (
    _deep_clone,
    collect_rollout,
    forward_fill_recovery,
    offset_root_positions,
    slice_state,
)


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
    """The poison single-env probe is gone: probing a num_envs==1 env must error."""

    class _Env:
        class unwrapped:
            num_envs = 1

    class _Ctl:
        def act(self, obs):
            return np.zeros(8), None, None

    import pytest

    with pytest.raises(ValueError, match="num_envs >= 2"):
        collect_rollout(_Env(), _Ctl(), recovery_controller=_Ctl())
