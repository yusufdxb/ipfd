"""Input-validation regression tests.

These lock in the fail-fast, clear-error behavior added by the input-hardening
pass: a malformed Rollout, an unknown detector weight, or a bad dt/t_failure must
raise a ValueError whose message explains what is wrong, rather than silently
producing NaN cascades or ignoring the input.
"""

import numpy as np
import pytest

from ipfd import detectors
from ipfd.types import Rollout


def _obs(T=10, d=3):
    return np.zeros((T, d))


def _act(T=10, d=2):
    return np.zeros((T, d))


# --- Rollout: shape / length ------------------------------------------------


def test_empty_rollout_rejected():
    with pytest.raises(ValueError, match=r"zero timesteps.*at least one timestep"):
        Rollout(observations=np.zeros((0, 3)), actions=np.zeros((0, 2)), success=False)


def test_zero_observation_dimension_rejected():
    with pytest.raises(ValueError, match=r"obs_dim=0"):
        Rollout(observations=np.zeros((10, 0)), actions=np.zeros((10, 2)), success=False)


def test_zero_action_dimension_rejected():
    with pytest.raises(ValueError, match=r"act_dim=0"):
        Rollout(observations=np.zeros((10, 3)), actions=np.zeros((10, 0)), success=False)


# --- Rollout: non-finite values ---------------------------------------------


def test_nan_observations_rejected():
    obs = _obs()
    obs[4, 1] = np.nan
    with pytest.raises(ValueError) as ei:
        Rollout(observations=obs, actions=_act(), success=False)
    msg = str(ei.value)
    assert "observations" in msg
    assert "non-finite" in msg
    assert "[4, 1]" in msg


def test_nan_actions_rejected():
    act = _act()
    act[2, 0] = np.nan
    with pytest.raises(ValueError) as ei:
        Rollout(observations=_obs(), actions=act, success=False)
    msg = str(ei.value)
    assert "actions" in msg
    assert "non-finite" in msg


def test_inf_observations_rejected():
    obs = _obs()
    obs[7, 2] = np.inf
    with pytest.raises(ValueError) as ei:
        Rollout(observations=obs, actions=_act(), success=False)
    assert "observations" in str(ei.value)
    assert "non-finite" in str(ei.value)


def test_nan_recovery_success_rejected():
    # NaN would silently cast to True under the bool conversion; it must be caught.
    rec = np.ones(10, dtype=float)
    rec[3] = np.nan
    with pytest.raises(ValueError) as ei:
        Rollout(observations=_obs(), actions=_act(), success=False, recovery_success=rec)
    msg = str(ei.value)
    assert "recovery_success" in msg
    assert "non-finite" in msg


def test_nan_entropy_rejected():
    entropy = np.ones(10)
    entropy[3] = np.nan
    with pytest.raises(ValueError, match="entropy.*non-finite"):
        Rollout(observations=_obs(), actions=_act(), success=False, entropy=entropy)


def test_nan_embeddings_rejected():
    embeddings = np.zeros((10, 4))
    embeddings[2, 1] = np.nan
    with pytest.raises(ValueError, match="embeddings.*non-finite"):
        Rollout(observations=_obs(), actions=_act(), success=False, embeddings=embeddings)


def test_inf_recovery_success_rejected():
    rec = np.ones(10, dtype=float)
    rec[5] = np.inf
    with pytest.raises(ValueError) as ei:
        Rollout(observations=_obs(), actions=_act(), success=False, recovery_success=rec)
    assert "recovery_success" in str(ei.value)


# --- Rollout: dt ------------------------------------------------------------


def test_dt_zero_rejected():
    with pytest.raises(ValueError) as ei:
        Rollout(observations=_obs(), actions=_act(), success=False, dt=0.0)
    msg = str(ei.value)
    assert "dt" in msg
    assert "> 0" in msg


def test_dt_negative_rejected():
    with pytest.raises(ValueError) as ei:
        Rollout(observations=_obs(), actions=_act(), success=False, dt=-0.5)
    msg = str(ei.value)
    assert "dt" in msg
    assert "-0.5" in msg


def test_inf_dt_rejected():
    with pytest.raises(ValueError, match="dt.*finite"):
        Rollout(observations=_obs(), actions=_act(), success=False, dt=float("inf"))


# --- Rollout: t_failure typing ----------------------------------------------


def test_float_t_failure_rejected():
    with pytest.raises(ValueError) as ei:
        Rollout(observations=_obs(), actions=_act(), success=False, t_failure=5.0)
    msg = str(ei.value)
    assert "t_failure" in msg
    assert "float" in msg


def test_numpy_float_t_failure_rejected():
    with pytest.raises(ValueError) as ei:
        Rollout(observations=_obs(), actions=_act(), success=False, t_failure=np.float64(5))
    assert "float" in str(ei.value)


def test_bool_t_failure_rejected():
    with pytest.raises(ValueError) as ei:
        Rollout(observations=_obs(), actions=_act(), success=False, t_failure=True)
    msg = str(ei.value)
    assert "t_failure" in msg
    assert "bool" in msg


def test_valid_t_failure_still_accepted():
    # Guard against over-tightening: a plain int in range must still work.
    r = Rollout(observations=_obs(), actions=_act(), success=False, t_failure=5)
    assert r.t_failure == 5
    r2 = Rollout(observations=_obs(), actions=_act(), success=False, t_failure=np.int64(5))
    assert r2.t_failure == 5


def test_success_with_t_failure_rejected():
    with pytest.raises(ValueError, match="success=True.*t_failure"):
        Rollout(observations=_obs(), actions=_act(), success=True, t_failure=5)


# --- Detector weights -------------------------------------------------------


def test_unknown_detector_weight_rejected():
    r = Rollout(observations=_obs(), actions=_act(), success=False)
    with pytest.raises(ValueError) as ei:
        detectors.failure_imminence_score(r, weights={"actoin_variance": 1.0})
    msg = str(ei.value)
    assert "actoin_variance" in msg  # the offending unknown key
    assert "valid keys" in msg
    assert "action_variance" in msg  # the valid keys are listed


def test_known_detector_weights_still_accepted():
    r = Rollout(observations=_obs(), actions=_act(), success=False)
    s = detectors.failure_imminence_score(r, weights={"drift": 0.5, "action_variance": 1.0})
    assert s.shape == (r.T,)


def test_nonfinite_detector_weight_rejected():
    r = Rollout(observations=_obs(), actions=_act(), success=False)
    with pytest.raises(ValueError, match=r"detector weight 'drift'.*finite"):
        detectors.failure_imminence_score(r, weights={"drift": np.nan})


def test_first_alarm_rejects_nonpositive_persistence():
    with pytest.raises(ValueError, match="persistence.*>= 1"):
        detectors.first_alarm(np.zeros(5), threshold=0.5, persistence=0)
