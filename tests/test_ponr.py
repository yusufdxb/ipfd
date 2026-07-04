import numpy as np

from ipfd.adapters.synthetic import make_silent_failure_rollout, make_success_rollout
from ipfd.ponr import point_of_no_return


def test_ponr_matches_ground_truth():
    r = make_silent_failure_rollout(seed=0, t_ponr=90, t_failure=160)
    assert point_of_no_return(r.recovery_success) == 90


def test_ponr_none_when_always_recoverable():
    r = make_success_rollout(seed=1)
    assert point_of_no_return(r.recovery_success) is None


def test_ponr_none_without_probe():
    assert point_of_no_return(None) is None


def test_ponr_uses_last_recoverable_step():
    # recoverable flickers back on at t=5, so PoNR is after the LAST True (t=6), i.e. 7
    rec = np.array([True, True, False, False, False, True, True, False, False, False])
    assert point_of_no_return(rec) == 7


def test_ponr_all_false_is_zero():
    assert point_of_no_return(np.zeros(10, dtype=bool)) == 0
