import numpy as np
import pytest

from ipfd.adapters.synthetic import make_silent_failure_rollout, make_success_rollout
from ipfd.ponr import aggregate_repeated_probes, point_of_no_return, point_of_no_return_repeated


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


def test_repeated_probe_requires_consensus_before_false():
    probes = np.array([[True, True, True], [False, True, False], [False, False, False]])
    stats = aggregate_repeated_probes(probes, min_repeats=3, min_confidence=0.8)
    assert stats.verdict.tolist() == [True, True, False]
    assert stats.false_fraction.tolist() == [0.0, 2 / 3, 1.0]
    assert point_of_no_return_repeated(probes, min_repeats=3, min_confidence=0.8) == 2


def test_repeated_probe_single_sample_is_conservative():
    probes = np.array([[False], [False]])
    assert point_of_no_return_repeated(probes, min_repeats=3) is None


def test_ponr_all_false_is_zero():
    assert point_of_no_return(np.zeros(10, dtype=bool)) == 0


@pytest.mark.parametrize(
    "values",
    [
        np.array([[True, False]]),
        np.array([0.0, 1.0]),
        np.array([True, np.nan]),
    ],
)
def test_ponr_rejects_non_boolean_vector(values):
    with pytest.raises(ValueError):
        point_of_no_return(values)


def test_repeated_probe_rejects_non_binary_integer_values():
    with pytest.raises(ValueError, match="0/1"):
        aggregate_repeated_probes(np.array([[0, 2, 1]]))


def test_empty_repeat_axis_reports_zero_confidence():
    stats = aggregate_repeated_probes(np.empty((2, 0), dtype=bool))
    assert stats.confidence.tolist() == [0.0, 0.0]
    assert stats.verdict.tolist() == [True, True]
