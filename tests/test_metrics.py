import numpy as np

from ipfd import metrics


def test_time_to_failure():
    assert metrics.time_to_failure(60, dt=0.1) == 6.0
    assert metrics.time_to_failure(None, dt=0.1) is None


def test_failure_lead_time_positive_when_alarm_early():
    # alarm at 100, visible failure at 160 -> 60 steps of warning
    assert metrics.failure_lead_time(100, 160, dt=0.1) == 6.0


def test_ponr_lead_time_sign():
    # alarm before PoNR -> positive (actionable)
    assert metrics.ponr_lead_time(80, 90, dt=1.0) == 10.0
    # alarm after PoNR -> negative (too late)
    assert metrics.ponr_lead_time(120, 90, dt=1.0) == -30.0


def test_silent_doom_window():
    assert metrics.silent_doom_window(90, 160, dt=0.1) == 7.0
    assert metrics.silent_doom_window(None, 160, dt=0.1) is None


def test_false_continuity_rate_all_quiet():
    imm = np.zeros(200)  # detector never fires during doom window -> 100% false continuity
    assert metrics.false_continuity_rate(imm, 90, 160, 0.5) == 1.0


def test_false_continuity_rate_all_loud():
    imm = np.ones(200)
    assert metrics.false_continuity_rate(imm, 90, 160, 0.5) == 0.0


def test_false_continuity_none_without_window():
    assert metrics.false_continuity_rate(np.zeros(10), None, 5, 0.5) is None


def test_drift_at_collapse():
    drift = np.arange(100, dtype=float)
    assert metrics.drift_magnitude_at_collapse(drift, 90) == 90.0
    assert metrics.drift_magnitude_at_collapse(np.zeros(0), 90) is None
