from pathlib import Path

from ipfd import build_report
from ipfd.actionability import evaluate_actionability, point_of_no_return_interval
from ipfd.replay import load_rollout

FIXTURE = Path(__file__).parent / "fixtures" / "learned_teleport_rollout.npz"


def _report():
    return build_report(load_rollout(FIXTURE))


def test_strided_ponr_is_reported_as_an_interval():
    interval = point_of_no_return_interval(_report(), probe_stride=8)
    assert interval is not None
    assert (interval.earliest, interval.latest) == (49, 56)


def test_alarm_before_known_disturbance_is_not_actionable():
    result = evaluate_actionability(_report(), disturbance_onset=40, probe_stride=8)
    assert result.alarm_relation == "pre_disturbance"
    assert result.valid_actionable_warning is False


def test_alarm_inside_ponr_interval_is_explicitly_ambiguous():
    report = _report()
    report.t_alarm = 50
    result = evaluate_actionability(report, disturbance_onset=48, probe_stride=8)
    assert result.alarm_relation == "ambiguous_within_ponr_interval"
    assert result.valid_actionable_warning is False


def test_alarm_after_disturbance_but_before_ponr_is_actionable():
    report = _report()
    report.t_alarm = 45
    result = evaluate_actionability(report, disturbance_onset=40, probe_stride=8)
    assert result.alarm_relation == "definitely_actionable"
    assert result.valid_actionable_warning is True
    assert result.alarm_delay_from_disturbance_s == 0.1


def test_disturbance_at_latest_ponr_has_no_actionability_window():
    result = evaluate_actionability(_report(), disturbance_onset=56, probe_stride=8)
    assert result.window_status == "empty"
    assert result.alarm_relation == "pre_disturbance"
    assert result.valid_actionable_warning is False


def test_disturbance_bounds_are_validated():
    report = _report()
    for onset in (-1, report.T):
        try:
            evaluate_actionability(report, disturbance_onset=onset)
        except ValueError as exc:
            assert "disturbance_onset" in str(exc)
        else:
            raise AssertionError("expected invalid disturbance onset to fail")
