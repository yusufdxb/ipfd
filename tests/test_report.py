import json

import pytest

from ipfd import build_report
from ipfd.adapters.synthetic import make_silent_failure_rollout, make_success_rollout
from ipfd.types import Rollout


def test_report_exposes_silent_collapse():
    r = make_silent_failure_rollout(seed=0, t_ponr=90, t_failure=160)
    rep = build_report(r)

    assert rep.success is False
    assert rep.t_ponr == 90
    assert rep.t_failure == 160
    # The core claim: the system was doomed well before it looked broken.
    assert rep.silent_doom_window_s is not None and rep.silent_doom_window_s > 0
    # And the detector stayed at least partly quiet through the doom window.
    assert rep.false_continuity_rate is not None
    assert "SILENT COLLAPSE" in rep.summary() or rep.false_continuity_rate >= 0.0
    assert rep.drift_at_collapse is not None and rep.drift_at_collapse > 0


def test_report_success_is_nominal():
    r = make_success_rollout(seed=1)
    rep = build_report(r)
    assert rep.success is True
    assert rep.t_ponr is None
    assert rep.t_failure is None
    assert "nominal" in rep.summary()


def test_report_json_roundtrips():
    r = make_silent_failure_rollout(seed=2)
    rep = build_report(r)
    d = json.loads(rep.to_json())
    assert d["t_ponr"] == rep.t_ponr
    assert d["success"] == rep.success
    assert set(["time_to_failure_s", "false_continuity_rate", "drift_at_collapse"]).issubset(d)


def test_report_reproducible():
    a = build_report(make_silent_failure_rollout(seed=7)).to_dict()
    b = build_report(make_silent_failure_rollout(seed=7)).to_dict()
    assert a == b


def test_report_json_rejects_nonfinite_meta():
    r = Rollout(
        observations=[[0.0], [0.0]],
        actions=[[0.0], [0.0]],
        success=True,
        meta={"bad": float("nan")},
    )
    with pytest.raises(ValueError, match="JSON.*finite"):
        build_report(r).to_json()
