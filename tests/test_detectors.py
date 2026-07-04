import numpy as np

from ipfd import detectors
from ipfd.adapters.synthetic import make_silent_failure_rollout, make_success_rollout


def test_action_variance_score_range_and_spike():
    r = make_silent_failure_rollout(seed=0, t_ponr=90, t_failure=160)
    s = detectors.action_variance_score(r.actions, baseline_window=20)
    assert s.shape == (r.T,)
    assert np.all((s >= 0) & (s <= 1))
    # thrash near observable failure must score higher than the calm early phase
    assert s[150:160].mean() > s[:80].mean() + 0.2


def test_entropy_collapse_fires_in_doom_window():
    r = make_silent_failure_rollout(seed=1, t_ponr=90, t_failure=160)
    s = detectors.entropy_collapse_score(r.entropy, baseline_window=20)
    assert s.size == r.T
    # entropy collapses (overconfidence) after PoNR -> detector should light up
    assert s[120:160].mean() > s[:80].mean()


def test_entropy_collapse_none_returns_empty():
    assert detectors.entropy_collapse_score(None).size == 0


def test_representation_drift_grows_after_ponr():
    r = make_silent_failure_rollout(seed=2, t_ponr=90, t_failure=160)
    d = detectors.representation_drift(r.embeddings, ref_window=10)
    assert d.size == r.T
    assert d[:80].mean() < d[120:].mean()


def test_drift_score_missing_embeddings():
    assert detectors.drift_score(None).size == 0


def test_success_rollout_stays_quiet():
    r = make_success_rollout(seed=3)
    imm = detectors.failure_imminence_score(r)
    # negative control: no sustained alarm on a clean success
    assert detectors.first_alarm(imm, threshold=0.5, persistence=3) is None


def test_first_alarm_persistence():
    score = np.zeros(50)
    score[10] = 1.0  # single blip must NOT trigger with persistence 3
    assert detectors.first_alarm(score, 0.5, 3) is None
    score[20:25] = 1.0
    assert detectors.first_alarm(score, 0.5, 3) == 20
