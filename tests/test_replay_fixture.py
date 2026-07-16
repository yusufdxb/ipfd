"""GPU-free regression test: replay real recorded rollouts and re-derive the report.

These fixtures were captured from a LIVE Isaac Lab 4.5.22 session driving IPFD on
NVIDIA's official published ``Isaac-Lift-Cube-Franka-v0`` rsl_rl checkpoint
(``scripts/verify_learned_policy.py --use_pretrained --probe``). The ``.npz`` holds
only the rollout's NumPy arrays; no simulator, GPU, or Isaac Lab is needed to reload
it. This test freezes IPFD's analysis contract: given the exact same rollout, the
report must come out byte-for-byte identical, forever. A change to any detector,
metric, or PoNR rule that moves a published number breaks this test loudly.

Regenerate the goldens ONLY when intentionally changing the analysis (and say so in
the changelog):

    python -c "from ipfd.replay import load_rollout; from ipfd import build_report; \
        import pathlib; \
        [pathlib.Path(f'tests/fixtures/{n}_report.json').write_text( \
            build_report(load_rollout(f'tests/fixtures/{n}_rollout.npz')).to_json() + chr(10)) \
         for n in ('learned_teleport','learned_slip')]"
"""

from __future__ import annotations

import pathlib

import pytest

from ipfd import build_report
from ipfd.replay import load_rollout, save_rollout

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CASES = ["learned_teleport", "learned_slip"]


@pytest.mark.parametrize("case", CASES)
def test_report_matches_frozen_golden_byte_for_byte(case: str) -> None:
    """The report re-derived from the recorded rollout equals the frozen JSON exactly."""
    rollout = load_rollout(FIXTURES / f"{case}_rollout.npz")
    got = build_report(rollout).to_json(indent=2) + "\n"
    golden = (FIXTURES / f"{case}_report.json").read_text()
    assert got == golden, (
        f"{case}: report no longer matches the frozen golden. If this change is "
        f"intentional, regenerate the golden (see this module's docstring)."
    )


def test_teleport_headline_claim() -> None:
    """The published headline: irrecoverable teleport -> PoNR at step 56, +0.72s lead."""
    report = build_report(load_rollout(FIXTURES / "learned_teleport_rollout.npz"))
    assert report.success is False
    assert report.t_ponr == 56
    assert report.t_failure == 57
    assert report.t_alarm == 20
    assert report.ponr_lead_time_s == pytest.approx(0.72)
    # Env-isolation integrity: probing never perturbed the pristine primary env.
    assert report.meta["primary_integrity_max_delta"] == 0.0


def test_slip_is_recoverable_no_ponr() -> None:
    """The recoverable slip case: no point of no return is claimed."""
    report = build_report(load_rollout(FIXTURES / "learned_slip_rollout.npz"))
    assert report.success is False
    assert report.t_ponr is None
    assert report.silent_doom_window_s is None
    assert report.meta["primary_integrity_max_delta"] == 0.0


@pytest.mark.parametrize("case", CASES)
def test_save_load_roundtrip_is_report_stable(case: str, tmp_path: pathlib.Path) -> None:
    """save_rollout -> load_rollout preserves every array the report depends on."""
    original = load_rollout(FIXTURES / f"{case}_rollout.npz")
    out = tmp_path / "rt.npz"
    save_rollout(original, str(out))
    reloaded = load_rollout(out)
    assert build_report(reloaded).to_json() == build_report(original).to_json()


def test_save_load_preserves_negative_seed(tmp_path: pathlib.Path) -> None:
    """A real negative seed must not collide with the legacy None sentinel."""
    original = load_rollout(FIXTURES / "learned_teleport_rollout.npz")
    original.seed = -1
    out = tmp_path / "negative_seed.npz"
    save_rollout(original, str(out))
    reloaded = load_rollout(out)
    assert reloaded.seed == -1
    assert build_report(reloaded).to_json() == build_report(original).to_json()
