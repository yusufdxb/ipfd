import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ipfd.replay import save_rollout
from ipfd.types import Rollout

_SPEC = importlib.util.spec_from_file_location(
    "build_actionability_evidence",
    Path(__file__).parents[1] / "scripts" / "build_actionability_evidence.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
build_evidence = _MODULE.build_evidence


def _rollout(seed: int) -> Rollout:
    return Rollout(
        observations=np.zeros((12, 2)),
        actions=np.zeros((12, 1)),
        success=False,
        t_failure=11,
        recovery_success=np.array([True] * 8 + [False] * 4),
        dt=0.02,
        seed=seed,
        meta={
            "source": "isaac_lab",
            "task": "task",
            "checkpoint_sha256": "a" * 64,
            "runtime": {
                "isaaclab": "4.5.22",
                "isaacsim": "6.0.0",
                "torch": "2.7.0",
            },
            "software": {
                "ipfd_version": "1.1.0.dev0",
                "git_commit": "b" * 40,
                "git_dirty": False,
            },
        },
    )


def test_build_evidence_derives_relation_and_hashes_rollout(tmp_path):
    rollout_path = tmp_path / "run.npz"
    save_rollout(_rollout(7), rollout_path)
    manifest = {
        "schema": "ipfd.actionability_manifest.v1",
        "task": "task",
        "checkpoint_sha256": "a" * 64,
        "cases": [
            {
                "case_id": "case-1",
                "rollout": "run.npz",
                "disturbance_onset": 3,
                "probe_stride": 1,
                "expected_relation": "no_alarm",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifact = build_evidence(manifest_path)
    assert artifact["cases"][0]["alarm_relation"] == "no_alarm"
    assert artifact["cases"][0]["seed"] == 7
    assert len(artifact["cases"][0]["rollout_sha256"]) == 64


def test_build_evidence_rejects_duplicate_rollout(tmp_path):
    rollout_path = tmp_path / "run.npz"
    save_rollout(_rollout(7), rollout_path)
    case = {
        "case_id": "case-1",
        "rollout": "run.npz",
        "disturbance_onset": 3,
        "expected_relation": "no_alarm",
    }
    manifest = {
        "schema": "ipfd.actionability_manifest.v1",
        "task": "task",
        "checkpoint_sha256": "a" * 64,
        "cases": [case, case | {"case_id": "case-2"}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicates another case"):
        build_evidence(manifest_path)
