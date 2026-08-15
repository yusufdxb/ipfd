from __future__ import annotations

import hashlib

import matplotlib.image as mpimg
import pytest

from ipfd.demo_report import render_demo_report


def _summary() -> dict:
    return {
        "system": "MuJoCo filtered-contact example",
        "focus_protocol": "minimal_visible",
        "trajectory": {
            "steps": [0, 1, 2, 3, 4, 5],
            "reference_position": [0.0, 0.1, 0.21, 0.33, 0.46, 0.60],
            "restored_position": [0.0, 0.1, 0.21, 0.34, 0.50, 0.69],
            "tolerance": 0.025,
            "reference_contact_steps": [1, 3],
            "restored_contact_steps": [1, 4],
            "reference_decision": "stable",
            "restored_decision": "unstable",
            "position_label": "Object position (m)",
        },
        "protocols": [
            {
                "name": "minimal_visible",
                "omitted_capabilities": ["solver warm-start state", "filtered contact state"],
                "fidelity": {
                    "l0_restore": "PASS",
                    "l1_one_step": "PASS",
                    "l2_by_horizon": {"1": "PASS", "5": "DEGRADED", "30": "FAIL"},
                    "l3_decision": "FAIL",
                },
            },
            {
                "name": "integration_with_warmstart",
                "omitted_capabilities": [],
                "fidelity": {
                    "l0_restore": "PASS",
                    "l1_one_step": "PASS",
                    "l2_by_horizon": {1: "PASS", 5: "PASS", 30: "PASS"},
                    "l3_decision": "PASS",
                },
            },
        ],
    }


def test_render_demo_report_creates_fixed_size_png(tmp_path):
    output = tmp_path / "nested" / "report.png"

    result = render_demo_report(_summary(), output)

    assert result == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert mpimg.imread(output).shape == (1440, 1728, 4)


def test_render_demo_report_is_byte_reproducible(tmp_path):
    first = render_demo_report(_summary(), tmp_path / "first.png")
    second = render_demo_report(_summary(), tmp_path / "second.png")

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()


def test_render_demo_report_rejects_inconsistent_trajectory_lengths(tmp_path):
    summary = _summary()
    summary["trajectory"]["restored_position"] = [0.0]

    with pytest.raises(ValueError, match="equal lengths"):
        render_demo_report(summary, tmp_path / "report.png")


def test_render_demo_report_requires_focus_protocol_in_grid(tmp_path):
    summary = _summary()
    summary["focus_protocol"] = "not_present"

    with pytest.raises(ValueError, match="focus_protocol"):
        render_demo_report(summary, tmp_path / "report.png")
