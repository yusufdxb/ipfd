"""Reproducible, simulator-free IPFD demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .actionability import evaluate_actionability
from .adapters.synthetic import make_silent_failure_rollout
from .ponr import aggregate_repeated_probes, point_of_no_return_repeated
from .report import build_report

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Run the offline demo and optionally persist its JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the demo result as JSON")
    args = parser.parse_args(argv)

    rollout = make_silent_failure_rollout(seed=0)
    report = build_report(rollout)
    assert rollout.recovery_success is not None
    probes = rollout.recovery_success.astype(bool)
    repeated = probes[:, None].repeat(5, axis=1)
    stats = aggregate_repeated_probes(repeated, min_repeats=3, min_confidence=0.8)
    result = {
        "demo": "offline_synthetic",
        "report": report.to_dict(),
        "ponr_repeated": point_of_no_return_repeated(repeated),
        "probe_confidence_min": float(stats.confidence.min()),
        "probe_repeats": int(stats.repeat_count[0]),
        "actionability": evaluate_actionability(
            report, disturbance_onset=40, probe_stride=1
        ).to_dict(),
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0
