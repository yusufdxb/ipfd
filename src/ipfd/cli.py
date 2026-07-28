"""Command-line entry point for zero-code rollout analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .actionability import evaluate_actionability
from .replay import load_rollout
from .report import build_report
from .viz import plot_timeline

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Analyze a recorded rollout without importing Isaac Lab."""
    parser = argparse.ArgumentParser(prog="ipfd", description="Analyze robot-policy rollouts offline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze a recorded .npz rollout")
    analyze.add_argument("rollout", type=Path, help="path to a rollout archive")
    analyze.add_argument("--plot", type=Path, help="write a PNG timeline")
    analyze.add_argument("--report", type=Path, help="write the report JSON")
    analyze.add_argument(
        "--disturbance-onset",
        type=int,
        help="known perturbation timestep for causal actionability analysis",
    )
    analyze.add_argument(
        "--probe-stride", type=int, default=None,
        help="probe stride; defaults to the rollout metadata, then 1",
    )
    args = parser.parse_args(argv)

    if args.command != "analyze":  # pragma: no cover - argparse enforces this
        parser.error(f"unknown command: {args.command}")

    rollout = load_rollout(args.rollout)
    report = build_report(rollout)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report.to_json(str(args.report))
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_timeline(rollout, report, str(args.plot))

    print(report.summary())
    if args.disturbance_onset is not None:
        probe_stride = args.probe_stride
        if probe_stride is None:
            probe_stride = int(rollout.meta.get("probe_stride", 1))
        actionability = evaluate_actionability(
            report,
            disturbance_onset=args.disturbance_onset,
            probe_stride=probe_stride,
        )
        print("\n=== Causal Actionability ===")
        print(json.dumps(actionability.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
