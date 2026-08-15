"""Command-line entry point for replay-fidelity auditing and legacy analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Run replay-fidelity audits, regressions, or the legacy rollout analysis."""
    parser = argparse.ArgumentParser(
        prog="ipfd",
        description="Audit simulator snapshot-and-restore fidelity contracts.",
    )
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
    audit = subparsers.add_parser(
        "audit",
        help="audit a declared snapshot-and-restore fidelity contract",
    )
    audit.add_argument("--config", type=Path, required=True, help="path to audit.yaml")
    regress = subparsers.add_parser(
        "regress",
        help="compare two machine-readable audit summaries",
    )
    regress.add_argument("--baseline", type=Path, required=True)
    regress.add_argument("--candidate", type=Path, required=True)
    regress.add_argument("--output", type=Path)
    fidelity = subparsers.add_parser(
        "fidelity",
        help="audit empirical counterfactual fidelity from branch records",
    )
    from .fidelity_cli import add_fidelity_arguments

    add_fidelity_arguments(fidelity)
    args = parser.parse_args(argv)

    if args.command == "fidelity":
        try:
            from .fidelity_cli import run_fidelity

            return run_fidelity(args)
        except Exception as exc:
            print(f"IPFD_FIDELITY_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    if args.command == "audit":
        try:
            from .fidelity.archive import run_archived_isaac_audit
            from .fidelity.audit import run_audit
            from .fidelity.config import load_config
            from .fidelity.reporting import write_audit_outputs

            config = load_config(args.config)
            if config.adapter["kind"] == "matrix":
                from .fidelity.matrix import run_audit_matrix

                matrix_summary = run_audit_matrix(config)
                print(
                    json.dumps(
                        {
                            "command": "ipfd audit",
                            "result": matrix_summary["result"],
                            "output_directory": str(config.output_directory),
                        },
                        sort_keys=True,
                    )
                )
                return 0
            if config.adapter["kind"] == "isaac_lab_archive":
                result = run_archived_isaac_audit(config)
            else:
                result = run_audit(config)
            outputs = write_audit_outputs(result, config)
        except Exception as exc:
            print(f"IPFD_AUDIT_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "command": "ipfd audit",
                    "result": outputs["summary"]["overall_result"],
                    "output_directory": str(config.output_directory),
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "regress":
        try:
            from .fidelity.regression import compare_audit_files

            comparison = compare_audit_files(args.baseline, args.candidate)
            rendered = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
        except Exception as exc:
            print(f"IPFD_REGRESSION_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command != "analyze":  # pragma: no cover - argparse enforces this
        parser.error(f"unknown command: {args.command}")

    from .actionability import evaluate_actionability
    from .replay import load_rollout
    from .report import build_report
    from .viz import plot_timeline

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
