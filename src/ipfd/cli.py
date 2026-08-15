"""Command-line entry point for replay-fidelity auditing and legacy analysis."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

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
    demo = subparsers.add_parser(
        "demo",
        help="run the bundled MuJoCo counterfactual-fidelity experiment",
    )
    demo.add_argument(
        "--output",
        type=Path,
        default=Path("ipfd-demo-results"),
        help="artifact directory (default: ./ipfd-demo-results)",
    )
    demo.add_argument("--json", type=Path, help="also copy summary.json to this path")
    adapter_check = subparsers.add_parser(
        "adapter-check",
        help="run the replay-adapter conformance suite",
    )
    adapter_check.add_argument(
        "target",
        help="Python import target for an adapter instance or zero-argument factory (module:object)",
    )
    adapter_check.add_argument("--decision", required=True, help="adapter decision name to exercise")
    adapter_check.add_argument("--output", type=Path, help="write the JSON report")
    regress = subparsers.add_parser(
        "regress",
        help="compare two machine-readable audit summaries",
    )
    regress.add_argument("--baseline", type=Path, required=True)
    regress.add_argument("--candidate", type=Path, required=True)
    regress.add_argument("--output", type=Path)
    compare = subparsers.add_parser(
        "compare",
        help="compare two machine-readable audit summaries",
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path)
    compare.add_argument(
        "--json",
        action="store_true",
        help="print the complete machine-readable comparison instead of the compact table",
    )
    fidelity = subparsers.add_parser(
        "fidelity",
        help="audit empirical counterfactual fidelity from branch records",
    )
    from .fidelity_cli import add_fidelity_arguments

    add_fidelity_arguments(fidelity)
    args = parser.parse_args(argv)

    if args.command == "demo":
        from .demo import main as demo_main

        demo_arguments = ["--output", str(args.output)]
        if args.json is not None:
            demo_arguments.extend(("--json", str(args.json)))
        return demo_main(demo_arguments)

    if args.command == "adapter-check":
        try:
            module_name, separator, attribute_name = args.target.partition(":")
            if not separator or not module_name or not attribute_name:
                raise ValueError("target must use module:object syntax")
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as first_error:
                cwd = str(Path.cwd())
                if cwd not in sys.path:
                    sys.path.insert(0, cwd)
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError:
                    raise first_error from None
            target = getattr(module, attribute_name)
            adapter = target() if callable(target) else target
            from .adapter_check import check_adapter

            try:
                adapter_report = check_adapter(adapter, decision=args.decision)
            finally:
                close = getattr(adapter, "close", None)
                if callable(close):
                    close()
            rendered = json.dumps(adapter_report.to_dict(), indent=2, sort_keys=True) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
        except Exception as exc:
            print(f"IPFD_ADAPTER_CHECK_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        return 0 if adapter_report.passed else 1

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
                return 0 if matrix_summary["result"] == "ALL_DECLARED_SCOPES_SUPPORTED" else 1
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
        return 0 if outputs["summary"]["overall_result"] == "SUPPORTED" else 1

    if args.command in {"regress", "compare"}:
        try:
            from .fidelity.regression import compare_audit_files

            comparison = compare_audit_files(args.baseline, args.candidate)
            rendered = (
                json.dumps(comparison, indent=2, sort_keys=True) + "\n"
                if args.command == "regress" or args.json
                else _compact_comparison(comparison, args.baseline, args.candidate)
            )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
        except Exception as exc:
            print(f"IPFD_REGRESSION_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        regression = bool(
            comparison["summary"]["previously_supported_became_unsupported"]
            or (args.command == "regress" and comparison["removed_scopes"])
        )
        return 1 if regression else 0

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


def _compact_comparison(
    comparison: dict[str, Any],
    baseline: Path,
    candidate: Path,
) -> str:
    """Render the decision-relevant protocol delta without dumping raw JSON."""

    rows = comparison["comparisons"]
    assert isinstance(rows, list)
    lines = [
        "IPFD AUDIT COMPARISON",
        "=====================",
        f"Baseline:  {baseline}",
        f"Candidate: {candidate}",
        "",
        "horizon  baseline     candidate    divergence       decision disagreement",
    ]
    for item in rows:
        assert isinstance(item, dict)
        key = item["comparison_key"]
        horizon = key.get("horizon", "?") if isinstance(key, dict) else "?"
        divergence = item["first_divergence"]
        decision = item["decision_disagreement"]
        assert isinstance(divergence, dict) and isinstance(decision, dict)
        before_divergence = divergence["baseline_step"]
        after_divergence = divergence["candidate_step"]
        before_text = "none" if before_divergence is None else str(before_divergence)
        after_text = "none" if after_divergence is None else str(after_divergence)
        lines.append(
            f"{str(horizon):>7}  {str(item['baseline_result']):<12} "
            f"{str(item['candidate_result']):<12} {before_text:>4} -> {after_text:<4}  "
            f"{str(decision['baseline']):>5} -> {str(decision['candidate']):<5}"
        )
    summary = comparison["summary"]
    assert isinstance(summary, dict)
    lines.extend(
        [
            "",
            f"Matched scopes: {summary['matched_configurations']}",
            f"Added scopes:   {summary['added_configurations']}",
            f"Removed scopes: {summary['removed_configurations']}",
            "Regression:     "
            + (
                "YES, a supported scope became unsupported"
                if summary["previously_supported_became_unsupported"]
                else "none detected"
            ),
            "Use --json for the complete machine-readable protocol delta.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
