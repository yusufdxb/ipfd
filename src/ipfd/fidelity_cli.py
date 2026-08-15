"""Command-line product surface for counterfactual fidelity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["add_fidelity_arguments", "main", "run_fidelity"]

_MANDATORY_GROUPS = ("protocol", "continuation", "predicate")
_ALLOWED_GROUPS = (*_MANDATORY_GROUPS, "disturbance", "phase")


def _group_fields(value: str) -> tuple[str, ...]:
    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    if not requested:
        raise argparse.ArgumentTypeError("--group-by must name at least one field")
    duplicates = sorted({field for field in requested if requested.count(field) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(f"--group-by contains duplicate fields: {', '.join(duplicates)}")
    unknown = sorted(set(requested) - set(_ALLOWED_GROUPS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--group-by fields must be drawn from {', '.join(_ALLOWED_GROUPS)}; got {', '.join(unknown)}"
        )
    return requested


def _protocol_pair(value: str) -> tuple[str, str]:
    protocols = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(protocols) != 2 or protocols[0] == protocols[1]:
        raise argparse.ArgumentTypeError("--compare-protocols requires two distinct comma-separated protocol names")
    return protocols[0], protocols[1]


def add_fidelity_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``ipfd fidelity`` arguments to a parser."""

    parser.add_argument("records", type=Path, help="branch-comparison JSONL evidence")
    parser.add_argument(
        "--max-disagreement",
        type=float,
        default=0.05,
        help="maximum observed disagreement rate for the empirical frontier (default: 0.05)",
    )
    parser.add_argument(
        "--group-by",
        type=_group_fields,
        default=_group_fields("protocol,continuation,disturbance"),
        metavar="FIELDS",
        help=("comma-separated scope fields; protocol, continuation, and predicate are always retained"),
    )
    parser.add_argument(
        "--minimum-independent-seeds",
        type=int,
        default=5,
        help="minimum independent seed groups required by the fidelity gate (default: 5)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="whole-seed bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20_260_729,
        help="deterministic bootstrap random seed (default: 20260729)",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="write output instead of stdout")
    parser.add_argument(
        "--provenance",
        type=Path,
        help="related study-provenance file to identify by SHA-256 in the manifest",
    )
    parser.add_argument("--protocol", action="append", dest="protocols")
    parser.add_argument("--continuation", action="append", dest="continuations")
    parser.add_argument("--disturbance", action="append", dest="disturbances")
    parser.add_argument("--phase", action="append", dest="phases")
    parser.add_argument("--predicate", action="append", dest="predicates")
    parser.add_argument(
        "--compare-protocols",
        type=_protocol_pair,
        metavar="BASELINE,CANDIDATE",
        help="add an exact paired restore-protocol comparison",
    )


def _filters(args: argparse.Namespace) -> dict[str, frozenset[str]]:
    filters: dict[str, frozenset[str]] = {}
    for field, argument in (
        ("protocol", "protocols"),
        ("continuation", "continuations"),
        ("disturbance", "disturbances"),
        ("phase", "phases"),
        ("predicate", "predicates"),
    ):
        values = getattr(args, argument, None)
        if values:
            filters[field] = frozenset(str(value) for value in values)
    return filters


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise TypeError(f"expected a mapping-compatible audit result, got {type(value).__name__}")


def _scope_label(scope: Mapping[str, Any]) -> str:
    return " | ".join(f"{name}: {value}" for name, value in scope.items())


def _format_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _render_text(payload: Mapping[str, Any]) -> str:
    lines = [
        "COUNTERFACTUAL FIDELITY AUDIT",
        "",
        f"Source SHA256: {payload['evidence_manifest']['source']['sha256']}",
        f"Branch comparisons analyzed: {payload['record_count']}",
        f"Independent seed groups observed: {payload['independent_seed_count']}",
        f"Required independent seed groups: {payload['configuration']['minimum_independent_seeds']}",
        "",
    ]
    envelopes = payload.get("envelopes", [])
    if not isinstance(envelopes, Sequence):
        raise TypeError("audit envelopes must be a sequence")
    for envelope_value in envelopes:
        envelope = _mapping(envelope_value)
        scope = _mapping(envelope["scope"])
        lines.extend((_scope_label(scope), ""))
        lines.append("Horizon  disagreements  branch rate  seeds  seed mean  seed 95% interval  seeds w/error  interpretation")
        for point_value in envelope["points"]:
            point = _mapping(point_value)
            disagreements = f"{point['disagreements']}/{point['comparisons']}"
            interpretation = str(point["threshold_interpretation"])
            bootstrap = _mapping(point["seed_bootstrap"])
            interval = f"[{_format_rate(bootstrap['lower'])}, {_format_rate(bootstrap['upper'])}]"
            lines.append(
                f"{int(point['horizon']):>7}  {disagreements:>13}  "
                f"{_format_rate(point['disagreement_rate']):>11}  "
                f"{int(point['independent_seed_count']):>5}  "
                f"{_format_rate(point['seed_mean_disagreement_rate']):>9}  "
                f"{interval:>17}  {int(point['seeds_with_disagreement']):>13}  {interpretation}"
            )

        error_profile = _mapping(envelope["error_profile"])
        pooled = envelope.get("pooled_observed_strata", {})
        if isinstance(pooled, Mapping) and pooled:
            pooled_label = "; ".join(f"{name}: {', '.join(map(str, values))}" for name, values in pooled.items())
            lines.extend(("", f"Pooled observed strata: {pooled_label}"))
        lines.extend(
            (
                "",
                "Error direction: "
                f"false-recoverable {error_profile['false_recoverable']}, "
                f"false-unrecoverable {error_profile['false_unrecoverable']}",
            )
        )
        frontier = _mapping(envelope["frontier"])
        lines.append(f"Observed frontier: {frontier['status']}")
        if frontier.get("last_acceptable_tested_horizon") is not None:
            lines.append(f"Last acceptable tested horizon: {frontier['last_acceptable_tested_horizon']}")
        if frontier.get("first_rejected_tested_horizon") is not None:
            lines.append(f"First rejected tested horizon: {frontier['first_rejected_tested_horizon']}")
        lines.extend(("",))

    comparison_value = payload.get("protocol_comparison")
    if comparison_value is not None:
        comparison = _mapping(comparison_value)
        outcomes = _mapping(comparison["paired_outcomes"])
        bootstrap = _mapping(comparison["seed_bootstrap"])
        lines.extend(
            (
                "PAIRED RESTORE PROTOCOL COMPARISON",
                f"Baseline: {comparison['protocol_a']}",
                f"Candidate: {comparison['protocol_b']}",
                f"Paired comparisons: {comparison['pairs']}",
                "Outcomes: "
                f"both agree {outcomes['both_agree']}, "
                f"fixed by candidate {outcomes['fixed_by_protocol_b']}, "
                f"introduced by candidate {outcomes['introduced_by_protocol_b']}, "
                f"both disagree {outcomes['both_disagree']}",
                "Equal-seed mean difference (candidate - baseline): "
                f"{_format_rate(comparison['seed_mean_difference_b_minus_a'])}",
                f"Descriptive seed-resampling interval: [{_format_rate(bootstrap['lower'])}, {_format_rate(bootstrap['upper'])}]",
                f"Schedule identity verified: {comparison['schedule_identity_verified']}",
                "",
            )
        )

    lines.extend(
        (
            "IMPORTANT:",
            "This empirical envelope is scoped only to the supplied evidence and visible scope fields.",
            "Branch-level rates are descriptive. Seed groups, not branch comparisons, are the independent units.",
            "No disagreement observed is not proof of validity, and no result is a simulator-wide guarantee.",
        )
    )
    return "\n".join(lines) + "\n"


def _write_or_print(rendered: str, output: Path | None) -> None:
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def run_fidelity(args: argparse.Namespace) -> int:
    """Load branch evidence, build empirical envelopes, and render the audit."""

    # Imports remain local so the legacy rollout CLI stays lightweight.
    from .fidelity.envelope import (
        compare_restore_protocols,
        evaluate_fidelity_gate,
        fidelity_envelope,
        first_untrusted_horizon,
    )
    from .fidelity.io import build_evidence_manifest, load_branch_comparisons

    if not 0.0 <= args.max_disagreement <= 1.0:
        raise ValueError("--max-disagreement must be in [0, 1]")
    if args.minimum_independent_seeds < 1:
        raise ValueError("--minimum-independent-seeds must be at least 1")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be at least 1")
    if args.output is not None and args.output.resolve() == args.records.resolve():
        raise ValueError("--output must not overwrite the source evidence file")
    if args.output is not None and args.provenance is not None and args.output.resolve() == args.provenance.resolve():
        raise ValueError("--output must not overwrite the provenance file")

    related_provenance = None
    if args.provenance is not None:
        provenance_payload = args.provenance.read_bytes()
        related_provenance = {
            "logical_name": args.provenance.name,
            "sha256": hashlib.sha256(provenance_payload).hexdigest(),
            "bytes": len(provenance_payload),
        }

    loaded = load_branch_comparisons(args.records)
    selected = list(loaded.records)
    filters = _filters(args)
    for field, allowed in filters.items():
        selected = [record for record in selected if str(getattr(record, field)) in allowed]
    if not selected:
        raise ValueError("no branch comparisons matched the requested scope filters")

    effective_group_by = args.group_by + tuple(field for field in _MANDATORY_GROUPS if field not in args.group_by)
    envelopes = fidelity_envelope(
        selected,
        group_by=effective_group_by,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    rendered_envelopes: list[dict[str, Any]] = []
    for envelope in envelopes:
        item = _mapping(envelope)
        scope = envelope.scope
        matching_records = [
            record for record in selected if all(str(getattr(record, field)) == value for field, value in scope.items())
        ]
        item["pooled_observed_strata"] = {
            field: sorted({str(getattr(record, field)) for record in matching_records})
            for field in _ALLOWED_GROUPS
            if field not in effective_group_by
        }
        for point, rendered_point in zip(envelope.points, item["points"], strict=True):
            gate = evaluate_fidelity_gate(
                envelope,
                horizon=point.horizon,
                max_disagreement=args.max_disagreement,
                minimum_independent_seeds=args.minimum_independent_seeds,
            )
            rendered_point["gate"] = gate.to_dict()
            if point.independent_seed_count < args.minimum_independent_seeds:
                threshold_interpretation = "insufficient independent seeds"
            elif point.disagreements == 0:
                threshold_interpretation = "no disagreement observed"
            elif point.summary.seed_mean_disagreement_rate <= args.max_disagreement:
                threshold_interpretation = "degraded within tolerance"
            else:
                threshold_interpretation = f"reject above {100.0 * args.max_disagreement:.2f}% tolerance"
            rendered_point["threshold_interpretation"] = threshold_interpretation
        item["frontier"] = _mapping(
            first_untrusted_horizon(
                envelope,
                max_disagreement=args.max_disagreement,
                minimum_independent_seeds=args.minimum_independent_seeds,
            )
        )
        rendered_envelopes.append(item)

    protocol_comparison = None
    if args.compare_protocols is not None:
        baseline, candidate = args.compare_protocols
        protocol_comparison = compare_restore_protocols(
            selected,
            protocol_a=baseline,
            protocol_b=candidate,
            samples=args.bootstrap_samples,
            random_seed=args.bootstrap_seed,
        ).to_dict()

    analysis_configuration = {
        "max_disagreement": args.max_disagreement,
        "requested_group_by": list(args.group_by),
        "effective_group_by": list(effective_group_by),
        "minimum_independent_seeds": args.minimum_independent_seeds,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_confidence": 0.95,
        "bootstrap_method": "equal_seed_macro_percentile_cluster_bootstrap",
        "bootstrap_rng": "python_random_mt19937",
        "gate_rate_basis": "EQUAL_SEED_MEAN_DISAGREEMENT_RATE",
        "compare_protocols": list(args.compare_protocols) if args.compare_protocols else None,
        "filters": {name: sorted(values) for name, values in filters.items()},
        "related_provenance": related_provenance,
    }
    manifest = build_evidence_manifest(loaded, configuration=analysis_configuration)
    seed_ids = {str(record.cluster_id) for record in selected}
    payload = {
        "schema_version": 1,
        "kind": "counterfactual_fidelity_audit",
        "record_count": len(selected),
        "independent_seed_count": len(seed_ids),
        "configuration": analysis_configuration,
        "envelopes": rendered_envelopes,
        "protocol_comparison": protocol_comparison,
        "evidence_manifest": _mapping(manifest),
        "interpretation_limit": (
            "This empirical envelope is scoped to the supplied evidence. No disagreement observed "
            "does not prove validity, and branch comparisons are not independent trials."
        ),
    }
    if args.format == "json":
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    else:
        rendered = _render_text(payload)
    _write_or_print(rendered, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point for the counterfactual fidelity auditor."""

    parser = argparse.ArgumentParser(
        prog="ipfd fidelity",
        description="Audit empirical counterfactual fidelity from branch-comparison records.",
    )
    add_fidelity_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run_fidelity(args)
    except Exception as exc:
        print(f"IPFD_FIDELITY_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
