"""Console entry point for the fail-closed evidence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence_gate import EvidenceCriteria, evaluate_evidence


def main(argv: list[str] | None = None) -> int:
    """Validate release evidence artifacts and return a shell-friendly status."""
    p = argparse.ArgumentParser(description="Fail-closed IPFD release evidence gate")
    for name in ("competence", "multiseed", "actionability"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--min-success-rate", type=float, default=0.80)
    p.add_argument("--min-seeds", type=int, default=5)
    p.add_argument("--min-actionability-cases", type=int, default=20)
    p.add_argument("--min-competence-sustain-steps", type=int, default=10)
    p.add_argument("--min-probe-repeats", type=int, default=3)
    p.add_argument("--min-probe-false-fraction", type=float, default=0.8)
    p.add_argument("--max-ponr-error-steps", type=int, default=10)
    p.add_argument("--max-reset-boundary-delta-m", type=float, default=1e-6)
    p.add_argument("--expected-git-commit")
    p.add_argument("--report", type=Path)
    a = p.parse_args(argv)

    def read(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"_error": str(exc)}

    try:
        criteria = EvidenceCriteria(
            min_success_rate=a.min_success_rate,
            min_seeds=a.min_seeds,
            min_actionability_cases=a.min_actionability_cases,
            min_competence_sustain_steps=a.min_competence_sustain_steps,
            min_probe_repeats=a.min_probe_repeats,
            min_probe_false_fraction=a.min_probe_false_fraction,
            max_ponr_error_steps=a.max_ponr_error_steps,
            max_reset_boundary_delta_m=a.max_reset_boundary_delta_m,
            required_git_commit=a.expected_git_commit,
        )
    except ValueError as exc:
        p.error(str(exc))
    artifacts = [
        read(getattr(a, name))
        for name in ("competence", "multiseed", "actionability")
    ]
    result = evaluate_evidence(
        artifacts[0],
        artifacts[1],
        artifacts[2],
        criteria=criteria,
    )
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result.passed else 1
