#!/usr/bin/env python3
"""Run a deterministic, CPU-only causal actionability benchmark.

The benchmark uses small scalar reports rather than a simulator.  This makes
the detector/actionability contract reproducible in CI while exercising
positive, negative, ambiguous, and natural-failure-like cases across seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ipfd.actionability import ActionabilityReport, evaluate_actionability
from ipfd.report import FailureDebugReport


@dataclass(frozen=True)
class BenchmarkCase:
    seed: int
    name: str
    disturbance_onset: int
    t_ponr: int | None
    t_alarm: int | None
    expected_relation: str


def _report(case: BenchmarkCase) -> FailureDebugReport:
    failure = case.t_ponr + 8 if case.t_ponr is not None else None
    return FailureDebugReport(
        success=case.name == "negative_control",
        T=100,
        dt=0.02,
        seed=case.seed,
        t_ponr=case.t_ponr,
        t_failure=failure,
        t_alarm=case.t_alarm,
        time_to_failure_s=None,
        failure_lead_time_s=None,
        ponr_lead_time_s=None,
        silent_doom_window_s=None,
        false_continuity_rate=None,
        drift_at_collapse=None,
    )


def make_cases(seeds: list[int]) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for seed in seeds:
        cases.extend(
            [
                BenchmarkCase(seed, "positive_control", 30, 60, 40, "definitely_actionable"),
                BenchmarkCase(seed, "negative_control", 30, None, None, "no_alarm"),
                BenchmarkCase(seed, "natural_failure", 30, 60, 65, "too_late"),
                BenchmarkCase(seed, "ambiguous_boundary", 30, 60, 55, "ambiguous_within_ponr_interval"),
            ]
        )
    return cases


def run(seeds: list[int], probe_stride: int = 8) -> dict:
    rows: list[dict] = []
    for case in make_cases(seeds):
        result: ActionabilityReport = evaluate_actionability(
            _report(case), disturbance_onset=case.disturbance_onset, probe_stride=probe_stride
        )
        row = asdict(result)
        row.update({"seed": case.seed, "case": case.name, "expected_relation": case.expected_relation})
        row["correct"] = result.alarm_relation == case.expected_relation
        rows.append(row)
    correct = sum(row["correct"] for row in rows)
    actionable = [row for row in rows if row["case"] == "positive_control"]
    return {
        "schema_version": 1,
        "source": "synthetic_contract",
        "probe_stride": probe_stride,
        "seeds": seeds,
        "n_cases": len(rows),
        "n_correct": correct,
        "accuracy": correct / len(rows) if rows else 1.0,
        "actionable_warning_rate": sum(r["valid_actionable_warning"] for r in actionable) / len(actionable)
        if actionable
        else 0.0,
        "relations": {
            relation: sum(r["alarm_relation"] == relation for r in rows)
            for relation in sorted({r["alarm_relation"] for r in rows})
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated deterministic seeds")
    parser.add_argument("--probe-stride", type=int, default=8)
    parser.add_argument("--json", type=Path, help="write aggregate and per-case JSON")
    parser.add_argument("--csv", type=Path, help="write per-case CSV")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    result = run(seeds, args.probe_stride)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            rows = result["cases"]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(payload, end="")


if __name__ == "__main__":
    main()
