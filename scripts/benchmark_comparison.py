#!/usr/bin/env python3
"""Compare IPFD actionability with a naive pre-failure alarm rule (CPU only).

The harness uses deterministic scalar reports. It measures event-level precision,
recall, false-alarm rate, lead time, and intervention success without making any
claim about simulator or learned-policy competence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from ipfd.actionability import evaluate_actionability
from ipfd.report import FailureDebugReport


@dataclass(frozen=True)
class Case:
    seed: int
    name: str
    disturbance_onset: int
    t_ponr: int | None
    t_alarm: int | None
    truly_actionable: bool


def make_cases(seeds: list[int]) -> list[Case]:
    result: list[Case] = []
    for seed in seeds:
        result += [
            Case(seed, "actionable", 30, 60, 40, True),
            Case(seed, "pre_disturbance_false_alarm", 45, 60, 40, False),
            Case(seed, "too_late", 30, 60, 65, False),
            Case(seed, "no_alarm_failure", 30, 60, None, False),
            Case(seed, "nominal_alarm_free", 30, None, None, False),
        ]
    return result


def _report(case: Case) -> FailureDebugReport:
    return FailureDebugReport(
        success=case.t_ponr is None,
        T=100, dt=0.02, seed=case.seed, t_ponr=case.t_ponr,
        t_failure=None if case.t_ponr is None else case.t_ponr + 8,
        t_alarm=case.t_alarm, time_to_failure_s=None, failure_lead_time_s=None,
        ponr_lead_time_s=None, silent_doom_window_s=None,
        false_continuity_rate=None, drift_at_collapse=None,
    )


def _metrics(rows: list[dict], key: str) -> dict:
    tp = sum(r[key] and r["truly_actionable"] for r in rows)
    fp = sum(r[key] and not r["truly_actionable"] for r in rows)
    fn = sum((not r[key]) and r["truly_actionable"] for r in rows)
    tn = sum((not r[key]) and not r["truly_actionable"] for r in rows)
    alarms = [r["lead_time_s"] for r in rows if r[key] and r["lead_time_s"] is not None]
    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "true_negatives": tn, "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "mean_lead_time_s": sum(alarms) / len(alarms) if alarms else None,
        "intervention_success_rate": tp / (tp + fp) if tp + fp else 0.0,
    }


def run(seeds: list[int], probe_stride: int = 8) -> dict:
    rows = []
    for case in make_cases(seeds):
        report = _report(case)
        ipfd = evaluate_actionability(report, disturbance_onset=case.disturbance_onset, probe_stride=probe_stride)
        naive = case.t_alarm is not None and case.t_ponr is not None and case.t_alarm < report.t_failure
        lead = None if case.t_alarm is None or case.t_ponr is None else (case.t_ponr - case.t_alarm) * report.dt
        rows.append({"seed": case.seed, "case": case.name, "truly_actionable": case.truly_actionable,
                     "ipfd": ipfd.valid_actionable_warning, "naive": naive,
                     "lead_time_s": lead, "ipfd_relation": ipfd.alarm_relation})
    return {"schema_version": 1, "source": "synthetic_contract",
            "seeds": seeds, "probe_stride": probe_stride,
            "n_cases": len(rows), "ipfd": _metrics(rows, "ipfd"),
            "naive": _metrics(rows, "naive"), "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--probe-stride", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run([int(x) for x in args.seeds.split(",") if x.strip()], args.probe_stride)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
