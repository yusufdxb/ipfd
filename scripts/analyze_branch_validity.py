"""Summarize paired counterfactual branch-validity reports and plot the result.

Example:
    python scripts/analyze_branch_validity.py \
        --input archived=oracle_equivalence.json \
        --output results/branch_validity/summary.json \
        --figure results/branch_validity/decision_fidelity.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from ipfd.branch_validity import summarize_branch_validity


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("input label cannot be empty")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input report does not exist: {path}")
    return label.strip(), path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return report


def write_figure(summary: dict[str, Any], path: Path) -> None:
    phases = [
        phase
        for phase in ("pre_manipulation", "mid_contact", "post_contact", "lifted", "unknown")
        if phase in summary["by_phase_and_continuation"]
    ]
    phases.extend(
        sorted(set(summary["by_phase_and_continuation"]) - set(phases))
    )
    modes = ["exact_action", "policy"]
    values = np.full((len(modes), len(phases)), np.nan)
    counts = np.zeros((len(modes), len(phases)), dtype=int)
    disagreements = np.zeros((len(modes), len(phases)), dtype=int)
    for column, phase in enumerate(phases):
        for row, mode in enumerate(modes):
            cell = summary["by_phase_and_continuation"][phase].get(mode)
            if cell is None:
                continue
            values[row, column] = float(cell["agreement_rate"])
            counts[row, column] = int(cell["comparisons"])
            disagreements[row, column] = int(cell["disagreements"])

    figure, axis = plt.subplots(figsize=(10.2, 3.8), constrained_layout=True)
    image = axis.imshow(values, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    for row in range(len(modes)):
        for column in range(len(phases)):
            if counts[row, column] == 0:
                continue
            axis.text(
                column,
                row,
                f"{values[row, column]:.0%}\n"
                f"{disagreements[row, column]}/{counts[row, column]} disagree",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    axis.set_xticks(range(len(phases)), [phase.replace("_", "\n") for phase in phases])
    axis.set_yticks(range(len(modes)), ["Recorded exact actions", "Closed-loop policy"])
    axis.set_xlabel("Task phase at snapshot")
    axis.set_title(
        "Immediate snapshot equality did not guarantee recovery-decision equality",
        loc="left",
        fontweight="bold",
    )
    overall = summary["overall"]
    axis.text(
        0.0,
        1.12,
        f"{overall['disagreements']} of {overall['comparisons']} restored branches "
        "changed the terminal success decision",
        transform=axis.transAxes,
        fontsize=10,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Decision agreement")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--maximum-disagreement-rate", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    cohorts = []
    for label, path in args.input:
        cohorts.append(
            summarize_branch_validity(
                load_report(path),
                source_label=label,
                source_sha256=sha256(path),
                maximum_disagreement_rate=args.maximum_disagreement_rate,
                confidence=args.confidence,
            )
        )
    output = {
        "schema_version": 1,
        "scientific_object": "counterfactual_branch_decision_fidelity",
        "cohorts": cohorts,
        "pooling_policy": (
            "Single cohort; no cross-cohort pooling was required."
            if len(cohorts) == 1
            else "Cohorts are not pooled because generator provenance and run conditions differ."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_figure(cohorts[0], args.figure)

    for cohort in cohorts:
        overall = cohort["overall"]
        print(
            f"{cohort['source']['label']}: {overall['disagreements']}/"
            f"{overall['comparisons']} decision disagreements, result={cohort['result']}"
        )
    print(f"summary: {args.output}")
    print(f"figure:  {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
