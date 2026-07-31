"""Audit and visualize the completed corrected snapshot-protocol study."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROTOCOL_A = "scene_plus_basic_manager_state"
PROTOCOL_B = "expanded_runtime_state"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"record {line_number} is not an object")
            records.append(value)
    return records


def rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    disagreements = sum(not bool(record["decision_match"]) for record in records)
    return {
        "records": total,
        "disagreements": disagreements,
        "disagreement_rate": disagreements / total if total else None,
        "false_recoverable": sum(
            not bool(record["reference_decision"]) and bool(record["candidate_decision"]) for record in records
        ),
        "false_unrecoverable": sum(
            bool(record["reference_decision"]) and not bool(record["candidate_decision"]) for record in records
        ),
    }


def grouped(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    values: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        values[(record["protocol"], str(record[field]))].append(record)
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (protocol, value), group in sorted(values.items()):
        result[value][protocol] = rate(group)
    return dict(result)


def create_figure(summary: dict[str, Any], path: Path) -> None:
    primary = summary["preregistered_primary_comparison"]
    rate_a = primary["protocol_a"]["disagreement_rate"]
    rate_b = primary["protocol_b"]["disagreement_rate"]
    required_b = rate_a * (1.0 - primary["required_relative_reduction"])

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    colors = ("#8f3b35", "#2f6f78")
    axes[0].bar(("Protocol A", "Protocol B"), (rate_a, rate_b), color=colors)
    axes[0].axhline(
        required_b,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label=f"Required B rate: at most {required_b:.2%}",
    )
    axes[0].set_ylim(0, max(rate_a * 1.35, 0.06))
    axes[0].set_ylabel("Exact-action sustained-lift disagreement")
    axes[0].set_title("Restoration fidelity", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    for index, item in enumerate(("protocol_a", "protocol_b")):
        value = primary[item]
        axes[0].text(
            index,
            value["disagreement_rate"] + 0.002,
            f"{value['disagreements']}/{value['records']}\n{value['disagreement_rate']:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for axis, title, message in (
        (
            axes[1],
            "Finite-sample validity gate",
            "Not run\nPositive-control threshold failed",
        ),
        (
            axes[2],
            "Downstream correction",
            "Not run\nStopping rule prohibited continuation",
        ),
    ):
        axis.set_facecolor("#f0f0f0")
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=12,
            color="#444444",
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.suptitle(
        "Expanded exposed state improved fidelity, but missed the preregistered threshold",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def refresh_manifest(output_dir: Path) -> None:
    manifest_path = output_dir / "artifact_manifest.json"
    existing = read_json(manifest_path)
    artifacts = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != manifest_path:
            artifacts[str(path.relative_to(output_dir))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    existing["artifacts"] = artifacts
    write_json(manifest_path, existing)


def externalize_worker_artifacts(study_dir: Path, external_dir: Path) -> None:
    worker_dir = study_dir / "workers"
    if not worker_dir.is_dir():
        raise FileNotFoundError(f"worker directory does not exist: {worker_dir}")
    if external_dir.exists():
        if not external_dir.is_dir() or any(external_dir.iterdir()):
            raise FileExistsError(
                f"external worker directory must be absent or empty: {external_dir}"
            )
    else:
        external_dir.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    for path in sorted(worker_dir.iterdir()):
        if path.is_file():
            entries[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    write_json(
        study_dir / "raw_worker_manifest.json",
        {
            "schema_version": 1,
            "storage": "external_to_repository",
            "artifacts": entries,
            "note": (
                "Local storage path is intentionally omitted from the repository "
                "artifact. Hashes and logical filenames preserve provenance."
            ),
        },
    )
    shutil.move(str(worker_dir), str(external_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--external-worker-dir", type=Path)
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    summary_path = study_dir / "protocol_comparison.json"
    records_path = study_dir / "per_branch_records.jsonl"
    summary = read_json(summary_path)
    records = read_records(records_path)
    if len(records) != summary["record_count"]:
        raise RuntimeError("per-branch record count does not match protocol summary")
    if summary["stopping_rule"]["decision"] != "STOP_BRANCH_VALIDITY_DIRECTION":
        raise RuntimeError("this analysis script expects the observed stopping decision")

    primary = [
        record for record in records if record["continuation"] == "exact_action" and record["predicate"] == "sustained_lift"
    ]
    tables = {
        "schema_version": 1,
        "source": {
            "protocol_comparison_sha256": sha256_file(summary_path),
            "per_branch_records_sha256": sha256_file(records_path),
            "records": len(records),
        },
        "primary_exact_action_sustained_lift": {
            "by_phase": grouped(primary, "phase"),
            "by_horizon": grouped(primary, "horizon"),
            "by_disturbance": grouped(primary, "disturbance"),
            "by_seed": grouped(primary, "base_seed"),
        },
        "exact_action_predicate_sensitivity": grouped(
            [record for record in records if record["continuation"] == "exact_action"],
            "predicate",
        ),
    }
    write_json(study_dir / "protocol_strata.json", tables)
    write_json(
        study_dir / "validity_gate_results.json",
        {
            "schema_version": 1,
            "status": "NOT_RUN_STOPPING_RULE",
            "reason": ("Protocol B reduced primary disagreement by 38.9 percent, below the preregistered 50 percent threshold."),
            "coverage": None,
            "accepted_branch_error": None,
            "held_out_comparison": None,
            "claim": "No empirical validity-gate claim was tested.",
        },
    )
    write_json(
        study_dir / "downstream_decision_results.json",
        {
            "schema_version": 1,
            "status": "NOT_RUN_STOPPING_RULE",
            "reason": (
                "The validity-gate stage was ineligible, so no PoNR, controller "
                "ranking, or checkpoint-selection correction was attempted."
            ),
            "before_gating": None,
            "after_gating": None,
            "uninterrupted_ground_truth": None,
            "claim": "No downstream-correction claim was demonstrated.",
        },
    )
    create_figure(summary, study_dir / "decisive_study.png")
    if args.external_worker_dir is not None:
        externalize_worker_artifacts(
            study_dir,
            args.external_worker_dir.expanduser().resolve(),
        )
    refresh_manifest(study_dir)
    print(f"protocol strata: {study_dir / 'protocol_strata.json'}")
    print(f"figure: {study_dir / 'decisive_study.png'}")
    print("gate: NOT_RUN_STOPPING_RULE")
    print("downstream: NOT_RUN_STOPPING_RULE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
