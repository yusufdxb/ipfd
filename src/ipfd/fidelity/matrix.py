"""One-command orchestration for a declared set of replay-fidelity audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .archive import run_archived_isaac_audit
from .audit import run_audit
from .config import AuditConfig, load_config
from .contracts import ContractVerdict
from .provenance import collect_provenance, sha256_file
from .reporting import write_audit_outputs

__all__ = ["run_audit_matrix"]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _public_output_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return "external-output/" + path.name


def run_audit_matrix(config: AuditConfig) -> dict[str, Any]:
    """Run every child audit in order and write a matrix-level evidence index."""

    if config.adapter.get("kind") != "matrix":
        raise ValueError("run_audit_matrix requires adapter.kind: matrix")
    raw_cases = config.adapter.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("adapter.cases must be a non-empty list of configuration paths")
    if not all(isinstance(value, str) and value for value in raw_cases):
        raise ValueError("every adapter.cases entry must be a non-empty path string")

    repo_root = Path(__file__).resolve().parents[3]
    cases: list[dict[str, Any]] = []
    for raw_path in raw_cases:
        child_path = Path(raw_path)
        if not child_path.is_absolute():
            child_path = (config.source_path.parent / child_path).resolve()
        child = load_config(child_path)
        if child.adapter.get("kind") == "matrix":
            raise ValueError("nested audit matrices are not supported")
        if child.adapter.get("kind") == "isaac_lab_archive":
            result = run_archived_isaac_audit(child)
        else:
            result = run_audit(child)
        output = write_audit_outputs(result, child)
        summary_path = child.output_directory / "audit_summary.json"
        cases.append(
            {
                "configuration": child.source_path.name,
                "configuration_sha256": sha256_file(child.source_path),
                "adapter_kind": child.adapter["kind"],
                "simulator_version": child.simulator_version,
                "environment": child.environment,
                "snapshot_protocol": child.snapshot_protocol,
                "overall_result": output["summary"]["overall_result"],
                "output_directory": _public_output_path(child.output_directory, repo_root),
                "audit_summary_sha256": sha256_file(summary_path),
            }
        )

    results = {case["overall_result"] for case in cases}
    if ContractVerdict.UNSUPPORTED.value in results:
        overall = "COMPLETED_WITH_UNSUPPORTED_SCOPES"
    elif ContractVerdict.INSUFFICIENT_EVIDENCE.value in results:
        overall = "COMPLETED_WITH_INSUFFICIENT_EVIDENCE"
    else:
        overall = "ALL_DECLARED_SCOPES_SUPPORTED"
    matrix_summary = {
        "schema_version": 1,
        "command": f"ipfd audit --config {config.source_path.name}",
        "result": overall,
        "case_count": len(cases),
        "cases": cases,
        "interpretation": (
            "This is an execution index across separately scoped audits, not a universal "
            "simulator verdict. Each child audit retains its own contract result."
        ),
    }
    output_directory = config.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "matrix_summary.json", matrix_summary)
    provenance = collect_provenance(
        adapter={
            "adapter": "audit_matrix",
            "case_count": len(cases),
            "child_configuration_sha256": {
                case["configuration"]: case["configuration_sha256"] for case in cases
            },
        },
        config_path=config.source_path,
        repo_root=repo_root,
        ignored_status_paths=(
            Path(__file__).resolve().parents[3] / "results" / "v2",
            config.output_directory,
        ),
    )
    _write_json(output_directory / "provenance.json", provenance)
    lines = [
        "# IPFD audit matrix",
        "",
        f"Execution result: **{overall}**",
        "",
        "Each row is a separately scoped fidelity contract. Unsupported rows are expected findings, not execution failures.",
        "",
        "| Configuration | Adapter | Protocol | Result |",
        "|---|---|---|---|",
    ]
    for case in cases:
        lines.append(
            f"| `{case['configuration']}` | {case['adapter_kind']} | "
            f"{case['snapshot_protocol']} | {case['overall_result']} |"
        )
    lines.append("")
    (output_directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return matrix_summary
