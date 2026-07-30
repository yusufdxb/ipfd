#!/usr/bin/env python3
"""Combine recovery-run artifacts into an auditable multi-seed bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate(paths: list[Path]) -> dict:
    """Load, validate, and deterministically order recovery-run artifacts."""
    if not paths:
        raise ValueError("at least one recovery-run artifact is required")
    runs = []
    for path in paths:
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
        if not isinstance(run, dict) or run.get("schema") != "ipfd.recovery_run.v1":
            raise ValueError(f"{path} is not an ipfd.recovery_run.v1 artifact")
        runs.append(run)
    runs.sort(key=lambda run: (int(run.get("seed", -1)), str(run.get("failure_mode", ""))))
    return {
        "schema": "ipfd.multiseed.v1",
        "status": "complete" if all(run.get("status") == "complete" for run in runs) else "incomplete",
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        bundle = aggregate(args.runs)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if bundle["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
