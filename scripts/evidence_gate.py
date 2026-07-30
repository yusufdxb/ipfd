#!/usr/bin/env python3
"""Fail-closed release gate for IPFD evidence."""

from ipfd.evidence_gate_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
