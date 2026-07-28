#!/usr/bin/env python3
"""Evaluate a recorded IPFD rollout against a known disturbance onset."""

from __future__ import annotations

import argparse
import json

from ipfd import build_report
from ipfd.actionability import evaluate_actionability
from ipfd.replay import load_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", help="Path to a rollout .npz fixture")
    parser.add_argument("--disturbance-onset", type=int, required=True)
    parser.add_argument("--probe-stride", type=int, default=1)
    args = parser.parse_args()
    result = evaluate_actionability(
        build_report(load_rollout(args.rollout)),
        disturbance_onset=args.disturbance_onset,
        probe_stride=args.probe_stride,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
