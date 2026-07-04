"""End-to-end IPFD demo with zero simulator dependency.

Generates a silent-failure rollout and a nominal success rollout, runs the full
analysis, prints both debug reports, and saves timeline plots + JSON to
``examples/figures/``. This is what CI and a first-time reader run.

    python examples/run_synthetic.py
"""

from __future__ import annotations

import os

from ipfd import build_report, plot_timeline
from ipfd.adapters.synthetic import make_silent_failure_rollout, make_success_rollout

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "figures")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    for name, rollout in [
        ("silent_failure", make_silent_failure_rollout(seed=0, t_ponr=90, t_failure=160)),
        ("success", make_success_rollout(seed=1)),
    ]:
        report = build_report(rollout)
        print("\n" + report.summary())
        png = plot_timeline(rollout, report, os.path.join(OUT, f"{name}.png"))
        report.to_json(os.path.join(OUT, f"{name}.json"))
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
