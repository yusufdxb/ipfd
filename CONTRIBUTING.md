# Contributing to IPFD

Thanks for your interest. IPFD is a small, deliberately narrow tool, and its value
comes from staying that way. Before you open a PR, please read the one architectural
rule below; it is the thing most likely to get a change sent back.

## The one rule: keep the analysis layer simulator-free

IPFD is split into two layers with very different dependencies:

- **Analysis layer** (`src/ipfd/detectors.py`, `ponr.py`, `metrics.py`, `report.py`,
  `viz.py`, `types.py`): pure NumPy and Matplotlib. No simulator, no torch, no GPU.
  This is what runs in CI on Python 3.10 and 3.11 with no hardware, and what makes
  IPFD reproducible.
- **Adapter / oracle layer** (`src/ipfd/adapters/isaac_lab.py`, `src/ipfd/oracles/*`):
  the only code allowed to import Isaac Lab, and it must do so **lazily** (inside the
  function, not at module top level) so that importing `ipfd` never pulls in a simulator.

If a change makes the analysis layer import a simulator or torch, it will not be
merged. That boundary is the whole reason the tool is trustworthy and testable.

## Scope

Supported scope is **Franka Emika Panda, single-object pick-and-place, Isaac Lab**.
This is not an oversight; a narrow, verified tool beats a broad, hand-wavy one.
Proposals that widen scope (new tasks, robots, simulators, or ML inside the
detectors) are welcome as issues first, so we can discuss the maintenance cost
before you write code.

## Development setup

```bash
git clone https://github.com/yusufdxb/ipfd
cd ipfd
pip install -e ".[dev]"     # analysis layer only; no GPU or Isaac Lab needed
pytest                      # pure-NumPy analysis tests
ruff check src tests
python3 examples/run_synthetic.py
```

If tests fail during collection in a ROS-sourced shell, run
`env -u PYTHONPATH pytest`. ROS can inject its `launch_testing` pytest plugin
through `PYTHONPATH`, along with dependencies that are unrelated to IPFD.

Isaac Lab is intentionally **not** a declared dependency. You only need it to run the
`scripts/verify_*.py` GPU experiments, and you install it yourself following the
[Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/).

## Before you open a PR

1. `ruff check src tests` is clean.
2. `pytest` passes; state the count in the PR.
3. New analysis-layer behavior has a test. Synthetic rollouts and reports are
   byte-reproducible (see `test_report_reproducible`); keep them that way.
4. If you touch the Isaac Lab path, run the relevant `scripts/verify_*.py` on a GPU
   and paste the machine-readable status block (`IPFD_RUNTIME_SMOKE`,
   `DUAL_PROBE_STATUS`, etc.) into the PR. We cannot merge GPU claims we cannot see.
5. Keep the PR small and focused. One idea per PR.

## Honesty about verification

This project distinguishes what is *verified by a run* from what is *claimed*. The
README's "Verified / Partially verified / Future work" structure is a feature, not a
placeholder. If your change adds a capability, say precisely what you ran and what you
observed. Do not label something verified unless a script or test demonstrates it.

## Reporting bugs and asking questions

- Bugs: use the bug-report issue template.
- Usage questions and method discussion: use GitHub Discussions, not issues.

## License

By contributing, you agree that your contributions are licensed under the MIT License
(see [LICENSE](LICENSE)).
