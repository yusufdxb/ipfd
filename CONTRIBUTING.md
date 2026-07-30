# Contributing to IPFD

IPFD supports one robot and one task. Before opening a PR, read the architectural
rule below. It is the constraint most likely to get a change sent back.

## The one rule: keep the analysis layer simulator-free

IPFD is split into two layers with very different dependencies:

- **Analysis layer** (`src/ipfd/detectors.py`, `ponr.py`, `metrics.py`, `report.py`,
  `viz.py`, `types.py`): pure NumPy and Matplotlib. No simulator, no torch, no GPU.
  This is what runs in CI on Python 3.10, 3.11, and 3.12 with no hardware, and what makes
  IPFD reproducible.
- **Adapter / oracle layer** (`src/ipfd/adapters/isaac_lab.py`, `src/ipfd/oracles/*`):
  the only code allowed to import Isaac Lab, and it must do so **lazily** (inside the
  function, not at module top level) so that importing `ipfd` never pulls in a simulator.

If a change makes the analysis layer import a simulator or torch, it will not be
merged. That boundary is what allows the analysis layer to be tested in CI without
a GPU.

## Scope

Supported scope is **Franka Emika Panda, single-object pick-and-place, Isaac Lab**.
Scope is limited so every supported path has a test or a recorded run.
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

1. `ruff check src tests scripts examples` is clean.
2. `mypy` and `pytest --cov=ipfd --cov-branch` pass.
3. New analysis-layer behavior has a test. Synthetic rollouts and reports are
   byte-reproducible (see `test_report_reproducible`); keep them that way.
4. If you touch the Isaac Lab path, run the relevant `scripts/verify_*.py` on a GPU
   and paste the machine-readable status block (`IPFD_RUNTIME_SMOKE`,
   `DUAL_PROBE_STATUS`, etc.) into the PR. GPU claims are merged only with a pasted
   status block.
5. Keep the PR small and focused. One idea per PR.

## Verification standard

This project distinguishes what a run verified from what is merely claimed. Keep the
README's evidence-status table accurate when you add a capability: say precisely what
you ran and what you observed. Do not label something verified unless a script or
test demonstrates it.

## Reporting bugs and asking questions

- Bugs: use the bug-report issue template.
- Usage questions and method discussion: use GitHub Discussions, not issues.

## License

By contributing, you agree that your contributions are licensed under the MIT License
(see [LICENSE](LICENSE)).
