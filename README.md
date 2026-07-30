# IPFD: Isaac Policy Failure Debugger

[![CI](https://github.com/yusufdxb/ipfd/actions/workflows/ci.yml/badge.svg)](https://github.com/yusufdxb/ipfd/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Latest release](https://img.shields.io/github/v/release/yusufdxb/ipfd)](https://github.com/yusufdxb/ipfd/releases/latest)

A success rate tells you an episode failed. IPFD tells you the step at which a
tested recovery controller stopped being able to save it, and whether any internal
policy signal changed before the failure was externally visible.

Scope is one robot (Franka Emika Panda), one task (`Isaac-Lift-Cube-Franka-v0`,
single-object lift in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)), and one
output (a per-rollout failure debug report). Detectors are deterministic NumPy. It is
not a benchmark suite, an Isaac Sim extension, or an ML-based detector.

## Evidence status

Read this before the feature list. The CPU analysis layer is stable and tested.
The simulator-side recovery evidence is mid-revalidation and is labeled as such.

| Area | Status | Backed by |
|---|---|---|
| Analysis layer (detectors, PoNR, metrics, report, plotting) | Stable. Pure NumPy, no simulator import. | 132 tests, 85.59% branch coverage, `ruff` and `mypy` clean, on Python 3.10, 3.11, 3.12 |
| Frozen-fixture reports are byte-stable when regenerated from a recorded rollout | Stable, GPU-free. | `tests/test_replay_fixture.py`. `test_report_reproducible` checks that two in-process builds of one synthetic rollout agree, which is narrower than byte-stability across the Python matrix. |
| IPFD attaches to a live Isaac Lab rollout (env, reset/step, obs structure, report) | Verified on one runtime. | [`scripts/verify_isaac_runtime.py`](scripts/verify_isaac_runtime.py) prints `IPFD_RUNTIME_SMOKE: overall PASS` |
| Probe writes do not move env 0 at the reset boundary | Measured, narrowly. | Historical runs measured max env-0 pose delta of 0.00e+00 m across probe `reset_to` calls. This is a reset-boundary measurement. It does not claim env 0 is static while the vectorized simulator steps every cell. |
| PoNR localizes an expected-PoNR disturbance on a trained policy | Historical fixture only. Classified `historical_fixture_only`; the release evidence gate is not satisfied. | The published fixture used a height-only recovery predicate that an airborne, out-of-reach object can satisfy, which is the teleport disturbance being injected. A conservative physical predicate now exists in the live path. Regenerated simulator evidence is required. See [docs/REVALIDATION.md](docs/REVALIDATION.md). |
| An expected no-PoNR control yields no PoNR | Historical fixture only. | The old slip fixture contains a non-monotone probe verdict sequence, so repeated probes are required before a verdict flip counts as evidence. |
| Imminence alarm localizes the fault | No. | On the trained policy tested, the alarm fires at the grasp transition, before the injected fault. Self-calibrated detectors are noisy across task phases. On a trained policy the usable signal is PoNR, not the alarm. |
| Entropy-collapse detector | No signal on this checkpoint. | The published checkpoint uses a state-independent action std, so the entropy proxy is constant and the detector does not fire. The report shows it flat. |
| The published NVIDIA Lift-Cube checkpoint is competent on the current local runtime | No. Measured 0.00% success. | `scripts/eval_checkpoint.py` over 64 environments reports `max_lift mean=0.000` and `SUCCESS_RATE 0.00%`, and recorded frames show the arm parked away from the cube. This blocks the learned-policy evidence bundle. See [docs/RELEASE_BLOCKERS.md](docs/RELEASE_BLOCKERS.md). |
| The scripted-oracle recovery experiment still reproduces on the current runtime | Yes, as historical diagnostic data. | `scripts/verify_pnor_grasped.py` reproduced its recorded numbers exactly: PoNR 138 against an injected slip at step 127, 68 recoverable and 92 unrecoverable probe verdicts, peak lift 0.179 m. The script labels its own output `HISTORICAL_ONLY`, since it predates the repeated physical predicate and the evidence schema. |

Everything simulator-side above rests on one machine and one checkpoint. Treat it
as a compatibility fingerprint, not a general result.

> **Validated runtime fingerprint:** locally installed `isaaclab` **4.5.22** with
> Isaac Sim **6.0.0.0**. The CPU analysis layer needs neither.

## What the report contains

Given one rollout, IPFD emits a Point of No Return index, the observable-failure
time, the detector alarm time, the timing metrics below, and a stacked timeline plot.

`build_report` computes that index from whatever `recovery_success` labels it is
given. It does not verify who produced them: controller identity, budget, predicate,
stride, and repeat count are optional metadata on `Rollout`. Provenance is enforced
only by the release evidence gate, described below.

The recording below is the failure IPFD is built to time. It is a live capture
from `scripts/verify_pnor_grasped.py` on Isaac Lab 4.5.22, driven by the vendored
scripted pick-lift oracle. The gripper is forced open once the cube is genuinely
grasped and lifted (step 127), the cube falls, and the arm keeps executing its
lift command afterward. In that run the recovery probe placed PoNR at step 138.

![Recorded Franka lift with an injected gripper slip](examples/figures/rollout_slip.gif)

Frames come from a Camera sensor in the scene. The headless viewport render
product returns all-zero frames on this runtime, so nothing is captured through
the viewport. This is the scripted oracle, not a learned policy, and the script
labels its own output `HISTORICAL_ONLY`.

![IPFD timeline on a trained policy](examples/figures/learned_teleport.png)

This figure is a historical trained-policy artifact, retained as a deterministic
analysis regression. Its alarm (orange) fires before the injected fault, and its
recovery labels come from the height-only predicate now under revalidation. The
title line ("SILENT FAILURE | seed=0") is an analysis verdict over the recorded
arrays, not current proof of physical irrecoverability.

## Point of No Return

A physical optimal-control PoNR cannot be read off a passive log, and one
controller's failure is not proof that a state was doomed. IPFD therefore measures
a narrower, operational quantity against the recovery controller actually run:

```
recovery_success[t] == True  <=>  the supplied recovery controller, restarted from
                                  the saved sim state at step t, satisfies the
                                  supplied success predicate within a fixed budget.

PoNR = the first step after which recovery never again succeeds.
```

Three properties follow, and they bound what the number means:

1. It is **oracle-relative**. A stronger recovery controller can push the measured
   step later, so when positive recovery verdicts are physically sound, the measured
   step is a **lower bound** on the optimal-control PoNR step.
2. A failed recovery attempt is **not** proof of physical irrecoverability.
3. Strided probing resolves PoNR to an interval, not an exact step.

Producing `recovery_success` requires a simulator that can save and restore state.
Consuming it does not. That split is why the analysis layer runs in CI with no GPU.

## Architecture

![IPFD architecture](examples/figures/architecture.png)

The recovery probe cannot run in the primary environment. In the validated runtime,
exposed scene state round-tripped exactly while the continued trajectory diverged
after evolved, contact-rich state. The experiment did not isolate which unexposed
simulator or task state caused that divergence. A separate two-environment
measurement showed `reset_to(..., env_ids=[1])` did not change env 0's object pose
at the reset boundary. IPFD therefore uses environment isolation in a decoupled
two-pass design:

- **env 0 (primary)** is rolled out and recorded, and is never `reset_to`.
- **env 1 (probe)** receives origin-shifted snapshots of the primary and runs the
  recovery oracle for a fixed budget. Its verdicts become `recovery_success[t]`.

The analysis layer (`detectors.py`, `ponr.py`, `metrics.py`, `report.py`, `viz.py`,
`types.py`) is pure NumPy and Matplotlib and never imports a simulator.
Simulator-facing code is confined to `adapters/isaac_lab.py` and `oracles/`, both
lazily imported.

To run IPFD on your own task, implement the recovery oracle. See
[**Bring your own recovery oracle**](docs/ORACLE_CONTRACT.md) for the callable
signatures, the exact meaning of `recovery_success[t]`, fixed-budget semantics, and
a copy-adaptable example.

## Quickstart: analysis layer (no GPU, no Isaac Lab)

```bash
pip install -e ".[dev]"
pytest
python3 examples/run_synthetic.py   # writes plots and JSON to examples/figures/
ipfd-demo                           # packaged offline demonstration
```

Given a recorded rollout archive, the same analysis runs as a single command with
no simulator present:

```bash
ipfd analyze rollout.npz --report report.json --plot timeline.png
ipfd analyze rollout.npz --disturbance-onset 56 --probe-stride 8
```

The disturbance arguments produce the conservative causal-actionability
classification described in [docs/CAUSAL_ACTIONABILITY.md](docs/CAUSAL_ACTIONABILITY.md),
which asks whether an alarm followed a known disturbance and preceded the
evidence-bounded PoNR, rather than crediting any early alarm.

```python
from ipfd import build_report, plot_timeline
from ipfd.adapters.synthetic import make_silent_failure_rollout

rollout = make_silent_failure_rollout(seed=0)
report  = build_report(rollout)
print(report.summary())
plot_timeline(rollout, report, "timeline.png")
```

If your shell has sourced ROS, run `env -u PYTHONPATH pytest` so ROS does not
inject its `launch_testing` plugin into this project's test collection.

## GPU validation driver (with Isaac Lab)

Prerequisites: a compatible Isaac Lab runtime, a CUDA GPU, and an Isaac Lab Python
environment. The only locally validated fingerprint is the one above. The first
learned-policy run downloads NVIDIA's published Lift-Cube checkpoint, so it can
take several minutes.

Run the compatibility preflight first:

```bash
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_isaac_runtime.py --headless
```

If your installation resolves assets from a different channel, point both commands
at the tested Isaac 4.5 production tree:

```bash
ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_isaac_runtime.py --headless --asset_root "$ASSET_ROOT"
```

The learned-policy driver rolls the checkpoint through the packaged
`collect_rollout`, injects a disturbance, runs the env-isolated recovery probe, and
writes the timeline figure and a machine-readable evidence record. The teleport case
is the expected-PoNR run:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  python3 scripts/verify_learned_policy.py --headless --use_pretrained \
         --probe --failure teleport --save_plot artifacts/timeline.png \
         --save_rollout artifacts/rollout.npz \
         --json artifacts/recovery-run.json \
         --asset_root "$ASSET_ROOT"
```

Swap `--failure slip` for the expected no-PoNR control. If the policy never
reaches the lift precondition, the command exits non-zero with
`fault_injection_triggered: NO`, which is a runtime or checkpoint compatibility
problem, not a result.

A single run is not evidence. The historical fixture reported PoNR at step 56, but
that value used the retracted height-only predicate and is not a current expected
result. A learned-policy headline is promoted only when the release evidence gate
accepts a complete bundle: a competence artifact, both failure modes across at
least five seeds, and real actionability cases. The thresholds and the exact
commands are in [docs/EVIDENCE_GATE.md](docs/EVIDENCE_GATE.md).

The legacy `scripts/verify_pnor_*.py` chain records historical diagnostic
experiments and cannot satisfy the current gate. See
[`scripts/README.md`](scripts/README.md) for the index.

## Metrics

| Metric | Question it answers |
|---|---|
| `time_to_failure` | When did failure become externally observable? |
| `failure_lead_time` | How early did the alarm fire relative to visible failure? |
| `ponr_lead_time` | Did the alarm precede the point of no return? (positive means alarm first) |
| `false_continuity_rate` | What fraction of the doomed window did the detector stay quiet? |
| `drift_magnitude_at_collapse` | How much had the representation drifted at PoNR? |

## Reproducibility

Fixed seeds make the synthetic rollouts deterministic, and reports regenerated from
the frozen recorded fixtures are byte-identical (`tests/test_replay_fixture.py`).
CI runs lint, type checks, branch coverage, source
builds, and installed-package smoke tests on Python 3.10, 3.11, and 3.12 with no
GPU. Live runs write auditable competence and recovery artifacts. The boundary
between what the published package reproduces and what only a live simulator can
establish is stated in
[`docs/GPU_REPRODUCIBILITY.md`](docs/GPU_REPRODUCIBILITY.md).

## Compatibility reports

The simulator results come from one setup, so independent reports broaden the
compatibility evidence.

- Follow the [validation checklist](docs/VALIDATION.md): clone, install, run the
  CPU path, and optionally the GPU driver. Every step emits a deterministic,
  copy-pasteable evidence block.
- Post the result in a
  [Tested on my machine](https://github.com/yusufdxb/ipfd/discussions) discussion
  (OS, Python, Isaac Lab version, GPU, what worked).
- For a version or platform mismatch, open a
  [compatibility report](https://github.com/yusufdxb/ipfd/issues/new?template=compatibility_report.yml).

Planned direction is in [ROADMAP.md](ROADMAP.md).

## License

MIT, see [`LICENSE`](LICENSE). `src/ipfd/oracles/pick_lift_sm.py` is vendored from
Isaac Lab under BSD-3-Clause, see
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
