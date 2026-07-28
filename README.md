# IPFD: Isaac Policy Failure Debugger

[![CI](https://github.com/yusufdxb/ipfd/actions/workflows/ci.yml/badge.svg)](https://github.com/yusufdxb/ipfd/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Latest release](https://img.shields.io/github/v/release/yusufdxb/ipfd)](https://github.com/yusufdxb/ipfd/releases/latest)

**Localize when a tested recovery controller stopped succeeding, and determine
whether an internal signal warned before the failure was visible.**

> **Validated runtime fingerprint:** locally installed `isaaclab` distribution
> **4.5.22** with Isaac Sim **6.0.0.0**. The CPU analysis layer supports Python
> **3.10 / 3.11 / 3.12** and runs without Isaac Lab.

IPFD is a small, headless debugging tool for robot-policy rollouts in
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/). The scope is narrow on purpose:
one robot (**Franka Emika Panda**), one task (**single-object pick-and-place**), one
job (**find the moment the policy was already doomed**). No benchmark suite, no Kit
extension, and no ML quietly hiding inside the detectors. One sharp tool instead of a
whole Swiss-army drawer.

---

## The problem

You train a manipulation policy. It reports, say, 85% success. The other 15% of
episodes end with the cube on the floor while the success metric politely declines to
explain why. That average tells you *that* those episodes failed. It cannot tell you
the two things you actually need in order to debug them:

1. **When did the tested recovery controller stop succeeding**, within the
   resolution of the probe schedule?
2. **Did any internal signal know** (action statistics, policy confidence, latent
   drift) *before* the failure was externally observable?

The dangerous case is a policy that stays smooth and confident well *after* it has
entered a doomed trajectory. Success-rate evaluation counts that episode as "fine"
right up until the object hits the floor. That silent interval, doomed but still
looking healthy, is what IPFD makes visible.

## What IPFD produces

Given one rollout, IPFD emits a **failure debug report**: the oracle-relative
Point of No Return,
the observable-failure time, the detector alarm time, a small set of timing
metrics, and a stacked timeline plot. The figure below is a historical trained-
policy artifact retained for deterministic analysis regression. Its alarm
(orange) fires before the injected fault, and its height-only recovery labels are
under physical-predicate revalidation:

![IPFD timeline on a trained policy](examples/figures/learned_teleport.png)

*The title line ("SILENT FAILURE | seed=0") is an analysis verdict for the
recorded arrays. It is not current proof of physical irrecoverability.*

---

## The one hard idea: Point of No Return

A physical optimal-control PoNR **cannot be read off a passive log**, and a failed
attempt by one controller is not proof. IPFD therefore measures a narrower,
operational PoNR against a **recovery probe**:

```
recovery_success[t] == True  <=>  a best-effort recovery controller, restarted from
                                  the saved sim state at step t, reaches the goal
                                  within a fixed budget.

PoNR = the first step after which recovery never again succeeds.
```

This is an **oracle-relative estimate**. Strided probing bounds PoNR to an
interval rather than an exact step. A better recovery controller can push the
measured timestep later, so when positive recovery verdicts are physically sound,
the measured timestep is a **lower bound on the optimal-control PoNR timestep**.
A failed recovery attempt is not proof of physical irrecoverability. Producing
`recovery_success` needs a simulator that can save/restore state; consuming it does
not. **That boundary is the whole architecture.**

## Architecture: the dual-environment recovery probe

![IPFD architecture](examples/figures/architecture.png)

The recovery probe cannot run in the primary environment. In the validated local
runtime, exposed scene state round-tripped exactly while the continued trajectory
diverged after evolved, contact-rich state. The experiment did not isolate which
unexposed simulator or task state caused the divergence. A separate two-environment
measurement showed that `reset_to(..., env_ids=[1])` did not change env 0's object
pose at the reset boundary. IPFD therefore uses **environment isolation** in a
**decoupled two-pass**:

- **env 0 (PRIMARY)** is rolled out and recorded, and is **never** `reset_to`.
- **env 1 (PROBE)** receives origin-shifted snapshots of the primary and runs the
  recovery oracle for a fixed budget; its verdicts become `recovery_success[t]`.

To run IPFD on your own task, implement the recovery oracle: see
[**Bring your own recovery oracle**](docs/ORACLE_CONTRACT.md) for the callable
signatures, the exact meaning of `recovery_success[t]`, fixed-budget semantics,
and one copy-adaptable example.

The analysis layer (detectors, PoNR, metrics, report, plotting) is **pure
NumPy/Matplotlib** and never imports a simulator. It runs in CI with no GPU. Only
`ipfd.adapters.isaac_lab` and `ipfd.oracles.*` touch Isaac Lab, and they are lazily
imported.

The core modules are `types.py` (rollout contract), `detectors.py`, `ponr.py`,
`metrics.py`, `report.py`, and `viz.py`. Simulator-facing code is confined to
`adapters/isaac_lab.py` and `oracles/`.

---

## Verified results

The committed rollout fixtures reproduce historical analysis output, not physical
oracle correctness. The learned-policy fixtures are retained as compatibility
regressions while the physical recovery predicate is revalidated. Checkpoint
behavior is sensitive to Isaac Lab, Isaac Sim, asset-channel, and task-version
compatibility. A run that never reaches the lift precondition exits non-zero and
must not be used as evidence.

Every claim below maps to a script or test. Historical GPU runs used the local
runtime fingerprint above. Analysis-layer claims run in CI without a GPU.

### Verified

| Claim | Evidence |
|---|---|
| The analysis layer is pure NumPy, tested, and byte-reproducible. | `pytest` passes, `ruff` clean; `test_report_reproducible`. CI runs lint, type checks, coverage, source builds, and installed-package smoke tests on Python 3.10, 3.11, and 3.12. |
| IPFD attaches to a **real** Isaac Lab rollout (import, env, reset/step, obs structure, `build_report`). | [`scripts/verify_isaac_runtime.py`](scripts/verify_isaac_runtime.py) → `IPFD_RUNTIME_SMOKE: overall PASS`. |
| Probe reset writes are isolated from env 0 at the measured boundary. | Historical runs measured `max env-0 pose delta = 0.00e+00 m` immediately across probe `reset_to` calls. This does not claim that env 0 remains static while the vectorized simulator steps all environments. |
| On a **genuinely competent trained policy**, PoNR localizes an irrecoverable failure. | **Under revalidation.** The original fixture used a height-only recovery predicate that can classify an airborne, out-of-reach cube as recovered. The live path now supports a conservative physical recovery predicate; regenerated simulator evidence is required before this becomes a verified claim. |
| A **recoverable** failure correctly yields **no PoNR**. | **Under revalidation.** Probe verdict stability and recovery semantics must be measured with repeated probes and a physical recovery predicate. |
| Historical fixtures remain analysis-compatible. | CI loads both fixtures, rebuilds reports byte-for-byte, and exercises the installed wheel. This validates serialization and analysis compatibility, not current simulator semantics. |

### Partially verified

| Claim | Honest bound |
|---|---|
| Silent-collapse **detection** on a trained policy. | The imminence alarm *fires*, but on this policy it fires at the natural **grasp transition**, before the injected fault. Self-calibrated detectors are noisy across a real policy's task phases. On a trained policy the reliable signal is **PoNR**, not the alarm. |
| **Entropy-collapse** detector. | The official checkpoint uses a *state-independent* action std, so the entropy signal is **flat** and that detector does not fire. IPFD reports it flat rather than hiding it. |
| Scripted-policy PoNR. | Historical grasped-region runs produced a transition near the injection. Pre-grasp checkpoints were noisy, and the responsible unexposed simulator or task state was not identified. These runs predate the current evidence contract. |

### Future work

- **Phase-aware detector calibration**, so the imminence alarm localizes the fault
  rather than task transitions.
- A **rendered rollout GIF** from a real run. Headless offscreen capture currently
  produces flawlessly empty frames, so for now you get honest plots instead of a
  fake GIF.
- Tasks beyond Franka single-object pick-and-place.

---

## Quickstart: analysis layer (no GPU, no Isaac Lab)

```bash
pip install -e ".[dev]"            # analysis layer only
pytest                             # pure-NumPy analysis tests
python3 examples/run_synthetic.py  # prints two reports, writes plots + JSON to examples/figures/
ipfd-demo                         # packaged offline demonstration
```

Once you have a recorded rollout archive, the same analysis is available as a
zero-code command. This works without Isaac Lab or a GPU:

```bash
ipfd analyze rollout.npz --report report.json --plot timeline.png
ipfd analyze rollout.npz --disturbance-onset 56 --probe-stride 8
```

The optional disturbance arguments produce the conservative causal-actionability
classification described in [the causal guide](docs/CAUSAL_ACTIONABILITY.md).

If the shell has sourced ROS, run `env -u PYTHONPATH pytest` to prevent ROS from
injecting its `launch_testing` plugin into this project's test collection.

```python
from ipfd import build_report, plot_timeline
from ipfd.adapters.synthetic import make_silent_failure_rollout

rollout = make_silent_failure_rollout(seed=0)
report  = build_report(rollout)
print(report.summary())
plot_timeline(rollout, report, "timeline.png")
```

## GPU validation driver (with Isaac Lab)

Prerequisites: a compatible Isaac Lab runtime, a CUDA GPU, and an Isaac Lab Python
environment. The only locally validated fingerprint is listed at the top of this
README.
The first learned-policy run downloads NVIDIA's published Lift-Cube checkpoint,
so it can take several minutes.

Run the compatibility preflight before the full demo:

```bash
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_isaac_runtime.py --headless
```

If the installed simulator defaults to a different asset channel, point both
commands at the tested Isaac 4.5 production tree:

```bash
ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_isaac_runtime.py --headless --asset_root "$ASSET_ROOT"
```

The learned-policy command rolls the checkpoint through the packaged
`collect_rollout`, injects an irrecoverable failure, runs the env-isolated recovery
probe, and writes the timeline figure:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  python3 scripts/verify_learned_policy.py --headless --use_pretrained \
         --probe --failure teleport --save_plot artifacts/timeline.png \
         --save_rollout artifacts/rollout.npz \
         --json artifacts/recovery-run.json \
         --asset_root "$ASSET_ROOT"
```

The command writes raw repeated probe verdicts and physical-predicate metadata.
Treat a run as evidence only after the release evidence gate accepts the complete
multi-seed bundle. The historical fixture reported PoNR at step 56, but that value
is under physical-oracle revalidation and is not a current expected result. If the
policy never reaches the lift precondition, the command exits non-zero with
`fault_injection_triggered: NO`; fix the runtime/checkpoint compatibility first.
Swap `--failure slip` for the recoverable negative control.

The legacy `scripts/verify_pnor_*.py` chain records historical diagnostic
experiments and cannot satisfy the current release gate; see
[`scripts/README.md`](scripts/README.md) for the index.

For the stricter causal question, see the [causal actionability guide](docs/CAUSAL_ACTIONABILITY.md).
It evaluates whether an alarm followed a known disturbance and definitely
preceded the evidence-bounded PoNR, rather than treating any early alarm as a
success.

---

## Metrics (the minimal set)

| Metric | Question it answers |
|---|---|
| `time_to_failure` | When did failure become externally observable? |
| `failure_lead_time` | How early did the alarm fire vs visible failure? |
| `ponr_lead_time` | Did the alarm precede the point of no return? (+ve = alarm before PoNR) |
| `false_continuity_rate` | What fraction of the doomed window did the detector stay quiet? |
| `drift_magnitude_at_collapse` | How much had the representation drifted at PoNR? |

## Reproducibility

Fixed seeds make the synthetic rollouts and reports byte-reproducible
(`test_report_reproducible`). CI runs lint, type checks, branch coverage, builds,
and installed-artifact smoke tests on Python 3.10, 3.11, and 3.12 with no GPU.
Current GPU runs write auditable competence and recovery-run artifacts. The
boundary between what the published package reproduces and what only a live
simulator can establish is stated in
[`docs/GPU_REPRODUCIBILITY.md`](docs/GPU_REPRODUCIBILITY.md).

## External validation

The historical GPU results were produced on one setup. If
you run IPFD, **independent reports are the most useful contribution**: they
broaden compatibility evidence beyond one setup.

- Follow the [validation checklist](docs/VALIDATION.md): clone, install, run the
  CPU synthetic path, and (optionally) the GPU learned-policy demo. Every step emits
  a deterministic, copy-pasteable evidence block.
- Share the result in a **[Tested on my machine](https://github.com/yusufdxb/ipfd/discussions)** discussion
  (OS, Python, Isaac Lab version, GPU, whether it worked, whether PoNR matched).
- Hit a version or platform mismatch? Open a
  [compatibility report](https://github.com/yusufdxb/ipfd/issues/new?template=compatibility_report.yml). The
  tested GPU runtime is the fingerprint at the top of this README; other versions
  are unverified. The CPU analysis matrix covers Python 3.10, 3.11, and 3.12.

## License

MIT: see [`LICENSE`](LICENSE). `src/ipfd/oracles/pick_lift_sm.py` is vendored from
Isaac Lab under BSD-3-Clause; see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
