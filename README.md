# IPFD — Isaac Policy Failure Debugger

**Find the exact moment a policy is doomed — before it visibly fails.**

IPFD is a small, headless debugging utility for robot-policy rollouts in
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/). It answers one question an
average-success-rate number can never answer:

> The policy failed at second 2.7. But *when did it actually become unrecoverable*,
> and *did any internal signal know* before the failure was externally visible?

Scope is deliberately narrow: **Franka Emika Panda, single-object pick-and-place,
Isaac Lab default environment.** No multi-robot zoo, no benchmark suite, no Kit
extension. One sharp tool that does one thing well.

---

## The failure mode it exposes

A well-trained policy can stay **high-confidence after entering an irreversible
failure trajectory**. Success-rate evaluation counts those episodes as "fine" right
up until the object hits the floor. IPFD makes the silent interval visible:

![silent failure timeline](examples/figures/silent_failure.png)

Reading the panels top to bottom (this is the real output of
`examples/run_synthetic.py`, seed 0):

- The object leaves the reachable workspace at the **Point of No Return** (red, 1.5s).
- **Action output stays perfectly calm for ~1 second afterward** — the policy has no
  idea it is doomed. It only thrashes near the externally-observable failure (grey, 2.7s).
- **Policy entropy collapses** (confidence *rises*) through the doomed window — the
  "confident but wrong" signature.
- **Representation drift** and the combined **imminence score** fire at the alarm
  (orange), ~1 second before the action-level thrash a human would notice.

```
=== IPFD Failure Debug Report ===
outcome            : FAILURE
point of no return : step 90 (1.50s)
observable failure : step 160 (2.67s)
detector alarm     : step 95 (1.58s)
failure lead time  : +1.08s   (alarm before visible failure)
PoNR lead time     : -0.08s   (alarm vs irrecoverable; +ve = actionable)
silent-doom window : +1.17s   (doomed but looked fine)
false continuity   : 6%       (of doomed window, detector stayed quiet)
verdict            : SILENT COLLAPSE -- alarm fired only AFTER the trajectory was doomed.
```

The honest reading: **internal signals bought ~1.08s of warning over external
visibility, but landed right at the point of no return** — enough for diagnosis, not
always for prevention. IPFD reports both, and never pretends the alarm was earlier
than it was.

---

## The one hard idea: Point of No Return

"The first timestep the task is irrecoverable, even under optimal control" **cannot
be read off a passive log** — you only know a state was doomed by *trying to recover
from it and failing*. So IPFD defines PoNR operationally against a **recovery probe**:

```
recovery_success[t] == True  <=>  a best-effort recovery controller, restarted from
                                  the saved sim state at step t, reaches the goal
                                  within a fixed budget.
```

`PoNR = the first step after which recovery never again succeeds.` This is a **sound
upper bound** on the true optimal-control PoNR (a better recovery controller can only
push it later). We say *"irrecoverable under the provided recovery oracle,"* not
*"provably irrecoverable."* Producing `recovery_success` needs a simulator that can
save/restore state; consuming it does not. That boundary is the whole architecture.

---

## Architecture

```
ipfd/
  types.py              Rollout — the single unit of analysis (pure NumPy arrays)
  detectors.py          action-variance / entropy-collapse / drift -> imminence score
  ponr.py               Point of No Return from a recovery-probe array
  metrics.py            time-to-failure, lead time, false continuity, drift@collapse
  report.py             build_report() -> FailureDebugReport (+ .summary(), .to_json())
  viz.py                plot_timeline() — the stacked-panel figure above (Agg, headless)
  adapters/
    synthetic.py        simulator-free rollouts for tests/examples/CI
    isaac_lab.py        real Franka rollout collection + recovery probe (GPU, gated)
```

**The analysis layer never imports a simulator.** Detectors, PoNR, metrics, report,
and viz are pure NumPy/Matplotlib and run in CI without a GPU. Only
`adapters/isaac_lab.py` talks to Isaac Lab, and it is imported lazily.

### Detectors (deliberately simple, no ML, no training)

Each returns a per-step score in `[0,1]`, self-calibrated against the rollout's own
calm early window (robust median + `max(MAD, std)` scale, so calm jitter doesn't
false-alarm). Combined by a weighted max into a single **failure-imminence score**;
`first_alarm` requires the score to persist above threshold to page.

| Detector | Fires on |
|---|---|
| action-variance spike | policy starts thrashing |
| entropy collapse | policy becomes overconfident as it commits |
| representation drift | latent embedding leaves its early-episode manifold |

---

## Quickstart

```bash
pip install -e ".[dev]"      # analysis layer only — no GPU, no Isaac Lab
pytest                        # 24 tests, all pure-NumPy
python examples/run_synthetic.py   # prints two reports, writes plots + JSON to examples/figures/
```

```python
from ipfd import build_report, plot_timeline
from ipfd.adapters.synthetic import make_silent_failure_rollout

rollout = make_silent_failure_rollout(seed=0)
report  = build_report(rollout)
print(report.summary())
plot_timeline(rollout, report, "timeline.png")
```

### With a real Franka rollout (needs Isaac Lab + a GPU)

```python
from ipfd.adapters.isaac_lab import collect_rollout
from ipfd import build_report

rollout = collect_rollout(env, my_policy, recovery_controller=my_recovery, seed=0)
print(build_report(rollout).summary())
```

> `adapters/isaac_lab.py` is **runtime-verified on Isaac Lab 4.5.22** with
> `Isaac-Lift-Cube-Franka-v0` on a live GPU — see
> [`scripts/verify_isaac_runtime.py`](scripts/verify_isaac_runtime.py), which reports
> `overall_verdict: REAL_COMPATIBLE`. Both flagged touchpoints are confirmed against
> the real sim: the observation key is `policy` (shape `(1, 36)`), and
> `env.unwrapped.scene` exposes `get_state()` / `reset_to()`, so the recovery probe
> runs real state save/restore. What is *not* yet shown: **meaningful** pre-failure
> PoNR/imminence needs a trained policy plus a real recovery controller — with an
> untrained oracle the probe never recovers and PoNR degenerates to step 0. That is a
> policy/oracle gap, not an API gap.

### Verifying it yourself

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  ~/Sim/isaac-sim-venv/bin/python scripts/verify_isaac_runtime.py --headless
```

The script launches a live Franka env, drives it through IPFD's own adapter (no
mocks), runs `build_report` on the real rollout, and prints a machine-readable
`IPFD_RUNTIME_COMPATIBILITY` block. It detects every API assumption at runtime rather
than trusting it.

### State-restore fidelity and the recovery-probe limitation

Two further scripts stress the recovery probe that PoNR depends on:

- [`scripts/verify_state_fidelity.py`](scripts/verify_state_fidelity.py) —
  save → replay a fixed action → `reset_to` → replay again. On Isaac Lab 4.5.22
  the round trip is **bit-exact**: write-back, observation, joint-state and
  object-state diffs are all `0.0`, reward diff `~1e-6`. Single-step state restore
  is faithful. **`STATE_RESTORE_FIDELITY: PASS`.**

- [`scripts/verify_real_policy.py`](scripts/verify_real_policy.py) — drives IPFD
  with a *competent* policy (Isaac Lab's scripted pick-and-lift state machine;
  no trained checkpoint exists on the test machine). **Honest result:** run
  uninterrupted (`--diagnose`) the policy lifts the cube on every seed, but with
  the recovery probe interleaved in the *same* env, the primary rollout is
  corrupted and PoNR degenerates. Single-step restore is exact, yet repeatedly
  restoring across a *contact-rich* grasp does not preserve the primary
  trajectory. So under a real policy, **PoNR is not yet meaningful in the loop**
  (`meaningful_pnor_detected: NO`) — an open limitation of the recovery-probe
  design, not of the detectors or analysis layer. See the `RESULT` block in that
  script for the full accounting.

- [`scripts/verify_probe_transparency.py`](scripts/verify_probe_transparency.py)
  — **root-causes** the above. It injects a probe before contact vs after the
  grasp and compares the primary trajectory with vs without the probe. Measured
  result: the entity **write-back diff is `0.0` in both cases** (`reset_to` does
  restore joint/object pose and velocity bit-exactly, even post-grasp), yet the
  post-grasp trajectory diverges *immediately* (A-vs-B observation diff `1.2`,
  onset step 0) while the pre-contact probe is transparent. The unrestored state
  is therefore the **PhysX contact-manifold / solver warm-start cache**, which is
  not part of Isaac Lab's `scene.get_state()`. `root_cause:
  CONTACT_STATE_NOT_RESTORED`.

- [`scripts/verify_pnor_decoupled.py`](scripts/verify_pnor_decoupled.py) — the
  implied fix was to **decouple** probing from the primary (record the primary
  once, evaluate `recovery_success[t]` in separate passes that never resume it).
  Building and running it surfaced a **deeper, directly confirmed blocker**:
  `reset_to_poisons_env: YES`. A fresh episode lifts the cube (`zmax 0.341`);
  after *one* `reset_to` probe followed by `env.reset(seed=0)`, the same seed no
  longer lifts (`zmax 0.045`). So a single `reset_to()` **permanently corrupts
  the PhysX sim and the corruption survives `env.reset()`** — which poisons
  pass-2 probes for each other and invalidates their verdicts. **PoNR is
  therefore not meaningful in a single shared env instance in this Isaac Lab
  version** (`meaningful_pnor_detected: NO`); a correct implementation needs env
  *isolation* (a separate sim instance per probe) or a `reset_to` that also
  restores PhysX solver/contact state. Single-step restore remains bit-exact;
  the block is the persistent `reset_to` side effect, measured against ground
  truth.

- [`scripts/verify_multienv_isolation.py`](scripts/verify_multienv_isolation.py)
  — tests whether the poison is *global* or *per-env*. Measured: churning env 1
  through 14 `reset_to` calls leaves **env 0 bit-identical** (lift `+0.320` with
  and without). So `reset_to` is **local to the reset env** — the poison only
  bites `num_envs=1` because there's nowhere else to run. `isolation_viable: YES`.

- [`scripts/verify_pnor_isolated.py`](scripts/verify_pnor_isolated.py) — uses
  that isolation: primary rollout recorded in env 0 (never reset), recovery
  probes farmed to env 1. This **solves the corruption/poison block** — the
  primary is pristine and lifts (`success=True`), and recovery verdicts use a
  *continue-the-policy* oracle. Run with `--auto_doom` it reaches
  **`meaningful_pnor_detected: YES` / `overall_status: VERIFIED`**: the failure is
  injected in the **grasped region** (cube teleported out of reach at step 131,
  eight steps after the nominal lift onset), and the recovery verdicts flip
  cleanly — checkpoints 126–137 recover (`True`), 138 onward do not (`False`) — so
  `point_of_no_return` fires at **step 138**, localising at the injected doom
  (within tolerance) with a **+0.22 s lead** over observable failure. The claim is
  bounded honestly: it holds where the recovery oracle can adjudicate. **Pre-grasp
  checkpoints stay noisy** — `reset_to` hands the probe a cold PhysX contact state
  that derails the scripted grasp's sub-cm approach (`--debug_step 30` traces the
  stall), so a stray early `True` appears among the pre-grasp `False`s. IPFD's
  analysis, single-step restore, and env-isolated probing are sound; the only
  residual limitation is recovery-oracle robustness to a cold-contact restart
  during fine *pre-grasp* manipulation — a controller property, not an IPFD gap.

---

## Metrics (the minimal set)

| Metric | Question it answers |
|---|---|
| `time_to_failure` | When did failure become externally observable? |
| `failure_lead_time` | How early did the alarm fire vs visible failure? |
| `ponr_lead_time` | Did the alarm precede the point of no return? (+ve = actionable) |
| `false_continuity_rate` | What fraction of the doomed window did the detector stay quiet? |
| `drift_magnitude_at_collapse` | How much had the representation drifted at PoNR? |

## Reproducibility

Fixed seeds throughout; synthetic rollouts and reports are byte-reproducible
(`test_report_reproducible`). CI runs lint + tests + a headless example smoke-run on
Python 3.10 and 3.11 with no GPU.

## License

MIT.
