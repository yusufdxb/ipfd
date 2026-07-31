# Engineering note: exposed-state restore fidelity in contact-rich Isaac Lab branching

**Status: unfiled draft. Not submitted upstream.** This document has not been filed
as an issue, opened as a pull request, sent to any maintainer, or published. It is
a draft prepared for later human review, and it should not be shared without that
review.

**Framing.** This is an engineering observation about what the documented
save/restore interfaces do and do not cover, and about a validity check that is
easy to get wrong when building on them. It is not a bug report and not a claim of
defect. Isaac Lab documents `InteractiveScene.get_state()` as entity root pose,
root velocity, joint position, and joint velocity, and does not describe it as a
complete task, manager, solver, or random-state snapshot. Everything below is
consistent with that documentation. The observation is that the gap has a
measurable, direction-biased effect on downstream *decisions*, not only on
trajectories, and that the obvious check for it does not detect it.

## 1. Minimal problem statement

Tooling that branches a recorded rollout, restores a mid-episode state into a fresh
vectorized environment, and continues it, is commonly validated by confirming that
the restored observation equals the recorded observation at the branch step. In a
contact-rich manipulation task, that check can pass on every branch while a
meaningful fraction of those branches still reach a different task outcome, even
when the continuation replays the recorded action sequence exactly.

## 2. Tested versions

| Component | Version |
|---|---|
| `isaaclab` (locally installed distribution reported) | 4.5.22 |
| Isaac Sim | 6.0.0.0 |
| Python | 3.12 |
| Device | single CUDA GPU, consumer NVIDIA (Blackwell) class |
| Physics time step | 0.01 s |
| Control (environment) time step | 0.02 s |

Isaac Lab repository state used for the study: `a4a7602f29e755e2673fe0022ea35566df6dd7d5`.

## 3. Task and checkpoint

- Task: `Isaac-Lift-Cube-Franka-v0` (Franka Emika Panda, single-object lift).
- Policy: published RSL-RL feed-forward actor, checkpoint SHA-256
  `fb658f989bf5ebf35b20347813275979a6778ade8d3823d12eb3190612f9e36d`.
- Assets: Isaac 4.5 production tree. Entry USD hashes for the Franka, DexCube, and
  Seattle lab table are recorded in the study provenance; transitive dependencies
  are not hashed.
- The policy is deterministic under the tested configuration and has no recurrent
  or observation-history state, verified rather than assumed.
- Nominal lift competence was checked per seed group as an experiment control and
  passed. Note separately that this same published checkpoint measured 0.00%
  success in an independent 64-environment evaluation on this runtime, so it should
  not be treated as representative of a well-trained policy.

## 4. Snapshot APIs used

- Capture: `InteractiveScene.get_state()`.
- Restore: `InteractiveScene.reset_to()`, with root poses translated between
  environment origins. `reset_to()` also initializes joint position and velocity
  targets from the restored joint state.
- Manager-level state was read and written directly through the action manager,
  command manager, reward manager, termination manager, event manager, and
  observation manager attributes for the expanded protocol.
- Contact forces were read through object-filtered contact sensors, one per
  finger, used for instrumentation only. These expose measurements but no supported
  state-restoration interface.

## 5. State captured by each protocol

**Protocol A, `scene_plus_basic_manager_state`:** articulation root pose and
velocity; articulation joint position and velocity; rigid-object root pose and
velocity; action-manager current action; action-manager previous action; command
tensor, time-to-resampling, and resampling counter; episode-length counter.

**Protocol B, `expanded_runtime_state`:** all of Protocol A, plus raw and processed
buffers for every action term; articulation joint position, velocity, and effort
targets; environment reward, reset, terminated, and timeout buffers; reward-manager
episodic sums, step rewards, and reward buffer; termination-manager term,
last-episode, terminated, and timeout buffers; command metrics and cached
world-frame command data where present; observation-history circular buffers where
configured; event-manager per-environment interval and reset-trigger buffers where
configured; external disturbance-scheduler type, start, duration, magnitude,
target, random values, applied-event state, and remaining duration; and the
presence or absence of policy recurrent state.

Audited but not independently restorable per environment: PyTorch CPU and CUDA RNG
state, NumPy and Python RNG state, process-global simulation and control counters,
and curriculum term configuration shared across the vectorized environments.

## 6. Unsupported solver and contact state

No supported per-environment interface was found to capture or restore:

- PhysX warm-start impulses;
- solver contact manifolds and persistent contact caches;
- broadphase pair caches;
- per-contact friction solver state;
- internal GPU solver scheduling state;
- contact-sensor history buffers.

Neither protocol is a complete simulator snapshot, and neither is described as one.

## 7. Smallest reproducible observation

`scripts/isaaclab_reset_to_contact_mre.py` isolates the effect without the full
study:

- With `--grasp_steps 0` (free space, no contact history), the exposed-state round
  trip is exact and all post-step observation gaps are zero.
- With `--grasp_steps 40` (the same round trip after the gripper has closed on the
  cube), the exposed-state round trip is still reported exact at the restore step,
  while continuing with the *recorded actions* produces a final exact-action
  observation gap of 0.1046.

Evolved contact-rich state is required to reproduce the divergence. The reproducer
demonstrates the effect; it does not identify which unexposed state causes it.

## 8. Exact-action decision-disagreement result

Preregistered paired study, five independent seed groups (101, 211, 307, 401, 503),
separate process workers, machine-derived branch phases, horizons defined as
executed control actions, and disturbance schedules asserted identical between each
branch and its uninterrupted reference. Primary decision: `sustained_lift`.
Continuation: replay of the recorded action sequence.

| | Protocol A | Protocol B |
|---|---:|---:|
| Paired records | 444 | 444 |
| Decision disagreements | 18 (4.05%) | 11 (2.48%) |
| False recoverable | 0 | 0 |
| False unrecoverable | 18 | 11 |

Seed-paired mean difference -1.57 percentage points, bootstrap 95% interval
[-2.48, -0.67] percentage points over 5 independent units, improving in 4 of 5 seed
groups.

By horizon (74 records per protocol per horizon):

| Control steps after branch | Protocol A | Protocol B |
|---:|---:|---:|
| 1, 3, 5 | 0 / 74 each | 0 / 74 each |
| 10 | 2 / 74 | 0 / 74 |
| 30 | 4 / 74 | 1 / 74 |
| 90 | 12 / 74 | 10 / 74 |

By disturbance family: `object_teleport` 2/210 to **0/210**;
`gripper_open_interruption` 16/234 to 11/234. Widening to all three decision
predicates preserves this: teleport 4/630 to 0/630, gripper-open 53/702 to 39/702.

Two things follow. Restoring the additional exposed runtime state in Protocol B
gives a consistent, measurable improvement and clears the teleport family entirely.
It does not eliminate disagreement in the family whose disturbance acts through
sustained finger-object contact, which is where unrestored solver and contact state
would be expected to matter. The study measures this association; it does not
attribute the residual to any specific missing state.

## 9. Why immediate observation equality is insufficient

In the preserved three-seed cohort, the restored policy observation matched the
recorded observation at the branch step in **120 of 120** branches. Thirteen of
those same 120 branches reached a different terminal decision, ten of them under
identical replayed actions.

The policy observation is a projection of simulator state. Two states that agree on
that projection can disagree on state the projection does not expose, including
contact and solver state that governs the next several hundred physics substeps.
Because the divergence is amplified by contact rather than damped, it grows with
horizon: zero primary disagreement through 10 control steps, 13.5% at 90.

The failure mode is quiet. A validity check at the restore step passes, short
smoke tests pass, and the error only appears at the horizons that a recovery probe,
a counterfactual rollout, or a long regression replay actually uses. It is also
direction-biased here: under exact-action continuation nearly every error is
false-unrecoverable, meaning the branch reports failure where the reference
succeeded.

## 10. Possible documentation or regression-test value

Offered as options for maintainers to weigh, not as requests:

1. A short documentation note next to the scene-state API stating that the returned
   state is not sufficient for decision-equivalent continuation in contact-rich
   tasks, and that observation equality at the restore step is not a validity check
   for it.
2. A worked list of the manager-level state a caller must restore alongside
   `get_state()` for a branched environment to continue faithfully. Protocol B in
   `SNAPSHOT_PROTOCOLS.md` is a concrete, tested starting point.
3. A regression test of the shape used here: branch a recorded rollout at a
   contact-rich state, continue with recorded actions for a long horizon (tens to
   ~100 control steps), and assert agreement on a task predicate rather than on the
   immediate observation. A short-horizon or observation-only assertion would not
   catch this.
4. If the residual is in fact PhysX solver and contact-cache state, an interface to
   capture and restore it per environment would be the direct fix. Establishing
   that it *is* the cause would require a targeted experiment that this work did
   not perform.

## 11. Reproduction

Minimal observation (§7):

```bash
OMNI_KIT_ACCEPT_EULA=YES <isaac-lab-python> \
  scripts/isaaclab_reset_to_contact_mre.py --headless \
  --asset_root https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5 \
  --grasp_steps 40      # compare against --grasp_steps 0
```

Full paired study (§8), requires a CUDA GPU:

```bash
export IPFD_ISAACLAB_ROOT=/path/to/IsaacLab
export IPFD_CHECKPOINT=/path/to/checkpoint.pt
export IPFD_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5

OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src \
  "$IPFD_ISAACLAB_ROOT/isaaclab.sh" -p \
  scripts/run_snapshot_protocol_study.py \
  --checkpoint "$IPFD_CHECKPOINT" \
  --asset-root "$IPFD_ASSET_ROOT" \
  --output-dir /tmp/ipfd-corrected-five-seed \
  --isaac-lab-root "$IPFD_ISAACLAB_ROOT"

python3 scripts/analyze_snapshot_protocol_study.py \
  --study-dir /tmp/ipfd-corrected-five-seed
```

Protocol definitions: `SNAPSHOT_PROTOCOLS.md`. Experimental design:
`CORRECTED_EXPERIMENT_PROTOCOL.md`. Recorded artifacts and hashes:
`results/branch_validity/`.

## 12. Limitations

- One task, one robot, one checkpoint, one Isaac Lab version, one Isaac Sim
  version, one machine, one GPU class. Nothing here establishes that the behavior
  generalizes to other tasks, robots, policies, or simulator versions.
- Five independent seed groups. Branches, horizons, predicates, and continuations
  within a seed group are correlated, so five is the effective sample size for any
  uncertainty statement.
- Two disturbance families and one primary decision predicate drive the headline
  numbers.
- The residual disagreement under Protocol B is **not attributed** to unexposed
  PhysX solver state, or to any other specific mechanism. The concentration in the
  sustained-contact disturbance family is suggestive and nothing more; no
  experiment isolating solver state was run.
- The checkpoint used is not a strong policy on this runtime (see §3). The result
  concerns restore fidelity, not policy quality, but a reviewer may reasonably ask
  whether a better policy would produce the same pattern.
- No physical-robot validation of any kind.
- These are the measurements this project made. They are not a general statement
  about Isaac Lab's reliability, correctness, or fitness for any purpose.
