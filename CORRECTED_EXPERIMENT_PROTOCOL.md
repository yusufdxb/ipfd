# Corrected branch-validity experiment protocol

Status: preregistered before inspecting Protocol A versus Protocol B outcomes.

## Research question

Does `expanded_runtime_state` preserve downstream lift decisions more
faithfully than the existing `scene_plus_basic_manager_state` protocol after a
cold target-environment reset?

The secondary question is attempted only if Protocol B meets the positive
control threshold: can a finite-sample validity gate reject unreliable
branches on held-out seed groups and correct a downstream robotics conclusion?

## Hypotheses and stopping rules

Primary null hypothesis:

> Protocol B does not materially reduce exact-action sustained-lift decision
> disagreement relative to Protocol A.

Protocol B supplies a meaningful improvement only if all conditions hold:

1. exact-action sustained-lift disagreement falls by at least 50 percent
   relative to Protocol A across the corrected records;
2. the per-seed disagreement rate improves in at least two of five independent
   seed groups;
3. the improvement is not obtained by changing the success predicate or
   dropping a disturbance, phase, or horizon stratum;
4. false-recoverable and false-unrecoverable counts are reported separately.

The study stops and the branch-validity research direction is archived if
Protocol B fails this threshold. If it passes, the study continues only when
both valid and invalid Protocol B examples remain available for held-out
gating. The final direction is also archived if gating changes no PoNR,
controller-ranking, or checkpoint-selection conclusion relative to an
uninterrupted control.

## Independent unit and seeds

The independent uncertainty unit is a seed group, not a branch, horizon,
predicate, or continuation. Results from multiple branch points and horizons
within one seed group are treated as correlated.

Five base seed groups are fixed:

`101, 211, 307, 401, 503`

Each trajectory records distinct fields for:

- simulator seed;
- environment/reset seed;
- deterministic policy seed;
- disturbance seed;
- branch-selection seed.

The PhysX evolution and the selected policy are deterministic under this
configuration. Their seed fields are still recorded to distinguish absence of
stochasticity from missing provenance. Teleport and gripper-interruption
trajectories use distinct environment seeds within a seed group. Separate
process workers prevent one trajectory's solver or random state from
contaminating the next.

## Task and checkpoint

- task: `Isaac-Lift-Cube-Franka-v0`;
- policy: published RSL-RL feed-forward actor;
- checkpoint SHA-256:
  `fb658f989bf5ebf35b20347813275979a6778ade8d3823d12eb3190612f9e36d`;
- physics time step: 0.01 seconds;
- control time step: 0.02 seconds;
- policy observation corruption: disabled;
- primary hardware scope: one CUDA GPU.

The checkpoint must pass a nominal lift competence control in every seed group.
Failure to lift is reported as a failed control, not converted into a branch
sample.

## Paired branch design

Each worker owns one uninterrupted reference environment and preallocated
probe environments. Every probe is reset immediately before restoration.
Protocol A and Protocol B are restored into separate cold probe environments
at the same source step. Each protocol has:

- a recorded identical-action continuation;
- a closed-loop learned-policy continuation.

The uninterrupted source, Protocol A, and Protocol B share the same absolute
disturbance schedule. The code asserts schedule equality before recording a
comparison.

Each phase is assigned once per trajectory at its first machine-derived
occurrence. No hand-selected lift-relative offsets are branch points.
Only one phase can claim a control state. If two predicates first become true
together, the earlier phase claims that state and the later phase remains
eligible at the next state. This prevents duplicated nominal checkpoints.

## Phase predicates

The following predicates are evaluated from runtime signals:

- `free_space_pre_manipulation`: initial state, no finger-object contact, object
  rise below 1 mm;
- `approach`: no contact, end-effector to object distance at most 18 cm, and
  distance decreases over the recent control history;
- `first_contact`: first control state where either finger's object-filtered
  contact force is at least 0.5 N;
- `stable_grasp`: both finger-object forces are at least 0.5 N, finger aperture
  is at most 5.5 cm, object to end-effector distance is at most 12 cm, for
  three consecutive control states;
- `initial_lift`: object rise is at least 5 mm while the stable-grasp geometric
  conditions hold;
- `sustained_lift`: object rise is at least 4 cm and object to end-effector
  distance is at most 12 cm for five consecutive control states;
- `disturbance_onset`: the exact state immediately after a teleport is applied,
  or immediately before the first forced-open gripper action;
- `post_disturbance_recovery`: the first five-state sustained-lift window after
  the disturbance schedule ends.

Finger-object forces come from two object-filtered contact sensors, one per
finger. Missing or unsupported contact measurements fail the phase-control
check.

## Continuation horizons

Horizon means the number of actual control actions executed after the branch.
The fixed horizons are:

`1, 3, 5, 10, 30, 90`

A terminal task state is absorbing for decision evaluation. A record is marked
censored if a required recorded-action suffix is unavailable before the
requested horizon. Censored cells are never counted as agreements.

## Disturbances

Two disturbance families are fixed:

- `object_teleport`: one object translation at the scheduled start step;
- `gripper_open_interruption`: a forced-open gripper command for eight control
  steps.

Each schedule record includes start, duration, vector or command magnitude,
target, generated random values, and a canonical schedule hash. A branch and
its uninterrupted reference must have identical schedule hashes.

## Decision predicates

Predicates are evaluated separately:

- `final_height`: final object rise is at least 6 cm and no failure termination
  has occurred;
- `sustained_lift`: object rise is at least 4 cm and object to end-effector
  distance is at most 12 cm over a five-state window ending at the decision
  horizon, with no failure termination;
- `stable_grasp`: bilateral object-filtered contact, aperture, and pose
  conditions hold over a three-state window ending at the decision horizon,
  with no failure termination.

The configured Isaac Lift task has no native success termination. Its native
`object_is_lifted` reward condition is recorded as a task signal but is not
misrepresented as task-native success.

The primary protocol-comparison decision is `sustained_lift`. Predicate
sensitivity is mandatory.

## State-equivalence measurements

For every branch and protocol, the experiment records immediate and first-step
maximum absolute differences for:

- policy observation;
- articulation root pose and velocity;
- joint position and velocity;
- object pose and velocity;
- articulation targets;
- action-manager and action-term buffers;
- command and task buffers;
- simulation and episode counters;
- disturbance-scheduler state;
- contact-force instrumentation.

Across each continuation it records first numerical, observation, contact,
action, and predicate divergence; terminal agreement; and maximum and terminal
trajectory error.

Immediate observation equality is an input feature, not a validity label.

## Statistical reporting

All raw per-branch records are retained. Descriptive rates are reported by
protocol, continuation, phase, horizon, disturbance, and predicate.
Uncertainty is reported through the five per-seed paired rates and a
seed-cluster bootstrap. The small number of independent units is stated
prominently. Branch-level binomial intervals are not used as publication-grade
uncertainty.

## Conditional validity gate

This stage is skipped if the positive-control threshold fails.

If eligible, the gate uses leave-one-seed-out evaluation. Inputs are restricted
to branch-time or first-step information:

- phase;
- contact state;
- horizon;
- disturbance type;
- protocol;
- immediate mismatch features;
- first-step mismatch features;
- controller-target and task-buffer mismatch.

It is compared with accept-all, reject-all, contact-only, immediate-observation
equality, phase-only, and protocol-only baselines. It reports coverage,
accepted-branch fidelity, invalid accepted branches, valid rejected branches,
and admitted false-recoverable and false-unrecoverable labels. It is called an
empirical finite-sample validity gate, not a certificate.

## Runtime and artifact controls

- asset entry URLs are checked before a long run;
- source commit, dirty state, checkpoint, configuration, asset entries, and
  artifacts are hashed;
- any worker exception or failed runtime control returns a nonzero process
  status;
- raw traces may stay outside git, but their paths, sizes, and hashes are
  recorded;
- the preserved three-seed artifacts are never overwritten.

## Reproduction

Set the task-specific paths and run the isolated five-seed study:

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
```

Audit and visualize the generated artifacts:

```bash
python3 scripts/analyze_snapshot_protocol_study.py \
  --study-dir /tmp/ipfd-corrected-five-seed
```
