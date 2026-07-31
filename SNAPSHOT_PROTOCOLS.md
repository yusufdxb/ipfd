# Snapshot protocols for the branch-validity study

This document specifies the restoration treatments before the corrected
five-seed comparison is run. Neither treatment is called a complete simulator
snapshot.

## Protocol A: `scene_plus_basic_manager_state`

This is the protocol used by the preserved three-seed baseline in
`results/branch_validity/clean_three_seed_report.json`.

Captured and restored per environment:

- articulation root pose and velocity;
- articulation joint position and velocity;
- rigid-object root pose and velocity;
- action manager current action;
- action manager previous action;
- command tensor, time to resampling, and resampling counter;
- episode-length counter.

The scene state is obtained with `InteractiveScene.get_state()` and written
with `InteractiveScene.reset_to()`. Root poses are translated between
environment origins. `reset_to()` also initializes joint position and velocity
targets from the restored joint state.

Not captured or restored:

- the raw and processed buffers owned by individual action terms;
- articulation position, velocity, and effort targets as they existed at the
  branch point;
- reward-manager accumulators;
- termination-manager buffers;
- environment reward, reset, terminated, and timeout buffers;
- command metrics and world-frame command cache;
- observation-history buffers;
- event-manager interval buffers;
- curriculum state;
- disturbance-scheduler state;
- random-number-generator state;
- policy recurrent state;
- contact-sensor history;
- PhysX warm-start impulses, contact manifolds, contact caches, broadphase
  state, or other solver-internal state;
- renderer or sensor scheduling state;
- process-global simulation counters.

## Protocol B: `expanded_runtime_state`

Protocol B restores everything in Protocol A plus the causally relevant
per-environment state exposed by this task and runtime.

Additional captured and restored state:

- raw and processed buffers for every action term;
- articulation joint position, velocity, and effort targets;
- environment reward, reset, terminated, and timeout buffers;
- reward-manager episodic sums, step rewards, and reward buffer;
- termination-manager term, last-episode, terminated, and timeout buffers;
- command metrics and cached world-frame command data when present;
- observation-history circular buffers when configured;
- event-manager per-environment interval and reset-trigger buffers when
  configured;
- external disturbance type, start, duration, magnitude, target, random
  values, applied-event state, and remaining duration;
- the presence or absence of policy recurrent/history state;
- simulation and control counters as audited metadata.

State that is audited but not independently restorable per environment:

- PyTorch CPU and CUDA random-number-generator state;
- NumPy and Python random-number-generator state;
- the environment's process-global simulation and control counters;
- curriculum term configuration shared by every vectorized environment.

The corrected experiment removes post-branch stochastic resampling within the
tested horizon. The policy checkpoint is a feed-forward MLP and has no
recurrent or policy-history state. The lift task has no configured
observation-history buffers or interval events. These are verified and
recorded rather than silently assumed.

State unavailable through the supported runtime interfaces:

- PhysX warm-start impulses;
- solver contact manifolds and persistent contact caches;
- broadphase pair caches;
- per-contact friction solver state;
- internal GPU solver scheduling state.

Contact-sensor buffers expose measurements but no supported state-restoration
API. They are used for phase and predicate instrumentation, not as policy
inputs or dynamics state.

## Positive-control interpretation

Protocol B is a positive control for omitted exposed runtime state. It is not a
positive control for unexposed PhysX solver state. If Protocol B does not reduce
exact-action terminal-decision disagreement by the preregistered amount, the
branch-validity direction stops under the study rule. A history-replay or
same-history diagnostic may localize the missing state, but it cannot rescue a
failed Protocol B result.

## Preserved baseline provenance

The previous result is immutable evidence:

| Item | SHA-256 |
|---|---|
| Clean three-seed report | `86189ab525c394d24d6e1b8b26427850ad5822bae9bd3789074e4c4985b9cad2` |
| Compressed traces | `dcf30076c60ddfea754ea7dd13e0db95377186d8f5c6febaff9aa8ddc5a11558` |
| Baseline generator | `17b41305c418feb57550373ecbe1894652f11c4fd0164c55e3a181348589769e` |
| Learned checkpoint | `fb658f989bf5ebf35b20347813275979a6778ade8d3823d12eb3190612f9e36d` |
| Baseline analysis summary | `f322c5d4f8846bf6a4508d3d163b8c86f2d91f7b2c9192b17b863222cefde6bb` |
| Baseline figure | `416d2a79cdce4e3d0b468590a77e6d2488c211204d41f5af998d49915a49f4f3` |

The live Stage 0 reproduction regenerated the report and trace files
byte-for-byte from the current working tree.
