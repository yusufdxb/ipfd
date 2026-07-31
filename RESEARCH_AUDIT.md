# IPFD Research Audit

## Blunt assessment

IPFD currently computes a controller-relative boundary over restored simulator
branches. It does not establish a physical Point of No Return. The strongest
defensible result is a negative one: in the tested contact-rich lift task,
immediate equality of exposed restored state did not guarantee either
trajectory-equivalent continuation or the same terminal recovery decision.

The current concept needs a major redesign. Restored-branch decision fidelity
must become an explicit precondition for every recoverability claim.

## Strongest claim and biggest weakness

Strongest defensible current claim:

> On one Isaac Lab manipulation task, IPFD can record a trajectory, restore
> exposed scene and selected manager state into another vectorized environment,
> and detect cases where the restored continuation changes the task-success
> decision relative to uninterrupted execution.

Biggest weakness:

> The current PoNR treats restored recovery labels as ground truth even though
> the branch-validity experiment recorded 13 outcome disagreements in 120 paired
> comparisons, including 10 under identical recorded actions.

## What the system does

1. A primary rollout stores observations, pre-mutation actions, optional entropy
   and embeddings, exposed scene snapshots, and a terminal failure index.
2. A probe pass shifts an exposed scene snapshot from environment 0 into
   environment 1.
3. A supplied recovery controller executes for a fixed budget and a supplied
   predicate produces a binary recovery label.
4. Repeated labels are reduced with a false-fraction threshold. Current repeats
   reuse the same simulator cell and have not been shown to be independent
   stochastic trials.
5. Unprobed timesteps are forward-filled.
6. `point_of_no_return` returns one plus the last successful recovery probe,
   which is the start of the final all-false suffix.
7. Action variance, entropy drop, and embedding drift detectors are calibrated
   from an early prefix and evaluated independently of recovery.
8. Reports place the alarm, PoNR, and visible failure on a common timeline.

## Architecture map

| Stage | Current implementation | Proof boundary |
|---|---|---|
| State capture | `scene.get_state()` plus rollout arrays | Exposed entity root, joint, and velocity state only |
| Restore | `scene.reset_to()` with environment-origin translation | Immediate exposed-state equality can be tested |
| Manager restore | Expanded experiment restores action history, previous action, episode length, and command state | Does not include every task, solver, contact, RNG, sensor, or policy state |
| Perturbation | Teleport, gripper-open interruption, scripted slips | Mostly controlled and injected |
| Continuation | Recorded exact actions or closed-loop learned policy | Exact-action mode isolates branch-state fidelity most cleanly |
| Recovery criterion | Final object height above a threshold | Does not establish a stable physical grasp |
| Search | Strided probing plus forward fill | Can miss recovery islands between probes |
| PoNR | One plus the last true label | Unique only by an array convention, not a universal trajectory property |
| Uncertainty | Repeat agreement and probe-stride interval | Neither is a statistical confidence interval for recoverability |
| Isolation | Immediate environment 0 object-pose delta around an environment 1 reset | Does not prove end-to-end primary invariance |

## Claim-to-evidence matrix

| Claim | Status | Executable evidence | Limitation |
|---|---|---|---|
| Pure NumPy analysis and packaging work without Isaac Lab | Verified | CPU test suite, Ruff, MyPy | Does not validate simulator claims |
| Isaac Lab task attachment | Verified with an explicit production asset root | Runtime smoke executes 16 steps and produces a report | Default staging asset resolution failed during this audit |
| Immediate exposed scene-state round trip is bit exact | Verified narrowly | `verify_state_fidelity.py` and reset reproducer | Excludes manager, policy, random, contact-solver, and other hidden state |
| Full trajectory restoration is exact | Falsified | Exact-action continuation diverged materially in 56 of 60 clean comparisons | One task and one snapshot protocol |
| Restored recovery decision equals uninterrupted decision | Falsified generally for the tested protocol | 13 of 120 clean comparisons disagreed | Samples within a seed are correlated |
| Dual-environment reset write does not immediately move the primary object | Partially verified | 51 measured reset boundaries had zero object-pose delta | Only one measured object and one instant |
| Historical scripted PoNR equals 138 | Numerically reproduced | `verify_pnor_grasped.py` | One seed, injected gripper opening, nonmonotone raw labels, height-only predicate |
| The historical PoNR precedes visible failure by 0.42 seconds | Unsupported | Script prints the number | `t_failure` is the terminal horizon, not the first visible drop |
| Learned teleport fixture provides an actionable alarm | Falsified | Alarm 20, disturbance 56, PoNR interval 49 to 56 | Alarm precedes the injected fault, so the actionable window is empty |
| Published learned policy has no task competence | Checkpoint-dependent, not a general claim | One cached checkpoint achieved 0 percent; another content-distinct checkpoint achieved 100 percent over 64 environments in one run | Asset and checkpoint provenance differ; neither run proves general competence |
| Recovery repeats provide confidence | Unsupported scientifically | API aggregates repeated Booleans | No independent sampling unit or calibrated interval |
| PoNR is stable across seeds, policies, tasks, timestep, and thresholds | Unverified | No accepted evidence matrix exists | Restore validity is already negative in the tested task |
| IPFD performs causal failure diagnosis | Unsupported | Only controlled synthetic contracts | No causal estimand, randomized attribution study, or planted-fault recovery |
| Backend independence, real-time operation, formal safety, or sim-to-real validity | Unsupported | None | These claims must not be made |

## Reproduced evidence

The audit baseline CPU suite passed 132 locally collected tests. Six were in a
locally excluded file, so the tracked tree alone reproduced 126. Branch coverage
was 85.59 percent. Ruff and MyPy passed.

The clean current-project branch-validity run used:

- task: `Isaac-Lift-Cube-Franka-v0`;
- a learned checkpoint identified by SHA-256
  `fb658f989bf5ebf35b20347813275979a6778ade8d3823d12eb3190612f9e36d`;
- seeds 0, 1, and 2;
- teleport and gripper-interruption disturbance families;
- recorded exact-action and closed-loop policy continuations;
- 120 paired decision comparisons.

It reproduced the negative classification with 13 disagreements. Ten of the 60
recorded exact-action branches disagreed. All 120 immediate restored policy
observations matched the captured observations, yet 56 of 60 exact-action
trajectories crossed a material divergence threshold. The raw report, generator
hash, checkpoint hash, trace hash, and command arguments are preserved under
[`results/branch_validity`](results/branch_validity).

## Ranked weaknesses

1. **Invalid scientific ground truth.** Restored recovery outcomes can differ
   from uninterrupted outcomes.
2. **Under-specified PoNR.** The result depends on recovery controller,
   constraints, horizon, predicate, restore protocol, and intervention family.
3. **Weak task evidence.** The headline historical result is a single scripted,
   injected, height-threshold example.
4. **No calibrated uncertainty.** Deterministic repeats and probe stride do not
   measure outcome probability or restore bias.
5. **No demonstrated user decision.** Current outputs have not improved policy
   selection, debugging, training, recovery design, or deployment decisions.

## Required redesign

Every counterfactual result should be fail-closed:

1. declare the snapshot, manager, policy, disturbance, and continuation
   protocol;
2. compare uninterrupted and restored branches on held-out paired states;
3. estimate decision disagreement separately from trajectory divergence;
4. reject PoNR or frontier estimates outside a declared validity envelope;
5. show that the gate corrects at least one downstream evaluation decision.

The current result falsifies universal branch validity. It does not yet certify
any phase as valid.
