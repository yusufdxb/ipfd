# IPFD: archived negative result

Verdict: **archive as an honest negative.** The branch-validity research direction
is stopped. IPFD did not demonstrate a flagship research capability. This document
is the standalone record of what was asked, what was measured, and why the work
ended. Implementation detail lives in the protocol documents it links to and is not
repeated here.

## 1. Research question

Under which task phases, continuation horizons, continuation controllers,
disturbances, and snapshot protocols does a restored simulator branch preserve the
downstream task decision of the corresponding uninterrupted branch?

The question is load-bearing rather than academic. Every recovery verdict IPFD
produces, and therefore every Point of No Return index derived from those verdicts,
is computed by restoring a recorded state into a probe environment and re-running
it. If the restored branch can reach a different decision than the uninterrupted
episode would have, the verdict is measuring the branch, not the episode.

## 2. Original assumption

> Restoring the exposed simulator state that Isaac Lab's `InteractiveScene`
> save/restore API provides, into a freshly reset environment of the same
> vectorized simulation, yields a branch whose downstream task decision matches the
> uninterrupted episode's, at least once the immediate restored observation is
> shown to be equal.

Immediate exposed-state equality was treated as sufficient evidence of a valid
branch. It is not.

## 3. Falsification result

The preserved three-seed cohort
([`results/branch_validity/clean_three_seed_report.json`](results/branch_validity/clean_three_seed_report.json),
SHA-256 `86189ab5…`) ran 120 paired comparisons on
`Isaac-Lift-Cube-Franka-v0`:

| Measurement | Value |
|---|---|
| Paired decision comparisons | 120 |
| Immediate restored observation equal to recorded observation | 120 of 120 |
| Terminal decision disagreements | 13 (10.8%) |
| Disagreements under identical recorded actions (exact-action) | 10 of 60 |
| Disagreements under closed-loop policy continuation | 3 of 60 |
| Exact-action trajectories crossing a material divergence threshold | 56 of 60 |

Classification: `FALSIFIED_UNIVERSAL_DECISION_FIDELITY`. Equality of the restored
observation at the branch point did not predict equality of the terminal decision.

That cohort had real confounds: duplicated branch points from hand-selected
lift-relative offsets, horizons that did not correspond to executed control
actions, and disturbance schedules that were not asserted identical between a
branch and its reference. The correct response was to rerun the measurement
properly, not to publish the number.

## 4. Corrected experimental design

Full specification:
[`CORRECTED_EXPERIMENT_PROTOCOL.md`](CORRECTED_EXPERIMENT_PROTOCOL.md),
preregistered before Protocol A and Protocol B outcomes were inspected.

- **Task and policy:** `Isaac-Lift-Cube-Franka-v0`, published RSL-RL feed-forward
  actor, checkpoint SHA-256 `fb658f98…`. Physics step 0.01 s, control step 0.02 s.
- **Independent unit:** the seed group, not the branch. Five fixed base seeds
  (101, 211, 307, 401, 503). Branches, horizons, predicates, and continuations
  within a seed group are treated as correlated.
- **Isolation:** separate process workers per seed and disturbance family, so one
  trajectory's solver or random state cannot contaminate the next. Each probe
  environment is reset immediately before restoration.
- **Branch points:** each phase assigned once per trajectory at its first
  machine-derived occurrence from contact-sensor and pose signals, no hand-picked
  offsets, no phase claiming a state another phase already claimed.
- **Horizons:** 1, 3, 5, 10, 30, 90, defined as control actions actually executed
  after the branch. Records whose recorded-action suffix is unavailable are marked
  censored and never counted as agreements.
- **Disturbances:** `object_teleport` and `gripper_open_interruption`, each with a
  canonical schedule hash that the code asserts identical between a branch and its
  uninterrupted reference before recording a comparison.
- **Decision predicates:** `sustained_lift` (primary), `final_height`,
  `stable_grasp`, evaluated separately. Predicate sensitivity mandatory.
- **Uncertainty:** five per-seed paired rates plus a 10,000-sample seed-cluster
  bootstrap. Branch-level binomial intervals are explicitly not used.
- **Controls:** ten worker control blocks, all passed, checking exact-action
  identity, horizon semantics, disturbance-onset observation, contact-measurement
  availability, unique branch steps, nominal pre-disturbance lift competence, and
  schedule equivalence.
- **Records:** 5,328 per-branch decision records.

## 5. Protocol A and Protocol B

Full state inventories: [`SNAPSHOT_PROTOCOLS.md`](SNAPSHOT_PROTOCOLS.md).

**Protocol A, `scene_plus_basic_manager_state`** (the treatment the three-seed
cohort used): articulation root pose and velocity, joint position and velocity,
rigid-object root pose and velocity, action-manager current and previous action,
command tensor and resampling state, episode-length counter.

**Protocol B, `expanded_runtime_state`** (the positive control): everything in
Protocol A plus raw and processed buffers for every action term, articulation
position/velocity/effort targets, environment reward/reset/terminated/timeout
buffers, reward-manager and termination-manager buffers, command metrics and
cached world-frame command data, observation-history buffers where configured,
event-manager interval buffers where configured, and full disturbance-scheduler
state.

Neither protocol is a complete simulator snapshot, and the document says so. What
Protocol B still does not restore, because the runtime exposes no supported
per-environment interface for it: PhysX warm-start impulses, solver contact
manifolds and persistent contact caches, broadphase pair caches, per-contact
friction solver state, and internal GPU solver scheduling state. Protocol B is
therefore a positive control for *omitted exposed* state only. It is not a test of
whether unexposed solver state matters.

Preregistered rule: Protocol B counts as a meaningful improvement only if
exact-action `sustained_lift` disagreement falls by **at least 50 percent**, the
per-seed rate improves in at least two of five seed groups, the improvement is not
obtained by changing the predicate or dropping a stratum, and false-recoverable and
false-unrecoverable counts are reported separately.

## 6. Primary results

Exact-action continuation, `sustained_lift` predicate, 444 paired records per
protocol
([`protocol_comparison.json`](results/branch_validity/corrected_five_seed/protocol_comparison.json)):

| | Protocol A | Protocol B |
|---|---:|---:|
| Records | 444 | 444 |
| Disagreements | **18** | **11** |
| Disagreement rate | 4.05% | 2.48% |
| False recoverable | 0 | 0 |
| False unrecoverable | 18 | 11 |

- Relative reduction: **38.9%**. Required: **50%**. `threshold_passed: false`.
- Seeds improved: 4 of 5 (seed 211 unchanged at 3.33%; 401 and 503 reached zero).
- Seed-paired mean difference: **-1.57 percentage points**, bootstrap 95% interval
  **[-2.48, -0.67]** percentage points over 5 independent units.
- Stopping decision: `STOP_BRANCH_VALIDITY_DIRECTION`,
  `positive_control_meaningfully_improved: false`.

The direction of the effect is real and consistently signed. Its size is not what
was registered as meaningful, and 5 independent units is a small basis for any
effect-size claim.

Across all three predicates and both continuations, Protocol B still produced 39 of
1,332 exact-action disagreements and 32 of 1,332 closed-loop disagreements. Note
the asymmetry: exact-action errors are almost entirely false-unrecoverable (38 of
39), while closed-loop errors split both ways (17 false-recoverable, 15 false-
unrecoverable). A restored branch replaying identical actions mostly fails where
the reference succeeded; a restored branch running the policy can fail either way.

## 7. Results by horizon and disturbance

Primary exact-action `sustained_lift`, 74 records per protocol per horizon
([`protocol_strata.json`](results/branch_validity/corrected_five_seed/protocol_strata.json)):

| Horizon (control steps) | Protocol A | Protocol B |
|---:|---:|---:|
| 1 | 0 / 74 | 0 / 74 |
| 3 | 0 / 74 | 0 / 74 |
| 5 | 0 / 74 | 0 / 74 |
| 10 | 2 / 74 | **0 / 74** |
| 30 | 4 / 74 | 1 / 74 |
| 90 | 12 / 74 | 10 / 74 |

Disagreement is a long-horizon phenomenon. Protocol B produced **zero primary
disagreements at every horizon through 10 control steps** (0 of 296) and 11 of 148
at 30 and 90 steps combined. At 90 steps its rate is 13.5%. Every residual error is
false-unrecoverable; the primary comparison recorded no false-recoverable errors at
any horizon under either protocol.

By disturbance family, primary comparison:

| Disturbance | Protocol A | Protocol B |
|---|---:|---:|
| `object_teleport` | 2 / 210 | **0 / 210** |
| `gripper_open_interruption` | 16 / 234 | 11 / 234 |

Widening to all three predicates under exact-action continuation preserves the
pattern: teleport 4/630 to **0/630**, gripper-open 53/702 to 39/702.

By branch phase, primary comparison, the residual is spread rather than isolated:
Protocol B recorded 0/60 in `free_space_pre_manipulation`, 1/60 in `approach`,
2/60 in `first_contact`, 1/60 in `stable_grasp`, 1/60 in `initial_lift`, 2/60 in
`sustained_lift`, 3/60 in `disturbance_onset`, and 1/24 in
`post_disturbance_recovery`. The largest Protocol A to B improvements were in
`post_disturbance_recovery` (5/24 to 1/24) and `approach` (4/60 to 1/60).

Predicate sensitivity, exact-action, 444 records per cell: `sustained_lift` 18 to
11, `stable_grasp` 16 to 11, `final_height` 23 to 17. Protocol B improves under
every predicate, and no predicate choice would have reached the 50 percent bar.

Teleport branches, where the object is displaced and the interaction is largely
ballistic, restore cleanly under Protocol B. Gripper-open branches, where the
disturbance acts through sustained finger-object contact, do not. This is the
expected signature of unrestored contact-solver state, though the study did not
isolate that mechanism and cannot attribute the residual to it.

## 8. Why this result blocks PoNR analysis

`PoNR = the first step after which recovery never again succeeds` is computed from
`recovery_success[t]`, and every `recovery_success[t]` comes from a restored
branch. The measurement above says those branches carry a nonzero, horizon-
dependent decision error, and that the error is asymmetric: under exact-action
continuation it is overwhelmingly **false-unrecoverable**, meaning the restored
branch reports "could not recover" where the uninterrupted episode succeeded.

A false-unrecoverable label at step *t* pulls the measured PoNR *earlier* than the
truth. The bias runs in the direction that makes IPFD's headline number look more
impressive, and its magnitude scales with the probe budget, because recovery probes
use long horizons and long horizons are exactly where disagreement concentrates.
The 90-step Protocol B rate of 13.5% is the relevant figure for a probe budget of
that length, not the 2.48% rate pooled over all six horizons.

Validity gating was the intended fix: learn on branch-time features which branches
to reject, then show that gating corrects a real downstream conclusion. That stage
required a positive control that worked. It did not, so the gate was **not eligible
to run**, and no PoNR, controller-ranking, or checkpoint-selection decision was
corrected. Both stages are recorded as `NOT_RUN_STOPPING_RULE` in
[`validity_gate_results.json`](results/branch_validity/corrected_five_seed/validity_gate_results.json)
and
[`downstream_decision_results.json`](results/branch_validity/corrected_five_seed/downstream_decision_results.json).
Nothing in this repository demonstrates a corrected downstream robotics decision.

## 9. Strongest reviewer criticism

**The finding may be a documented API limitation rather than a research result.**
Isaac Lab documents `InteractiveScene.get_state()` as entity root pose, root
velocity, joint position, and joint velocity, and does not claim it is a complete
task, manager, solver, or random-state snapshot. MuJoCo states outright that
reproducible continuation requires all integration-state components including
warm-start state, and that contact systems amplify small differences. A reviewer is
entitled to say: you used a partial snapshot API as if it were complete, in a
contact-rich task, and observed divergence; this is expected behavior, and the
finding is a usage note.

That criticism largely lands. The intended defense was decision-relativity: not
"trajectories diverge" (uninteresting) but "*decisions reverse*, here is where, and
here is a gate that catches it before a downstream conclusion is drawn." The first
half of that defense survives, and it is not free, because 120 of 120 immediate
observation matches would have satisfied most practitioners' validity check. The
second half does not survive, because the gate was never eligible to run. Without a
gate and a corrected decision, the contribution reduces to a well-instrumented
reproducer of a known-in-principle limitation.

Secondary criticisms that also stand: five independent seed groups is a thin basis
for the effect-size claim; a single task, robot, checkpoint, and machine cannot
support generalization; the checkpoint used for the study is the same published
Lift-Cube checkpoint that measured 0.00% competence in a separate evaluation on
this runtime (the study's own nominal-lift control passed per seed group, but this
weakens any argument about typical policy behavior); and residual disagreement
concentrating in one disturbance family invites the alternative explanation that
the whole effect is one contact-mode artifact.

## 10. Engineering relevance

The narrow claim is useful to anyone building simulator-branching infrastructure:

1. **Immediate observation equality after restore is not a validity check.** In the
   preserved cohort it passed 120 out of 120 times on branches that included 13
   decision reversals. If your replay, regression, or counterfactual tooling
   validates a restore by comparing observations at the restore step, it is not
   testing what you think it is.
2. **Restore fidelity is horizon-dependent, and short horizons look fine.** Zero
   primary disagreement through 10 control steps, 13.5% at 90. A tool validated at
   short horizons can be silently wrong at the horizons an actual recovery probe
   uses.
3. **Restoring more exposed runtime state genuinely helps.** Action-term buffers,
   articulation targets, manager buffers, and scheduler state moved the primary
   rate from 4.05% to 2.48% and cleared the teleport family entirely. If you are
   building on `InteractiveScene.get_state()` alone, you are leaving accuracy on
   the table.
4. **The remainder concentrates in sustained-contact disturbances.** Consistent
   with unrestored PhysX solver and contact-cache state, which the supported
   per-environment interfaces do not expose.

An upstream-facing writeup of points 1 through 4, framed as an engineering
observation rather than a defect report, is drafted in
[`ISAACLAB_ENGINEERING_NOTE.md`](ISAACLAB_ENGINEERING_NOTE.md). It has not been
filed, submitted, or shown to anyone, and requires human review before it is.

## 11. Limitations

- One task (`Isaac-Lift-Cube-Franka-v0`), one robot (Franka Panda), one checkpoint,
  one simulator (Isaac Lab 4.5.22 / Isaac Sim 6.0.0.0), one machine, one GPU class.
- Five independent seed groups. Everything within a seed group is correlated.
- Two disturbance families, two continuation modes, three decision predicates, six
  horizons. Not a sampling of the space, a fixed grid.
- Protocol B does not restore unexposed PhysX solver, contact-cache, broadphase, or
  friction-solver state. The residual disagreement is **not attributed** to any
  specific missing state; the study measures an effect, not a mechanism.
- Entry-USD asset hashes do not cover transitive dependencies.
- No held-out validity gate was trained or evaluated. No downstream decision was
  corrected. No causal attribution was performed. No formal certification of any
  kind is offered or implied.
- No physical-robot validation. Nothing here says anything about real hardware.
- The negative result does not show that simulators are unreliable, that Isaac Lab
  is defective, or that a state is physically irrecoverable. It shows that two
  specific restoration protocols, in one setting, did not preserve one specific
  class of downstream decision.

## 12. Stopping decision

The preregistered stopping rule fired on its own terms:

```
stopping_rule.decision                            = STOP_BRANCH_VALIDITY_DIRECTION
stopping_rule.gate_eligible                       = false
stopping_rule.positive_control_meaningfully_improved = false
```

The rule was written before the outcome was known, it was applied without
amendment, and no post-hoc rescue was attempted: no new predicate, no dropped
stratum, no re-weighted horizon, no additional seeds run until the number moved.
Research development on IPFD stops here. The repository is retained as a rigorous
negative result with its evidence trail intact.

## 13. Reproduction commands

CPU only, no simulator required. Re-derives the strata tables, the figure, the
not-run gate records, and refreshed artifact hashes from the preserved study
outputs:

```bash
python3 scripts/analyze_snapshot_protocol_study.py \
  --study-dir results/branch_validity/corrected_five_seed
```

This requires `per_branch_records.jsonl`, which is retained outside git (see §14).
The script fails closed: it exits nonzero if the record count disagrees with
`protocol_comparison.json` or if the stopping decision is not the observed one.

Analysis-layer test suite, lint, and types:

```bash
env -u PYTHONPATH python3 -m pytest -q
python3 -m ruff check src tests scripts
python3 -m mypy src/ipfd
```

Regenerating the five-seed study itself needs a CUDA GPU and an Isaac Lab runtime.
The exact invocation, environment variables, and preconditions are in
[`CORRECTED_EXPERIMENT_PROTOCOL.md`](CORRECTED_EXPERIMENT_PROTOCOL.md). It was not
rerun for this archival pass; every recorded artifact hash verified against its
manifest instead.

## 14. Artifact locations

**In the repository** (`results/branch_validity/`):

| Artifact | Contents |
|---|---|
| `clean_three_seed_report.json` | Preserved three-seed raw paired-branch report |
| `clean_three_seed_provenance.json` | Generator, checkpoint, command, and artifact hashes for that cohort |
| `summary.json` | Three-seed cohort analysis with intervals and the fail-closed gate |
| `decision_fidelity.png` | Three-seed result visualization |
| `corrected_five_seed/protocol_comparison.json` | Corrected-study comparison, controls, stopping rule |
| `corrected_five_seed/protocol_strata.json` | Primary comparison by phase, horizon, disturbance, seed, predicate |
| `corrected_five_seed/study_provenance.json` | Config, checkpoint, generator, asset, and git provenance |
| `corrected_five_seed/artifact_manifest.json` | SHA-256 and byte counts for every study artifact |
| `corrected_five_seed/raw_worker_manifest.json` | SHA-256 and byte counts for the 20 external raw worker files |
| `corrected_five_seed/validity_gate_results.json` | `NOT_RUN_STOPPING_RULE` |
| `corrected_five_seed/downstream_decision_results.json` | `NOT_RUN_STOPPING_RULE` |
| `corrected_five_seed/decisive_study.png` | Final visualization |

The three-seed and five-seed cohorts are separate and are never pooled.

**Outside the repository**, retained under
`~/.local/share/ipfd/branch_validity/`, excluded from git as large raw traces, with
every size and SHA-256 recorded in the manifests above:

| File | Size | SHA-256 (prefix) | Manifest |
|---|---:|---|---|
| `corrected_five_seed/workers/*.json` and `*.log` (20 files) | ~13 MB | see manifest | `raw_worker_manifest.json` |
| `corrected_five_seed/per_branch_records.jsonl` (5,328 records) | 10,166,696 B | `1c55862c…` | `artifact_manifest.json` |
| `clean_three_seed/traces.npz` | 5,329,453 B | `dcf30076…` | `clean_three_seed_provenance.json` |

All three groups were verified byte-for-byte against their recorded hashes during
this archival pass.

## 15. Supporting documents

| Document | Role |
|---|---|
| [`RESEARCH_AUDIT.md`](RESEARCH_AUDIT.md) | Claim-to-evidence matrix and ranked weaknesses |
| [`NOVELTY_REVIEW.md`](NOVELTY_REVIEW.md) | Prior art, closest work, and the differentiation test that was not met |
| [`HYPOTHESES.md`](HYPOTHESES.md) | Directions considered, the one selected, and the final stopping decision |
| [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) | Original three-seed protocol |
| [`CORRECTED_EXPERIMENT_PROTOCOL.md`](CORRECTED_EXPERIMENT_PROTOCOL.md) | Preregistered five-seed protocol |
| [`SNAPSHOT_PROTOCOLS.md`](SNAPSHOT_PROTOCOLS.md) | Exact state captured and omitted by Protocol A and Protocol B |
| [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md) | Every claim with its command, artifact, status, and limitation |
| [`CLAIM_AUDIT.md`](CLAIM_AUDIT.md) | Material claims changed during archival and why |
| [`ISAACLAB_ENGINEERING_NOTE.md`](ISAACLAB_ENGINEERING_NOTE.md) | Unfiled draft engineering observation for later human review |
