# IPFD Research Hypotheses

## Final stopping decision

The selected Counterfactual Branch Validity Envelope direction is stopped.
In the corrected five-seed comparison, `expanded_runtime_state` reduced the
primary exact-action sustained-lift disagreement rate from 18/444 (4.05
percent) to 11/444 (2.48 percent). This is a 38.9 percent relative reduction,
below the preregistered 50 percent threshold, despite improvement in four of
five seed groups. The held-out validity gate and downstream correction were not
run because continuing would violate the stopping rule. The project therefore
receives the final verdict `Archive as an honest negative`.

## Ranked decision

Scores are on a five-point scale. Expected value weighs importance and likely
reviewer survival against implementation and falsification risk.

| Rank | Direction | Novelty | Importance | Tractability | NVIDIA relevance | Reviewer survival | Expected value |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Counterfactual Branch Validity Envelope | 4 if completed | 5 | 4 | 5 | 3 | Highest |
| 2 | Validity-gated recovery-conditional intervention frontier | 3 | 4 | 2 | 5 | 2 | Medium |
| 3 | Randomized cross-stack failure attribution | 4 | 4 | 2 | 4 | 3 | Medium-low |

## 1. Selected: Counterfactual Branch Validity Envelope

### Research question

Under which task phases, continuation horizons, controllers, disturbances, and
snapshot protocols does a restored simulator branch preserve the downstream
decision of the corresponding uninterrupted branch?

### Falsifiable hypothesis

For contact-rich learned-policy execution, the probability that a restored
branch reverses a recovery or task-success decision is structured by phase,
continuation mode, horizon, and snapshot completeness. A fail-closed empirical
validity envelope will identify invalid branches on held-out data better than a
simple contact versus no-contact heuristic.

### One-sentence contribution

A decision-relative audit for simulator counterfactuals that separates harmless
trajectory divergence from branch errors that reverse recovery conclusions and
blocks downstream analysis outside a measured validity envelope.

### Closest prior work

Simulator state APIs and reproducibility documentation describe which state must
be restored. Safety and shielding work defines controller-relative
recoverability. The open intersection is a practical, finite-sample decision
audit for black-box policy branches in high-dimensional contact tasks.

### Minimum falsifier

Correct the matched-disturbance and task-state confounds, add a more complete
positive-control snapshot protocol, and evaluate held-out states. Stop if
decision disagreements disappear, if a contact flag explains all of them, or if
the positive control does not improve validity.

### Effort, compute, and risk

- Implementation: approximately one focused experiment, not a platform rewrite.
- Compute: one learned manipulation task, paired branches, and tens to hundreds
  of GPU-parallel continuations.
- Major risk: the result reduces to documented misuse of a partial scene API.
- NVIDIA relevance: direct relevance to Isaac Lab evaluation, replay, regression
  testing, and large-scale counterfactual branching.
- Publication potential: moderate only with a positive control, held-out
  prediction, and a corrected downstream decision.
- Real-robot path: none required for branch certification. Later sim-to-real
  analysis would inherit, not eliminate, model error.
- Demo: two visually identical restored snapshots evolve under identical
  actions and reverse the same recovery decision, while the validity gate blocks
  the false conclusion.

## 2. Deferred: Validity-gated recovery-conditional intervention frontier

### Research question

Which recovery controllers and intervention strengths remain effective at each
time along a failed learned-policy trajectory, after branch validity is
established?

### Falsifiable hypothesis

The time by intervention surface changes a policy-ranking, controller-selection,
or retraining-state decision that a scalar controller-specific threshold gets
wrong.

### Why deferred

The capability is valuable, but recovery-policy dependence is already known.
The experiment is scientifically blocked while restored branches can reverse
terminal labels.

### Minimum falsifier

Use two materially different recovery controllers and two severities. Stop if
the surface is a shifted monotone threshold and changes no downstream decision.

### Effort, compute, and risk

- Implementation: medium.
- Compute: substantial intervention sweeps over multiple seeds.
- Major risk: a more expensive visualization of a known reachability concept.
- NVIDIA relevance: high for GPU-parallel policy evaluation and controller
  design.
- Publication potential: moderate if it changes decisions and includes
  uncertainty.
- Real-robot path: reconstruct conservative candidate branches in simulation,
  then validate only low-risk interventions.
- Demo: a scalar PoNR says "too late" while a stronger recovery controller still
  succeeds, with all branch cells passing the validity gate.

## 3. Deferred: Randomized cross-stack failure attribution

### Research question

Can matched counterfactual interventions distinguish planted faults in actions,
observations, latency, dynamics, actuator constraints, and policy memory?

### Falsifiable hypothesis

A randomized interventional design identifies planted fault sources and
interactions more accurately than one-factor ablations.

### Why deferred

Potential novelty is stronger, but implementation and causal-design risk are
high. It also depends on valid branch state and carefully matched disturbance
schedules.

### Minimum falsifier

Plant at least five fault families with hidden labels. Stop if attribution does
not beat simple ablations or if interaction effects make the source
unidentifiable.

### Effort, compute, and risk

- Implementation: high.
- Compute: high due factorial interventions.
- Major risk: a synthetic planted-fault benchmark with little external value.
- NVIDIA relevance: high for simulation-based failure triage.
- Publication potential: moderate to high with real failure transfer.
- Real-robot path: infer candidate sources in simulation, then run conservative
  diagnostic checks on recorded real trajectories.
- Demo: the same visible drop is correctly attributed to latency, actuator
  saturation, or policy memory under different planted cases.

## Historical selection decision

The first direction is selected because it is logically prior to the other two,
has a real falsification result now, and has the highest expected value under the
available compute. The current slice is an audit and falsification result, not a
finished validity certificate.

This historical selection was superseded by the corrected five-seed stopping
decision above.
