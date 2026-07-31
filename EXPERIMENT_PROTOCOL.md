# Counterfactual Branch Validity Protocol

## Scientific objects

Let the complete environment-policy state at time \(t\) be

\[
s_t = (x_t, h_t, m_t, r_t),
\]

where \(x_t\) is physical simulator state, \(h_t\) is policy state, \(m_t\) is
task and manager state, and \(r_t\) contains relevant random or exogenous state.

- **Trajectory:** \(\tau = (s_0, a_0, s_1, \ldots, s_T)\).
- **Failure event:** a declared predicate \(F(\tau_{0:t})\), including its
  persistence rule and first evaluation time.
- **Intervention:** a declared change \(I=(u,\lambda,d,c)\), where \(u\) is the
  intervention type, \(\lambda\) its strength, \(d\) its duration or delay, and
  \(c\) its constraints.
- **Intervention time:** the policy step at which \(I\) first changes the
  uninterrupted execution.
- **Intervention space:** the finite or measurable set of allowed interventions,
  including controller family, action limits, observation access, and latency.
- **Recovery policy:** the continuation controller \(\kappa\), including hidden
  state and reset semantics.
- **Recovery criterion:** a declared predicate \(G(\tau_{t:t+H})\).
- **Recovery horizon:** the maximum continuation length \(H\).
- **Recoverability:** for a declared intervention distribution and restoration
  protocol \(q\),

\[
R_q(t,I,\kappa,H)
=
\Pr\left(G(\tilde{\tau}_{q(s_t),I,\kappa,H})=1\right).
\]

- **Intervention frontier:** a level set of \(R_q\) over time, intervention, and
  controller capability. It is meaningful only where \(q\) passes branch
  validity.
- **Controller-conditional PoNR:** after fixing \(I,\kappa,H,G,q\), constraints,
  and threshold \(\rho\), one possible convention is the first point in a final
  suffix for which \(R_q < \rho\).

PoNR need not be unique or monotone before a convention is imposed. Later states
can be more recoverable because contact mode, phase, or allowed intervention
changes. A different recovery controller or intervention type can change the
boundary. Under stochasticity, PoNR is an estimated probability level set, not
a deterministic index.

If restore is imperfect, it induces a restore kernel
\(Q_q(\tilde{s}\mid s)\). Reported recoverability then includes restore error.
"Not recovered in the sampled branch" means only that one declared controller,
horizon, predicate, and restore draw failed. It does not mean physically
unrecoverable.

## Branch-validity estimand

For snapshot protocol \(q\), continuation controller \(\kappa\), horizon \(H\),
and downstream decision functional \(D\), define

\[
E(q,s,\kappa,H,D)
=
\mathbf{1}\left[
D(\tau_{s,\kappa,H})
\ne
D(\tilde{\tau}_{q(s),\kappa,H})
\right].
\]

For a state stratum \(z\), define the disagreement probability

\[
e_q(z,\kappa,H,D)
=
\Pr(E=1 \mid z,\kappa,H,D).
\]

At allowed error \(\epsilon\) and confidence \(1-\beta\), a validity envelope is

\[
\mathcal{V}_{\epsilon,\beta}(q)
=
\left\{(z,\kappa,H,D):
\operatorname{UCB}_{1-\beta}(e_q)\le\epsilon\right\}.
\]

Recorded exact-action continuation isolates branch-state fidelity more cleanly
than closed-loop policy continuation. Trajectory equivalence is stronger than
decision equivalence. Immediate exposed-state equality is weaker than both.

## Implemented experiment

### Task and policy

- Isaac Lab task: `Isaac-Lift-Cube-Franka-v0`.
- Learned policy checkpoint SHA-256:
  `fb658f989bf5ebf35b20347813275979a6778ade8d3823d12eb3190612f9e36d`.
- Policy competence control: 64 of 64 environments sustained the lift criterion
  in one 200-step evaluation with this content-distinct checkpoint.
- Physics step: 0.01 seconds.
- Environment step: 0.02 seconds.
- Parallel environments: four.

### Snapshot protocol

The expanded protocol restores:

- exposed scene entity state;
- command-manager command and timing state;
- current and previous action-manager state;
- episode length.

It does not claim to restore every solver, contact, task, sensor, RNG, or policy
state.

### Conditions

- Continuations: uninterrupted reference, recorded exact actions, closed-loop
  learned policy.
- Disturbances: object teleport and an eight-step gripper-open interruption.
- Checkpoint strata: pre-manipulation, contact onset, mid-contact, post-contact,
  lifted disturbance points, and offsets 1, 3, 5, and 10 steps from nominal
  lift.
- Clean run: seeds 0, 1, and 2, one run per disturbance, 120 paired
  comparisons.
- Branch budget: 90 steps.
- Decision: terminal object height more than 0.06 m above initial rest height
  without termination.

The values 1, 3, 5, and 10 are offsets to the nominal lift event, not varied
continuation horizons. The current artifact must not be presented as a horizon
sweep.

### Baselines

1. no restore, uninterrupted continuation;
2. recorded identical action continuation after restore;
3. closed-loop policy continuation after restore;
4. immediate exposed observation equality;
5. free-space control in `isaaclab_reset_to_contact_mre.py` with
   `--grasp_steps 0`;
6. evolved-state reset reproducer with `--grasp_steps 40`.

### Metrics

- terminal-decision agreement;
- false-recoverable and false-unrecoverable counts;
- first material trajectory-divergence step;
- maximum observation, joint, and object-pose divergence;
- immediate observation round-trip error;
- comparison-level Wilson interval, labeled descriptive;
- fail-closed upper-bound gate per phase and continuation.

Comparisons within a seed are correlated. Current Wilson intervals do not
constitute a deployment certificate or independent-seed confidence interval.

## Reproduction

Set task-specific paths:

```bash
export IPFD_ISAACLAB_ROOT=/path/to/IsaacLab
export IPFD_CHECKPOINT=/path/to/competent/checkpoint.pt
export IPFD_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5
```

Run the clean three-seed experiment:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  "$IPFD_ISAACLAB_ROOT/isaaclab.sh" -p scripts/validate_recovery_oracle.py \
  --headless \
  --checkpoint "$IPFD_CHECKPOINT" \
  --seeds 0,1,2 \
  --repeats 1 \
  --num-envs 4 \
  --probe-budget 90 \
  --horizons 1,3,5,10 \
  --disturbances teleport,gripper_interrupt \
  --asset-root "$IPFD_ASSET_ROOT" \
  --output-dir /tmp/ipfd-branch-validity
```

Analyze one or more raw reports:

```bash
python3 scripts/analyze_branch_validity.py \
  --input clean=/tmp/ipfd-branch-validity/oracle_equivalence.json \
  --output results/branch_validity/summary.json \
  --figure results/branch_validity/decision_fidelity.png
```

## Current results and interpretation

The clean three-seed run produced 13 disagreements in 120 comparisons:

- recorded exact actions: 10 of 60;
- closed-loop policy: 3 of 60;
- exact-action material trajectory divergence: 56 of 60;
- immediate observation equality: 120 of 120.

No phase and continuation cell passed the five-percent, 95-percent fail-closed
gate. This does not prove that every cell is invalid. It means the current
sample is too small to certify cells and contains enough disagreements to reject
universal decision fidelity.

## Confounds and corrections required

1. Nominal checkpoints are repeated across disturbance loops and are not
   independent states.
2. The closed-loop gripper-interruption branch does not reapply the forced-open
   schedule, while the uninterrupted branch does. The exact-action comparison
   remains the cleaner evidence.
3. The terminal height criterion is not a sustained physical grasp criterion.
4. There is no more-complete positive-control snapshot protocol yet.
5. One task and one learned checkpoint do not establish transfer.

## Success and stopping criteria

Promote this direction only if a corrected five-seed study:

- reproduces decision reversals with matched disturbance schedules;
- varies actual horizons 1, 3, 5, 10, 30, and 90;
- shows predicted improvement from a more complete positive-control protocol;
- predicts held-out invalid branches better than a contact flag;
- corrects a downstream PoNR, controller-ranking, or policy-selection result.

Reduce the work to an engineering reproducer if disagreements disappear after
confounds are corrected, are explained entirely by documented partial-state
usage, or do not change any downstream decision.
