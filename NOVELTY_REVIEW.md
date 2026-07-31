# IPFD Novelty Review

## Verdict

A universal Point of No Return is not a novel or well-defined scientific
object. Recoverability has long been conditioned on a backup controller,
constraints, dynamics, horizon, and uncertainty. A recovery-controller-specific
heatmap is also not enough by itself.

The narrow open gap is decision-relative validation of restored counterfactual
branches in high-dimensional contact simulation:

> Before simulator branches are used to label recoverability, determine where a
> declared snapshot and continuation protocol preserves the downstream decision,
> and fail closed where it does not.

This is currently a hypothesis and protocol direction, not a completed original
contribution.

## Closest work and capability comparison

| Work | Existing capability | Threat to IPFD | Narrow remaining gap |
|---|---|---|---|
| [Statistical Model Predictive Shielding, RSS 2021](https://trustml.github.io/docs/rss21.pdf) | Defines recoverability relative to a backup policy and finite horizon, samples stochastic rollouts, and derives high-probability safety results | Controller-conditional probabilistic recoverability is established with stronger statistics | Post-hoc audit of whether a high-dimensional simulator branch itself preserves the recovery decision |
| [Recovery RL, 2020/2021](https://arxiv.org/abs/2010.15920) | Learns recovery zones and a separate recovery policy across multiple domains and hardware | Recovery zones and recovery-policy conditioning are not new | Comparing the validity of counterfactual labels used to construct such zones |
| [Feasible Reachable Policy Iteration, ICML 2024](https://proceedings.mlr.press/v235/qin24d.html) | Defines whether some policy can reach a goal safely within finite time | A universal scalar PoNR is conceptually under-specified | Black-box empirical diagnosis when dynamics and optimal recovery are unavailable |
| [HJ safety and liveness filtering](https://arxiv.org/abs/2312.15347) | Characterizes safe or live control sets and projects nominal controls into them | Intervention frontiers and minimum corrections have formal precedents | Empirical contact-rich regimes where reachability is computationally intractable |
| [PREFAIL, IROS 2026](https://zshanggu.github.io/zeyu-prefail/) | Identifies a latest intervention time and applies speed scaling in simulation and reality | Broad novelty claims for actionable latest intervention time are untenable | Multiple explicit recovery capabilities and validity-gated simulation branches |
| [TAIL-Safe, RSS 2026](https://roboticsconference.org/program/papers/207/) | Learns a safe set for imitation policies and computes recovery actions with digital-twin and physical Franka evidence | Strong direct competition for learned-policy recovery and safe sets | Auditing branch decision fidelity rather than proposing another recovery controller |
| [Sentinel, CoRL 2024](https://github.com/agiachris/sentinel) | Evaluates consistency and task-progress monitors on simulation and real robot-policy datasets | Generic failure detection is much better developed than IPFD's current detectors | Whether an alarm occurs while a declared intervention remains effective |
| [Model-Based Runtime Monitoring, ICRA 2024](https://rpl.cs.utexas.edu/publications/2024/05/13/liu-icra24-siriusrm/) | Predicts future failure with latent dynamics and improves simulation and hardware success | Prediction plus downstream benefit is established | Simulator-grounded audit of monitor actionability under explicit branch validity |
| [VERIFAI, CAV 2019](https://arxiv.org/abs/1902.04245) | Simulator-guided falsification, parameter synthesis, counterexamples, and analysis | Generic failure search and counterexample generation are old | Decision fidelity of restored branches along a recorded learned-policy trajectory |
| [Bayesian failure sampling, CoRL 2023](https://proceedings.mlr.press/v229/dawson23a.html) | Efficiently finds diverse failures and repairs designs | Active failure search is established | Active sampling of a calibrated branch-validity envelope |
| [Sequential counterfactual explanations, NeurIPS 2021](https://arxiv.org/abs/2107.02776) | Finds small action-sequence changes that improve an outcome | Minimum altered actions are not new | Physically grounded interventions whose simulator branches first pass a validity test |
| [Why Did I Fail?, RA-L 2022](https://arxiv.org/abs/2204.04483) | Learns causal Bayesian networks from randomized simulation and generates contrastive explanations | Generic causal failure explanation is established | Planted cross-stack fault attribution with matched interventions and valid branches |
| [NVIDIA RoboLab](https://github.com/NVlabs/RoboLab) | Large Isaac Lab task coverage, automated predicates, replay, and evaluation analysis | IPFD cannot compete on evaluation breadth or infrastructure scale | Per-failure validity-gated counterfactual recovery information |

## Simulator-state prior art

Isaac Lab documents `InteractiveScene.get_state()` as entity root pose, root
velocity, joint position, and joint velocity state. It does not document the
result as a complete task, policy, manager, contact-solver, or random state
snapshot. See the
[Isaac Lab scene-state API](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.scene.html).

MuJoCo explicitly documents that reproducible continuation requires all
integration-state components, including warm-start state for bitwise equality,
and that contact systems amplify small differences. See the
[MuJoCo reproducibility notes](https://mujoco.readthedocs.io/en/3.8.0/computation.html)
and
[simulation-state documentation](https://mujoco.readthedocs.io/en/latest/programming/simulation.html).

Therefore, "partial snapshots diverge in contact" is not novel. The possible
contribution must be decision-relative:

- distinguish harmless trajectory divergence from decision reversal;
- quantify false-recoverable and false-unrecoverable labels;
- compare explicit snapshot protocols;
- provide finite-sample, fail-closed validity rules;
- show a downstream PoNR, controller-ranking, or policy-evaluation decision
  that the validity gate corrects.

## Ideas rejected

1. **Universal PoNR as a trajectory property.** Recoverability is capability and
   model dependent.
2. **PoNR depends on recovery policy.** Correct but already inherent in
   shielding and recovery-zone definitions.
3. **Minimal sufficient intervention alone.** Closely adjacent to reachability
   filters and sequential counterfactual explanations.
4. **Active search without a new branch-efficiency result.** Falsification and
   Bayesian failure sampling already address efficient search.
5. **Causal onset from deterministic replay.** Invalid without an explicit
   intervention, matched exogenous variables, and causal estimand.
6. **Generic failure detection.** Existing monitors have broader simulation and
   hardware evidence.
7. **Boundary-state retraining without comparative gain.** Reverse curricula,
   intervention learning, and counterexample augmentation already motivate it.

## Differentiation test

The branch-validity direction is genuinely differentiated only if it satisfies
all of the following:

1. a clean paired experiment reproduces decision reversals after fault schedule,
   task state, and policy state are matched;
2. disagreement changes predictably with phase, actual continuation horizon,
   continuation controller, and snapshot protocol;
3. a more complete positive-control protocol improves the validity envelope;
4. the envelope predicts held-out invalid branches better than a contact flag;
5. the gate prevents a wrong PoNR, recovery-controller ranking, or policy
   selection.

Without these results, the work is an engineering reproducer, not a research
contribution.
