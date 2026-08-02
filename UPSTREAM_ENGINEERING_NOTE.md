# Upstream engineering contribution draft for Isaac Lab

**Status:** prepared for human review. Not submitted. No issue was filed, no pull
request was opened, and no maintainer was contacted.

## Proposed contribution

The strongest contribution supported by the preserved evidence is a narrow
documentation clarification beside `InteractiveScene.get_state()` and
`InteractiveScene.reset_to()`, accompanied by a finite-horizon regression-test
sketch. This is not a defect report. The documented APIs operate on scene-entity
state, and the proposed text clarifies the boundary between that contract and a
complete simulator/task snapshot.

The archived observation is bounded to one tested Isaac Lab task and version. It
shows why the distinction matters for replay tooling, but it does not establish
that expected contact divergence is an Isaac Lab or PhysX defect.

## Suggested documentation text

The following note is upstream-ready in scope and tone:

> `get_state()` returns the state exposed by scene entities, and `reset_to()` writes
> that scene-entity state. These methods do not by themselves capture every source
> of continuation state owned by an environment, its managers, controllers,
> sensors, random generators, or the physics solver. Applications that require
> replay-equivalent or decision-equivalent continuation should restore all relevant
> application state and validate the result after stepping identical actions over
> the intended horizon. Equality of scene state or observations immediately after
> `reset_to()` establishes only equality of those measured values.

An optional second sentence can point users to environment-specific reset methods
when those methods restore more task state than `InteractiveScene.reset_to()`.
The wording should remain descriptive and should not promise a complete snapshot
through either API.

Official API context:

- [`InteractiveScene.get_state()` and `reset_to()`](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html)
- [Manager-based environment `reset_to()`](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html)

## Why this clarification is supported

The archived standalone reproducer,
`scripts/isaaclab_reset_to_contact_mre.py`, measures two separate properties:

1. the scene state returned by `get_state()` round-trips through `reset_to()`;
2. uninterrupted and restored environments continue under identical action
   tensors for a declared number of control steps.

In the tested configuration, the free-space control with `--grasp_steps 0`
reported an exact scene-state round trip and zero post-step observation gaps. The
evolved `--grasp_steps 40` case still reported exact exposed-state restoration but
developed a final exact-action observation gap of 0.1046 over 12 comparison steps.
The reproducer does not isolate contact solver state as the cause. Evolved task,
manager, controller, sensor, or solver state remains a possible contributor.

The corrected five-seed study provides a longer-horizon reason to document the
boundary. In the preregistered exact-action `sustained_lift` comparison, the
expanded runtime protocol recorded no primary decision disagreements through 10
control steps, then 1 of 74 at 30 steps and 10 of 74 at 90 steps. Those counts are
specific to the archived task, checkpoint, protocols, branches, and tolerances.
They show that a restore-step assertion and a short smoke test can answer a
different question than the eventual counterfactual decision.

## Finite-horizon regression-test sketch

The proposed upstream test should verify only behavior Isaac Lab intends to
support. It should not encode the archived contact-rich divergence as a failing
defect test.

```python
@pytest.mark.parametrize("horizon", [1, 5, 10, 30])
def test_scene_restore_identical_action_fidelity_free_space(horizon):
    env = make_small_deterministic_scene(num_envs=2, contacts_disabled=True)
    reference_id, restored_id = 0, 1

    # Advance both instances through the same deterministic setup.
    actions = fixed_action_sequence(horizon)
    snapshot = clone_tree(env.scene.get_state(is_relative=True), reference_id)

    env.scene.reset_to(snapshot, env_ids=[restored_id], is_relative=True)
    restored = clone_tree(env.scene.get_state(is_relative=True), restored_id)
    assert_tree_equal(snapshot, restored)  # scene-state contract

    reference_trace = []
    restored_trace = []
    for action in actions:
        step_both_with_identical_actions(env, reference_id, restored_id, action)
        reference_trace.append(read_declared_scene_state(env, reference_id))
        restored_trace.append(read_declared_scene_state(env, restored_id))

    assert_trajectory_close(reference_trace, restored_trace, declared_tolerances())
```

The actual test should use an upstream-owned minimal scene with no policy
checkpoint, downloaded asset, network dependency, or random event. The reference
and restored instances should run in the same test process only if Isaac Lab's
environment isolation contract makes that a valid comparison. Otherwise, the test
should use separate deterministic runs.

The test records or asserts:

- the exact state fields included by `get_state()`;
- the restore-step comparison;
- identical action values at every step;
- first numerical divergence and maximum error over the full horizon;
- simulator and test configuration when a failure occurs.

A contact-rich diagnostic may be kept as an example or non-gating reproducer. It
should report finite-horizon differences without asserting that scene-state APIs
promise complete contact-state restoration.

## Suggested test acceptance boundary

The upstream regression is suitable only if maintainers confirm that deterministic
free-space continuation is intended behavior for the chosen fixture. Its tolerance
must be derived from that fixture and execution backend, not copied from IPFD as a
universal threshold.

The test must fail with useful context when:

- immediate scene-state restoration changes;
- one-step fidelity changes;
- divergence begins earlier than the declared baseline;
- action identity is violated;
- required provenance is missing.

It must not claim that passing the fixture proves contact-rich replay, downstream
decision fidelity, or complete simulator snapshots.

## Maintainer-facing reproduction option

If maintainers prefer a diagnostic example over a gated regression test, the
standalone MRE can be reduced to:

```bash
OMNI_KIT_ACCEPT_EULA=YES <isaac-lab-python> \
  scripts/isaaclab_reset_to_contact_mre.py --headless --grasp_steps 0

OMNI_KIT_ACCEPT_EULA=YES <isaac-lab-python> \
  scripts/isaaclab_reset_to_contact_mre.py --headless --grasp_steps 40
```

Before any upstream use, the script should be rebased onto a current upstream-owned
task or minimal scene, remove project-specific policy loading if possible, record
exact dependency versions, and confirm that required assets are available through
supported public paths.

## Evidence and limitations to include

- Tested evidence is one contact-rich manipulation task, one policy checkpoint,
  five independent seed groups, two snapshot protocols, and one archived runtime.
- Immediate observation equality and exact scene-state round trip do not measure
  hidden task or solver state.
- Restoring additional exposed runtime state improved the archived result but did
  not meet its preregistered threshold.
- Residual disagreements concentrated in a sustained-contact disturbance family,
  but no experiment identified the missing causal state.
- The contribution does not assert universal nondeterminism, a PhysX defect,
  physical irrecoverability, or a valid recovery boundary.
- The original study and failed calibration successor remain stopped. This draft
  uses them only to motivate clearer API-contract language and a better regression
  shape.

## Submission gate

Human approval is required before filing an issue, opening a pull request, posting
the reproducer, or contacting maintainers. Before approval, validate the proposed
text against the exact target Isaac Lab version, run the reduced test on that
version, scrub machine-specific paths and hardware identifiers, and inspect the
rendered contribution as an external reader would.

No upstream action has been taken.
