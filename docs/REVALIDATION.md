# Learned-policy evidence revalidation

The committed learned-policy fixtures are historical analysis artifacts. They
are retained to test archive loading and deterministic report generation, but
they are not current proof that the Isaac Lab recovery oracle is physically
correct.

The old teleport fixture used a height-only recovery predicate. An airborne,
out-of-reach object can satisfy that predicate temporarily, so its reported
PoNR is under revalidation. The old slip fixture also contains a non-monotone
probe verdict sequence, so repeated probes are required before treating a flip
as evidence.

A learned-policy result can be promoted to verified only when the run records:

- the exact checkpoint SHA-256 and simulator/task versions;
- the IPFD version and a clean source commit;
- the disturbance timestep and injection parameters;
- the physical recovery predicate configuration;
- raw probe verdicts with repeat count and confidence;
- reset-boundary cross-cell pose measurements;
- successful lift preconditions; and
- regenerated report and rollout hashes.

The current driver records the checkpoint digest, disturbance onset, named
physical predicate, repeated raw verdicts, probe configuration, and reset-boundary
env-0 delta in `ipfd.recovery_run.v1` JSON. Five seeds with both irrecoverable and
recoverable controls are required by the multi-seed gate. Real actionability cases
remain a separate required artifact.

Until then, the result is classified as `historical_fixture_only`.
