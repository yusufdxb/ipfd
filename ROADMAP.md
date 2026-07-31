# Roadmap (historical)

**This project is archived.** Nothing below is planned, scheduled, or in progress.
The file is retained so the record shows what was intended and what closed it out.
See [ARCHIVED_NEGATIVE_RESULT.md](ARCHIVED_NEGATIVE_RESULT.md) for the result that
ended development, and the archival section at the top of [README.md](README.md)
for the summary.

## Why the roadmap closed

Every direction below assumed that a restored simulator branch stands in for the
uninterrupted episode it was branched from. The corrected five-seed study measured
that assumption directly and it did not hold: under the expanded restoration
protocol, 11 of 444 primary paired branches still reversed the task decision, a
38.9 percent relative reduction against a preregistered 50 percent requirement.
Broadening tasks, runtimes, or detectors would have propagated an unvalidated
label rather than fixing it.

## Guiding principles (retained; these held up)

- **The analysis layer stays simulator-free.** Pure NumPy, testable in CI without a
  GPU. This invariant was maintained throughout.
- **New scope requires validation at the existing bar.** This is the principle that
  stopped the project rather than letting it expand.
- **Every capability is backed by a run.** Every claim in the README names the script
  or test that produced it.

## Directions that were planned and are now dropped

### Make it usable on your own task
Lowering the barrier to running IPFD on a task other than the shipped demo, via a
documented recovery-oracle contract. The contract was written
([docs/ORACLE_CONTRACT.md](docs/ORACLE_CONTRACT.md)) and still describes the
interface accurately. Dropped because the quantity it computes was not validated.

### Broaden simulator-version support
The recovery probe was exercised against one local runtime whose `isaaclab`
distribution reported 4.5.22. Dropped with no further runtimes tested.

### Strengthen the safety net
Extending automated coverage of the GPU path via recorded-rollout fixtures replayed
through the analysis layer in CI. Partially delivered
(`tests/test_replay_fixture.py`); not extended further.

### Sharpen failure localization
Phase-aware detector calibration, so the alarm localizes the fault rather than the
grasp transition. Never attempted. On the trained policy tested, the alarm fires
before the injected fault, so the actionable window measured empty.

### Beyond single-object pick-and-place
Additional tasks and manipulators. Never attempted.

## Out of scope (then and now)

- Turning the analysis-layer detectors into learned/ML components.
- A general benchmark suite or leaderboard. IPFD is a debugger, not a benchmark.
- A Kit/Omniverse extension UI.
