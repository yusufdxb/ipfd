# Roadmap

IPFD supports one robot and one task. Its 1.0 CPU analysis surface is stable. The
Franka learned-policy recovery evidence is under physical-predicate revalidation,
so simulator results remain explicitly provisional. This roadmap describes the
direction of the project without presenting planned work as shipped evidence.

Priorities are ordered by how much they help a new Isaac Lab user, not by novelty.
Tracked work lives in [GitHub issues and milestones](https://github.com/yusufdxb/ipfd/milestones).
This document describes direction; the issues track specific work.

## Guiding principles

- **The analysis layer stays simulator-free.** Pure NumPy, testable in CI without a
  GPU. Changes that break this invariant are rejected.
- **New scope requires validation at the existing bar.** Scope is added only when it
  can be validated to the same standard as the existing results.
- **Every capability is backed by a run.** Every claim in the README names the script
  or test that produced it.

## Direction

### Make it usable on your own task
The highest-value direction is lowering the barrier to running IPFD on a task other
than the shipped demo. That means a clearly documented, tested contract for supplying
your own recovery oracle, so PoNR can be computed for your policy without reading the
adapter source.

### Broaden simulator-version support
The recovery probe was exercised against one local runtime whose `isaaclab`
distribution reported 4.5.22. We want to identify version-specific behavior and
support additional runtimes with a clear warning when a version is untested.

### Strengthen the safety net
Extend automated coverage of the GPU path via recorded-rollout fixtures that replay
through the analysis layer in CI (no GPU required), so the probe's output contract is
continuously protected against regressions.

### Sharpen failure localization
The imminence alarm currently localizes to task phase transitions on trained
policies. Phase-aware detector calibration, so the alarm localizes the fault rather
than the grasp transition, is a research direction that will ship only if it improves
alarm localization against the actionability benchmark.

### Beyond single-object pick-and-place
Additional tasks and manipulators are of interest long-term. Each will be added only
with validation matching the current bar, and only when the recovery-oracle contract
generalizes cleanly to it.

## Out of scope

- Turning the analysis-layer detectors into learned/ML components.
- A general benchmark suite or leaderboard. IPFD is a debugger, not a benchmark.
- A Kit/Omniverse extension UI.

If you want to help move any direction forward, open an issue to discuss it before
writing code, see [CONTRIBUTING.md](CONTRIBUTING.md).
