# Roadmap

IPFD is a stable, deliberately narrow tool. Its 1.0 surface, a pure-NumPy analysis
layer plus a verified env-isolated recovery probe for Franka single-object
pick-and-place, is complete and supported. This roadmap describes the direction we
intend to grow in, without compromising that stability.

Priorities are ordered by how much they help a new Isaac Lab user, not by novelty.
Tracked work lives in [GitHub issues and milestones](https://github.com/yusufdxb/ipfd/milestones);
this document is the "why," the issues are the "what."

## Guiding principles

- **The analysis layer stays simulator-free.** Pure NumPy, testable in CI without a
  GPU. This is the invariant that makes IPFD trustworthy, and it is not negotiable.
- **Narrow and verified beats broad and hand-wavy.** New scope is added only when it
  can be validated to the same standard as the existing results.
- **Every capability is backed by a run.** Claims map to scripts and tests, not to
  intentions. The README's verified/partially-verified structure is permanent.

## Direction

### Make it usable on your own task
The highest-value direction is lowering the barrier to running IPFD on a task other
than the shipped demo. That means a clearly documented, tested contract for supplying
your own recovery oracle, so PoNR can be computed for your policy without reading the
adapter source.

### Broaden simulator-version support
The recovery probe is validated against Isaac Lab 4.5.22. We want to understand which
parts of the adapter are version-specific and support additional Isaac Lab versions,
with a clear runtime signal when a version is untested.

### Strengthen the safety net
Extend automated coverage of the GPU path via recorded-rollout fixtures that replay
through the analysis layer in CI (no GPU required), so the probe's output contract is
continuously protected against regressions.

### Sharpen failure localization
The imminence alarm currently localizes to task phase transitions on trained
policies. Phase-aware detector calibration, so the alarm localizes the fault rather
than the grasp transition, is a research direction we are interested in but will only
ship if it demonstrably beats reporting PoNR alone.

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
