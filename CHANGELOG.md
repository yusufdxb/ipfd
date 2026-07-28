# Changelog

All notable changes to IPFD are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). See
[RELEASE.md](RELEASE.md) for how versions are cut.

## [Unreleased]

### Added
- A causal actionability evaluator that treats strided PoNR as an uncertainty
  interval and refuses to credit alarms that fired before a known disturbance.
- A zero-code `ipfd analyze` command for replay archives, JSON reports, timeline
  plots, and causal actionability output.
- Packaged `ipfd-demo` and `ipfd-evidence-gate` commands that work from an
  installed wheel outside the source tree.
- A fail-closed evidence schema for checkpoint competence, paired multi-seed
  recovery controls, raw repeated probes, source/runtime provenance, and
  code-derived actionability cases.
- A manifest-based actionability artifact builder and deterministic multi-seed
  recovery-run aggregator.
- Explicit Isaac asset-root overrides for the GPU entry points.
- A warn-once runtime guard names the tested Isaac Lab 4.5.22 target when a
  different installed version enters the adapter. The warning never blocks a run.
- Regression coverage for NumPy-backed report metadata, degenerate detector input,
  detector weights, single-feature visualization, and Isaac Lab version checks.

### Fixed
- The wheel's demo command no longer imports the repository-only `scripts`
  namespace.
- PoNR, detector, rollout, archive, and analysis-configuration inputs now reject
  malformed shapes, non-finite values, non-binary verdicts, and unsafe archives
  instead of silently coercing them.
- Learned checkpoints use restricted tensor-only loading with strict actor key
  and shape checks for current and legacy layouts.
- The physical pick/lift recovery predicate now requires sustained lift,
  reachability, end-effector proximity, and gripper closure, and fails closed
  when signals are unavailable.
- Repeated recovery probes reset stateful policies and predicates and persist
  their raw verdicts.
- The documented PoNR bound direction was backwards. An oracle-relative PoNR
  timestep is an early estimate and, under sound positive verdicts, a lower bound
  on the optimal-control PoNR timestep, not an upper bound. Public documentation
  also downgrades unsupported simulator root-cause claims and labels historical
  height-only fixtures as analysis regressions rather than physical evidence.
- The learned-policy adapter now loads both current and legacy published rsl_rl
  checkpoint layouts.
- The learned-policy demo seeds the Isaac environment for repeatable runs.
- The runtime smoke test exits nonzero when its machine-readable result is FAIL.
- `action_variance_score` now rejects wrong-rank and non-numeric object arrays with
  a clear shape or numeric-contract `ValueError`.
- Report JSON conversion now handles every NumPy scalar type supported by JSON,
  including `np.bool_`, as well as arrays.
- The learned-policy demo states its real prerequisites, runs the compatibility
  preflight first, and no longer contains author-specific interpreter paths.
- README and contributor troubleshooting now cover ROS-injected pytest plugins.

### Changed
- CI now covers Python 3.10, 3.11, and 3.12 with lint, typing, branch coverage,
  wheel/sdist builds, installed-command smoke tests, and immutable action pins.
- Release publishing is fail-closed on clean, commit-linked GPU evidence and uses
  PyPI trusted publishing. Until the required `release-evidence/` artifacts are
  committed, any tag build stops at the evidence step and nothing is published.
- Both workflows declare read-only repository content at the workflow level, and
  every third-party action is pinned to an immutable commit SHA.

### Earlier hardening in this development series

A hardening and adoption pass. These changes were never tagged as v1.0.1.

#### Added
- **Input validation on `Rollout`**: fail-fast `ValueError`s (naming the field, the
  offending value, and what was expected) for empty rollouts, NaN/Inf in
  `observations` / `actions` / `recovery_success`, `dt <= 0`, and non-integer /
  boolean / float `t_failure`. Previously these produced silent NaN cascades or a
  silently mis-cast recovery array.
- **Detector-weight key validation**: `failure_imminence_score` now raises on unknown
  weight keys (listing the unknown and the valid keys) instead of silently ignoring them.
- **Isaac runtime smoke UX**: `verify_isaac_runtime.py` detects a missing Isaac Lab and
  exits cleanly with a pointer to the install docs, instead of a raw `ModuleNotFoundError`.
- Regression tests for every new validation.
- **Adoption docs**: `docs/VALIDATION.md`, `docs/TESTED_ON_MY_MACHINE.md`,
  `docs/COMPATIBILITY.md`; a "Tested on my machine" discussion template and a
  compatibility-report issue template; README badge row and an External validation section.

#### Changed
- `LICENSE` is now pure MIT (so GitHub classifies the repo as MIT); the vendored
  Isaac Lab BSD-3-Clause notice moved to `THIRD_PARTY_LICENSES.md`.

## [1.0.0] - 2026-07-12

First public release. The core architecture, the recovery-probe method, and the
learned-policy validation are complete, so the analysis-layer API and the report JSON
schema are now stable under semantic versioning.

### Added
- **Analysis layer** (pure NumPy / Matplotlib, no simulator): `Rollout` type,
  imminence detectors (action-variance, entropy-collapse, drift), `point_of_no_return`,
  the minimal timing-metric set, `build_report`, and the stacked-panel `plot_timeline`.
- **Point of No Return** defined operationally against a recovery probe.
- **Dual-environment recovery probe** (`adapters/isaac_lab`): decoupled two-pass
  design that never calls `reset_to` on the recorded environment. Historical
  runs measured a zero immediate env-0 pose delta across probe-cell resets; this
  is not an end-to-end trajectory-isolation result.
- **Recovery oracles**: `pick_lift_sm` (scripted, vendored from Isaac Lab under
  BSD-3-Clause) and `rsl_rl_policy` (trained).
- Historical learned-policy fixtures remain compatible with the analysis layer.
  Their height-only recovery labels are under revalidation and are not current
  physical evidence.
- `scripts/verify_*.py` diagnostic chain with machine-readable status blocks.
- CI on Python 3.10, 3.11, and 3.12: lint, type checks, branch coverage, build,
  and installed-package smoke tests, with no GPU.
- Project governance: issue and pull-request templates, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `ROADMAP.md`, and `RELEASE.md`.

### Known limitations
- Silent-collapse *detection* on a trained policy fires at the grasp transition, not
  the injected fault; on trained policies the reliable signal is PoNR, not the alarm.
- Entropy-collapse detector is flat on checkpoints with state-independent action std.
- Historical scripted-policy PoNR runs were limited to the grasped region;
  pre-grasp continuation divergence had no identified root cause.
- A historical adapter runtime smoke used a local Isaac Lab 4.5.22 distribution;
  learned-policy recovery semantics are under revalidation.

[Unreleased]: https://github.com/yusufdxb/ipfd/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yusufdxb/ipfd/releases/tag/v1.0.0
