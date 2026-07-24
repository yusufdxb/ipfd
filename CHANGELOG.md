# Changelog

All notable changes to IPFD are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). See
[RELEASE.md](RELEASE.md) for how versions are cut.

## [Unreleased]

### Added
- A warn-once runtime guard names the tested Isaac Lab 4.5.22 target when a
  different installed version enters the adapter. The warning never blocks a run.
- Regression coverage for NumPy-backed report metadata, degenerate detector input,
  detector weights, single-feature visualization, and Isaac Lab version checks.

### Fixed
- `action_variance_score` now rejects wrong-rank and non-numeric object arrays with
  a clear shape or numeric-contract `ValueError`.
- Report JSON conversion now handles every NumPy scalar type supported by JSON,
  including `np.bool_`, as well as arrays.
- The learned-policy demo states its real prerequisites, runs the compatibility
  preflight first, and no longer contains author-specific interpreter paths.
- README and contributor troubleshooting now cover ROS-injected pytest plugins.

### Changed
- CI now declares read-only repository-content permissions explicitly.

## [1.0.1] - 2026-07-13

A hardening and adoption release. No new features, no algorithm or API changes: valid
inputs behave exactly as in 1.0.0, and the report JSON schema is unchanged.

### Added
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

### Changed
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
- **Point of No Return** defined operationally against a recovery probe, documented as
  a sound upper bound under the provided recovery oracle.
- **Dual-environment recovery probe** (`adapters/isaac_lab`): env-isolated,
  decoupled two-pass design that never `reset_to` the primary environment. Verified
  `max env-0 pose delta = 0.0` across probe resets.
- **Recovery oracles**: `pick_lift_sm` (scripted, vendored from Isaac Lab under
  BSD-3-Clause) and `rsl_rl_policy` (trained).
- **Verified results** on NVIDIA's official published `rsl_rl` Lift-Cube checkpoint:
  PoNR localizes an injected irrecoverable failure; a recoverable slip correctly
  yields no PoNR.
- `scripts/verify_*.py` evidence chain, each emitting a machine-readable status block.
- CI on Python 3.10 / 3.11: lint, tests, and a headless example smoke-run, all
  with no GPU.
- Project governance: issue and pull-request templates, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `ROADMAP.md`, and `RELEASE.md`.

### Known limitations
- Silent-collapse *detection* on a trained policy fires at the grasp transition, not
  the injected fault; on trained policies the reliable signal is PoNR, not the alarm.
- Entropy-collapse detector is flat on checkpoints with state-independent action std.
- Scripted-policy PoNR holds only in the grasped region; pre-grasp `reset_to` hands
  the probe a cold PhysX contact state.
- The Isaac Lab adapter is validated against Isaac Lab 4.5.22 only.

[Unreleased]: https://github.com/yusufdxb/ipfd/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/yusufdxb/ipfd/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/yusufdxb/ipfd/releases/tag/v1.0.0
