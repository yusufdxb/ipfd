# Changelog

All notable changes to IPFD are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). See
[RELEASE.md](RELEASE.md) for how versions are cut.

## [Unreleased]

_Nothing yet._

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
- CI on Python 3.10 / 3.11: lint, 31 tests, and a headless example smoke-run, all
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

[Unreleased]: https://github.com/yusufdxb/ipfd/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yusufdxb/ipfd/releases/tag/v1.0.0
