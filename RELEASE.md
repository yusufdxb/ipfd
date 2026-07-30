# Release process and versioning

## Versioning strategy

IPFD follows [Semantic Versioning](https://semver.org/). The public API for semver
purposes is the analysis layer's importable surface (`ipfd.__all__`) and the JSON
schema emitted by `FailureDebugReport.to_json()`. The Isaac Lab adapter and oracles
are a lower-stability surface, because they depend on a specific Isaac Lab version
(currently 4.5.22).

Releases are defined by **verified capabilities**, not by feature count. A new
detector, metric, or oracle does **not**, on its own, justify a release. It qualifies
only once it is verified by a run or a test, documented, and reproducible. Code that
exists but has no evidence behind it is not a shipped capability and does not move the
version. This follows the project's operating philosophy: verified claims over
speculative ones, reproducibility over novelty, maintainability over feature count,
and stable APIs over rapid expansion.

- **MAJOR**: a breaking change to the analysis-layer API or the report JSON schema.
- **MINOR**: a new **verified, documented, backward-compatible capability**, a
  detector/metric/oracle or adapter capability that ships with the run or test that
  demonstrates it. Unverified additions do not qualify.
- **PATCH**: bug fixes, documentation, and non-breaking runtime guards, with no change
  to the API or report schema.

No release, at any level, may contain a claim that has not been reproduced on the
tagged commit. The GPU evidence gate must pass before any tag promotes a claim
that depends on the adapter or oracles.

### What 1.0.0 means here

1.0.0 is the first public release. It is a promise of API stability: the analysis-layer
surface (`ipfd.__all__`) and the report JSON schema will not break without a major
version bump. The 1.0 stability promise covers the offline analysis API and
report schema. Historical learned-policy fixtures remain deterministic regression
inputs, but their recovery semantics are under revalidation and are not current
release evidence.

The Isaac Lab adapter is the lower-stability surface. A historical runtime smoke
used a local Isaac Lab 4.5.22 distribution, and learned-policy semantics are under
revalidation. Support for simulator versions may adjust in minor releases.

## Release checklist

1. `ruff check src tests scripts examples`, `mypy`, and coverage-gated `pytest`
   are green locally and in CI.
2. Bump `version` in `pyproject.toml` and `__version__` in `src/ipfd/__init__.py`
   (keep them identical).
3. Move the `[Unreleased]` section of `CHANGELOG.md` under the new version with the
   date; update the compare links.
4. Update `CITATION.cff` `version` and `date-released`.
5. If the adapter or oracles changed, regenerate all fail-closed GPU evidence
   artifacts from the tagged commit. Do not promote learned-policy claims unless
   the competence, multi-seed, and actionability gates pass.
   Store the accepted artifacts as `release-evidence/competence.json`,
   `release-evidence/multiseed.json`, and `release-evidence/actionability.json`.
   Commit those artifacts without changing `src/`, `scripts/`, or
   `pyproject.toml`. The release workflow requires the recorded source commit to
   be an ancestor of the tag and rejects runtime-code changes after that commit.
6. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z"` and push the tag.
7. Cut a GitHub Release from the tag, pasting the changelog section.
8. Confirm the clean-wheel and source-distribution smoke jobs passed.
9. PyPI publication uses the tag-triggered trusted-publishing workflow. Do not
   upload from a developer workstation.

## Deprecation policy

Deprecated API stays for one minor release with a `DeprecationWarning` before removal
in the next minor (pre-1.0) or major (post-1.0). Removals are always in the changelog.
