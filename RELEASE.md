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
tagged commit. The GPU headline is re-run on `main` before every tag that touches the
adapter or oracles.

### What 1.0.0 means here

1.0.0 is the first public release. It is a promise of API stability: the analysis-layer
surface (`ipfd.__all__`) and the report JSON schema will not break without a major
version bump. The project ships at 1.0 because the core architecture, the recovery-probe
method, and the learned-policy validation are complete and verified, that is the
substance a stable API rests on.

The Isaac Lab adapter is the lower-stability surface: it is validated against Isaac Lab
4.5.22, and support for other versions may adjust it in minor releases. This boundary is
stated in the README and the changelog so users know exactly what is covered by the
stability guarantee.

## Release checklist

1. `ruff check src tests` clean; `pytest` green locally and in CI.
2. Bump `version` in `pyproject.toml` and `__version__` in `src/ipfd/__init__.py`
   (keep them identical).
3. Move the `[Unreleased]` section of `CHANGELOG.md` under the new version with the
   date; update the compare links.
4. Update `CITATION.cff` `version` and `date-released`.
5. Re-run the GPU evidence chain if the adapter or oracles changed, and confirm the
   figures in `examples/figures/` still match the README claims.
6. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z"` and push the tag.
7. Cut a GitHub Release from the tag, pasting the changelog section.
8. (When published to PyPI) `python -m build && twine upload dist/*`.

## Deprecation policy

Deprecated API stays for one minor release with a `DeprecationWarning` before removal
in the next minor (pre-1.0) or major (post-1.0). Removals are always in the changelog.
