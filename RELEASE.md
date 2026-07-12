# Release process and versioning

## Versioning strategy

IPFD follows [Semantic Versioning](https://semver.org/). The public API for semver
purposes is the analysis layer's importable surface (`ipfd.__all__`) and the JSON
schema emitted by `FailureDebugReport.to_json()`. The Isaac Lab adapter and oracles
are considered a lower-stability surface until 1.0, because they depend on a specific
Isaac Lab version (currently 4.5.22).

- **MAJOR**: breaking change to the analysis-layer API or the report JSON schema.
- **MINOR**: new detectors, metrics, oracles, or adapter capabilities, backward
  compatible.
- **PATCH**: bug fixes and documentation.

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
