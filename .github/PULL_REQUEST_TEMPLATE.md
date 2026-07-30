<!-- Keep PRs small and focused. One idea per PR. -->

## What and why

<!-- What does this change, and what problem does it solve? Link the issue it closes. -->

Closes #

## The architecture boundary (required check)

IPFD's core invariant is that the analysis layer (`detectors`, `ponr`, `metrics`,
`report`, `viz`) is pure NumPy/Matplotlib and never imports a simulator. Only
`adapters/isaac_lab` and `oracles/*` may touch Isaac Lab, and only via lazy import.

- [ ] This PR does not add a simulator or torch import to the analysis layer.
- [ ] Any GPU-only code is lazily imported and guarded (`importorskip` in tests).

## Verification

<!-- State what you ran and what you observed. State the command and its output. -->

- [ ] `ruff check src tests` is clean.
- [ ] `pytest` passes locally (state the count).
- [ ] If this touches the Isaac Lab path, I ran the relevant `scripts/verify_*.py`
      on a GPU and pasted the machine-readable status block below.

```
<paste test count and/or IPFD_* status block here>
```

## Scope

- [ ] This PR stays within the supported scope (Franka single-object pick-and-place),
      or the PR description justifies why widening it is worth the maintenance cost.
