# Release blockers (PyPI publish path)

Status as measured on 2026-07-29 against the local Isaac Lab 4.5.22 runtime with
Isaac Sim 6.0.0.0. This file records why the tag-triggered publish path is inert,
so the next attempt does not start by tagging and hoping.

## The publish path is fail-closed on GPU evidence

`.github/workflows/release.yml` runs `build` before `publish`, and `build`
enforces, in order:

1. `tag == pyproject version`. `pyproject.toml` currently declares
   `1.1.0.dev0`, so a `v1.0.1` tag fails this step immediately.
2. The built wheel installs and runs outside the source tree.
3. `release-evidence/competence.json`, `release-evidence/multiseed.json`, and
   `release-evidence/actionability.json` all exist and pass
   `ipfd-evidence-gate`, and the evidence commit must be an ancestor of the
   tagged commit with `src`, `scripts`, and `pyproject.toml` byte-identical
   between them.

The `release-evidence/` directory does not exist in the repository. Nothing is
published until it does.

## Why that evidence cannot currently be produced

`ipfd.evidence_gate` requires `min_success_rate = 0.80` on the competence
artifact. Measured competence on this runtime is **0.00%**:

```
scripts/eval_checkpoint.py --headless --num_envs 64 --steps 200 \
  --checkpoint <published NVIDIA rsl_rl Lift-Cube checkpoint>

max_lift  mean=0.000  median=0.000  p90=0.000  max=0.000
  frac lifted >0.02m at some point: 0.00%
SUCCESS_RATE (>0.06m for final 10 steps): 0.00%
```

Recorded frames from the same checkpoint confirm this is real policy behavior
and not a measurement artifact: the arm stays folded away from the cube and
never approaches it. The loader reports loading the published checkpoint
"through strict actor-only mapping" for a legacy layout, so the most likely
cause is that the legacy actor weights no longer correspond to the current
network or observation layout on this Isaac Sim release.

Consequences:

- No competence artifact can pass the gate, so no multi-seed or actionability
  bundle can be assembled on top of it.
- The retracted learned-policy PoNR headline cannot be revalidated on this
  runtime, because revalidation requires a genuinely competent policy.

## Ordered unblock path

1. Resolve checkpoint competence. Either fix the legacy-to-current actor
   mapping in `ipfd/oracles/rsl_rl_policy.py`, or train a checkpoint against the
   installed runtime, and confirm with `eval_checkpoint.py` that success rate is
   at least 0.80 over at least 32 environments.
2. Produce the evidence bundle per [EVIDENCE_GATE.md](EVIDENCE_GATE.md):
   competence, then `teleport` and `slip` runs for at least five seeds
   aggregated by `aggregate_recovery_runs.py`, then at least 20 real
   actionability cases via `build_actionability_evidence.py`.
3. Run `ipfd-evidence-gate` locally until it exits 0, then commit the three
   JSON files under `release-evidence/` in the same commit that finalizes
   `src`, `scripts`, and `pyproject.toml`.
4. Set the release version in `pyproject.toml` and `src/ipfd/__init__.py` to the
   version being shipped (the CI `package` job guards drift between them).
5. Repository owner action, which cannot be automated: create a PyPI Trusted
   Publisher for project `ipfd` (repository `yusufdxb/ipfd`, workflow
   `release.yml`, environment `pypi`), then set the repository variable
   `PYPI_PUBLISH=true`.
6. Push the matching `v<version>` tag.

Step 1 is the real blocker. Steps 5 and 6 are cheap but pointless before it.
