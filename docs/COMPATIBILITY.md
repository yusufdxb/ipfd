# Compatibility reporting guide

IPFD's headline results are verified on **one** configuration:

| Axis | Tested |
|---|---|
| Isaac Lab | 4.5.22 |
| Python | 3.10, 3.11 |
| OS | Linux (Ubuntu) |
| GPU | single CUDA device |

Anything outside this matrix is **unverified**, not unsupported — and a report on
another configuration is exactly the evidence needed to widen the matrix.

## When to file a compatibility report

- A different **Isaac Lab version** (any 4.5.x other than 4.5.22, or 5.x).
- A different **OS** (Windows, other Linux distro).
- A different **GPU / driver / CUDA** combination.
- The analysis layer failing on a Python version other than 3.10/3.11.

Use the
[compatibility report template](../../../issues/new?template=compatibility_report.yml).

## The most likely incompatibility: the recovery probe

The Point-of-No-Return probe depends on Isaac Lab's `reset_to` state save/restore.
On 4.5.22, IPFD works around a specific PhysX behavior: a single `reset_to` corrupts
a `num_envs == 1` sim after a grasp because the contact/solver cache is not part of
`scene.get_state()`, and the corruption survives `env.reset()`. IPFD sidesteps this
with **environment isolation** (the probe runs in a second env that never touches the
recorded primary).

If you run on a different Isaac Lab version, the **probe / PoNR path is where a
mismatch is most likely to appear** — the analysis layer (steps 1–4 of the
[checklist](VALIDATION.md)) is simulator-free and version-independent. When filing,
please note which step first failed:

`import` · `AppLauncher` · `env creation` · `reset_to` · `probe` · `analysis`

That maps directly onto IPFD's code boundaries and makes triage immediate.

## What to include

The [template](../../../issues/new?template=compatibility_report.yml) collects OS,
Python, Isaac Lab version, GPU, the first-failing step, and the machine-readable
status block(s). Paste the block verbatim — it is enough to reproduce most issues
without a back-and-forth.
