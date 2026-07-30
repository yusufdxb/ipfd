# Compatibility reporting guide

IPFD's historical GPU results were collected on **one** configuration:

| Axis | Tested |
|---|---|
| Local `isaaclab` distribution | 4.5.22 |
| CPU analysis Python | 3.10, 3.11, 3.12 |
| OS | Linux (Ubuntu) |
| GPU | single CUDA device |

Anything outside this matrix is **unverified**, not unsupported. Reports on other
configurations widen the tested matrix.

## When to file a compatibility report

- A different **Isaac Lab version** (any 4.5.x other than 4.5.22, or 5.x).
- A different **OS** (Windows, other Linux distro).
- A different **GPU / driver / CUDA** combination.
- The analysis layer failing on Python 3.10, 3.11, or 3.12.

Use the
[compatibility report template](https://github.com/yusufdxb/ipfd/issues/new?template=compatibility_report.yml).

## The most likely incompatibility: the recovery probe

The Point-of-No-Return probe depends on Isaac Lab's `reset_to` state save/restore.
In the historical runtime, exposed scene state round-tripped while the continued
trajectory diverged after evolved, contact-rich state. The missing simulator or
task state was not isolated. IPFD avoids restoring the recorded primary by using
**environment isolation** for recovery probes.

If you run on a different Isaac Lab version, the **probe / PoNR path is where a
mismatch is most likely to appear**. The analysis layer (steps 1 to 4 of the
[checklist](VALIDATION.md)) is simulator-free and version-independent. When filing,
please note which step first failed:

`import` · `AppLauncher` · `env creation` · `reset_to` · `probe` · `analysis`

That maps directly onto IPFD's code boundaries and makes triage immediate.

## What to include

The [template](https://github.com/yusufdxb/ipfd/issues/new?template=compatibility_report.yml) collects OS,
Python, Isaac Lab version, GPU, the first-failing step, and the machine-readable
status block(s). Paste the block verbatim. It is enough to reproduce most issues
without a back-and-forth.
