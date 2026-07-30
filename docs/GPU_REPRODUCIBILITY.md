# GPU reproducibility boundary

IPFD's CPU analysis layer is reproducible from the Python package metadata. The
Isaac Lab adapter was exercised only on one local Isaac Lab/Isaac Sim
installation. The installed `isaaclab` distribution reported `4.5.22`; that is a
provenance fingerprint, not a claim that every public installation channel
provides the same build. This repository does not claim that `pip install ipfd`
recreates the GPU stack.

Live runs must record simulator package versions, task name, checkpoint SHA-256,
seed, probe stride, budget, repeated raw verdicts, physical recovery predicate,
and the reset-boundary isolation measurement. Do not publish local checkpoint
paths or exact private hardware identifiers. Use
`scripts/eval_checkpoint.py --json result.json` and
`scripts/verify_learned_policy.py --json run.json` for machine-readable artifacts.
A failed or incomplete run is never evidence of policy competence.

Set `IPFD_EXPECTED_ISAAC_LAB_VERSION` only when reproducing a known environment;
the adapter otherwise avoids warning against non-public version strings.
