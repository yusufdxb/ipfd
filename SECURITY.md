# Security Policy

IPFD is an offline research and debugging tool. It has no network surface, no server,
and no persistent service. The realistic risk surface is small, but two items are
worth stating plainly.

## Untrusted model checkpoints

The Isaac Lab path (`scripts/verify_learned_policy.py`, `eval_checkpoint.py`, and the
`rsl_rl` oracle) loads policy checkpoints. Loading a checkpoint from an untrusted
source can execute arbitrary code, because that is how PyTorch checkpoint
deserialization works, not a flaw specific to IPFD. Only load checkpoints you trust.
The documented demo uses NVIDIA's official published Lift-Cube checkpoint.

## The recovery probe writes simulator state

The dual-environment probe restores saved sim state into a probe environment. It is
verified to never perturb the primary environment (`max env-0 pose delta = 0.0`), but
it is still code that manipulates simulator internals. Run it only against
environments you control.

## Reporting a vulnerability

If you find a security issue, please report it privately rather than opening a public
issue. Email **yusuf.a.guenena@gmail.com** with a description and, if possible, a
reproduction. Expect an acknowledgement within one week. Once a fix is available we
will credit you in the release notes unless you prefer otherwise.

## Supported versions

IPFD is pre-1.0. Security fixes land on the latest release only.
