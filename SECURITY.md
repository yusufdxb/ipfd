# Security Policy

IPFD is an offline research and debugging tool. It has no network surface, no server,
and no persistent service. The realistic risk surface is small, but two items are
worth stating plainly.

## Model checkpoints

The Isaac Lab path loads policy checkpoints with PyTorch's restricted
`weights_only=True` mode and maps actor tensors with strict key and shape checks.
IPFD refuses checkpoint formats that require unrestricted pickle loading. This
reduces deserialization risk, but checkpoints should still come from a trusted
source and their SHA-256 should be recorded. The documented driver uses NVIDIA's
published Lift-Cube checkpoint.

## The recovery probe writes simulator state

The dual-environment probe restores saved simulator state into a vectorized probe
cell. Historical runs measured no immediate env-0 object-pose change across the
env-1 reset boundary. This is not a full trajectory-isolation guarantee: all
vectorized cells still advance during a recovery rollout. Run probes only against
environments you control.

## Rollout archives

Rollouts load with NumPy pickle support disabled. IPFD also rejects malformed
containers, archives with excessive member counts, and archives that declare
more than 512 MiB of uncompressed array data. These limits reduce accidental or
malicious resource exhaustion; treat third-party artifacts as untrusted inputs.

## Reporting a vulnerability

If you find a security issue, please report it privately rather than opening a public
issue. Email **yusuf.a.guenena@gmail.com** with a description and, if possible, a
reproduction. Expect an acknowledgement within one week. Once a fix is available we
will credit you in the release notes unless you prefer otherwise.

## Supported versions

Security fixes land on the latest release only.
