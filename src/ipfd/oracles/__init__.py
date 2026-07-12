"""Recovery / reference controllers ("oracles") for IPFD.

These modules provide the best-effort recovery controllers whose success or
failure *defines* the Point of No Return (see :mod:`ipfd.ponr`). They are kept in
their own subpackage and are **not** imported here, because each one pulls in a
heavy GPU/sim runtime (``torch`` + ``warp``). Import the concrete module directly,
and only *after* Isaac Lab's ``AppLauncher`` has started the sim:

    from ipfd.oracles.pick_lift_sm import PickAndLiftSm, sm_action

The analysis layer (detectors, PoNR, metrics, report, viz) never touches this
subpackage, so it continues to run in CI without a GPU.
"""
