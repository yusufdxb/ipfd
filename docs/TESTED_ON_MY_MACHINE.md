# "Tested on my machine" guide

The single most useful contribution to IPFD right now is an **independent report
that it ran on hardware and software that isn't the author's.** It converts a
one-machine result into broader compatibility evidence. This takes about five minutes and
requires **no understanding of IPFD's internals.**

## How to report

1. Run the [validation checklist](VALIDATION.md) (steps 1–4 are CPU-only; 5–7 need
   Isaac Lab + a GPU; do as many as you can).
2. Open a new **Tested on my machine** discussion:
   [start here](https://github.com/yusufdxb/ipfd/discussions/new?category=tested-on-my-machine).
3. The form asks only for facts you can copy-paste; no prose required.

## What the form collects (and how to get each value)

| Field | How to get it |
|---|---|
| OS | e.g. `Ubuntu 22.04`, `Windows 11` |
| Python version | `python --version` |
| Isaac Lab version | the version you installed (skip if CPU-only run) |
| GPU | `nvidia-smi --query-gpu=name --format=csv,noheader` (skip if CPU-only) |
| Did it run end to end? | Yes / Yes with warnings / No |
| Did PoNR match? | teleport → `YES`, slip → `NO` (skip if you didn't run the GPU demo) |
| Runtime | wall-clock seconds of the demo |
| Status blocks | copy everything between the `====` lines: `IPFD_RUNTIME_SMOKE`, `IPFD_LEARNED_STATUS`, `DUAL_PROBE_STATUS` |
| Logs / screenshot | optional; paste errors, drag-and-drop the timeline PNG |

## CPU-only reports are welcome

You do **not** need a GPU or Isaac Lab. A report that steps 1–4 passed on your OS
and Python version is genuinely useful; it validates the analysis layer and the
install path across environments.

## What happens next

Reports are read and, where they surface a real incompatibility, turned into a
labeled issue. There is no obligation to debug anything; the raw evidence is the
contribution.
