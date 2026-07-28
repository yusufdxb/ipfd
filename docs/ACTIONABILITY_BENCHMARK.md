# Causal actionability benchmark

`benchmark_actionability.py` is a simulator-free regression benchmark for the
IPFD actionability contract. It runs four labeled cases per seed: a positive
actionable warning, a no-alarm negative control, a late natural-failure-like
alarm, and an alarm inside the strided PoNR uncertainty interval.

Run it from the repository root:

```bash
python scripts/benchmark_actionability.py --seeds 0,1,2,3,4 \\
  --json benchmark.json --csv benchmark.csv
```

The output is deterministic and includes the schema version, seed list,
aggregate accuracy, actionable-warning rate, relation counts, and one row per
case. This is a CPU-only contract test, not evidence of simulator or learned
policy competence. Isaac Lab results must be reported separately with the
runtime validation and learned-policy evaluation scripts.
