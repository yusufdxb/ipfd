# CPU baseline comparison

`python scripts/benchmark_comparison.py` compares IPFD's causal actionability
decision with a naive rule that accepts any alarm before observable failure.
The fixture includes pre-disturbance, too-late, no-alarm, nominal, and actionable
episodes across deterministic seeds. Results report precision, recall, false-alarm
rate, PoNR lead time, and intervention-success proxy.

This is a contract benchmark only. It does not establish Isaac Lab or learned
policy competence. Run it with `--json artifacts/comparison.json` for a machine-
readable record.
