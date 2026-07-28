import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_comparison",
    Path(__file__).parents[1] / "scripts" / "benchmark_comparison.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
run = _MODULE.run


def test_ipfd_beats_naive_on_causal_fixture():
    result = run([0, 1, 2])
    assert result["n_cases"] == 15
    assert result["ipfd"]["precision"] == 1.0
    assert result["ipfd"]["recall"] == 1.0
    assert result["naive"]["false_alarm_rate"] > result["ipfd"]["false_alarm_rate"]


def test_metrics_are_deterministic():
    assert run([7]) == run([7])
