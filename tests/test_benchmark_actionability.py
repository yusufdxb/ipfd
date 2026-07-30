import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_actionability", Path(__file__).parents[1] / "scripts" / "benchmark_actionability.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
run = _MODULE.run


def test_benchmark_is_deterministic_and_covers_controls():
    first = run([0, 7])
    second = run([0, 7])
    assert first == second
    assert first["n_cases"] == 8
    assert first["accuracy"] == 1.0
    assert first["actionable_warning_rate"] == 1.0
    assert set(first["relations"]) == {
        "ambiguous_within_ponr_interval",
        "definitely_actionable",
        "no_alarm",
        "too_late",
    }
