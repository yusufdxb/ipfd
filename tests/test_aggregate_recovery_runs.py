import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_recovery_runs",
    Path(__file__).parents[1] / "scripts" / "aggregate_recovery_runs.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
aggregate = _MODULE.aggregate
main = _MODULE.main


def _run(seed, failure, status="complete"):
    return {
        "schema": "ipfd.recovery_run.v1",
        "status": status,
        "seed": seed,
        "failure_mode": failure,
    }


def test_aggregate_orders_runs_and_writes_bundle(tmp_path):
    second = tmp_path / "second.json"
    first = tmp_path / "first.json"
    second.write_text(json.dumps(_run(1, "teleport")))
    first.write_text(json.dumps(_run(0, "slip")))
    output = tmp_path / "nested" / "bundle.json"

    assert main([str(second), str(first), "--output", str(output)]) == 0
    bundle = json.loads(output.read_text())
    assert bundle["schema"] == "ipfd.multiseed.v1"
    assert [run["seed"] for run in bundle["runs"]] == [0, 1]


def test_aggregate_marks_incomplete_bundle(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_run(0, "slip", status="incomplete")))
    assert aggregate([path])["status"] == "incomplete"
