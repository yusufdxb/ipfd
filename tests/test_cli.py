import json

from ipfd.cli import main


def test_analyze_cli_replays_fixture_and_writes_outputs(tmp_path, capsys):
    fixture = "tests/fixtures/learned_teleport_rollout.npz"
    report_path = tmp_path / "report.json"
    plot_path = tmp_path / "timeline.png"

    assert main(
        [
            "analyze",
            fixture,
            "--report",
            str(report_path),
            "--plot",
            str(plot_path),
            "--disturbance-onset",
            "56",
            "--probe-stride",
            "8",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "IPFD Failure Debug Report" in output
    assert '"alarm_relation": "pre_disturbance"' in output
    payload = json.loads(report_path.read_text())
    assert (payload["t_ponr"], payload["t_failure"]) == (56, 57)
    assert plot_path.stat().st_size > 0
