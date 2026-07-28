import json

from ipfd.demo import main


def test_demo_writes_reproducible_json(tmp_path):
    output = tmp_path / "demo.json"
    assert main(["--json", str(output)]) == 0
    text = output.read_text()
    assert '"demo": "offline_synthetic"' in text
    assert '"ponr_repeated"' in text
    assert json.loads(text)["report"]["t_ponr"] == 90
