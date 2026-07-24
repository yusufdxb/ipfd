import numpy as np

from ipfd import build_report, plot_timeline
from ipfd.types import Rollout


def test_plot_timeline_accepts_single_observation_dimension(tmp_path):
    rollout = Rollout(
        observations=np.zeros((10, 1)),
        actions=np.zeros((10, 2)),
        success=False,
    )
    output = tmp_path / "timeline.png"

    result = plot_timeline(rollout, build_report(rollout), str(output))

    assert result == str(output)
    assert output.stat().st_size > 0
