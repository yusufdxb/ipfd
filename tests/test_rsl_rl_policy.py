import pytest

from ipfd.oracles.rsl_rl_policy import _load_checkpoint_compat

# torch is an Isaac-Lab-side dependency and is deliberately not declared in
# [dev], so CI installs the analysis layer without it. Skip rather than fail.
torch = pytest.importorskip("torch")


class _FakeActor:
    def __init__(self):
        self.params = {
            "mlp.0.weight": torch.zeros(2, 3),
            "mlp.0.bias": torch.zeros(2),
            "distribution.std_param": torch.zeros(2),
        }
        self.loaded = None

    def state_dict(self):
        return self.params

    def load_state_dict(self, mapped, strict):
        assert strict is True
        self.loaded = mapped


class _FakeRunner:
    def __init__(self):
        self.actor = _FakeActor()
        self.alg = type("Alg", (), {"get_policy": lambda _self: self.actor})()

    def load(self, *_args, **_kwargs):
        raise AssertionError("safe actor-only paths should not call runner.load")


def test_legacy_checkpoint_maps_actor_weights_strictly(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": {
                "actor.0.weight": torch.ones(2, 3),
                "actor.0.bias": torch.ones(2),
                "std": torch.full((2,), 0.5),
            }
        },
        checkpoint,
    )
    runner = _FakeRunner()

    _load_checkpoint_compat(runner, str(checkpoint), "cpu")

    assert torch.equal(runner.actor.loaded["mlp.0.weight"], torch.ones(2, 3))
    assert torch.equal(runner.actor.loaded["distribution.std_param"], torch.full((2,), 0.5))


def test_current_checkpoint_loads_actor_state_strictly(tmp_path):
    checkpoint = tmp_path / "current.pt"
    state = {
        "mlp.0.weight": torch.ones(2, 3),
        "mlp.0.bias": torch.ones(2),
        "distribution.std_param": torch.full((2,), 0.25),
    }
    torch.save({"actor_state_dict": state}, checkpoint)
    runner = _FakeRunner()

    _load_checkpoint_compat(runner, str(checkpoint), "cpu")

    assert runner.actor.loaded is not None
    assert torch.equal(runner.actor.loaded["distribution.std_param"], torch.full((2,), 0.25))
