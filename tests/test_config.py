from gradient_starvation.config import load_config


def test_nested_overrides_are_typed():
    config = load_config(
        "configs/smoke.yaml",
        ["training.steps=3", "task.rho_values=[4]", "experiment.device=cpu"],
    )
    assert config["training"]["steps"] == 3
    assert config["task"]["rho_values"] == [4]
    assert config["experiment"]["device"] == "cpu"

