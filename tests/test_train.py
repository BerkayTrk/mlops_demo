import json

from train import train


def load_params():
    with open("params.json") as f:
        return json.load(f)


def test_model_beats_baseline():
    """Guardrail: CI fails if a change drops accuracy below 90%."""
    metrics = train(load_params())
    assert metrics["accuracy"] >= 0.90


def test_metrics_have_expected_keys():
    metrics = train(load_params())
    assert {"model", "accuracy", "f1"} <= metrics.keys()
