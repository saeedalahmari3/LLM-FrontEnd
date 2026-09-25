import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def test_generation_hallucination_metric_uses_label_as_reference(monkeypatch):
    embedding_module = ModuleType("get_phi_embeddings")
    embedding_module.get_phi_embeddings = lambda texts: [
        [1.0, 0.0],
        [0.9, 0.1],
    ]

    hallucination_calls = []
    hallucination_module = ModuleType("hallucination_score")

    def fake_compute_h(reference, candidate):
        hallucination_calls.append((reference, candidate))
        return 0.25

    hallucination_module.compute_h = fake_compute_h
    monkeypatch.setitem(sys.modules, "get_phi_embeddings", embedding_module)
    monkeypatch.setitem(sys.modules, "hallucination_score", hallucination_module)

    module_path = Path(__file__).parents[1] / "get_cosine_similarity.py"
    spec = importlib.util.spec_from_file_location(
        "get_cosine_similarity_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metrics = module.compute_hallucination_metrics(
        ["clean response", "perturbed response"],
        "ground-truth label",
    )

    assert metrics["hallucination_score_clean_vs_label"] == 0.25
    assert hallucination_calls[0] == (
        "ground-truth label",
        "clean response",
    )
    assert hallucination_calls[1][0] == "ground-truth label"
    assert hallucination_calls[1][1] == metrics["center_response"]
    assert hallucination_calls[2] == (
        "clean response",
        metrics["center_response"],
    )
    assert metrics["hallucination_score_center_vs_clean"] == 0.25
    assert metrics["hallucination-score"] == 0.25
