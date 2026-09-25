import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def load_hallucination_module(monkeypatch, sentence_transformer):
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = sentence_transformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    module_path = Path(__file__).parents[1] / "hallucination_score.py"
    spec = importlib.util.spec_from_file_location(
        "hallucination_score_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_model_is_forced_to_cpu_float32(monkeypatch):
    constructor_calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_path, device):
            constructor_calls.append((model_path, device))
            self.float_called = False

        def float(self):
            self.float_called = True
            return self

    module = load_hallucination_module(monkeypatch, FakeSentenceTransformer)
    monkeypatch.setattr(
        module,
        "_get_cached_model_path",
        lambda model_name: "/cached/minilm",
    )

    model = module._get_semantic_model()

    assert constructor_calls == [("/cached/minilm", "cpu")]
    assert model.float_called is True


def test_compute_s_sem_returns_finite_cosine(monkeypatch):
    class FakeEncoder:
        def encode(self, texts, **kwargs):
            assert texts == ["ground truth", "model output"]
            assert kwargs == {
                "show_progress_bar": False,
                "convert_to_numpy": True,
            }
            return np.array([[1.0, 0.0], [0.6, 0.8]])

    module = load_hallucination_module(monkeypatch, object)
    monkeypatch.setattr(module, "_get_semantic_model", lambda: FakeEncoder())

    score = module.compute_s_sem("ground truth", "model output")

    assert score == pytest.approx(0.6)
    assert np.isfinite(score)
    assert -1.0 <= score <= 1.0


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_compute_s_sem_rejects_non_finite_embeddings(
    monkeypatch,
    invalid_value,
):
    class FakeEncoder:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, invalid_value], [0.0, 1.0]])

    module = load_hallucination_module(monkeypatch, object)
    monkeypatch.setattr(module, "_get_semantic_model", lambda: FakeEncoder())

    with pytest.raises(RuntimeError, match="non-finite embedding values"):
        module.compute_s_sem("ground truth", "model output")


def test_compute_s_sem_rejects_malformed_embedding_shape(monkeypatch):
    class FakeEncoder:
        def encode(self, texts, **kwargs):
            return np.array([1.0, 2.0])

    module = load_hallucination_module(monkeypatch, object)
    monkeypatch.setattr(module, "_get_semantic_model", lambda: FakeEncoder())

    with pytest.raises(RuntimeError, match="invalid embedding shape"):
        module.compute_s_sem("ground truth", "model output")
