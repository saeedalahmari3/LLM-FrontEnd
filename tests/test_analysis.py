import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import analysis


def test_compute_bleu_score_is_case_insensitive_and_smoothed():
    reference = "The clean answer has exactly five words."

    assert analysis.compute_bleu_score(reference, reference.lower()) == pytest.approx(1.0)

    unrelated_score = analysis.compute_bleu_score(
        reference,
        "Completely unrelated tokens appear over here.",
    )
    assert 0.0 < unrelated_score < 0.1
    assert analysis.compute_bleu_score(reference, "") == 0.0


def test_metric_statistics_ignore_non_finite_values():
    dataframe = pd.DataFrame(
        {
            "metric": ["1", "3", float("inf"), float("-inf"), None],
            "empty_metric": [None, None, None, None, None],
        }
    )

    metric_values = analysis.collect_metric_values(
        dataframe,
        ("metric", "empty_metric"),
    )
    averages, standard_deviations = analysis.compute_metric_statistics(
        metric_values
    )

    assert metric_values == {"metric": [1.0, 3.0], "empty_metric": []}
    assert averages["metric"] == 2.0
    assert standard_deviations["metric"] == 1.0
    assert math.isnan(averages["empty_metric"])
    assert math.isnan(standard_deviations["empty_metric"])


def test_boxplot_paths_and_all_empty_plot(tmp_path):
    csv_path = tmp_path / "responses.csv"
    csv_path.touch()

    plot_path = analysis.resolve_boxplot_path(
        csv_path,
        tmp_path / "nested" / "combined_metrics",
    )
    assert plot_path == tmp_path / "nested" / "combined_metrics.png"
    assert plot_path.parent.is_dir()

    analysis.save_metric_boxplot(
        {"metric_one": [], "metric_two": []},
        plot_path,
    )
    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="cannot overwrite"):
        analysis.resolve_boxplot_path(csv_path, csv_path)
    with pytest.raises(ValueError, match=".png or .pdf"):
        analysis.resolve_boxplot_path(csv_path, tmp_path / "plot.txt")


def test_compute_text_metrics_uses_each_response_as_claim_and_candidate(
    monkeypatch,
):
    class FakeBertScorer:
        def __init__(self):
            self.calls = []
            self.outputs = iter(([0.71, 0.72], [0.73], [0.74, 0.75]))

        def score(self, *, cands, refs, batch_size):
            self.calls.append((cands, refs, batch_size))
            scores = next(self.outputs)
            return [0.0] * len(scores), [0.0] * len(scores), scores

    class FakeAlignScorer:
        def __init__(self):
            self.calls = []
            self.outputs = iter(([0.81, 0.82], [0.83], [0.84, 0.85]))

        def score(self, *, contexts, claims):
            self.calls.append((contexts, claims))
            return next(self.outputs)

    bert_scorer = FakeBertScorer()
    align_scorer = FakeAlignScorer()
    monkeypatch.setattr(
        analysis,
        "create_bertscore_scorer",
        lambda *args: bert_scorer,
    )
    monkeypatch.setattr(
        analysis,
        "create_alignscore_scorer",
        lambda *args: align_scorer,
    )

    clean_responses = ["clean one", "clean two"]
    clean_labels = ["label one", "label two"]
    center_responses = ["center one"]
    center_labels = ["center label one"]
    center_clean_responses = ["center clean one", "center clean two"]
    center_clean_references = ["clean ref one", "clean ref two"]
    scores = analysis.compute_text_metrics(
        clean_responses=clean_responses,
        clean_labels=clean_labels,
        center_responses=center_responses,
        center_labels=center_labels,
        center_clean_responses=center_clean_responses,
        center_clean_references=center_clean_references,
        alignscore_checkpoint="unused.ckpt",
        alignscore_model="roberta-base",
        bertscore_model=None,
        batch_size=8,
        device="cpu",
    )

    assert bert_scorer.calls == [
        (clean_responses, clean_labels, 8),
        (center_responses, center_labels, 8),
        (center_clean_responses, center_clean_references, 8),
    ]
    assert align_scorer.calls == [
        (clean_labels, clean_responses),
        (center_labels, center_responses),
        (center_clean_references, center_clean_responses),
    ]
    assert scores == {
        analysis.BERTSCORE_CLEAN_LABEL_COLUMN: [0.71, 0.72],
        analysis.BERTSCORE_CENTER_LABEL_COLUMN: [0.73],
        analysis.BERTSCORE_CENTER_CLEAN_COLUMN: [0.74, 0.75],
        analysis.ALIGNSCORE_CLEAN_LABEL_COLUMN: [0.81, 0.82],
        analysis.ALIGNSCORE_CENTER_LABEL_COLUMN: [0.83],
        analysis.ALIGNSCORE_CENTER_CLEAN_COLUMN: [0.84, 0.85],
    }


def test_analyze_csv_writes_requested_metrics_and_handles_missing_pairs(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "responses.csv"
    dataframe = pd.DataFrame(
        {
            analysis.RESPONSE_COLUMN: [
                str(["alpha beta gamma delta epsilon", "perturbed response"]),
                str(["red blue green yellow white", "perturbed response"]),
                str(["", "perturbed response"]),
            ],
            analysis.CENTER_RESPONSE_COLUMN: [
                "alpha beta gamma delta",
                None,
                "north south east west center",
            ],
            analysis.LABEL_COLUMN: [
                "alpha beta gamma delta epsilon zeta",
                "red blue green yellow black",
                "north south east west center",
            ],
        }
    )
    dataframe.to_csv(csv_path, index=False)

    hallucination_calls = []

    def fake_hallucination(reference, candidate):
        hallucination_calls.append((reference, candidate))
        return 0.1 if candidate.startswith("alpha") else 0.2

    def fake_text_metrics(**kwargs):
        assert kwargs["clean_responses"] == [
            "alpha beta gamma delta epsilon",
            "red blue green yellow white",
        ]
        assert kwargs["clean_labels"] == [
            "alpha beta gamma delta epsilon zeta",
            "red blue green yellow black",
        ]
        assert kwargs["center_responses"] == [
            "alpha beta gamma delta",
            "north south east west center",
        ]
        assert kwargs["center_labels"] == [
            "alpha beta gamma delta epsilon zeta",
            "north south east west center",
        ]
        assert kwargs["center_clean_responses"] == [
            "alpha beta gamma delta",
        ]
        assert kwargs["center_clean_references"] == [
            "alpha beta gamma delta epsilon",
        ]
        return {
            analysis.BERTSCORE_CLEAN_LABEL_COLUMN: [0.31, 0.32],
            analysis.BERTSCORE_CENTER_LABEL_COLUMN: [0.33, 0.34],
            analysis.BERTSCORE_CENTER_CLEAN_COLUMN: [0.35],
            analysis.ALIGNSCORE_CLEAN_LABEL_COLUMN: [0.41, 0.42],
            analysis.ALIGNSCORE_CENTER_LABEL_COLUMN: [0.43, 0.44],
            analysis.ALIGNSCORE_CENTER_CLEAN_COLUMN: [0.45],
        }

    monkeypatch.setattr(
        analysis,
        "compute_hallucination_score",
        fake_hallucination,
    )
    real_bleu = analysis.compute_bleu_score
    bleu_calls = []

    def recording_bleu(reference, candidate):
        bleu_calls.append((reference, candidate))
        return real_bleu(reference, candidate)

    monkeypatch.setattr(analysis, "compute_bleu_score", recording_bleu)
    monkeypatch.setattr(analysis, "compute_text_metrics", fake_text_metrics)

    (
        updated_path,
        averages,
        standard_deviations,
        boxplot_path,
    ) = analysis.analyze_csv(
        csv_path,
        alignscore_checkpoint="unused.ckpt",
        batch_size=2,
        device="cpu",
    )
    updated = pd.read_csv(updated_path)

    requested_columns = {
        analysis.CLEAN_HALLUCINATION_COLUMN,
        analysis.HALLUCINATION_CENTER_LABEL_COLUMN,
        analysis.HALLUCINATION_CENTER_CLEAN_COLUMN,
        analysis.BERTSCORE_CLEAN_LABEL_COLUMN,
        analysis.BERTSCORE_CENTER_LABEL_COLUMN,
        analysis.BERTSCORE_CENTER_CLEAN_COLUMN,
        analysis.ALIGNSCORE_CLEAN_LABEL_COLUMN,
        analysis.ALIGNSCORE_CENTER_LABEL_COLUMN,
        analysis.ALIGNSCORE_CENTER_CLEAN_COLUMN,
        analysis.BLEU_CLEAN_LABEL_COLUMN,
        analysis.BLEU_CENTER_CLEAN_COLUMN,
        analysis.BLEU_CENTER_LABEL_COLUMN,
    }
    assert requested_columns.issubset(updated.columns)
    assert set(averages) == requested_columns
    assert set(standard_deviations) == requested_columns
    assert boxplot_path == csv_path.with_name("responses_metrics_boxplot.png")
    assert boxplot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert hallucination_calls == [
        (
            "alpha beta gamma delta epsilon zeta",
            "alpha beta gamma delta epsilon",
        ),
        (
            "alpha beta gamma delta epsilon zeta",
            "alpha beta gamma delta",
        ),
        (
            "alpha beta gamma delta epsilon",
            "alpha beta gamma delta",
        ),
        (
            "red blue green yellow black",
            "red blue green yellow white",
        ),
        (
            "north south east west center",
            "north south east west center",
        ),
    ]
    assert bleu_calls == [
        (
            "alpha beta gamma delta epsilon zeta",
            "alpha beta gamma delta epsilon",
        ),
        (
            "alpha beta gamma delta epsilon zeta",
            "alpha beta gamma delta",
        ),
        (
            "alpha beta gamma delta epsilon",
            "alpha beta gamma delta",
        ),
        (
            "red blue green yellow black",
            "red blue green yellow white",
        ),
        (
            "north south east west center",
            "north south east west center",
        ),
    ]

    assert updated.loc[0, analysis.CLEAN_HALLUCINATION_COLUMN] == pytest.approx(0.1)
    assert updated.loc[1, analysis.CLEAN_HALLUCINATION_COLUMN] == pytest.approx(0.2)
    assert math.isnan(updated.loc[2, analysis.CLEAN_HALLUCINATION_COLUMN])
    assert updated.loc[0, analysis.HALLUCINATION_CENTER_LABEL_COLUMN] == pytest.approx(0.1)
    assert math.isnan(updated.loc[1, analysis.HALLUCINATION_CENTER_LABEL_COLUMN])
    assert updated.loc[2, analysis.HALLUCINATION_CENTER_LABEL_COLUMN] == pytest.approx(0.2)
    assert updated.loc[0, analysis.HALLUCINATION_CENTER_CLEAN_COLUMN] == pytest.approx(0.1)
    assert math.isnan(updated.loc[1, analysis.HALLUCINATION_CENTER_CLEAN_COLUMN])
    assert math.isnan(updated.loc[2, analysis.HALLUCINATION_CENTER_CLEAN_COLUMN])
    assert updated.loc[0, analysis.BERTSCORE_CENTER_LABEL_COLUMN] == pytest.approx(0.33)
    assert math.isnan(updated.loc[1, analysis.BERTSCORE_CENTER_LABEL_COLUMN])
    assert updated.loc[2, analysis.BERTSCORE_CENTER_LABEL_COLUMN] == pytest.approx(0.34)
    assert updated.loc[0, analysis.BERTSCORE_CENTER_CLEAN_COLUMN] == pytest.approx(0.35)
    assert math.isnan(updated.loc[1, analysis.BERTSCORE_CENTER_CLEAN_COLUMN])
    assert math.isnan(updated.loc[2, analysis.BERTSCORE_CENTER_CLEAN_COLUMN])
    assert updated.loc[0, analysis.ALIGNSCORE_CENTER_LABEL_COLUMN] == pytest.approx(0.43)
    assert math.isnan(updated.loc[1, analysis.ALIGNSCORE_CENTER_LABEL_COLUMN])
    assert updated.loc[2, analysis.ALIGNSCORE_CENTER_LABEL_COLUMN] == pytest.approx(0.44)
    assert updated.loc[0, analysis.ALIGNSCORE_CENTER_CLEAN_COLUMN] == pytest.approx(0.45)
    assert math.isnan(updated.loc[1, analysis.ALIGNSCORE_CENTER_CLEAN_COLUMN])
    assert math.isnan(updated.loc[2, analysis.ALIGNSCORE_CENTER_CLEAN_COLUMN])
    assert updated.loc[0, analysis.BLEU_CLEAN_LABEL_COLUMN] == pytest.approx(
        math.exp(-0.2)
    )
    assert not math.isnan(updated.loc[1, analysis.BLEU_CLEAN_LABEL_COLUMN])
    assert math.isnan(updated.loc[2, analysis.BLEU_CLEAN_LABEL_COLUMN])
    assert updated.loc[0, analysis.BLEU_CENTER_CLEAN_COLUMN] == pytest.approx(
        math.exp(-0.25)
    )
    assert math.isnan(updated.loc[1, analysis.BLEU_CENTER_CLEAN_COLUMN])
    assert math.isnan(updated.loc[2, analysis.BLEU_CENTER_CLEAN_COLUMN])
    assert updated.loc[0, analysis.BLEU_CENTER_LABEL_COLUMN] == pytest.approx(
        math.exp(-0.5)
    )
    assert math.isnan(updated.loc[1, analysis.BLEU_CENTER_LABEL_COLUMN])
    assert updated.loc[2, analysis.BLEU_CENTER_LABEL_COLUMN] == pytest.approx(1.0)
    assert averages[analysis.CLEAN_HALLUCINATION_COLUMN] == pytest.approx(0.15)
    assert averages[analysis.HALLUCINATION_CENTER_LABEL_COLUMN] == pytest.approx(
        0.15
    )
    assert averages[analysis.HALLUCINATION_CENTER_CLEAN_COLUMN] == pytest.approx(
        0.1
    )
    assert averages[analysis.BERTSCORE_CENTER_LABEL_COLUMN] == pytest.approx(
        0.335
    )
    assert averages[analysis.BERTSCORE_CENTER_CLEAN_COLUMN] == pytest.approx(
        0.35
    )
    assert averages[analysis.ALIGNSCORE_CENTER_LABEL_COLUMN] == pytest.approx(
        0.435
    )
    assert averages[analysis.ALIGNSCORE_CENTER_CLEAN_COLUMN] == pytest.approx(
        0.45
    )
    expected_standard_deviations = {
        analysis.CLEAN_HALLUCINATION_COLUMN: 0.05,
        analysis.HALLUCINATION_CENTER_LABEL_COLUMN: 0.05,
        analysis.HALLUCINATION_CENTER_CLEAN_COLUMN: 0.0,
        analysis.BERTSCORE_CLEAN_LABEL_COLUMN: 0.005,
        analysis.BERTSCORE_CENTER_LABEL_COLUMN: 0.005,
        analysis.BERTSCORE_CENTER_CLEAN_COLUMN: 0.0,
        analysis.ALIGNSCORE_CLEAN_LABEL_COLUMN: 0.005,
        analysis.ALIGNSCORE_CENTER_LABEL_COLUMN: 0.005,
        analysis.ALIGNSCORE_CENTER_CLEAN_COLUMN: 0.0,
        analysis.BLEU_CLEAN_LABEL_COLUMN: abs(
            math.exp(-0.2) - (0.2 ** 0.25)
        ) / 2,
        analysis.BLEU_CENTER_CLEAN_COLUMN: 0.0,
        analysis.BLEU_CENTER_LABEL_COLUMN: (1 - math.exp(-0.5)) / 2,
    }
    for metric_name, expected_std in expected_standard_deviations.items():
        assert standard_deviations[metric_name] == pytest.approx(expected_std)


def test_analyze_csv_computes_center_label_when_all_clean_responses_are_missing(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "center_only.csv"
    pd.DataFrame(
        {
            analysis.RESPONSE_COLUMN: [str([""])],
            analysis.CENTER_RESPONSE_COLUMN: ["center response text"],
            analysis.LABEL_COLUMN: ["ground truth text"],
        }
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        analysis,
        "compute_hallucination_score",
        lambda reference, candidate: 0.25,
    )

    def fake_text_metrics(**kwargs):
        assert kwargs["clean_responses"] == []
        assert kwargs["clean_labels"] == []
        assert kwargs["center_responses"] == ["center response text"]
        assert kwargs["center_labels"] == ["ground truth text"]
        assert kwargs["center_clean_responses"] == []
        assert kwargs["center_clean_references"] == []
        return {
            analysis.BERTSCORE_CENTER_LABEL_COLUMN: [0.55],
            analysis.ALIGNSCORE_CENTER_LABEL_COLUMN: [0.65],
        }

    monkeypatch.setattr(analysis, "compute_text_metrics", fake_text_metrics)

    (
        updated_path,
        averages,
        standard_deviations,
        boxplot_path,
    ) = analysis.analyze_csv(
        csv_path,
        alignscore_checkpoint="unused.ckpt",
        device="cpu",
    )
    updated = pd.read_csv(updated_path)

    assert updated.loc[0, analysis.HALLUCINATION_CENTER_LABEL_COLUMN] == 0.25
    assert updated.loc[0, analysis.BERTSCORE_CENTER_LABEL_COLUMN] == 0.55
    assert updated.loc[0, analysis.ALIGNSCORE_CENTER_LABEL_COLUMN] == 0.65
    assert boxplot_path.is_file()
    unavailable_metrics = (
        analysis.CLEAN_HALLUCINATION_COLUMN,
        analysis.BERTSCORE_CLEAN_LABEL_COLUMN,
        analysis.ALIGNSCORE_CLEAN_LABEL_COLUMN,
        analysis.BLEU_CLEAN_LABEL_COLUMN,
        analysis.HALLUCINATION_CENTER_CLEAN_COLUMN,
        analysis.BERTSCORE_CENTER_CLEAN_COLUMN,
        analysis.ALIGNSCORE_CENTER_CLEAN_COLUMN,
        analysis.BLEU_CENTER_CLEAN_COLUMN,
    )
    for metric_name in unavailable_metrics:
        assert math.isnan(updated.loc[0, metric_name])
        assert math.isnan(averages[metric_name])
        assert math.isnan(standard_deviations[metric_name])
    assert standard_deviations[analysis.HALLUCINATION_CENTER_LABEL_COLUMN] == 0.0
    assert standard_deviations[analysis.BERTSCORE_CENTER_LABEL_COLUMN] == 0.0
    assert standard_deviations[analysis.ALIGNSCORE_CENTER_LABEL_COLUMN] == 0.0
    assert standard_deviations[analysis.BLEU_CENTER_LABEL_COLUMN] == 0.0


def test_main_prints_standard_deviation_next_to_average(monkeypatch, capsys):
    monkeypatch.setattr(
        analysis,
        "parse_args",
        lambda: SimpleNamespace(
            batch_size=4,
            csv_file="responses.csv",
            alignscore_checkpoint="alignscore.ckpt",
            alignscore_model="roberta-base",
            bertscore_model=None,
            device="cpu",
            boxplot_file=None,
        ),
    )
    monkeypatch.setattr(
        analysis,
        "analyze_csv",
        lambda *args, **kwargs: (
            Path("responses.csv"),
            {"metric_name": 0.25},
            {"metric_name": 0.05},
            Path("responses_metrics_boxplot.png"),
        ),
    )

    analysis.main()

    output = capsys.readouterr().out
    assert "metric_name: average=0.250000, std=0.050000" in output
    assert "Combined metric box plot: responses_metrics_boxplot.png" in output


@pytest.mark.parametrize("response_count", [9, 19, 39])
def test_analyze_response_positions_writes_means_without_changing_source(
    tmp_path,
    monkeypatch,
    response_count,
):
    csv_path = tmp_path / "responses.csv"
    first_row_responses = [f"a-{position}" for position in range(response_count)]
    second_row_responses = [
        "b-0",
        "",
        "b-2",
        "b-3",
        "b-4",
        "b-5",
        "b-6",
        "b-7",
    ]
    center_responses = ["center-a", "center-b"]
    pd.DataFrame(
        {
            analysis.RESPONSE_COLUMN: [
                str(first_row_responses),
                str(second_row_responses),
            ],
            analysis.CENTER_RESPONSE_COLUMN: center_responses,
            analysis.LABEL_COLUMN: ["label a", "label b"],
            "existing_output": ["keep a", "keep b"],
        }
    ).to_csv(csv_path, index=False)
    source_before = csv_path.read_bytes()

    base_scores = {
        **{
            f"a-{position}": (position + 1) / 100
            for position in range(response_count)
        },
        **{
            f"b-{position}": (position + 11) / 100
            for position in (0, 2, 3, 4, 5, 6, 7)
        },
        "center-a": 0.21,
        "center-b": 0.31,
    }
    hallucination_calls = []
    bleu_calls = []

    def fake_hallucination(reference, candidate):
        hallucination_calls.append((reference, candidate))
        return base_scores[candidate]

    def fake_bleu(reference, candidate):
        bleu_calls.append((reference, candidate))
        return base_scores[candidate] + 0.6

    class FakeBertScorer:
        def __init__(self):
            self.calls = []

        def score(self, *, cands, refs, batch_size):
            self.calls.extend(zip(refs, cands))
            scores = [base_scores[candidate] + 0.2 for candidate in cands]
            return [0.0] * len(scores), [0.0] * len(scores), scores

    class FakeAlignScorer:
        def __init__(self):
            self.calls = []

        def score(self, *, contexts, claims):
            self.calls.extend(zip(contexts, claims))
            return [base_scores[claim] + 0.4 for claim in claims]

    bert_scorer = FakeBertScorer()
    align_scorer = FakeAlignScorer()
    bert_factory_calls = []
    align_factory_calls = []

    def fake_bert_factory(model_name, batch_size, device):
        bert_factory_calls.append((model_name, batch_size, device))
        return bert_scorer

    def fake_align_factory(checkpoint_path, model_name, batch_size, device):
        align_factory_calls.append(
            (checkpoint_path, model_name, batch_size, device)
        )
        return align_scorer

    monkeypatch.setattr(
        analysis,
        "compute_hallucination_score",
        fake_hallucination,
    )
    monkeypatch.setattr(analysis, "compute_bleu_score", fake_bleu)
    monkeypatch.setattr(
        analysis,
        "create_bertscore_scorer",
        fake_bert_factory,
    )
    monkeypatch.setattr(
        analysis,
        "create_alignscore_scorer",
        fake_align_factory,
    )

    output_path = analysis.analyze_response_positions(
        csv_path,
        alignscore_checkpoint="unused.ckpt",
        alignscore_model="roberta-large",
        bertscore_model="bert-model",
        batch_size=4,
        device="cpu",
    )

    assert output_path == tmp_path / "responses_response_position_metrics.csv"
    assert csv_path.read_bytes() == source_before
    summary = pd.read_csv(output_path)
    assert list(summary.columns) == [
        "response_position",
        "mean_hallucination_score",
        "mean_bertscore_f1",
        "mean_alignscore",
        "mean_bleu",
    ]
    assert summary["response_position"].tolist() == [
        "C",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        *(f"P{position}" for position in range(9, response_count)),
        "Center",
    ]

    # The blank P1 in row two and absent P8 are excluded from their means.
    expected_base_means = [
        0.06,
        0.02,
        0.08,
        0.09,
        0.10,
        0.11,
        0.12,
        0.13,
        0.09,
        *((position + 1) / 100 for position in range(9, response_count)),
        0.26,
    ]
    assert summary["mean_hallucination_score"].tolist() == pytest.approx(
        expected_base_means
    )
    assert summary["mean_bertscore_f1"].tolist() == pytest.approx(
        [score + 0.2 for score in expected_base_means]
    )
    assert summary["mean_alignscore"].tolist() == pytest.approx(
        [score + 0.4 for score in expected_base_means]
    )
    assert summary["mean_bleu"].tolist() == pytest.approx(
        [score + 0.6 for score in expected_base_means]
    )

    expected_pairs = [
        *(("label a", response) for response in first_row_responses),
        *(
            ("label b", response)
            for response in second_row_responses
            if response
        ),
        ("label a", "center-a"),
        ("label b", "center-b"),
    ]
    assert sorted(hallucination_calls) == sorted(expected_pairs)
    assert sorted(bleu_calls) == sorted(expected_pairs)
    assert sorted(bert_scorer.calls) == sorted(expected_pairs)
    assert sorted(align_scorer.calls) == sorted(expected_pairs)
    assert bert_factory_calls == [("bert-model", 4, "cpu")]
    assert align_factory_calls == [
        ("unused.ckpt", "roberta-large", 4, "cpu")
    ]


def test_analyze_response_positions_rejects_source_as_output(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "responses.csv"
    pd.DataFrame(
        {
            analysis.RESPONSE_COLUMN: [str(["clean response"])],
            analysis.LABEL_COLUMN: ["ground truth"],
        }
    ).to_csv(csv_path, index=False)
    source_before = csv_path.read_bytes()

    monkeypatch.setattr(
        analysis,
        "compute_hallucination_score",
        lambda reference, candidate: 0.1,
    )
    monkeypatch.setattr(
        analysis,
        "compute_bleu_score",
        lambda reference, candidate: 0.2,
    )

    class FakeBertScorer:
        def score(self, *, cands, refs, batch_size):
            return [0.0], [0.0], [0.3]

    class FakeAlignScorer:
        def score(self, *, contexts, claims):
            return [0.4]

    monkeypatch.setattr(
        analysis,
        "create_bertscore_scorer",
        lambda *args: FakeBertScorer(),
    )
    monkeypatch.setattr(
        analysis,
        "create_alignscore_scorer",
        lambda *args: FakeAlignScorer(),
    )

    with pytest.raises(ValueError, match="(?i)(overwrite|separate)"):
        analysis.analyze_response_positions(
            csv_path,
            alignscore_checkpoint="unused.ckpt",
            output_file=csv_path,
            device="cpu",
        )

    assert csv_path.read_bytes() == source_before


@pytest.mark.parametrize(
    "response, expected",
    [
        ("1. Did Not Happen: explanation", "did not happen"),
        ("  **2. FAR FUTURE** because...", "far future"),
        ("3) nonsense. explanation", "nonsense"),
        ('"obscure" unknown information', "obscure"),
        ("4. obscure", "obscure"),
        ("I think obscure", None),
        ("obscurely worded", None),
        ("2", None),
        ("", None),
    ],
)
def test_extract_mbzuai_category(response, expected):
    from category_metrics import extract_category

    assert extract_category(response) == expected


@pytest.mark.parametrize("dataset_name", ["mbzuai", None])
def test_mbzuai_accuracy_saved_in_row_and_position_metrics(
    tmp_path, monkeypatch, dataset_name,
):
    csv_path = tmp_path / "mbzuai.csv"
    pd.DataFrame({
        "dataset_name": ["MBZUAI/LaMini-Hallucination"] * 3,
        "category": ["did not happen", "far future", "obscure"],
        "label_gt": ["reference"] * 3,
        "LLM_response": [
            str(["1. did not happen explanation", "nonsense"]),
            str(["2. FAR FUTURE explanation", "far future"]),
            str(["I think obscure", ""]),
        ],
        "center_response": ["did not happen", "nonsense", ""],
    }).to_csv(csv_path, index=False)
    monkeypatch.setattr(analysis, "compute_hallucination_score", lambda *a: 0.1)
    monkeypatch.setattr(analysis, "compute_bleu_score", lambda *a: 0.2)
    monkeypatch.setattr(analysis, "save_metric_boxplot", lambda *a: None)

    def fake_text_metrics(**kwargs):
        return {
            analysis.BERTSCORE_CLEAN_LABEL_COLUMN: [0.3] * len(kwargs['clean_responses']),
            analysis.ALIGNSCORE_CLEAN_LABEL_COLUMN: [0.4] * len(kwargs['clean_responses']),
        }

    monkeypatch.setattr(analysis, "compute_text_metrics", fake_text_metrics)
    _, averages, stds, _ = analysis.analyze_csv(
        csv_path, "unused", device="cpu", dataset_name=dataset_name,
    )
    result = pd.read_csv(csv_path)
    assert result[analysis.CLEAN_ACCURACY_COLUMN].tolist() == [1, 1, 0]
    assert result[analysis.CENTER_ACCURACY_COLUMN][:2].tolist() == [1, 0]
    assert math.isnan(result[analysis.CENTER_ACCURACY_COLUMN][2])
    assert averages[analysis.CLEAN_ACCURACY_COLUMN] == pytest.approx(2 / 3)
    assert stds[analysis.CENTER_ACCURACY_COLUMN] == pytest.approx(0.5)
    source_before = csv_path.read_bytes()
    output = analysis.analyze_response_positions(
        csv_path, "unused", device="cpu", dataset_name=dataset_name,
    )
    summary = pd.read_csv(output)
    assert summary['response_position'].tolist() == ['C', 'P1', 'Center']
    assert summary['mean_accuracy'].tolist() == pytest.approx([2 / 3, 0.5, 0.5])
    assert csv_path.read_bytes() == source_before


def test_mbzuai_category_validation_and_other_datasets():
    frame = pd.DataFrame({'label_gt': ['4. Obscure']})
    assert analysis.get_mbzuai_categories(frame, 'mbzuai') == ['obscure']
    frame['label_gt'] = "I don't have an answer for this question"
    with pytest.raises(ValueError, match='Row 2: invalid MBZUAI category'):
        analysis.get_mbzuai_categories(frame, 'mbzuai')
    assert analysis.get_mbzuai_categories(frame, 'imdb') is None


@pytest.mark.parametrize('label,response,expected', [
    ('did_not_happen', 'True: no such event.', 1),
    ('far_future', '**TRUE**: far ahead.', 1),
    ('nonsense', 'The answer is true.', 1),
    ('obscure', 'False', 0),
    ('ordinary', 'false: known answer.', 1),
    ('ordinary', 'True', 0),
    ('nonsense', 'untrue', 0),
    ('ordinary', 'No boolean answer', 0),
    ('nonsense', 'True or False', 0),
])
def test_true_false_accuracy(label, response, expected):
    from category_metrics import compute_true_false_accuracy
    assert compute_true_false_accuracy(label, response) == expected


def test_true_false_accuracy_empty():
    from category_metrics import compute_true_false_accuracy
    for response in (None, '', '   '):
        assert math.isnan(compute_true_false_accuracy('obscure', response))


@pytest.mark.parametrize('category_column', [True, False])
def test_true_false_output_row_and_position_metrics(tmp_path, monkeypatch, category_column):
    csv_path = tmp_path / 'boolean.csv'
    data = {
        'dataset_name': ['mbzuai'] * 3,
        'label_gt': ['did_not_happen', 'ordinary', 'obscure'],
        'LLM_response': [str(['True: never happened', 'False']),
                         str(['False', 'invalid']), str(['False', ''])],
        'center_response': ['True', 'True', ''],
    }
    if category_column:
        data['category'] = data['label_gt']
        data['label_gt'] = ['reference answer'] * 3
    pd.DataFrame(data).to_csv(csv_path, index=False)
    monkeypatch.setattr(analysis, 'compute_hallucination_score', lambda *a: 0.1)
    monkeypatch.setattr(analysis, 'compute_bleu_score', lambda *a: 0.2)
    monkeypatch.setattr(analysis, 'save_metric_boxplot', lambda *a: None)
    monkeypatch.setattr(analysis, 'compute_text_metrics', lambda **kw: {
        analysis.BERTSCORE_CLEAN_LABEL_COLUMN: [0.3] * len(kw['clean_responses']),
        analysis.ALIGNSCORE_CLEAN_LABEL_COLUMN: [0.4] * len(kw['clean_responses']),
    })
    _, averages, _, _ = analysis.analyze_csv(csv_path, 'unused', true_false_output=True)
    result = pd.read_csv(csv_path)
    assert result[analysis.CLEAN_ACCURACY_COLUMN].tolist() == [1, 1, 0]
    assert result[analysis.CENTER_ACCURACY_COLUMN][:2].tolist() == [1, 0]
    assert math.isnan(result[analysis.CENTER_ACCURACY_COLUMN][2])
    assert averages[analysis.CLEAN_ACCURACY_COLUMN] == pytest.approx(2 / 3)
    source_before = csv_path.read_bytes()
    output = analysis.analyze_response_positions(csv_path, 'unused', true_false_output=True)
    assert pd.read_csv(output)['mean_accuracy'].tolist() == pytest.approx([2 / 3, 0, 0.5])
    assert csv_path.read_bytes() == source_before


def test_true_false_output_cli(monkeypatch):
    monkeypatch.setattr('sys.argv', ['analysis.py', '--csv_file', 'input.csv',
                                  '--alignscore-checkpoint', 'unused', '--true-false-output'])
    assert analysis.parse_args().true_false_output is True
