"""Compute response-quality metrics and add them to a result CSV."""
""" How to run:

python analysis.py --csv_file ../results/resultsFor9PercentPeturbation/MBZUAILaMini-Hallucinationsynonyms_responses_output_synonyms_EntireTestSet_MerriamWebster_gemma3_12b_20260903_130001.csv --alignscore-checkpoint ./AlignScore/AlignScore-base.ckpt --response-position-output ../results/resultsFor9PercentPeturbation/per_response_results_MBZUAI.csv
"""
import argparse
import ast
import inspect
import logging
import math
import os
import re
import tempfile
import warnings
from collections import Counter
from pathlib import Path

from category_metrics import compute_true_false_accuracy, compute_category_accuracy, is_mbzuai, normalize_category

import pandas as pd
import torch
from tqdm import tqdm

RESPONSE_COLUMN = "LLM_response"
CENTER_RESPONSE_COLUMN = "center_response"
LABEL_COLUMN = "label_gt"
CLEAN_HALLUCINATION_COLUMN = "hallucination_score_clean_vs_label"
HALLUCINATION_CENTER_LABEL_COLUMN = "hallucination_score_center_vs_label"
HALLUCINATION_CENTER_CLEAN_COLUMN = "hallucination_score_center_vs_clean"
BERTSCORE_CLEAN_LABEL_COLUMN = "bertscore_f1_clean_vs_label"
BERTSCORE_CENTER_LABEL_COLUMN = "bertscore_f1_center_vs_label"
BERTSCORE_CENTER_CLEAN_COLUMN = "bertscore_f1_center_vs_clean"
ALIGNSCORE_CLEAN_LABEL_COLUMN = "alignscore_clean_vs_label"
ALIGNSCORE_CENTER_LABEL_COLUMN = "alignscore_center_vs_label"
ALIGNSCORE_CENTER_CLEAN_COLUMN = "alignscore_center_vs_clean"
BLEU_CLEAN_LABEL_COLUMN = "bleu_clean_vs_label"
BLEU_CENTER_CLEAN_COLUMN = "bleu_center_vs_clean"
BLEU_CENTER_LABEL_COLUMN = "bleu_center_vs_label"
BLEU_MAX_ORDER = 4
BLEU_SMOOTHING_EPSILON = 0.1
BLEU_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
BOXPLOT_SUFFIX = "_metrics_boxplot.png"
CENTER_RESPONSE_POSITION_LABEL = "Center"
RESPONSE_POSITION_COLUMN = "response_position"
MEAN_HALLUCINATION_COLUMN = "mean_hallucination_score"
MEAN_BERTSCORE_COLUMN = "mean_bertscore_f1"
MEAN_ALIGNSCORE_COLUMN = "mean_alignscore"
MEAN_BLEU_COLUMN = "mean_bleu"
RESPONSE_POSITION_OUTPUT_SUFFIX = "_response_position_metrics.csv"


CLEAN_ACCURACY_COLUMN = "accuracy_clean_vs_label"
CENTER_ACCURACY_COLUMN = "accuracy_center_vs_label"
MEAN_ACCURACY_COLUMN = "mean_accuracy"


def get_mbzuai_categories(dataframe, dataset_name):
    """Validate category labels before computing expensive metrics."""
    enabled = is_mbzuai(dataset_name)
    if dataset_name is None and "dataset_name" in dataframe:
        enabled = dataframe["dataset_name"].map(is_mbzuai).all()
    if not enabled:
        return None
    column = "category" if "category" in dataframe else LABEL_COLUMN
    return [
        normalize_category(value, row_number)
        for row_number, value in enumerate(dataframe[column], start=2)
    ]


def compute_hallucination_score(reference, candidate):
    """Load the existing hallucination metric only when it is evaluated."""
    try:
        from hallucination_score import compute_h
    except ImportError as exc:
        raise RuntimeError(
            "The hallucination-score dependencies are not installed. "
            "Install `sentence-transformers`, `transformers`, and "
            "`huggingface-hub` before running this analysis."
        ) from exc
    return float(compute_h(reference, candidate))


def tokenize_for_bleu(text):
    """Tokenize text for case-insensitive, punctuation-aware sentence BLEU."""
    return BLEU_TOKEN_PATTERN.findall(str(text).casefold())


def _get_ngrams(tokens, order):
    """Return the token n-grams used by modified BLEU precision."""
    return Counter(
        tuple(tokens[index:index + order])
        for index in range(len(tokens) - order + 1)
    )


def compute_bleu_score(reference, candidate):
    """Compute smoothed sentence-level BLEU-4 for one candidate/reference pair.

    This is a small, dependency-free implementation of modified n-gram
    precision, brevity penalty, and Chen-Cherry method-1 smoothing. Keeping the
    implementation local avoids corpus downloads and compatibility issues in
    older NLTK versions. The argument order is deliberately
    ``(reference, candidate)`` so all directional comparisons are explicit.
    """
    reference_tokens = tokenize_for_bleu(reference)
    candidate_tokens = tokenize_for_bleu(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    log_precision_sum = 0.0
    weight = 1.0 / BLEU_MAX_ORDER
    for order in range(1, BLEU_MAX_ORDER + 1):
        candidate_ngrams = _get_ngrams(candidate_tokens, order)
        reference_ngrams = _get_ngrams(reference_tokens, order)
        possible_matches = max(1, sum(candidate_ngrams.values()))
        clipped_matches = sum(
            min(count, reference_ngrams[ngram])
            for ngram, count in candidate_ngrams.items()
        )
        if clipped_matches:
            precision = clipped_matches / possible_matches
        else:
            precision = BLEU_SMOOTHING_EPSILON / possible_matches
        log_precision_sum += weight * math.log(precision)

    reference_length = len(reference_tokens)
    candidate_length = len(candidate_tokens)
    brevity_penalty = (
        1.0
        if candidate_length > reference_length
        else math.exp(1.0 - reference_length / candidate_length)
    )
    return float(brevity_penalty * math.exp(log_precision_sum))


def mps_supports_required_ops():
    """Return whether MPS implements operations required by RoBERTa."""
    if not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return False

    try:
        values = torch.ones((1, 2), dtype=torch.long, device="mps")
        torch.cumsum(values, dim=1)
    except (NotImplementedError, RuntimeError):
        return False
    return True


def resolve_device(device):
    """Resolve the requested device without selecting incomplete MPS support."""
    if device != "auto":
        if device == "mps" and not mps_supports_required_ops():
            print(
                "MPS lacks operations required by RoBERTa in this PyTorch "
                "version; using CPU instead."
            )
            return "cpu"
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if mps_supports_required_ops():
        return "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(
            "MPS lacks operations required by RoBERTa in this PyTorch "
            "version; using CPU instead."
        )
    return "cpu"


def create_bertscore_scorer(model_name, batch_size, device):
    """Create a cached BERTScore model with an actionable import error."""
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore is not installed. Install it with "
            "`python -m pip install bert-score`."
        ) from exc

    scorer_options = {
        "lang": "en",
        "batch_size": batch_size,
        "device": device,
    }
    if model_name:
        scorer_options["model_type"] = model_name

    from transformers.utils import logging as transformers_logging

    previous_verbosity = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`resume_download` is deprecated.*",
                category=FutureWarning,
            )
            return BERTScorer(**scorer_options)
    finally:
        transformers_logging.set_verbosity(previous_verbosity)


def create_alignscore_scorer(
    checkpoint_path,
    model_name,
    batch_size,
    device,
):
    """Create AlignScore using an explicitly supplied local checkpoint."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            from alignscore import AlignScore
    except ImportError as exc:
        raise RuntimeError(
            "AlignScore is not installed. Install it from the official "
            "repository and install its spaCy `en_core_web_sm` model."
        ) from exc

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"AlignScore checkpoint not found: {checkpoint}"
        )

    from transformers.utils import logging as transformers_logging

    previous_verbosity = transformers_logging.get_verbosity()
    lightning_logger = logging.getLogger(
        "pytorch_lightning.utilities.migration.utils"
    )
    previous_lightning_level = lightning_logger.level
    try:
        transformers_logging.set_verbosity_error()
        lightning_logger.setLevel(logging.WARNING)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`resume_download` is deprecated.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Lightning automatically upgraded your loaded checkpoint.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Found keys that are not in the model state dict but in "
                    "the checkpoint.*"
                ),
            )
            scorer = AlignScore(
                model=model_name,
                batch_size=batch_size,
                device=device,
                ckpt_path=str(checkpoint),
                evaluation_mode="nli_sp",
                verbose=False,
            )

            # AlignScore already loads spaCy, so use it for sentence splitting
            # instead of requiring a separate NLTK punkt_tab data download.
            from alignscore import inference as alignscore_inference

            def spacy_sent_tokenize(text):
                return [
                    sentence.text
                    for sentence in scorer.model.spacy(text).sents
                ]

            alignscore_inference.sent_tokenize = spacy_sent_tokenize
            return scorer
    finally:
        lightning_logger.setLevel(previous_lightning_level)
        transformers_logging.set_verbosity(previous_verbosity)


def to_float_list(scores):
    """Convert a tensor, NumPy array, or sequence of scores to floats."""
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().tolist()
    elif hasattr(scores, "tolist"):
        scores = scores.tolist()
    return [float(score) for score in scores]


def compute_text_metrics(
    clean_responses,
    clean_labels,
    center_responses,
    center_labels,
    center_clean_responses,
    center_clean_references,
    alignscore_checkpoint,
    alignscore_model,
    bertscore_model,
    batch_size,
    device,
):
    """Compute BERTScore F1 and AlignScore for all requested text pairs."""
    comparisons = (
        (
            BERTSCORE_CLEAN_LABEL_COLUMN,
            ALIGNSCORE_CLEAN_LABEL_COLUMN,
            clean_responses,
            clean_labels,
        ),
        (
            BERTSCORE_CENTER_LABEL_COLUMN,
            ALIGNSCORE_CENTER_LABEL_COLUMN,
            center_responses,
            center_labels,
        ),
        (
            BERTSCORE_CENTER_CLEAN_COLUMN,
            ALIGNSCORE_CENTER_CLEAN_COLUMN,
            center_clean_responses,
            center_clean_references,
        ),
    )
    for _, _, candidates, references in comparisons:
        if len(candidates) != len(references):
            raise ValueError(
                "Every BERTScore/AlignScore candidate list must have a "
                "reference of the same length."
            )

    scores = {}
    bert_scorer = create_bertscore_scorer(
        bertscore_model,
        batch_size,
        device,
    )
    for bert_column, _, candidates, references in comparisons:
        if not candidates:
            continue
        _, _, f1_scores = bert_scorer.score(
            cands=candidates,
            refs=references,
            batch_size=batch_size,
        )
        scores[bert_column] = to_float_list(f1_scores)
    del bert_scorer

    # Load the two large scorers sequentially instead of retaining both model
    # instances at once. This substantially reduces peak memory on CPU hosts.
    align_scorer = create_alignscore_scorer(
        alignscore_checkpoint,
        alignscore_model,
        batch_size,
        device,
    )

    # AlignScore is directional: the reference context supports (or
    # contradicts) the candidate claim.
    for _, align_column, candidates, references in comparisons:
        if not candidates:
            continue
        align_scores = align_scorer.score(
            contexts=references,
            claims=candidates,
        )
        scores[align_column] = to_float_list(align_scores)

    return scores


def get_first_response(value, row_number):
    """Extract the first response, returning None when it is empty."""
    if value is None or (
        not isinstance(value, (list, tuple, str)) and pd.isna(value)
    ):
        return None

    if isinstance(value, (list, tuple)):
        responses = value
    elif isinstance(value, str):
        if not value.strip():
            return None
        try:
            responses = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Row {row_number}: {RESPONSE_COLUMN!r} is not a valid list."
            ) from exc
    else:
        raise ValueError(
            f"Row {row_number}: {RESPONSE_COLUMN!r} must contain a list."
        )

    if not isinstance(responses, (list, tuple)):
        raise ValueError(
            f"Row {row_number}: {RESPONSE_COLUMN!r} must contain a list."
        )
    if not responses:
        return None

    first_response = responses[0]
    if pd.isna(first_response) or not str(first_response).strip():
        return None

    return str(first_response)


def get_response_positions(value, row_number):
    """Parse and normalize all clean and perturbed responses in a row."""
    if value is None or (
        not isinstance(value, (list, tuple, str)) and pd.isna(value)
    ):
        return []

    if isinstance(value, (list, tuple)):
        responses = value
    elif isinstance(value, str):
        if not value.strip():
            return []
        try:
            responses = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Row {row_number}: {RESPONSE_COLUMN!r} is not a valid list."
            ) from exc
    else:
        raise ValueError(
            f"Row {row_number}: {RESPONSE_COLUMN!r} must contain a list."
        )

    if not isinstance(responses, (list, tuple)):
        raise ValueError(
            f"Row {row_number}: {RESPONSE_COLUMN!r} must contain a list."
        )
    normalized = [None] * len(responses)
    for position_index, response in enumerate(responses):
        try:
            response_is_missing = pd.isna(response)
            if response_is_missing:
                continue
        except (TypeError, ValueError):
            position_name = "C" if position_index == 0 else f"P{position_index}"
            raise ValueError(
                f"Row {row_number}: response {position_name} must be text."
            ) from None

        response_text = str(response)
        if response_text.strip():
            normalized[position_index] = response_text
    return normalized


def save_csv_in_place(dataframe, csv_path):
    """Replace the source CSV only after the updated file is written safely."""
    temporary_path = None
    original_mode = csv_path.stat().st_mode
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f".{csv_path.stem}_",
            dir=csv_path.parent,
            delete=False,
            encoding="utf-8",
            newline="",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            dataframe.to_csv(temporary_file, index=False)

        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, csv_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def collect_metric_values(dataframe, metric_names):
    """Return finite numeric values for each metric, preserving name order."""
    metric_values = {}
    for metric_name in metric_names:
        values = pd.to_numeric(
            dataframe[metric_name],
            errors="coerce",
        ).dropna()
        values = values[values.map(math.isfinite)]
        metric_values[metric_name] = values.astype(float).tolist()
    return metric_values


def compute_metric_statistics(metric_values):
    """Return per-metric means and population standard deviations."""
    averages = {}
    standard_deviations = {}
    for metric_name, values in metric_values.items():
        if values:
            series = pd.Series(values, dtype=float)
            averages[metric_name] = float(series.mean())
            standard_deviations[metric_name] = float(series.std(ddof=0))
        else:
            averages[metric_name] = float("nan")
            standard_deviations[metric_name] = float("nan")
    return averages, standard_deviations


def resolve_boxplot_path(csv_path, boxplot_file=None):
    """Resolve a safe output path for the combined metric box plot."""
    if boxplot_file is None:
        plot_path = csv_path.with_name(f"{csv_path.stem}{BOXPLOT_SUFFIX}")
    else:
        plot_path = Path(boxplot_file).expanduser()
        if not plot_path.suffix:
            plot_path = plot_path.with_suffix(".png")
        plot_path = plot_path.resolve()

    if plot_path == csv_path:
        raise ValueError("The box-plot path cannot overwrite the source CSV.")
    if plot_path.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("The box plot must use a .png or .pdf extension.")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    return plot_path


def save_metric_boxplot(metric_values, plot_path):
    """Save every metric distribution in one horizontal box-plot figure."""
    cache_root = Path(tempfile.gettempdir()) / "llmfrontend-plot-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to create the metric box plot. Install "
            "it with `python -m pip install matplotlib`."
        ) from exc

    metric_names = list(metric_values)
    if not metric_names:
        raise ValueError("At least one metric is required for the box plot.")

    figure = Figure(
        figsize=(14, max(7, 0.55 * len(metric_names))),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    boxplot_options = {
        "vert": False,
        "patch_artist": True,
        "showmeans": True,
        "meanprops": {
            "marker": "D",
            "markerfacecolor": "#D62728",
            "markeredgecolor": "#D62728",
            "markersize": 4,
        },
        "medianprops": {"color": "#FF7F0E", "linewidth": 1.5},
    }
    label_parameter = (
        "tick_labels"
        if "tick_labels" in inspect.signature(axis.boxplot).parameters
        else "labels"
    )
    boxplot_options[label_parameter] = metric_names
    artists = axis.boxplot(
        [metric_values[metric_name] for metric_name in metric_names],
        **boxplot_options,
    )
    for box in artists["boxes"]:
        box.set_facecolor("#4C78A8")
        box.set_alpha(0.65)

    axis.set_title("Distribution of Evaluation Metrics")
    axis.set_xlabel("Score")
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.tick_params(axis="y", labelsize=8)
    if not any(metric_values.values()):
        axis.text(
            0.5,
            0.5,
            "No finite metric values available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    figure.savefig(plot_path, dpi=300)
    figure.clear()
    return plot_path


def analyze_csv(
    csv_file,
    alignscore_checkpoint,
    alignscore_model="roberta-base",
    bertscore_model=None,
    batch_size=32,
    device="auto",
    boxplot_file=None,
    dataset_name=None,
    true_false_output=False,
):
    """Score rows, update the CSV, save a box plot, and return metric stats."""
    csv_path = Path(csv_file).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    boxplot_path = resolve_boxplot_path(csv_path, boxplot_file)

    # Preserve literal labels such as "N/A"; only truly empty strings are
    # treated as missing by the validation below.
    dataframe = pd.read_csv(csv_path, keep_default_na=False)
    missing_columns = {
        RESPONSE_COLUMN,
        CENTER_RESPONSE_COLUMN,
        LABEL_COLUMN,
    }.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required column(s): {missing}")
    if dataframe.empty:
        raise ValueError("CSV contains no data rows.")

    categories = (
        dataframe["category" if "category" in dataframe else LABEL_COLUMN].tolist()
        if true_false_output else get_mbzuai_categories(dataframe, dataset_name)
    )
    accuracy_function = (
        compute_true_false_accuracy if true_false_output else compute_category_accuracy
    )
    row_count = len(dataframe)
    metric_columns = {
        CLEAN_HALLUCINATION_COLUMN: [float("nan")] * row_count,
        HALLUCINATION_CENTER_LABEL_COLUMN: [float("nan")] * row_count,
        HALLUCINATION_CENTER_CLEAN_COLUMN: [float("nan")] * row_count,
        BERTSCORE_CLEAN_LABEL_COLUMN: [float("nan")] * row_count,
        BERTSCORE_CENTER_LABEL_COLUMN: [float("nan")] * row_count,
        BERTSCORE_CENTER_CLEAN_COLUMN: [float("nan")] * row_count,
        ALIGNSCORE_CLEAN_LABEL_COLUMN: [float("nan")] * row_count,
        ALIGNSCORE_CENTER_LABEL_COLUMN: [float("nan")] * row_count,
        ALIGNSCORE_CENTER_CLEAN_COLUMN: [float("nan")] * row_count,
        BLEU_CLEAN_LABEL_COLUMN: [float("nan")] * row_count,
        BLEU_CENTER_CLEAN_COLUMN: [float("nan")] * row_count,
        BLEU_CENTER_LABEL_COLUMN: [float("nan")] * row_count,
    }
    if categories is not None:
        for column in (CLEAN_ACCURACY_COLUMN, CENTER_ACCURACY_COLUMN):
            metric_columns[column] = [float("nan")] * row_count
    valid_clean_indices = []
    valid_center_indices = []
    valid_center_clean_indices = []
    skipped_response_rows = []
    clean_responses = []
    clean_labels = []
    center_responses = []
    center_labels = []
    center_clean_responses = []
    center_clean_references = []
    rows = dataframe[
        [RESPONSE_COLUMN, CENTER_RESPONSE_COLUMN, LABEL_COLUMN]
    ].itertuples(
        index=False,
        name=None,
    )
    for row_number, (responses_value, center_response, label) in enumerate(
        tqdm(rows, total=row_count, desc="Computing hallucination and BLEU"),
        start=2,
    ):
        dataframe_index = row_number - 2
        if pd.isna(label) or not str(label).strip():
            raise ValueError(f"Row {row_number}: {LABEL_COLUMN!r} is empty.")
        label = str(label)

        clean_response = get_first_response(responses_value, row_number)
        center_is_empty = (
            pd.isna(center_response) or not str(center_response).strip()
        )
        center_response = None if center_is_empty else str(center_response)

        if categories is not None:
            for column, response in (
                (CLEAN_ACCURACY_COLUMN, clean_response),
                (CENTER_ACCURACY_COLUMN, center_response),
            ):
                if response is not None:
                    metric_columns[column][dataframe_index] = accuracy_function(
                        categories[dataframe_index], response
                    )

        empty_columns = []
        if clean_response is None:
            empty_columns.append(RESPONSE_COLUMN)
        else:
            metric_columns[CLEAN_HALLUCINATION_COLUMN][dataframe_index] = (
                compute_hallucination_score(label, clean_response)
            )
            metric_columns[BLEU_CLEAN_LABEL_COLUMN][dataframe_index] = (
                compute_bleu_score(label, clean_response)
            )
            valid_clean_indices.append(dataframe_index)
            clean_responses.append(clean_response)
            clean_labels.append(label)

        if center_response is None:
            empty_columns.append(CENTER_RESPONSE_COLUMN)
        else:
            metric_columns[HALLUCINATION_CENTER_LABEL_COLUMN][
                dataframe_index
            ] = compute_hallucination_score(label, center_response)
            metric_columns[BLEU_CENTER_LABEL_COLUMN][dataframe_index] = (
                compute_bleu_score(label, center_response)
            )
            valid_center_indices.append(dataframe_index)
            center_responses.append(center_response)
            center_labels.append(label)
            if clean_response is not None:
                metric_columns[HALLUCINATION_CENTER_CLEAN_COLUMN][
                    dataframe_index
                ] = compute_hallucination_score(
                    clean_response,
                    center_response,
                )
                metric_columns[BLEU_CENTER_CLEAN_COLUMN][dataframe_index] = (
                    compute_bleu_score(clean_response, center_response)
                )
                valid_center_clean_indices.append(dataframe_index)
                center_clean_responses.append(center_response)
                center_clean_references.append(clean_response)

        if empty_columns:
            skipped_response_rows.append((row_number, empty_columns))

    if clean_responses or center_responses:
        print("Computing BERTScore and AlignScore for available response pairs...")
        valid_metrics = compute_text_metrics(
            clean_responses=clean_responses,
            clean_labels=clean_labels,
            center_responses=center_responses,
            center_labels=center_labels,
            center_clean_responses=center_clean_responses,
            center_clean_references=center_clean_references,
            alignscore_checkpoint=alignscore_checkpoint,
            alignscore_model=alignscore_model,
            bertscore_model=bertscore_model,
            batch_size=batch_size,
            device=resolve_device(device),
        )
        metric_indices = {
            BERTSCORE_CLEAN_LABEL_COLUMN: valid_clean_indices,
            ALIGNSCORE_CLEAN_LABEL_COLUMN: valid_clean_indices,
            BERTSCORE_CENTER_LABEL_COLUMN: valid_center_indices,
            ALIGNSCORE_CENTER_LABEL_COLUMN: valid_center_indices,
            BERTSCORE_CENTER_CLEAN_COLUMN: valid_center_clean_indices,
            ALIGNSCORE_CENTER_CLEAN_COLUMN: valid_center_clean_indices,
        }
        for column_name, scores in valid_metrics.items():
            response_indices = metric_indices[column_name]
            if len(scores) != len(response_indices):
                raise ValueError(
                    f"{column_name!r} returned {len(scores)} scores for "
                    f"{len(response_indices)} valid response pairs."
                )
            for dataframe_index, score in zip(response_indices, scores):
                metric_columns[column_name][dataframe_index] = score

    if skipped_response_rows:
        displayed_rows = ", ".join(
            f"{row_number} ({', '.join(columns)})"
            for row_number, columns in skipped_response_rows[:10]
        )
        if len(skipped_response_rows) > 10:
            displayed_rows += ", ..."
        print(
            f"Skipped unavailable metric comparisons for "
            f"{len(skipped_response_rows)} row(s) with an empty response: "
            f"{displayed_rows}"
        )

    for column_name, scores in metric_columns.items():
        dataframe[column_name] = scores

    metric_values = collect_metric_values(dataframe, metric_columns)
    averages, standard_deviations = compute_metric_statistics(metric_values)
    save_metric_boxplot(metric_values, boxplot_path)
    # Commit the CSV only after the plot succeeds, so plotting errors do not
    # leave the source file partially updated.
    save_csv_in_place(dataframe, csv_path)
    return csv_path, averages, standard_deviations, boxplot_path


def resolve_response_position_output_path(csv_path, output_file=None):
    """Resolve a CSV path that cannot overwrite the source dataset."""
    if output_file is None:
        output_path = csv_path.with_name(
            f"{csv_path.stem}{RESPONSE_POSITION_OUTPUT_SUFFIX}"
        )
    else:
        output_path = Path(output_file).expanduser()
        if not output_path.suffix:
            output_path = output_path.with_suffix(".csv")
        output_path = output_path.resolve()

    if output_path == csv_path:
        raise ValueError(
            "Response-position metrics must be saved in a separate CSV; "
            "the output cannot overwrite the source CSV."
        )
    if output_path.suffix.lower() != ".csv":
        raise ValueError("The response-position output must be a CSV file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _mean_finite_scores(scores):
    """Return the arithmetic mean of finite scores, or NaN when absent."""
    finite_scores = []
    for score in scores:
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_score):
            finite_scores.append(numeric_score)
    if not finite_scores:
        return float("nan")
    return float(sum(finite_scores) / len(finite_scores))


def analyze_response_positions(
    csv_file,
    alignscore_checkpoint,
    alignscore_model="roberta-base",
    bertscore_model=None,
    batch_size=32,
    device="auto",
    output_file=None,
    dataset_name=None,
    true_false_output=False,
):
    """Save mean label/response metrics for every position and the center.

    The source CSV is read but never modified. Blank or unavailable responses
    are excluded from the mean for their position. The longest response list
    determines the positions: C, P1 through P(N-1), followed by the response
    nearest the embedding center (Center).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    csv_path = Path(csv_file).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    output_path = resolve_response_position_output_path(csv_path, output_file)

    # As in ``analyze_csv``, keep literal labels such as "N/A" as text.
    dataframe = pd.read_csv(csv_path, keep_default_na=False)
    missing_columns = {
        RESPONSE_COLUMN,
        CENTER_RESPONSE_COLUMN,
        LABEL_COLUMN,
    }.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required column(s): {missing}")
    if dataframe.empty:
        raise ValueError("CSV contains no data rows.")

    metric_columns = (
        MEAN_HALLUCINATION_COLUMN,
        MEAN_BERTSCORE_COLUMN,
        MEAN_ALIGNSCORE_COLUMN,
        MEAN_BLEU_COLUMN,
    )
    categories = (
        dataframe["category" if "category" in dataframe else LABEL_COLUMN].tolist()
        if true_false_output else get_mbzuai_categories(dataframe, dataset_name)
    )
    accuracy_function = (
        compute_true_false_accuracy if true_false_output else compute_category_accuracy
    )
    if categories is not None:
        metric_columns += (MEAN_ACCURACY_COLUMN,)
    parsed_responses = [
        get_response_positions(value, row_number)
        for row_number, value in enumerate(dataframe[RESPONSE_COLUMN], start=2)
    ]
    position_count = max(1, max(map(len, parsed_responses)))
    summary_labels = ("C",) + tuple(
        f"P{position}" for position in range(1, position_count)
    ) + (CENTER_RESPONSE_POSITION_LABEL,)
    position_scores = [
        {metric_name: [] for metric_name in metric_columns}
        for _ in summary_labels
    ]
    candidates = []
    references = []
    candidate_positions = []

    rows = dataframe[
        [RESPONSE_COLUMN, CENTER_RESPONSE_COLUMN, LABEL_COLUMN]
    ].itertuples(
        index=False,
        name=None,
    )
    for row_number, (responses_value, center_response, label) in enumerate(
        tqdm(
            rows,
            total=len(dataframe),
            desc="Computing response-position hallucination and BLEU",
        ),
        start=2,
    ):
        if pd.isna(label) or not str(label).strip():
            raise ValueError(f"Row {row_number}: {LABEL_COLUMN!r} is empty.")
        label = str(label)
        responses = parsed_responses[row_number - 2]
        responses = responses + [None] * (position_count - len(responses))
        center_is_empty = (
            pd.isna(center_response) or not str(center_response).strip()
        )
        responses.append(
            None if center_is_empty else str(center_response)
        )

        for position_index, response in enumerate(responses):
            if response is None:
                continue
            if categories is not None:
                position_scores[position_index][MEAN_ACCURACY_COLUMN].append(
                    accuracy_function(categories[row_number - 2], response)
                )
            position_scores[position_index][
                MEAN_HALLUCINATION_COLUMN
            ].append(compute_hallucination_score(label, response))
            position_scores[position_index][MEAN_BLEU_COLUMN].append(
                compute_bleu_score(label, response)
            )
            candidates.append(response)
            references.append(label)
            candidate_positions.append(position_index)

    if candidates:
        print(
            "Computing BERTScore and AlignScore for all response positions..."
        )
        text_metrics = compute_text_metrics(
            clean_responses=candidates,
            clean_labels=references,
            center_responses=[],
            center_labels=[],
            center_clean_responses=[],
            center_clean_references=[],
            alignscore_checkpoint=alignscore_checkpoint,
            alignscore_model=alignscore_model,
            bertscore_model=bertscore_model,
            batch_size=batch_size,
            device=resolve_device(device),
        )
        score_mappings = (
            (BERTSCORE_CLEAN_LABEL_COLUMN, MEAN_BERTSCORE_COLUMN),
            (ALIGNSCORE_CLEAN_LABEL_COLUMN, MEAN_ALIGNSCORE_COLUMN),
        )
        for source_column, output_column in score_mappings:
            scores = text_metrics.get(source_column)
            if scores is None:
                raise ValueError(
                    f"The text metric scorer did not return {source_column!r}."
                )
            if len(scores) != len(candidates):
                raise ValueError(
                    f"{source_column!r} returned {len(scores)} scores for "
                    f"{len(candidates)} valid response pairs."
                )
            for position_index, score in zip(candidate_positions, scores):
                position_scores[position_index][output_column].append(score)

    summary_rows = []
    for position_name, scores_by_metric in zip(
        summary_labels,
        position_scores,
    ):
        summary_row = {RESPONSE_POSITION_COLUMN: position_name}
        for metric_name in metric_columns:
            summary_row[metric_name] = _mean_finite_scores(
                scores_by_metric[metric_name]
            )
        summary_rows.append(summary_row)

    summary = pd.DataFrame(
        summary_rows,
        columns=(RESPONSE_POSITION_COLUMN,) + metric_columns,
    )
    summary.to_csv(output_path, index=False)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute hallucination score, BERTScore, and AlignScore for "
            "clean/label, center/clean, and center/label, plus BLEU for the "
            "same text pairs; then update the CSV, print average/std values, "
            "and save one combined box plot."
        )
    )
    parser.add_argument(
        "--true-false-output", action="store_true",
        help=("Compute boolean accuracy: did_not_happen, far_future, nonsense, "
              "and obscure map to True; all other labels map to False."),
    )
    parser.add_argument(
        "--dataset-name", "--dataset_name",
        default=None,
        help="Set to mbzuai to also compute category accuracy (or use CSV dataset_name).",
    )
    parser.add_argument(
        "--csv_file",
        required=True,
        help="Path to the result CSV file.",
    )
    parser.add_argument(
        "--alignscore-checkpoint",
        required=True,
        help="Path to a downloaded AlignScore .ckpt file.",
    )
    parser.add_argument(
        "--alignscore-model",
        default="roberta-base",
        choices=("roberta-base", "roberta-large"),
        help=(
            "AlignScore backbone matching the checkpoint "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--bertscore-model",
        default=None,
        help=(
            "Optional Hugging Face BERTScore model name "
            "(default: English model)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size for both metrics (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as cpu, cuda:0, or mps (default: auto).",
    )
    parser.add_argument(
        "--boxplot-file",
        default=None,
        help=(
            "Optional PNG/PDF path for the combined metric box plot "
            "(default: beside the CSV)."
        ),
    )
    parser.add_argument(
        "--response-position-output",
        nargs="?",
        const="",
        default=None,
        help=(
            "Also compute mean label/response metrics for C, all perturbed positions, and "
            "the embedding-center response. "
            "Optionally provide a separate output CSV path; when omitted, "
            "the file is saved beside the source CSV."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    csv_path, averages, standard_deviations, boxplot_path = analyze_csv(
        args.csv_file,
        alignscore_checkpoint=args.alignscore_checkpoint,
        alignscore_model=args.alignscore_model,
        bertscore_model=args.bertscore_model,
        batch_size=args.batch_size,
        device=args.device,
        boxplot_file=args.boxplot_file,
        dataset_name=getattr(args, "dataset_name", None),
        true_false_output=getattr(args, "true_false_output", False),
    )
    print(f"Updated CSV: {csv_path}")
    print("Dataset metrics (average and population standard deviation):")
    for metric_name, average in averages.items():
        standard_deviation = standard_deviations[metric_name]
        print(
            f"  {metric_name}: average={average:.6f}, "
            f"std={standard_deviation:.6f}"
        )
    print(f"Combined metric box plot: {boxplot_path}")

    response_position_output = getattr(
        args,
        "response_position_output",
        None,
    )
    if response_position_output is not None:
        response_position_path = analyze_response_positions(
            args.csv_file,
            alignscore_checkpoint=args.alignscore_checkpoint,
            alignscore_model=args.alignscore_model,
            bertscore_model=args.bertscore_model,
            batch_size=args.batch_size,
            device=args.device,
            output_file=response_position_output or None,
            dataset_name=getattr(args, "dataset_name", None),
            true_false_output=getattr(args, "true_false_output", False),
        )
        print(f"Response-position metric means: {response_position_path}")


if __name__ == "__main__":
    main()
