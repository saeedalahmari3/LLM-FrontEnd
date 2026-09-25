"""Category matching for the MBZUAI hallucination dataset."""
import re


MBZUAI_CATEGORIES = ("did_not_happen", "far_future", "nonsense", "obscure")
_CATEGORY_PREFIX = re.compile(
    r'''^\s*[\s*_'"`]*(?:\(?\d+\)?[.:)\-]?\s*)?[\s*_'"`]*'''
    r"(did_not_happen|far_future|nonsense|obscure)(?!\w)",
    re.IGNORECASE,
)


def is_mbzuai(dataset_name):
    return str(dataset_name).strip().casefold() in {
        "mbzuai", "mbzuai/lamini-hallucination"
    }


def extract_category(response):
    """Read an option only at the beginning, allowing numbering/formatting."""
    match = _CATEGORY_PREFIX.match(str(response))
    return " ".join(match.group(1).casefold().split()) if match else None


def normalize_category(label, row_number):
    category = " ".join(str(label).strip().casefold().split())
    category = re.sub(r"^\(?\d+\)?[.:)\-]?\s*", "", category)
    if category not in MBZUAI_CATEGORIES:
        raise ValueError(
            f"Row {row_number}: invalid MBZUAI category {label!r}; "
            "provide the dataset category in 'category' or 'label_gt'."
        )
    return category


def compute_category_accuracy(category, response):
    """Return 1 for a matching leading option and 0 for other answers."""
    if response is None or not str(response).strip():
        return float("nan")
    return float(extract_category(response) == category)


_BOOLEAN_TOKEN = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def compute_true_false_accuracy(label, response):
    """Score an unambiguous boolean answer against the category mapping.

    Empty responses are excluded, as with category accuracy. Missing or
    conflicting boolean tokens in nonempty responses count as incorrect.
    """
    if response is None or not str(response).strip():
        return float("nan")
    category = "_".join(str(label).strip().casefold().split())
    expected = "true" if category in MBZUAI_CATEGORIES else "false"
    answers = {token.casefold() for token in _BOOLEAN_TOKEN.findall(str(response))}
    return float(answers == {expected})
