from functools import lru_cache

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


@lru_cache(maxsize=None)
def _get_cached_model_path(model_name):
    """Resolve a Hugging Face model ID to its cached local snapshot path."""
    try:
        return snapshot_download(
            repo_id=model_name,
            local_files_only=True,
        )
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            f"The cached model '{model_name}' is unavailable. "
            "Download it once while online before running this program offline."
        ) from exc


@lru_cache(maxsize=1)
def _get_semantic_model():
    """Load the cached semantic model on a numerically stable device."""
    try:
        # SentenceTransformers 2.x does not accept ``local_files_only`` here.
        # The resolved snapshot is already a local directory, so passing only
        # that path still guarantees that this constructor cannot download it.
        #
        # Do not let SentenceTransformers automatically select Apple MPS. The
        # PyTorch 1.x MPS backend can occasionally emit non-finite MiniLM
        # embeddings. This model is small, and deterministic CPU inference is
        # preferable to corrupting a metric (or crashing in scikit-learn).
        model = SentenceTransformer(
            _get_cached_model_path(SEMANTIC_MODEL_NAME),
            device="cpu",
        )
        return model.float()
    except OSError as exc:
        raise RuntimeError(
            f"The cached model '{SEMANTIC_MODEL_NAME}' is unavailable. "
            "Download it once while online before running this program offline."
        ) from exc


@lru_cache(maxsize=1)
def _get_nli_components():
    """Load the cached NLI tokenizer and model only when they are first used."""
    model_path = _get_cached_model_path(NLI_MODEL_NAME)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            # Avoid fast-tokenizer JSON incompatibilities across versions.
            use_fast=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"The cached model '{NLI_MODEL_NAME}' is unavailable. "
            "Download it once while online before running this program offline."
        ) from exc

    model.eval()
    return tokenizer, model

# 2. Semantic similarity S_sem
def compute_s_sem(gt: str, mo: str) -> float:
    if not isinstance(gt, str) or not isinstance(mo, str):
        raise TypeError("compute_s_sem expects two strings.")

    sem_model = _get_semantic_model()
    embeddings = np.asarray(
        sem_model.encode(
            [gt, mo],
            show_progress_bar=False,
            convert_to_numpy=True,
        ),
        dtype=np.float64,
    )
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != 2
        or embeddings.shape[1] == 0
    ):
        raise RuntimeError(
            "The semantic model returned an invalid embedding shape: "
            f"{embeddings.shape}; expected (2, embedding_dimension)."
        )
    if not np.isfinite(embeddings).all():
        invalid_rows = np.flatnonzero(
            ~np.isfinite(embeddings).all(axis=1)
        ).tolist()
        raise RuntimeError(
            "The semantic model returned non-finite embedding values for "
            f"input row(s) {invalid_rows}, despite CPU float32 inference. "
            "Restart the process and verify the cached model and PyTorch "
            "installation."
        )

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0.0):
        raise RuntimeError(
            "The semantic model returned an embedding with an invalid norm."
        )

    sim = cosine_similarity(embeddings[:1], embeddings[1:2])[0][0]
    if not np.isfinite(sim):
        raise RuntimeError("Cosine similarity produced a non-finite value.")
    return float(np.clip(sim, -1.0, 1.0))

# 3. Factual score S_fact' using NLI (scaled 0..1)
def compute_s_fact(gt: str, mo: str) -> float:
    nli_tokenizer, nli_model = _get_nli_components()
    # premise = ground truth, hypothesis = model output
    inputs = nli_tokenizer(
        gt, mo,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        logits = nli_model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    # label order for many NLI models is: contradiction, neutral, entailment
    p_contr = probs[0].item()
    p_neutr = probs[1].item()
    p_ent   = probs[2].item()
    # print('Probability of contradiction is {}'.format(p_contr))
    # print('Probability of neutral is {}'.format(p_neutr))
    # print('Proabability of entailment is {}'.format(p_ent))

    # S_fact in [-1,1], then rescale to [0,1]
    #s_fact_raw = p_ent - p_contr
    s_fact = max(p_ent,p_neutr)
    #s_fact = (s_fact_raw + 1.0) / 2.0
    return float(s_fact)

# 4. Hallucination score H
def compute_h(gt: str, mo: str):
    s_sem = compute_s_sem(gt, mo)
    s_fact = compute_s_fact(gt, mo)
    h = s_sem * (1.0 - s_fact)
    return h

# Example
# gt = "Both halves may survive, but only the front half can regenerate its tail."
# mo = "Each half can potentially regenerate into two smaller worms, but survival depends on retaining vital parts"

# s_sem, s_fact, h = compute_h(gt, mo)
# print(f"S_sem (semantic similarity): {s_sem:.4f}")
# print(f"S_fact (factual consistency, 0..1): {s_fact:.44f}")
# print(f"H (hallucination score)          : {h:.4f}")
