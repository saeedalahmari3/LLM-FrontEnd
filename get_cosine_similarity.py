from sklearn.metrics.pairwise import cosine_similarity
from get_phi_embeddings import get_phi_embeddings
import numpy as np
from sklearn.manifold import TSNE
#from sentence_transormers import SentenceTransformer
from hallucination_score import compute_h

CLEAN_HALLUCINATION_KEY = 'hallucination_score_clean_vs_label'
CENTER_LABEL_HALLUCINATION_KEY = 'hallucination_score_center_vs_label'
CENTER_CLEAN_HALLUCINATION_KEY = 'hallucination_score_center_vs_clean'

def get_cosine_similarity(sentence1,sentence2):
    # Get embeddings
    #model = SentenceTransformer('sentence-transformer/all-mpnet-base-v2')
    embedding1, embedding2 = get_embeddings([sentence1, sentence2])
    # Compute cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    #print(f"Cosine Similarity: {similarity}")
    return similarity

def get_embeddings(texts):
    """Embed a list of texts into a 2D numpy array."""
    embeddings = np.asarray(get_phi_embeddings(texts), dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding array, received shape {embeddings.shape}."
        )
    return embeddings

def get_embedding_center(embeddings):
    """Compute a robust center by dropping distant outliers before averaging."""
    mean_emb = np.mean(embeddings, axis=0)
    dists = np.linalg.norm(embeddings - mean_emb, axis=1)
    threshold = dists.mean() + dists.std()
    mask = dists <= threshold
    filtered = embeddings[mask]
    if filtered.size == 0:
        return mean_emb, dists, mask
    return np.mean(filtered, axis=0), dists, mask

def compute_hallucination_metrics(responses, label=None, reference_answer=None):
    """Select the center response and compute directional hallucination scores.

    ``reference_answer`` is retained as a fallback for existing callers;
    ``label`` is the preferred ground-truth reference for clean/label and
    center/label, while the clean response is the reference for center/clean.
    """
    if not responses:
        return {
            'center_response': None,
            'center_similarity_mean': 0.0,
            'clean_to_center_similarity': 0.0,
            'cluster_dispersion': 0.0,
            'reference_similarity': None,
            CLEAN_HALLUCINATION_KEY: 1.0,
            CENTER_LABEL_HALLUCINATION_KEY: 1.0,
            CENTER_CLEAN_HALLUCINATION_KEY: 1.0,
            # Backward-compatible aliases.
            'hallucination-score': 1.0,
            'hallucination-score-center-vs-label': 1.0,
            'hallucination-score-center-vs-clean': 1.0,
        }

    ground_truth = label if label is not None else reference_answer
    ground_truth = None if ground_truth is None else str(ground_truth)
    clean_response = str(responses[0])
    embeddings = get_embeddings(responses)
    center, dists, mask = get_embedding_center(embeddings)
    sims_to_center = cosine_similarity([center], embeddings)[0]
    center_idx = int(np.argmax(sims_to_center))
    center_response = str(responses[center_idx])
    clean_embedding = embeddings[0]
    clean_to_center_similarity = cosine_similarity([center], [clean_embedding])[0][0]
    center_similarity_mean = float(np.mean(sims_to_center))
    kept_dists = dists[mask] if np.any(mask) else dists
    cluster_dispersion = float(np.mean(kept_dists))

    clean_label_score = (
        float('nan')
        if ground_truth is None
        else compute_h(ground_truth, clean_response)
    )
    center_label_score = (
        float('nan')
        if ground_truth is None
        else compute_h(ground_truth, center_response)
    )
    center_clean_score = compute_h(clean_response, center_response)

    metrics = {
        'center_response': center_response,
        'center_similarity_mean': center_similarity_mean,
        'clean_to_center_similarity': float(clean_to_center_similarity),
        'cluster_dispersion': cluster_dispersion,
        'reference_similarity': None,
        CLEAN_HALLUCINATION_KEY: clean_label_score,
        CENTER_LABEL_HALLUCINATION_KEY: center_label_score,
        CENTER_CLEAN_HALLUCINATION_KEY: center_clean_score,
        # Backward-compatible aliases for callers using the old schema.
        'hallucination-score': clean_label_score,
        'hallucination-score-center-vs-label': center_label_score,
        'hallucination-score-center-vs-clean': center_clean_score,
    }

    return metrics

def visualize_embedding_space(responses, labels=None):
    """Visualize the embedding space of responses using t-SNE.
    
    Args:
        responses: List of response strings
        labels: Optional list of labels for each response
    """
    import matplotlib.pyplot as plt
    
    if not responses:
        return None
    
    # Get embeddings
    embeddings = get_embeddings(responses)
    
    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(responses)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(len(responses)), cmap='viridis', s=100)
    
    # Add response text annotations
    for i, response in enumerate(responses):
        plt.annotate(response, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.7)
    
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9, fontweight='bold')
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Embedding Space Visualization')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig('embedding_space_visualization.pdf')
    plt.close()

def select_closest_response(responses, label=None):
    """Return the response closest to the center of the embedding distribution.

    Steps:
      1. Compute sentence embeddings for every string in ``responses``.
      2. Average the embeddings to obtain a mean vector.
      3. Calculate the Euclidean distance of each embedding to that mean.
      4. Drop the embeddings whose distance is greater than mean+std (i.e. farthest outliers).
      5. Recompute the center from the remaining embeddings.
      6. Return the original response whose embedding is closest (highest cosine
         similarity) to the new center.

    If all responses are removed by the threshold, the original mean is reused.
    """
    if not responses:
        return None

    metrics = compute_hallucination_metrics(responses, label)
    return metrics['center_response']
