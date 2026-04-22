from sklearn.metrics.pairwise import cosine_similarity
from get_bert_word_embedding import get_sentence_embedding,get_sentence_transformer_embedding
from get_phi_embeddings import get_phi_embeddings
import numpy as np
from sklearn.manifold import TSNE
#from sentence_transormers import SentenceTransformer

def get_cosine_similarity(sentence1,sentence2):
    # Get embeddings
    #model = SentenceTransformer('sentence-transformer/all-mpnet-base-v2')
    embedding1 = np.squeeze(np.array(get_phi_embeddings(sentence1)))
    embedding2 = np.squeeze(np.array(get_phi_embeddings(sentence2))) 
    # Compute cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    #print(f"Cosine Similarity: {similarity}")
    return similarity

def get_embeddings(texts):
    """Embed a list of texts into a 2D numpy array."""
    return np.array([np.squeeze(np.array(get_phi_embeddings(text))) for text in texts])

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

def compute_hallucination_metrics(responses, reference_answer=None):
    """Measure response agreement and, when available, agreement with reference.

    Consistency alone is not enough to detect hallucination because a model can
    be consistently wrong. This function therefore returns:
      - cluster dispersion across perturbed responses
      - clean-response distance from the cluster center
      - center-response similarity to the reference answer (if provided)
      - a combined hallucination score in [0, 1]
    """
    if not responses:
        return {
            'center_response': None,
            'center_similarity_mean': 0.0,
            'clean_to_center_similarity': 0.0,
            'cluster_dispersion': 0.0,
            'reference_similarity': None,
            'hallucination_score': 1.0,
        }

    embeddings = get_embeddings(responses)
    center, dists, mask = get_embedding_center(embeddings)
    sims_to_center = cosine_similarity([center], embeddings)[0]
    center_idx = int(np.argmax(sims_to_center))
    center_response = responses[center_idx]

    clean_embedding = embeddings[0]
    clean_to_center_similarity = cosine_similarity([center], [clean_embedding])[0][0]
    center_similarity_mean = float(np.mean(sims_to_center))

    kept_dists = dists[mask] if np.any(mask) else dists
    cluster_dispersion = float(np.mean(kept_dists))

    metrics = {
        'center_response': center_response,
        'center_similarity_mean': center_similarity_mean,
        'clean_to_center_similarity': float(clean_to_center_similarity),
        'cluster_dispersion': cluster_dispersion,
        'reference_similarity': None,
        'hallucination_score': float(np.clip((1 - center_similarity_mean) / 2, 0.0, 1.0)),
    }

    if reference_answer is not None and str(reference_answer).strip():
        reference_embedding = np.squeeze(np.array(get_phi_embeddings(reference_answer)))
        reference_similarity = cosine_similarity([center], [reference_embedding])[0][0]

        # Higher score means more likely hallucination.
        # We weight reference support more than consistency because consistent
        # wrong answers are still hallucinations.
        inconsistency_score = float(np.clip((1 - center_similarity_mean) / 2, 0.0, 1.0))
        unsupported_score = float(np.clip((1 - reference_similarity) / 2, 0.0, 1.0))
        clean_divergence_score = float(np.clip((1 - clean_to_center_similarity) / 2, 0.0, 1.0))
        hallucination_score = (0.2 * inconsistency_score +
                               0.2 * clean_divergence_score +
                               0.6 * unsupported_score)

        metrics['reference_similarity'] = float(reference_similarity)
        metrics['hallucination_score'] = float(np.clip(hallucination_score, 0.0, 1.0))

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
    embeddings = np.array([np.squeeze(np.array(get_phi_embeddings(r))) for r in responses])
    
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

def select_closest_response(responses):
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

    metrics = compute_hallucination_metrics(responses)
    return metrics['center_response']



