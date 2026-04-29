"""
Compute embedding-based diversity metrics for downstream evaluation.
Uses mean pooling of hidden states from a pretrained encoder.
Metrics: pairwise cosine distance, embedding variance, centroid distance.
"""
import json, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/llm_judge")
OUTPUT_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/downstream")
MODELS = ["greedy_57", "qd_57", "random_57", "full"]

# Use a small multilingual model for encoding
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Fallback: use the Qwen model's embedding layer, or a simpler approach

def mean_pool(hidden_states, attention_mask):
    """Mean pooling over token embeddings."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def compute_diversity(embeddings):
    """Compute pairwise cosine distance and variance metrics."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    normed = embeddings / norms

    # Pairwise cosine similarity
    cos_sim = normed @ normed.T
    n = len(embeddings)

    # Average pairwise cosine distance (1 - similarity)
    upper_tri = cos_sim[np.triu_indices(n, k=1)]
    avg_cos_dist = float(np.mean(1 - upper_tri))

    # Embedding variance (spread)
    centroid = np.mean(normed, axis=0)
    dist_to_centroid = np.linalg.norm(normed - centroid, axis=1)
    avg_dist_centroid = float(np.mean(dist_to_centroid))
    std_dist_centroid = float(np.std(dist_to_centroid))

    # Effective dimensionality (number of PCA components explaining 95% variance)
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    eff_dim = int(np.searchsorted(cumvar, 0.95) + 1)

    return {
        "avg_cosine_distance": round(avg_cos_dist, 4),
        "avg_dist_to_centroid": round(avg_dist_centroid, 4),
        "std_dist_to_centroid": round(std_dist_centroid, 4),
        "effective_dim_95": eff_dim,
    }

def main():
    print("Loading encoder model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        print(f"Loaded {MODEL_NAME}")
    except Exception as e:
        print(f"Failed to load {MODEL_NAME}: {e}")
        print("Using fallback: computing diversity from character n-gram TF-IDF")
        # Fallback: use TF-IDF character n-grams as "embeddings"
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)

        results = {}
        for model_name in MODELS:
            path = RESULTS_DIR / f"responses_{model_name}.json"
            with open(path) as f:
                responses = json.load(f)
            texts = [r["response"] for r in responses]
            tfidf = vectorizer.fit_transform(texts).toarray()
            metrics = compute_diversity(tfidf)
            metrics["model"] = model_name
            metrics["n_responses"] = len(texts)
            metrics["method"] = "tfidf_char_ngram"
            results[model_name] = metrics
            print(f"  {model_name}: cos_dist={metrics['avg_cosine_distance']:.4f}, "
                  f"centroid_dist={metrics['avg_dist_to_centroid']:.4f}, "
                  f"eff_dim={metrics['effective_dim_95']}")

        out_path = OUTPUT_DIR / "embedding_diversity.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved to {out_path}")
        return

    results = {}
    for model_name in MODELS:
        path = RESULTS_DIR / f"responses_{model_name}.json"
        with open(path) as f:
            responses = json.load(f)
        texts = [r["response"] for r in responses]

        # Encode all responses
        embeddings = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            emb = mean_pool(outputs.last_hidden_state, inputs["attention_mask"]).numpy()
            embeddings.append(emb)

        embeddings = np.vstack(embeddings)
        metrics = compute_diversity(embeddings)
        metrics["model"] = model_name
        metrics["n_responses"] = len(texts)
        metrics["embedding_dim"] = embeddings.shape[1]
        metrics["method"] = MODEL_NAME
        results[model_name] = metrics

        print(f"  {model_name}: cos_dist={metrics['avg_cosine_distance']:.4f}, "
              f"centroid_dist={metrics['avg_dist_to_centroid']:.4f}, "
              f"eff_dim={metrics['effective_dim_95']}")

    out_path = OUTPUT_DIR / "embedding_diversity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
