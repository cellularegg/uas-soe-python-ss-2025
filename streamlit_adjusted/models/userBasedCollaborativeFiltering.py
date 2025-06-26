import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import faiss

def recommend_top_n_faiss_hybrid_fast_structured(
    user_id,
    sparse_matrix,
    user_map,
    movie_map,
    reverse_movie_map,
    movies_df,
    n=10,
    k=50,
    min_overlap=3,
    min_neighbors=15
):
    fallback_min_overlap = 1
    fallback_min_neighbors = 5

    if user_id not in user_map:
        return pd.DataFrame(columns=["userId", "movieId", "title", "genres", "predicted_rating", "neighbors_used"])

    user_idx = user_map[user_id]
    user_vector = sparse_matrix[user_idx]
    if user_vector.nnz == 0:
        return pd.DataFrame(columns=["userId", "movieId", "title", "genres", "predicted_rating", "neighbors_used"])

    user_dense = user_vector.toarray().astype("float32")[0]
    user_rated_mask = user_dense != 0
    user_rated_count = np.count_nonzero(user_rated_mask)

    predictions = []

    for movie_idx in range(sparse_matrix.shape[1]):
        if user_dense[movie_idx] != 0:
            continue

        column = sparse_matrix[:, movie_idx]
        rated_user_indices = column.nonzero()[0]
        if len(rated_user_indices) < fallback_min_neighbors:
            continue

        rated_user_vectors = sparse_matrix[rated_user_indices].toarray().astype("float32")
        dense_subset = normalize(rated_user_vectors)
        d = dense_subset.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(dense_subset)

        norm_target = normalize(user_dense.reshape(1, -1).astype("float32"))
        D, I = index.search(norm_target, min(k, len(rated_user_indices)))
        similarities = D[0]
        top_indices = I[0]

        def compute_weighted_rating(min_ol, min_nhbrs):
            scores, weights = [], []
            for sim, local_idx in zip(similarities, top_indices):
                neighbor_vector = rated_user_vectors[local_idx]
                rating = neighbor_vector[movie_idx]
                if rating == 0:
                    continue
                overlap = np.sum(user_rated_mask & (neighbor_vector != 0))
                if overlap < min_ol:
                    continue
                weight = sim * (overlap / (user_rated_count + 1e-10))
                scores.append(rating * weight)
                weights.append(weight)
            return scores, weights

        weighted_scores, weights = compute_weighted_rating(min_overlap, min_neighbors)

        # Fallback if not enough neighbors
        if len(weights) < min_neighbors:
            weighted_scores, weights = compute_weighted_rating(fallback_min_overlap, fallback_min_neighbors)

        if len(weights) < fallback_min_neighbors:
            continue

        pred_rating = np.sum(weighted_scores) / np.sum(weights)
        pred_rating = float(np.clip(pred_rating, 0.5, 5.0))

        movie_id = reverse_movie_map[movie_idx]
        title_row = movies_df.loc[movies_df['movieId'] == movie_id, 'title']
        genres_row = movies_df.loc[movies_df['movieId'] == movie_id, 'genres']
        title = title_row.values[0] if not title_row.empty else "Unknown"
        genres = genres_row.values[0] if not genres_row.empty else "Unknown"

        predictions.append({
            'userId': user_id,
            'movieId': movie_id,
            'title': title,
            'genres': genres,
            'predicted_rating': round(pred_rating, 2),
            'neighbors_used': len(weights)
        })

    if not predictions:
        return pd.DataFrame(columns=["userId", "movieId", "title", "genres", "predicted_rating", "neighbors_used"])

    df = pd.DataFrame(predictions)
    return df.sort_values(by="predicted_rating", ascending=False).head(n).reset_index(drop=True)
