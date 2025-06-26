import os
import pickle
import pandas as pd
import requests
import streamlit as st
from streamlit import session_state as state
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import normalize
import faiss
import random
import numpy as np

# -------------------------------
# TMDB Poster Caching
# -------------------------------
TMDB_API_TOKEN = "token"  # replace with your actual token
TMDB_BASE_IMG_URL = os.getenv("TMDB_BASE_IMG_URL", "https://image.tmdb.org/t/p/w200")
CACHE_CSV = os.getenv("MR_CACHE_POSTERS_URLS", ".cache/posters.csv")

def _get_poster_url_from_cache(tmdb_id):
    cached = state.df_movies_poster_cache[state.df_movies_poster_cache.tmdb_id == tmdb_id]
    if not cached.empty:
        return cached.iloc[0].poster_url
    return None

def _get_poster_url_from_api(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/images"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_TOKEN}"
    }
    try:
        res = requests.get(url, headers=headers, timeout=3)
        posters = res.json().get("posters", [])
        poster_url = None
        for poster in posters:
            if poster.get("iso_639_1") == "en":
                poster_url = f"{TMDB_BASE_IMG_URL}{poster['file_path']}"
                break
        if not poster_url and posters:
            poster_url = f"{TMDB_BASE_IMG_URL}{posters[0]['file_path']}"
        if not poster_url:
            poster_url = "https://via.placeholder.com/200x300?text=No+Image"
    except:
        poster_url = "https://via.placeholder.com/200x300?text=No+Image"

    state.df_movies_poster_cache = pd.concat([
        state.df_movies_poster_cache,
        pd.DataFrame([[tmdb_id, poster_url]], columns=["tmdb_id", "poster_url"])
    ], ignore_index=True).drop_duplicates(subset=["tmdb_id"], keep="last")
    state.df_movies_poster_cache.to_csv(CACHE_CSV, index=False)
    return poster_url

@st.cache_data(show_spinner=False)
def get_poster_url(tmdb_id):
    url = _get_poster_url_from_cache(tmdb_id)
    if url:
        return url
    return _get_poster_url_from_api(tmdb_id)

def init_cache():
    cache_dir = os.path.dirname(CACHE_CSV)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(CACHE_CSV):
        pd.DataFrame(columns=["tmdb_id", "poster_url"]).to_csv(CACHE_CSV, index=False)
    if "df_movies_poster_cache" not in state:
        state.df_movies_poster_cache = pd.read_csv(CACHE_CSV)

# -------------------------------
# Data Loading
# -------------------------------
@st.cache_data
def load_csv():
    movies = pd.read_csv("../data/movies.csv")
    links = pd.read_csv("../data/links.csv")
    merged = movies.merge(links, on="movieId", how="left").dropna(subset=["tmdbId"])
    merged["tmdbId"] = merged["tmdbId"].astype(int)
    return merged

@st.cache_data
def load_ratings():
    return pd.read_csv("../data/ratings.csv")

def get_random_movies(count: int):
    return random.sample(list(state.df_movies['movieId']), count)

# -------------------------------
# Local FAISS Index Builder
# -------------------------------
def build_local_faiss(user_ratings_dict, ratings_df, top_users=20):
    from sklearn.metrics.pairwise import cosine_similarity

    # Get all users who rated any of the movies the user rated
    user_movies = list(user_ratings_dict.keys())
    subset = ratings_df[ratings_df["movieId"].isin(user_movies)]

    if subset.empty:
        raise ValueError("No similar users found.")

    # Pivot to user-movie matrix
    user_item = subset.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    input_vector = pd.Series({mid: m["rating"] for mid, m in user_ratings_dict.items()})
    aligned_input = user_item[user_item.columns.intersection(input_vector.index)]

    # Compute cosine similarity
    sim = cosine_similarity(aligned_input, input_vector.values.reshape(1, -1)).flatten()
    top_user_ids = user_item.index[np.argsort(sim)[-top_users:]]  # most similar N users

    # Final matrix: only their full ratings + current user
    final_users = ratings_df[ratings_df["userId"].isin(top_user_ids)]
    all_movie_ids = final_users["movieId"].unique()
    movie_map = {mid: idx for idx, mid in enumerate(sorted(all_movie_ids))}
    reverse_movie_map = {idx: mid for mid, idx in movie_map.items()}
    user_map = {uid: idx for idx, uid in enumerate(sorted(top_user_ids))}

    row = final_users["userId"].map(user_map)
    col = final_users["movieId"].map(movie_map)
    data = final_users["rating"]

    sparse_matrix = csr_matrix((data, (row, col)), shape=(len(user_map), len(movie_map)))

    # Add new user
    new_vec = np.zeros(len(movie_map))
    for mid, m in user_ratings_dict.items():
        if mid in movie_map:
            new_vec[movie_map[mid]] = m["rating"]
    sparse_matrix = vstack([sparse_matrix, csr_matrix([new_vec])])

    dense = normalize(sparse_matrix.toarray().astype("float32"))
    d = dense.shape[1]

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, 10, faiss.METRIC_INNER_PRODUCT)
    index.train(dense)
    index.add(dense)

    user_map[len(user_map)] = sparse_matrix.shape[0] - 1
    return index, sparse_matrix, user_map, movie_map, reverse_movie_map
