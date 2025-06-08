import os
import pandas as pd
import requests
import streamlit as st
from streamlit import session_state as state
import random

TMDB_API_TOKEN = os.getenv("TMDB_API_TOKEN")
TMDB_BASE_IMG_URL = os.getenv("TMDB_BASE_IMG_URL", "https://image.tmdb.org/t/p/w200")
CACHE_CSV = os.getenv("MR_CACHE_POSTERS_URLS", ".cache/posters.csv")

##### Helpers #####
def get_random_movies():
    return random.sample(list(state.df_movies['movieId']), 20)

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
        # Find the first poster with iso_639_1 == "en"
        poster_url = None
        for poster in posters:
            if poster.get("iso_639_1") == "en":
                poster_url = f"{TMDB_BASE_IMG_URL}{poster['file_path']}"
                break
        # If not found, fallback to first poster if available
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
    """
    Returns the poster URL for a given TMDB movie ID.
    If the poster is cached, it retrieves it from the cache.
    If not cached, it fetches the poster from TMDB API and caches it.
    """
    url = _get_poster_url_from_cache(tmdb_id)
    if url:
        return url
    return _get_poster_url_from_api(tmdb_id)

##### Session State #####
def init_cache():
    """
    Initializes the cache directory and CSV file for poster URLs.
    """

    cache_dir = os.path.dirname(CACHE_CSV)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # Ensure the cache file exists
    if not os.path.exists(CACHE_CSV):
        pd.DataFrame(columns=["tmdb_id", "poster_url"]).to_csv(CACHE_CSV, index=False)

    if "df_movies_poster_cache" not in state:
        state.df_movies_poster_cache = pd.read_csv(CACHE_CSV)

@st.cache_data
def load_csv():
    """
    Loads movies and links CSV files, merges them and saves into session state.
    """

    movies = pd.read_csv("../data/csv/movies.csv")
    links = pd.read_csv("../data/csv/links.csv")

    merged = movies.merge(links, on="movieId", how="left").dropna(subset=["tmdbId"])
    merged["tmdbId"] = merged["tmdbId"].astype(int)
    return merged
