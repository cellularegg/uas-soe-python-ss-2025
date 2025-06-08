import os
import pandas as pd
import requests
import streamlit as st
import random
import pickle
from lenskit.data import RecQuery, ItemList
from lenskit import recommend

TMDB_API_TOKEN = os.getenv("TMDB_API_TOKEN")
TMDB_BASE_IMG_URL = "https://image.tmdb.org/t/p/w200"
CACHE_CSV = ".cache/posters.csv"

##### Helpers #####
def get_random_movies():
    # Store only the IDs of the random movies
    return random.sample(list(st.session_state.movies['movieId']), 20)

def _get_poster_url_from_cache(tmdb_id):
    global poster_cache_df
    cached = poster_cache_df[poster_cache_df.tmdb_id == tmdb_id]
    if not cached.empty:
        return cached.iloc[0].poster_url
    return None

def _get_poster_url_from_api(tmdb_id):
    global poster_cache_df
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/images"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_TOKEN}"
    }
    try:
        res = requests.get(url, headers=headers, timeout=3)
        posters = res.json().get("posters", [])
        if posters:
            poster_url = f"{TMDB_BASE_IMG_URL}{posters[10]['file_path']}" if len(posters) > 10 else f"{TMDB_BASE_IMG_URL}{posters[0]['file_path']}"
        else:
            poster_url = "https://via.placeholder.com/200x300?text=No+Image"
    except:
        poster_url = "https://via.placeholder.com/200x300?text=No+Image"

    poster_cache_df = pd.concat([
        poster_cache_df,
        pd.DataFrame([[tmdb_id, poster_url]], columns=["tmdb_id", "poster_url"])
    ], ignore_index=True).drop_duplicates(subset=["tmdb_id"], keep="last")
    
    poster_cache_df.to_csv(CACHE_CSV, index=False)
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
def _init_cache():
    """
    Initializes the cache directory and CSV file for poster URLs.
    """
    # Ensure the cache folder exists
    cache_dir = os.path.dirname(CACHE_CSV)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # Ensure the cache file exists
    if not os.path.exists(CACHE_CSV):
        pd.DataFrame(columns=["tmdb_id", "poster_url"]).to_csv(CACHE_CSV, index=False)

@st.cache_data
def _load_csv():
    """
    Loads movies and links CSV files, merges them and saves into session state.
    """

    movies = pd.read_csv("../data/csv/movies.csv")
    links = pd.read_csv("../data/csv/links.csv")

    merged = movies.merge(links, on="movieId", how="left").dropna(subset=["tmdbId"])
    merged["tmdbId"] = merged["tmdbId"].astype(int)
    return merged

def init_session_state():
    """
    Loads CSV files, Initializes movie posters links cache and session state variables
    """
    global poster_cache_df
    _init_cache()
    poster_cache_df = pd.read_csv(CACHE_CSV)

    if "movies" not in st.session_state:
        st.session_state.movies = _load_csv()
    if "movies_grid_ids" not in st.session_state:
        st.session_state.movies_grid_ids = get_random_movies()
    if "movies_ratings" not in st.session_state:
        st.session_state.movies_ratings = {}

##### models #####
@st.cache_resource
def _load_itemitemcollaborativefiltering_model():
  # Load model (itemitem collaborative filtering)
  MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/item-based-collaborative-filtering.pkl'))
  if "itemcf_model" not in st.session_state:
      with open(MODEL_PATH, 'rb') as f:
          st.session_state.itemcf_model = pickle.load(f)

def itemitemcollaborativefiltering():
    if "itemcf_model" not in st.session_state:
        _load_itemitemcollaborativefiltering_model()

    # Prepare user history DataFrame with only valid items
    user_hist = []
    for movieId, movie in st.session_state.movies_ratings.items():
        user_hist.append({'user_id': 9999, 'item_id': movieId, 'rating': movie['rating']})

    user_hist_df = pd.DataFrame(user_hist)
    hist_items = ItemList.from_df(user_hist_df, keep_user=False)
    query = RecQuery(user_id=9999, user_items=hist_items)
    rec = recommend(st.session_state.itemcf_model, query, n=10)
    rec_df = rec.to_df()
    # Store only the recommended movie IDs
    st.session_state.movies_grid_ids = rec_df['item_id'].tolist()
    return st.session_state.movies_grid_ids
