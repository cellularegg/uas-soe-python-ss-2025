from models.model import Model
import pandas as pd
import os
import streamlit as st
import pickle
from streamlit import session_state as state
from lenskit.data import RecQuery, ItemList
from lenskit import recommend

class itemBasedCollaborativeFiltering(Model):
  def load(self, path) -> None:
    # Load model (itemitem collaborative filtering)
    path_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if "mode_itemcf" not in state:
      with open(path_abs, 'rb') as f:
          state.mode_itemcf = pickle.load(f)
      print(f"Model loaded from: {path_abs}")
    else:
      print(f"Model already present: {path_abs}")


  def recommend(self, movies_rating: dict) -> list:
    # Prepare user history DataFrame with only valid items
    user_hist = []
    for movieId, movie in movies_rating.items():
        user_hist.append({'user_id': 9999, 'item_id': movieId, 'rating': movie['rating']})

    user_hist_df = pd.DataFrame(user_hist)
    hist_items = ItemList.from_df(user_hist_df, keep_user=False)
    query = RecQuery(user_id=9999, user_items=hist_items)
    rec = recommend(state.mode_itemcf, query, n=20)
    rec_df = rec.to_df()
    # Store only the recommended movie IDs

    return rec_df['item_id'].tolist()