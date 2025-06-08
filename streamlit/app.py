import streamlit as st
import os
from utils import get_poster_url, init_session_state, itemitemcollaborativefiltering, get_random_movies

mode = os.getenv("MODE", "production")
random_movies_count = os.getenv("RANDOM_MOVIES_COUNT", "production")

# page config
st.set_page_config(
    layout="wide",
    page_title="Movie Recommender",
    page_icon=":movie_camera:"
  )

# load movies
init_session_state()

##### debug ######
if mode == "dev":
  st.markdown(f"movies_grid_ids len: {len(st.session_state.movies_grid_ids)}")
  st.markdown(f"recommended_movies_grid_ids len: {len(st.session_state.recommended_movies_grid_ids)}")
  st.markdown(f"movies_ratings len: {len(st.session_state.movies_ratings)}")

##### Page Layout #####
st.title(f"{st.session_state.movies.shape[0]:,} available movies")

#### movie search
left_col, right_col = st.columns([12, 1])  # slightly more room for input
with left_col:
  search_query = st.text_input(
    label="🔍 Search for a movie title",
    label_visibility="visible",
    placeholder="Type a title...",
    key="search_query"
  )

with right_col:
  st.markdown("<div style='padding-top: 1.7em'>", unsafe_allow_html=True)  # aligns with input
  st.button(
    "🔄",
    key="new_random_movies_btn",
    help="Fetch a new set of random movies.",
    on_click=lambda: st.session_state.update(movies_grid_ids=get_random_movies(), recommended_movies_grid_ids=[], search_query="")
  )
  st.markdown("</div>", unsafe_allow_html=True)

if search_query:
    filtered_df = st.session_state.movies[
      st.session_state.movies["title"]\
        .str.lower()\
        .str.contains(search_query.lower())
    ].head(15)
else:
    filtered_df = st.session_state.movies.loc[
        st.session_state.movies["movieId"]\
          .isin(st.session_state.movies_grid_ids)
    ]
movies_grid = filtered_df.to_dict("records")

#### movies grid
if len(st.session_state.recommended_movies_grid_ids) == 0:
   st.markdown("## Random movies")
else:
   st.markdown("## Recommended movies")
   
st.markdown("---")
cols = st.columns(5)

for i, movie in enumerate(movies_grid):
    with cols[i % 5]:
        poster_url = get_poster_url(movie["tmdbId"])
        
        # Use HTML/CSS for fixed-size poster container
        st.markdown(f'''
            <div style="width: 200px; height: 300px; display: flex; align-items: center; justify-content: center; background: #f0f0f0; border-radius: 8px; overflow: hidden; margin: 0 auto 8px auto;">
                <img src="{poster_url}" style="max-width: 100%; max-height: 100%; object-fit: contain;"/>
            </div>
            <p style="text-align:center; height:50px">{movie['title']} </p>
        ''', unsafe_allow_html=True)
        
        # Extract rating value if exists, else 0
        slider_id = f"movie_rating_{movie['movieId']}"
        rating_val = st.session_state.movies_ratings.get(movie["movieId"], 0)
        if isinstance(rating_val, dict):
            rating_val = rating_val.get("rating", 0)

        if slider_id not in st.session_state:
            st.session_state[slider_id] = rating_val

        rating = st.slider("Rate", 0, 5, key=slider_id)

        if rating != rating_val:
          movie['rating'] = rating
          st.session_state.movies_ratings[movie["movieId"]] = movie

##### sidebar #####
with st.sidebar:
  st.markdown("# Filter")
  st.radio(
     "Choose a model to get reccomendations",
      [
        "Item Based Collaborative Filtering", 
        "Client Based Collaborative Filtering"
      ],
  )

  st.markdown("---")

  can_reccomend = len(st.session_state.movies_ratings) >= 5

  st.button(
    "Get Recommendations",
    disabled=not can_reccomend,
    use_container_width=True,
    type="primary",
    help="You need to rate at least 5 movies to get recommendations.",
    on_click=lambda: st.session_state.update(recommended_movies_grid_ids=itemitemcollaborativefiltering(), search_query="")
  )

  st.button("Clear Ratings",
    use_container_width=True,
    type="secondary",
    help="Clear all your movie ratings.",
    on_click=lambda: st.session_state.update(movies_ratings={}, search_query="")
    )

  st.header(f"⭐ Your movie ratings ({len(st.session_state.movies_ratings)})")
  if st.session_state.movies_ratings:
    for movieId, movie in st.session_state.movies_ratings.items():
      title = next((m["title"] for m in st.session_state.movies.to_dict("records") if m["movieId"] == movieId), "Unknown")
      # Display image, title, and stars side by side
      img_col, info_col = st.columns([1, 2])
      with img_col:
        poster_url = get_poster_url(movie["tmdbId"])
        st.markdown(f'''
            <div style="width: 100px; height: 150px; display: flex; align-items: center; justify-content: center; background: #f0f0f0; border-radius: 8px; overflow: hidden; margin: 0 auto 8px auto;">
                <img src="{poster_url}" style="max-width: 100%; max-height: 100%; object-fit: contain;"/>
            </div>
        ''', unsafe_allow_html=True)
      with info_col:
        st.markdown(f"## {title}")
        st.markdown(f"# {'★'*movie['rating']}{'☆'*(5-movie['rating'])}")
  else:
    st.info("No movies rated yet.")

# print("Session state:", st.session_state.movies_ratings)

