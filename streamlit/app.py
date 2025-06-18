import streamlit as st
from streamlit import session_state as state
from utils import init_cache, load_csv, get_random_movies, get_poster_url
from models.itemBasedCollaborativeFiltering import itemBasedCollaborativeFiltering
from models.itemBasedCollaborativeFilteringTest import itemBasedCollaborativeFilteringTest
import os

##### Init streamlit WebUI ######
st.set_page_config(
    layout="wide",
    page_title="Movie Recommender",
    page_icon=":movie_camera:"
  )

##### CONFIG ######
random_movies_count = int(os.getenv("MR_RANDOM_MOVIES_COUNT", 10))

##### Manage state ######
init_cache()

if "df_movies" not in st.session_state:
  st.session_state.df_movies = load_csv()
if "list_movies_grid_ids" not in st.session_state:
  st.session_state.list_movies_grid_ids = get_random_movies(random_movies_count)
if "dict_movies_ratings" not in st.session_state:
  st.session_state.dict_movies_ratings = {}
if "recommended" not in st.session_state:
  st.session_state.recommended = False

##### Page Layout #####
st.title(f"{state.df_movies.shape[0]:,} available movies")

#### movie search #####
col_search_bar, col_refresh_button = st.columns([12, 1])  # slightly more room for input
with col_search_bar:
  search_query = st.text_input(
    label="🔍 Search for a movie title",
    label_visibility="visible",
    placeholder="Type a title...",
    key="search_query"
  )

with col_refresh_button:
  st.markdown("<div style='padding-top: 1.7em'>", unsafe_allow_html=True)  # aligns with input
  st.button(
    "🔄",
    key="new_random_movies_btn",
    help="Fetch a new set of random movies.",
    on_click=lambda: state.update(list_movies_grid_ids=get_random_movies(random_movies_count), search_query="", recommended=False)
  )
  st.markdown("</div>", unsafe_allow_html=True)

# search functionality
if search_query:
    filtered_df = state.df_movies[
      state.df_movies["title"]\
        .str.lower()\
        .str.contains(search_query.lower())
    ].head(20)
else:
    filtered_df = state.df_movies.loc[
        state.df_movies["movieId"]\
          .isin(state.list_movies_grid_ids)
    ]
movies_grid = filtered_df.to_dict("records")

#### movies grid #####
if state.recommended:
  st.markdown("## Recommended movies")
else:
  st.markdown("## Random movies")
   
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
        rating_val = state.dict_movies_ratings.get(movie["movieId"], 0)
        if isinstance(rating_val, dict):
            rating_val = rating_val.get("rating", 0)

        if slider_id not in state:
            state[slider_id] = rating_val

        rating = st.slider("Rate", 0, 5, key=slider_id)

        if rating != rating_val:
          movie['rating'] = rating
          state.dict_movies_ratings[movie["movieId"]] = movie

##### sidebar #####
with st.sidebar:
  st.markdown("# Filter")
  selected_model = st.radio(
     "Choose a model to get reccomendations",
      [
        "Item Based Collaborative Filtering", 
        "Item Based Collaborative Filtering Test"
      ],
      key="selected_model"
  )

  # Load the selected model on radio change
  if selected_model == "Item Based Collaborative Filtering":
    model = itemBasedCollaborativeFiltering()
    model.load('../../models/item-based-collaborative-filtering.pkl')
  if selected_model == "Item Based Collaborative Filtering Test":
    model = itemBasedCollaborativeFilteringTest()
    model.load('../../models/item-based-collaborative-filtering-test.pkl')

  st.markdown("---")

  can_reccomend = len(state.dict_movies_ratings) >= 5

  st.button(
    "Get Recommendations",
    disabled=not can_reccomend,
    use_container_width=True,
    type="primary",
    help="You need to rate at least 5 movies to get recommendations.",
    on_click=lambda: state.update(
       list_movies_grid_ids=model.recommend(state.dict_movies_ratings, 10), 
       search_query="", 
       recommended=True)
  )

  st.button("Clear Ratings",
    use_container_width=True,
    type="secondary",
    help="Clear all your movie ratings.",
    on_click=lambda: state.update(
       dict_movies_ratings={},
       search_query="")
    )

  st.header(f"⭐ Your movie ratings ({len(state.dict_movies_ratings)})")
  if state.dict_movies_ratings:
    for movieId, movie in state.dict_movies_ratings.items():
      title = next((m["title"] for m in state.df_movies.to_dict("records") if m["movieId"] == movieId), "Unknown")
      # Display image, title, and stars side by side
      img_col, info_col = st.columns([1, 4])
      with img_col:
        poster_url = get_poster_url(movie["tmdbId"])
        st.markdown(f'''
            <div style="width: 60px; height: 90px; display: flex; align-items: center; justify-content: center; background: #f0f0f0; border-radius: 8px; overflow: hidden; margin: 0 0 10px 0 ;">
                <img src="{poster_url}" style="max-width: 100%; max-height: 100%; object-fit: contain;"/>
            </div>
        ''', unsafe_allow_html=True)
      with info_col:
        st.markdown(f'''
            <p style="margin-bottom: 0px">{title}</p>
            <p> {'★'*movie['rating']}{'☆'*(5-movie['rating'])}</p>
        ''', unsafe_allow_html=True)
  else:
    st.info("No movies rated yet.")

# print("Session state:", state.dict_movies_ratings)

