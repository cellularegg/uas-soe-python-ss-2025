import streamlit as st
from streamlit import session_state as state
from utils_adjusted import (
    init_cache,
    load_csv,
    get_random_movies,
    get_poster_url,
    load_ratings,
    build_local_faiss
)
from models.itemBasedCollaborativeFiltering import itemBasedCollaborativeFiltering
from models.userBasedCollaborativeFiltering import recommend_top_n_faiss_hybrid_fast_structured
import os
import ollama

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

if "ratings_df" not in st.session_state:
    st.session_state.ratings_df = load_ratings()

if "df_movies" not in st.session_state:
    df_movies_full = load_csv()
    rated_movie_ids = set(st.session_state.ratings_df["movieId"].unique())
    st.session_state.df_movies = df_movies_full[df_movies_full["movieId"].isin(rated_movie_ids)]

if "list_movies_grid_ids" not in st.session_state:
    st.session_state.list_movies_grid_ids = get_random_movies(random_movies_count)
if "dict_movies_ratings" not in st.session_state:
    st.session_state.dict_movies_ratings = {}
if "recommended" not in st.session_state:
    st.session_state.recommended = False

##### Page Layout #####
st.title(f"{state.df_movies.shape[0]:,} available movies")

#### movie search #####
col_search_bar, col_refresh_button = st.columns([12, 1])
with col_search_bar:
    search_query = st.text_input(
        label="🔍 Search for a movie title",
        label_visibility="visible",
        placeholder="Type a title...",
        key="search_query"
    )

with col_refresh_button:
    st.markdown("<div style='padding-top: 1.7em'>", unsafe_allow_html=True)
    st.button(
        "🔄",
        key="new_random_movies_btn",
        help="Fetch a new set of random movies.",
        on_click=lambda: state.update(
            list_movies_grid_ids=get_random_movies(random_movies_count),
            search_query="",
            recommended=False
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)

# search functionality
if search_query:
    filtered_df = state.df_movies[
        state.df_movies["title"].str.lower().str.contains(search_query.lower())
    ].head(20)
else:
    filtered_df = state.df_movies.loc[
        state.df_movies["movieId"].isin(state.list_movies_grid_ids)
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
        st.markdown(f'''
            <div style="width: 200px; height: 300px; display: flex; align-items: center; justify-content: center; background: #f0f0f0; border-radius: 8px; overflow: hidden; margin: 0 auto 8px auto;">
                <img src="{poster_url}" style="max-width: 100%; max-height: 100%; object-fit: contain;"/>
            </div>
            <p style="text-align:center; height:50px">{movie['title']} </p>
        ''', unsafe_allow_html=True)

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
        "Choose a model to get recommendations",
        [
            "Item-Based Collaborative Filtering",
            "User-Based Collaborative Filtering"
        ],
        key="selected_model"
    )

    if selected_model == "Item-Based Collaborative Filtering":
        model = itemBasedCollaborativeFiltering()
        model.load('../../models/item-based-collaborative-filtering.pkl')
    else:
        model = None

    st.markdown("---")

    can_reccomend = len(state.dict_movies_ratings) >= 5

    def recommend_user_based():
    # Always use the latest ratings from the user
        user_ratings_dict = {
            movieId: movie for movieId, movie in state.dict_movies_ratings.items()
            if "rating" in movie and movie["rating"] > 0
        }

        # Ensure we build a fresh FAISS and matrix every time
        index, sparse_matrix, user_map, movie_map, reverse_movie_map = build_local_faiss(
            user_ratings_dict, state.ratings_df
        )
        new_user_idx = sparse_matrix.shape[0] - 1

        top_recs_df = recommend_top_n_faiss_hybrid_fast_structured(
            user_id=new_user_idx,
            sparse_matrix=sparse_matrix,
            user_map=user_map,
            movie_map=movie_map,
            reverse_movie_map=reverse_movie_map,
            movies_df=state.df_movies,
            n=10,
            k=50,
            min_overlap=3,
            min_neighbors=15
        )

        state.list_movies_grid_ids = top_recs_df["movieId"].tolist()
        state.search_query = ""
        state.recommended = True


    st.button(
        "Get Recommendations",
        disabled=not can_reccomend,
        use_container_width=True,
        type="primary",
        help="You need to rate at least 5 movies to get recommendations.",
        on_click=(
        recommend_user_based
        if selected_model == "User-Based Collaborative Filtering"
        else lambda: state.update(
            list_movies_grid_ids=model.recommend(state.dict_movies_ratings, 10),
            search_query="",
            recommended=True
    )
)
)

    st.button("Clear Ratings",
        use_container_width=True,
        type="secondary",
        help="Clear all your movie ratings.",
        on_click=lambda: state.update(
            dict_movies_ratings={},
            search_query=""
        )
    )

    st.header(f"⭐ Your movie ratings ({len(state.dict_movies_ratings)})")
    if state.dict_movies_ratings:
        for movieId, movie in state.dict_movies_ratings.items():
            title = next((m["title"] for m in state.df_movies.to_dict("records") if m["movieId"] == movieId), "Unknown")
            img_col, info_col = st.columns([1, 4])
            with img_col:
                poster_url = get_poster_url(movie["tmdbId"])
                st.markdown(f'''
                    <div style="width: 60px; height: 90px; display: flex; align-items: center; justify-content: center; background: #f0f0f0; border-radius: 8px; overflow: hidden; margin: 0 0 10px 0;">
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
if state.recommended:
    st.markdown("---")

    if "movie_explanations" not in state:
        state.movie_explanations = {}
    if "insights_clicked" not in state:
        state.insights_clicked = False

    if st.button("🧠 Summarize Insights"):
        state.insights_clicked = True

        def explain_movies(recommended_df, rated_movies, ratings_df):
            explanations = {}

            liked_titles = [
                m["title"]
                for m in state.df_movies.to_dict("records")
                if m["movieId"] in rated_movies and rated_movies[m["movieId"]]["rating"] >= 4
            ]

            ratings_summary = ratings_df[
                ratings_df["movieId"].isin(recommended_df["movieId"])
            ].groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()

            summary_map = {
                row["movieId"]: (round(row["mean"], 2), row["count"])
                for _, row in ratings_summary.iterrows()
            }

            for _, row in recommended_df.iterrows():
                movie_id = row["movieId"]
                title = row["title"]
                genres = row["genres"]
                avg_rating, count = summary_map.get(movie_id, (None, None))

                if liked_titles:
                    prompt = f"The user liked the following movies: {', '.join(liked_titles)}.\n"
                else:
                    prompt = "The user has rated several movies, but none very highly.\n"

                if avg_rating is not None and count:
                    prompt += (
                        f"The recommended movie '{title}' (Genres: {genres}) "
                        f"has an average rating of {avg_rating}/5 from {count} similar users.\n"
                    )
                prompt += f"Why might the user enjoy '{title}'? Respond in 2–3 friendly sentences."

                try:
                    result = ollama.chat(
                        model="llama3",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a friendly movie assistant. "
                                    "Explain movie recommendations clearly using personal preferences and crowd ratings."
                                )
                            },
                            {"role": "user", "content": prompt}
                        ]
                    )
                    explanation = result["message"]["content"]
                except Exception as e:
                    explanation = f"(Explanation failed: {str(e)})"

                explanations[movie_id] = explanation

            return explanations

        user_rated = {
            movieId: movie for movieId, movie in state.dict_movies_ratings.items()
            if "rating" in movie and movie["rating"] > 0
        }
        rec_df = state.df_movies[state.df_movies["movieId"].isin(state.list_movies_grid_ids)]
        state.movie_explanations = explain_movies(rec_df, user_rated, state.ratings_df)

    if state.insights_clicked and state.movie_explanations:
        st.markdown("### AI-generated Insights:")
        for movieId in state.list_movies_grid_ids:
            explanation = state.movie_explanations.get(movieId)
            if explanation:
                title = next((m["title"] for m in state.df_movies.to_dict("records") if m["movieId"] == movieId), "")
                st.markdown(f"**{title}**: {explanation}")