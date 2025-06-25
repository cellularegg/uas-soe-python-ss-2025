import numpy as np
from itertools import chain, combinations
from typing import List
from scipy.sparse import load_npz
import pickle

# Load matrix and mappings once at import
user_movie_sparse = load_npz("user_movie_likes.npz")

with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

idx_to_movieId = mappings["idx_to_movieId"]
movieId_to_title = mappings["movieId_to_title"]

def powerset(s):
    "Generate all non-empty subsets of a set"
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))

def titles_from_indices(indices, idx_to_movieId, movieId_to_title):
    titles = []
    for idx in indices:
        mid = idx_to_movieId.get(idx)
        titles.append(movieId_to_title.get(mid, f"Unknown({mid})"))
    return titles

def generate_rule_tooltips_oneline(
    user_ratings: dict,
    recommended_movieIds: list,
    min_lift: float = 1.0
) -> dict:
    """
    Returns: dict {recommended_movieId: tooltip string}
    Only includes rules where lift >= min_lift.
    Tooltip is a plain one-line string suitable for HTML title attribute.
    Only considers last 5 liked movies and antecedents of size at most 3.
    """

    n_users = user_movie_sparse.shape[0]

    # Get the last 5 liked movie IDs (rating >= 4)
    liked_movie_ids = [mid for mid, rating in user_ratings.items() if rating >= 4][-5:]

    # Map those to indices
    antecedent_movie_indices = [
        k for mid in liked_movie_ids
        for k, v in idx_to_movieId.items() if v == mid
    ]

    # Get indices of recommended movies
    consequent_movie_indices = [
        k for mid in recommended_movieIds
        for k, v in idx_to_movieId.items() if v == mid
    ]

    def users_liked_all(movies):
        if not movies:
            return np.ones(n_users, dtype=bool)
        subset_matrix = user_movie_sparse[:, movies].toarray()
        return subset_matrix.all(axis=1)

    def bounded_powerset(items, max_size=2):
        return chain.from_iterable(combinations(items, r) for r in range(1, max_size + 1))

    result = {}

    for con_idx in consequent_movie_indices:
        best_rule = None
        best_lift = 0.0
        best_ant_indices = []

        for ant_indices in bounded_powerset(antecedent_movie_indices, max_size=1):
            if con_idx in ant_indices:
                continue

            ant_users = users_liked_all(list(ant_indices))
            con_users = users_liked_all([con_idx])
            both_users = ant_users & con_users

            ant_support = ant_users.sum()
            con_support = con_users.sum()
            both_support = both_users.sum()

            if ant_support == 0 or con_support == 0:
                continue

            confidence = both_support / ant_support
            lift = confidence / (con_support / n_users)

            if lift >= min_lift and lift > best_lift:
                best_rule = (ant_indices, con_idx)
                best_lift = lift
                best_ant_indices = list(ant_indices)

        con_movieId = idx_to_movieId.get(con_idx)
        con_title = movieId_to_title.get(con_movieId, "Unknown")

        if best_rule:
            ant_titles_raw = [movieId_to_title.get(idx_to_movieId[a], "Unknown") for a in best_ant_indices]
            ant_titles_wrapped = [f"<<{title}>>" for title in ant_titles_raw]

            if len(ant_titles_wrapped) == 1:
                ant_part = ant_titles_wrapped[0]
            else:
                ant_part = ", ".join(ant_titles_wrapped[:-1]) + " and " + ant_titles_wrapped[-1]

            lift_pct = (best_lift - 1) * 100
            tooltip = f"Users who liked {ant_part} are {lift_pct:.1f}% more likely to also like {con_title}."
            result[con_movieId] = tooltip
        else:
            result[con_movieId] = ""

    return result
