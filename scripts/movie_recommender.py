"""Content-based movie recommender using movielens_engineered.csv.

No per-user ratings exist in this file, so collaborative filtering isn't
possible here. Instead this builds a similarity space per movie from
genres, tags, year, and rating signal, then recommends nearest neighbors.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/movielens_engineered.csv"
GENRE_COLS = [
    "genre_Action", "genre_Adventure", "genre_Animation", "genre_Children",
    "genre_Comedy", "genre_Crime", "genre_Documentary", "genre_Drama",
    "genre_Fantasy", "genre_Film-Noir", "genre_Horror", "genre_IMAX",
    "genre_Musical", "genre_Mystery", "genre_Romance", "genre_Sci-Fi",
    "genre_Thriller", "genre_War", "genre_Western",
]


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    genres = df[GENRE_COLS].to_numpy(dtype=float)

    tags = df["all_tags"].fillna("")
    tfidf = TfidfVectorizer(max_features=300, stop_words="english")
    tag_vecs = tfidf.fit_transform(tags).toarray()

    numeric = df[["year", "avg_rating", "rating_count", "tag_count"]].fillna(0)
    numeric_scaled = MinMaxScaler().fit_transform(numeric)

    # Down-weight genres/numeric relative to sparse tag space so they don't
    # get drowned out by 300 tag dimensions.
    return np.hstack([genres * 2.0, numeric_scaled * 1.5, tag_vecs])


def recommend(title_query: str, df: pd.DataFrame, sim_matrix: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    matches = df[df["title"].str.contains(title_query, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"No movie found matching '{title_query}'")
    idx = matches.index[0]
    scores = sim_matrix[idx]
    similar_idx = np.argsort(scores)[::-1]
    similar_idx = [i for i in similar_idx if i != idx][:top_n]
    return df.loc[similar_idx, ["title", "genres", "avg_rating", "rating_count"]].assign(
        similarity=scores[similar_idx]
    )


def main():
    df = pd.read_csv(DATA_PATH)
    features = build_feature_matrix(df)
    sim_matrix = cosine_similarity(features)

    query = "Toy Story"
    print(f"Movies similar to a title matching '{query}':\n")
    print(recommend(query, df, sim_matrix, top_n=10).to_string(index=False))


if __name__ == "__main__":
    main()
