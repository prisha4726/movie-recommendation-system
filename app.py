import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Load Dataset
# -----------------------------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# -----------------------------
# Collaborative Filtering Setup
# -----------------------------
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)

def recommend_real(user_id, top_n=5):

    if user_id not in user_movie_matrix.index:
        return []

    user_index = user_movie_matrix.index.tolist().index(user_id)
    similarity_scores = user_similarity[user_index]

    similarity_series = pd.Series(similarity_scores, index=user_movie_matrix.index)
    similarity_series = similarity_series.drop(user_id)
    similarity_series = similarity_series.sort_values(ascending=False)

    top_similar_users = similarity_series[:5]

    recommendations = []

    for other_user in top_similar_users.index:
        other_user_ratings = user_movie_matrix.loc[other_user]
        high_rated_movies = other_user_ratings[other_user_ratings >= 4]

        for movie_id in high_rated_movies.index:
            if user_movie_matrix.loc[user_id, movie_id] == 0:
                recommendations.append(movie_id)

    recommendations = list(set(recommendations))[:top_n]

    movie_titles = movies[movies['movieId'].isin(recommendations)]['title']

    return movie_titles.tolist()

# -----------------------------
# Content-Based Filtering Setup
# -----------------------------
movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf.fit_transform(movies['genres'])

genre_similarity = cosine_similarity(genre_matrix, genre_matrix)

def recommend_by_genre(movie_title, top_n=5):

    if movie_title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_title].index[0]

    similarity_scores = list(enumerate(genre_similarity[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n+1]

    movie_indices = [i[0] for i in similarity_scores]

    return movies['title'].iloc[movie_indices].tolist()

# -----------------------------
# Hybrid Model
# -----------------------------
def hybrid_recommend(user_id, movie_title, alpha=0.7):

    collab_recs = recommend_real(user_id)
    genre_recs = recommend_by_genre(movie_title)

    hybrid_scores = {}

    for movie in collab_recs:
        hybrid_scores[movie] = alpha

    for movie in genre_recs:
        if movie in hybrid_scores:
            hybrid_scores[movie] += (1 - alpha)
        else:
            hybrid_scores[movie] = (1 - alpha)

    sorted_recommendations = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_recommendations

# -----------------------------
# Netflix Style UI
# -----------------------------

st.set_page_config(layout="wide")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .title {
        font-size: 50px;
        font-weight: bold;
        color: #E50914;
    }
    .subtitle {
        font-size: 20px;
        color: #bbbbbb;
    }
    .movie-card {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">NETFLIX RECOMMENDER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find your next favorite movie 🎬</div>', unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input("User ID", min_value=1, step=1)

with col2:
    movie_title = st.selectbox("Pick a Movie", movies['title'].sort_values().unique())

st.write("")

if st.button("🍿 Get Recommendations"):

    results = hybrid_recommend(user_id, movie_title)

    if results:
        st.subheader("🔥 Recommended For You")

        cols = st.columns(5)

        for i, (movie, score) in enumerate(results):
            with cols[i % 5]:
                st.markdown(f"""
                    <div class="movie-card">
                        <h4>{movie}</h4>
                        <p>⭐ {round(score,2)}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations found.")
