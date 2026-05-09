import ast
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

def fetch_poster(movie_id):

    API_KEY = os.getenv("API_KEY")

    try:

        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"

        response = requests.get(url, timeout=10)

        data = response.json()

        poster_path = data.get('poster_path')

        if poster_path:

            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path

            return full_path

        return "https://via.placeholder.com/500x750?text=No+Image"

    except:

        return "https://via.placeholder.com/500x750?text=Error"

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

def safe_convert(obj):
    result = []

    if isinstance(obj, list):
        for i in obj:
            if isinstance(i, dict) and 'name' in i:
                result.append(i['name'])
            elif isinstance(i, str):
                result.append(i)

    elif isinstance(obj, str):
        try:
            parsed = ast.literal_eval(obj)

            for i in parsed:
                if isinstance(i, dict) and 'name' in i:
                    result.append(i['name'])
                elif isinstance(i, str):
                    result.append(i)

        except:
            result = []

    return result

def fetch_director(obj):
    result = []

    try:
        parsed = ast.literal_eval(obj)

        for i in parsed:
            if i['job'] == 'Director':
                result.append(i['name'])

    except:
        result = []

    return result

movies['genres'] = movies['genres'].apply(safe_convert)
movies['keywords'] = movies['keywords'].apply(safe_convert)
movies['cast'] = movies['cast'].apply(lambda x: safe_convert(x)[:3])
movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].fillna('')

movies['tags'] = (
    movies['overview'] + ' ' +
    movies['genres'].apply(lambda x: ' '.join(x)) + ' ' +
    movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
    movies['cast'].apply(lambda x: ' '.join(x)) + ' ' +
    movies['crew'].apply(lambda x: ' '.join(x))
)

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:

        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)

        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

st.title("AI Movie Recommendation System")

selected_movie = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

if st.button("Recommend"):

    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(posters[0])
        st.text(names[0])

    with col2:
        st.image(posters[1])
        st.text(names[1])

    with col3:
        st.image(posters[2])
        st.text(names[2])

    with col4:
        st.image(posters[3])
        st.text(names[3])

    with col5:
        st.image(posters[4])
        st.text(names[4])