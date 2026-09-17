Voici un projet complet de recommandation de films, avec :

    le téléchargement du dataset ;

    le nettoyage des données ;

    un moteur basé sur les genres et les tags ;

    la personnalisation selon les notes de l’utilisateur ;

    une interface Streamlit ;

    une API FastAPI ;

    les tests ;

    une évaluation simple ;

    un guide d’installation et d’utilisation.

Le dataset ml-latest-small contient notamment 100836 évaluations, 9742 films, 610 utilisateurs et les fichiers movies.csv, ratings.csv, tags.csv et links.csv. Nous utiliserons TfidfVectorizer pour transformer les genres et tags en vecteurs numériques, puis la similarité cosinus pour comparer les films.
1. Fonctionnement du projet

Le projet aura deux moteurs :
Recommandation basée sur le contenu

Elle recommande des films ayant des genres et des tags similaires.

Exemple :

text
Toy Story (1995)

Peut produire :

text
Toy Story 2 (1999)
Monsters, Inc. (2001)
Finding Nemo (2003)
The Incredibles (2004)

Recommandation personnalisée

Elle utilise les films qu’un utilisateur a bien notés.

Par exemple :

    l’utilisateur note fortement les films d’action ;

    le système détecte ses genres préférés ;

    il recommande des films d’action qu’il n’a pas encore notés.

2. Structure finale

Crée cette structure :

text
movie-recommender/
│
├── data/
│   └── ml-latest-small/
│       ├── movies.csv
│       ├── ratings.csv
│       ├── tags.csv
│       └── links.csv
│
├── src/
│   ├── __init__.py
│   ├── recommender.py
│   └── evaluation.py
│
├── tests/
│   └── test_recommender.py
│
├── app.py
├── api.py
├── requirements.txt
├── README.md
└── .gitignore

3. Créer l’environnement Python

Dans Linux ou le terminal de VS Code :

bash
mkdir movie-recommender
cd movie-recommender

python3 -m venv venv
source venv/bin/activate

Sous Windows :

bash
python -m venv venv
venv\Scripts\activate

Crée requirements.txt :

text
pandas
numpy
scikit-learn
streamlit
fastapi
uvicorn[standard]
pydantic
pytest

Installe les dépendances :

bash
pip install -r requirements.txt

4. Télécharger MovieLens

Télécharge l’archive ml-latest-small.zip depuis la page officielle MovieLens de GroupLens.

Linux :

bash
mkdir -p data
cd data

wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip

cd ..

La structure doit être :

text
data/ml-latest-small/movies.csv
data/ml-latest-small/ratings.csv
data/ml-latest-small/tags.csv
data/ml-latest-small/links.csv

Vérifie les fichiers :

bash
ls data/ml-latest-small

5. Code du moteur de recommandation

Crée le fichier src/__init__.py :

python
"""
Package du système de recommandation de films.
"""

Crée ensuite src/recommender.py :

python
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """
    Système de recommandation de films.

    Il propose :
    - des recommandations basées sur le contenu ;
    - des recommandations personnalisées ;
    - la recherche de films ;
    - des statistiques sur les films et les utilisateurs.
    """

    def __init__(
        self,
        data_directory: str = "data/ml-latest-small"
    ):
        self.data_directory = Path(data_directory)

        self.movies_path = self.data_directory / "movies.csv"
        self.ratings_path = self.data_directory / "ratings.csv"
        self.tags_path = self.data_directory / "tags.csv"

        self._check_files()
        self._load_data()
        self._prepare_features()
        self._build_similarity_matrix()

    def _check_files(self) -> None:
        """
        Vérifie l'existence des fichiers nécessaires.
        """
        required_files = [
            self.movies_path,
            self.ratings_path,
            self.tags_path
        ]

        missing_files = [
            str(file)
            for file in required_files
            if not file.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "Fichiers manquants : "
                + ", ".join(missing_files)
            )

    def _load_data(self) -> None:
        """
        Charge les fichiers CSV.
        """
        self.movies = pd.read_csv(self.movies_path)
        self.ratings = pd.read_csv(self.ratings_path)
        self.tags = pd.read_csv(self.tags_path)

        self.movies["title"] = (
            self.movies["title"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        self.movies["genres"] = (
            self.movies["genres"]
            .fillna("")
            .astype(str)
            .str.replace("|", " ", regex=False)
        )

        self.tags["tag"] = (
            self.tags["tag"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

    def _prepare_features(self) -> None:
        """
        Fusionne les genres et les tags pour créer les caractéristiques
        textuelles de chaque film.
        """
        tags_by_movie = (
            self.tags[
                self.tags["tag"].str.len() > 0
            ]
            .groupby("movieId")["tag"]
            .apply(lambda values: " ".join(values))
            .reset_index()
        )

        tags_by_movie = tags_by_movie.rename(
            columns={"tag": "movie_tags"}
        )

        self.movies = self.movies.merge(
            tags_by_movie,
            on="movieId",
            how="left"
        )

        self.movies["movie_tags"] = (
            self.movies["movie_tags"]
            .fillna("")
            .astype(str)
        )

        self.movies["features"] = (
            self.movies["genres"]
            + " "
            + self.movies["movie_tags"]
        ).str.lower()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )

        self.feature_matrix = self.vectorizer.fit_transform(
            self.movies["features"]
        )

    def _build_similarity_matrix(self) -> None:
        """
        Calcule la similarité entre tous les films.
        """
        self.similarity_matrix = cosine_similarity(
            self.feature_matrix
        )

        self.title_to_index = pd.Series(
            self.movies.index,
            index=self.movies["title"]
        ).drop_duplicates()

    def _find_movie_index(self, title: str) -> Optional[int]:
        """
        Trouve l'index d'un film à partir de son titre.
        La recherche accepte aussi une partie du titre.
        """
        if not title or not title.strip():
            return None

        title = title.strip()

        exact_matches = self.movies[
            self.movies["title"].str.lower() == title.lower()
        ]

        if not exact_matches.empty:
            return int(exact_matches.index[0])

        partial_matches = self.movies[
            self.movies["title"].str.contains(
                title,
                case=False,
                na=False,
                regex=False
            )
        ]

        if not partial_matches.empty:
            return int(partial_matches.index[0])

        return None

    def search_movies(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Recherche des films par titre.
        """
        if not query or not query.strip():
            return []

        results = self.movies[
            self.movies["title"].str.contains(
                query.strip(),
                case=False,
                na=False,
                regex=False
            )
        ].head(limit)

        return results[
            ["movieId", "title", "genres"]
        ].to_dict(orient="records")

    def recommend_similar(
        self,
        title: str,
        number_of_recommendations: int = 10
    ) -> List[Dict]:
        """
        Recommande des films similaires à un film donné.
        """
        movie_index = self._find_movie_index(title)

        if movie_index is None:
            return []

        similarity_scores = list(
            enumerate(self.similarity_matrix[movie_index])
        )

        similarity_scores.sort(
            key=lambda item: item[1],
            reverse=True
        )

        recommendations = []

        for index, score in similarity_scores:
            if index == movie_index:
                continue

            movie = self.movies.iloc[index]

            recommendations.append({
                "movieId": int(movie["movieId"]),
                "title": movie["title"],
                "genres": movie["genres"],
                "similarity": round(float(score), 4)
            })

            if len(recommendations) >= number_of_recommendations:
                break

        return recommendations

    def _calculate_rating_statistics(self) -> pd.DataFrame:
        """
        Calcule la note moyenne et le nombre de notes par film.
        """
        statistics = (
            self.ratings
            .groupby("movieId")["rating"]
            .agg(
                average_rating="mean",
                number_of_ratings="count"
            )
            .reset_index()
        )

        return statistics

    def recommend_for_user(
        self,
        user_id: int,
        number_of_recommendations: int = 10,
        minimum_rating: float = 4.0
    ) -> List[Dict]:
        """
        Recommande des films personnalisés à un utilisateur.

        Les films bien notés par l'utilisateur construisent son profil.
        """
        user_ratings = self.ratings[
            self.ratings["userId"] == user_id
        ]

        if user_ratings.empty:
            return []

        liked_movies = user_ratings[
            user_ratings["rating"] >= minimum_rating
        ]

        if liked_movies.empty:
            return []

        liked_movie_ids = liked_movies["movieId"].tolist()

        liked_indices = self.movies[
            self.movies["movieId"].isin(liked_movie_ids)
        ].index.tolist()

        if not liked_indices:
            return []

        liked_ratings = liked_movies.merge(
            self.movies[["movieId"]],
            on="movieId",
            how="inner"
        )

        rating_by_movie = dict(
            zip(
                liked_ratings["movieId"],
                liked_ratings["rating"]
            )
        )

        scores = np.zeros(len(self.movies))

        for movie_index in liked_indices:
            movie_id = int(
                self.movies.iloc[movie_index]["movieId"]
            )

            user_rating = rating_by_movie.get(
                movie_id,
                minimum_rating
            )

            scores += (
                self.similarity_matrix[movie_index]
                * float(user_rating)
            )

        already_rated_ids = set(
            user_ratings["movieId"].tolist()
        )

        candidate_indices = [
            index
            for index in range(len(self.movies))
            if int(self.movies.iloc[index]["movieId"])
            not in already_rated_ids
        ]

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: scores[index],
            reverse=True
        )

        recommendations = []

        for index in ranked_indices[
            :number_of_recommendations
        ]:
            movie = self.movies.iloc[index]

            recommendations.append({
                "movieId": int(movie["movieId"]),
                "title": movie["title"],
                "genres": movie["genres"],
                "score": round(float(scores[index]), 4)
            })

        return recommendations

    def recommend_by_genre(
        self,
        genre: str,
        number_of_recommendations: int = 10,
        minimum_ratings: int = 20
    ) -> List[Dict]:
        """
        Recommande les films les mieux notés d'un genre donné.
        """
        if not genre or not genre.strip():
            return []

        statistics = self._calculate_rating_statistics()

        movies_with_statistics = self.movies.merge(
            statistics,
            on="movieId",
            how="left"
        )

        filtered = movies_with_statistics[
            movies_with_statistics["genres"].str.contains(
                genre.strip(),
                case=False,
                na=False,
                regex=False
            )
            &
            (
                movies_with_statistics["number_of_ratings"]
                >= minimum_ratings
            )
        ]

        filtered = filtered.sort_values(
            by=["average_rating", "number_of_ratings"],
            ascending=False
        ).head(number_of_recommendations)

        results = []

        for _, movie in filtered.iterrows():
            results.append({
                "movieId": int(movie["movieId"]),
                "title": movie["title"],
                "genres": movie["genres"],
                "average_rating": round(
                    float(movie["average_rating"]),
                    2
                ),
                "number_of_ratings": int(
                    movie["number_of_ratings"]
                )
            })

        return results

    def get_user_ids(self) -> List[int]:
        """
        Retourne les identifiants des utilisateurs.
        """
        return sorted(
            self.ratings["userId"]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )

    def get_genres(self) -> List[str]:
        """
        Retourne la liste des genres disponibles.
        """
        genres = set()

        for movie_genres in self.movies["genres"]:
            for genre in movie_genres.split():
                if genre and genre != "(no":
                    genres.add(genre)

        return sorted(genres)

    def get_statistics(self) -> Dict:
        """
        Retourne des statistiques générales sur le dataset.
        """
        return {
            "number_of_movies": int(
                self.movies["movieId"].nunique()
            ),
            "number_of_users": int(
                self.ratings["userId"].nunique()
            ),
            "number_of_ratings": int(
                len(self.ratings)
            ),
            "average_rating": round(
                float(self.ratings["rating"].mean()),
                2
            )
        }

6. Explication du moteur
Chargement des données

python
self.movies = pd.read_csv(self.movies_path)
self.ratings = pd.read_csv(self.ratings_path)
self.tags = pd.read_csv(self.tags_path)

Cette partie charge les fichiers CSV dans des DataFrames Pandas.

    movies contient les informations des films.

    ratings contient les notes des utilisateurs.

    tags contient les mots-clés associés aux films.

Création des caractéristiques

Les genres ont cette forme :

text
Action|Adventure|Sci-Fi

Nous les transformons en :

text
Action Adventure Sci-Fi

Les tags sont ensuite ajoutés :

text
Action Adventure Sci-Fi futuristic space robot

Transformation TF-IDF

TfidfVectorizer transforme un ensemble de textes en matrice de caractéristiques numériques. Cette méthode donne plus d’importance aux mots utiles et réduit l’importance des mots trop fréquents.

Exemple conceptuel :

text
Film A : action adventure space
Film B : action adventure robot
Film C : romance drama love

Les films A et B auront des vecteurs proches, tandis que le film C sera plus éloigné.
Similarité cosinus

La similarité cosinus donne une valeur généralement comprise entre 0 et 1 :

    1 : films très similaires ;

    0 : films sans caractéristiques communes.

7. Tester le moteur

Crée test_recommender.py à la racine :

python
from src.recommender import MovieRecommender


def main():
    recommender = MovieRecommender()

    print("Statistiques :")
    print(recommender.get_statistics())

    print("\nRecherche de films :")
    search_results = recommender.search_movies("Toy Story")

    for movie in search_results:
        print(movie)

    print("\nFilms similaires :")
    recommendations = recommender.recommend_similar(
        "Toy Story (1995)",
        number_of_recommendations=10
    )

    for movie in recommendations:
        print(movie)

    print("\nRecommandations pour l'utilisateur 1 :")
    user_recommendations = recommender.recommend_for_user(
        user_id=1,
        number_of_recommendations=10
    )

    for movie in user_recommendations:
        print(movie)

    print("\nMeilleurs films d'action :")
    action_movies = recommender.recommend_by_genre(
        genre="Action",
        number_of_recommendations=10
    )

    for movie in action_movies:
        print(movie)


if __name__ == "__main__":
    main()

Lance :

bash
python test_recommender.py

8. Interface Streamlit complète

Crée app.py :

python
import streamlit as st

from src.recommender import MovieRecommender


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def load_recommender():
    """
    Charge le modèle une seule fois.
    """
    return MovieRecommender()


@st.cache_data
def get_statistics():
    """
    Met en cache les statistiques.
    """
    return recommender.get_statistics()


try:
    recommender = load_recommender()
except FileNotFoundError as error:
    st.error(str(error))
    st.info(
        "Vérifie que les fichiers MovieLens sont dans "
        "data/ml-latest-small/"
    )
    st.stop()


st.title("🎬 Système intelligent de recommandation de films")

st.write(
    "Explore des films similaires, personnalisés ou populaires "
    "par genre."
)

with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Choisis une fonctionnalité",
        [
            "Films similaires",
            "Recommandation personnalisée",
            "Meilleurs films par genre",
            "Recherche",
            "Statistiques"
        ]
    )


if page == "Films similaires":
    st.header("Recommander des films similaires")

    movie_titles = sorted(
        recommender.movies["title"].tolist()
    )

    selected_movie = st.selectbox(
        "Sélectionne un film",
        movie_titles
    )

    number_of_recommendations = st.slider(
        "Nombre de recommandations",
        min_value=1,
        max_value=30,
        value=10
    )

    if st.button(
        "Trouver des films similaires",
        type="primary"
    ):
        results = recommender.recommend_similar(
            selected_movie,
            number_of_recommendations
        )

        if not results:
            st.warning(
                "Aucune recommandation trouvée."
            )
        else:
            st.subheader(
                f"Films similaires à : {selected_movie}"
            )

            for position, movie in enumerate(
                results,
                start=1
            ):
                st.write(
                    f"**{position}. {movie['title']}**"
                )
                st.write(
                    f"Genres : {movie['genres']}  \n"
                    f"Similarité : {movie['similarity']}"
                )
                st.divider()


elif page == "Recommandation personnalisée":
    st.header("Recommandation personnalisée")

    user_ids = recommender.get_user_ids()

    selected_user = st.selectbox(
        "Sélectionne un utilisateur",
        user_ids
    )

    minimum_rating = st.slider(
        "Note minimale considérée comme positive",
        min_value=3.0,
        max_value=5.0,
        value=4.0,
        step=0.5
    )

    number_of_recommendations = st.slider(
        "Nombre de recommandations",
        min_value=1,
        max_value=30,
        value=10
    )

    if st.button(
        "Recommander pour cet utilisateur",
        type="primary"
    ):
        results = recommender.recommend_for_user(
            user_id=selected_user,
            number_of_recommendations=number_of_recommendations,
            minimum_rating=minimum_rating
        )

        if not results:
            st.warning(
                "Cet utilisateur ne possède pas assez de notes "
                "positives."
            )
        else:
            st.subheader(
                f"Recommandations pour l'utilisateur "
                f"{selected_user}"
            )

            for position, movie in enumerate(
                results,
                start=1
            ):
                st.write(
                    f"**{position}. {movie['title']}**"
                )
                st.write(
                    f"Genres : {movie['genres']}  \n"
                    f"Score : {movie['score']}"
                )
                st.divider()


elif page == "Meilleurs films par genre":
    st.header("Meilleurs films par genre")

    genres = recommender.get_genres()

    selected_genre = st.selectbox(
        "Choisis un genre",
        genres
    )

    number_of_recommendations = st.slider(
        "Nombre de films",
        min_value=1,
        max_value=30,
        value=10
    )

    minimum_ratings = st.slider(
        "Nombre minimal de notes",
        min_value=1,
        max_value=200,
        value=20
    )

    if st.button(
        "Afficher les meilleurs films",
        type="primary"
    ):
        results = recommender.recommend_by_genre(
            genre=selected_genre,
            number_of_recommendations=
            number_of_recommendations,
            minimum_ratings=minimum_ratings
        )

        if not results:
            st.warning(
                "Aucun film trouvé pour ce genre."
            )
        else:
            for position, movie in enumerate(
                results,
                start=1
            ):
                st.write(
                    f"**{position}. {movie['title']}**"
                )
                st.write(
                    f"Genres : {movie['genres']}  \n"
                    f"Note moyenne : "
                    f"{movie['average_rating']}  \n"
                    f"Nombre de notes : "
                    f"{movie['number_of_ratings']}"
                )
                st.divider()


elif page == "Recherche":
    st.header("Rechercher un film")

    query = st.text_input(
        "Titre ou partie du titre"
    )

    limit = st.slider(
        "Nombre maximal de résultats",
        min_value=1,
        max_value=50,
        value=10
    )

    if query:
        results = recommender.search_movies(
            query=query,
            limit=limit
        )

        if not results:
            st.info(
                "Aucun film correspondant trouvé."
            )
        else:
            for movie in results:
                st.write(
                    f"**{movie['title']}**"
                )
                st.caption(
                    f"Genres : {movie['genres']}"
                )


elif page == "Statistiques":
    st.header("Statistiques du dataset")

    statistics = recommender.get_statistics()

    column_1, column_2, column_3, column_4 = st.columns(4)

    column_1.metric(
        "Films",
        statistics["number_of_movies"]
    )

    column_2.metric(
        "Utilisateurs",
        statistics["number_of_users"]
    )

    column_3.metric(
        "Évaluations",
        statistics["number_of_ratings"]
    )

    column_4.metric(
        "Note moyenne",
        statistics["average_rating"]
    )

    st.write(
        "Les données utilisées proviennent de MovieLens."
    )

Lance l’application avec :

bash
streamlit run app.py

La documentation Streamlit indique également la commande python -m streamlit run app.py comme alternative.

Si la commande streamlit n’est pas reconnue :

bash
python -m streamlit run app.py

9. API FastAPI complète

Crée api.py :

python
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.recommender import MovieRecommender


app = FastAPI(
    title="Movie Recommendation API",
    description=(
        "API de recommandation de films basée sur MovieLens."
    ),
    version="1.0.0"
)

recommender = MovieRecommender()


class MovieRecommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    similarity: Optional[float] = None
    score: Optional[float] = None
    average_rating: Optional[float] = None
    number_of_ratings: Optional[int] = None


@app.get("/")
def home():
    return {
        "message": "Movie Recommendation API",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get(
    "/movies/search",
    response_model=List[dict]
)
def search_movies(
    query: str = Query(
        ...,
        min_length=1,
        description="Titre ou partie du titre"
    ),
    limit: int = Query(
        10,
        ge=1,
        le=100
    )
):
    return recommender.search_movies(
        query=query,
        limit=limit
    )


@app.get(
    "/recommendations/similar",
    response_model=List[MovieRecommendation]
)
def similar_movies(
    title: str = Query(
        ...,
        min_length=1
    ),
    limit: int = Query(
        10,
        ge=1,
        le=100
    )
):
    recommendations = recommender.recommend_similar(
        title=title,
        number_of_recommendations=limit
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="Film introuvable."
        )

    return recommendations


@app.get(
    "/recommendations/user/{user_id}",
    response_model=List[MovieRecommendation]
)
def user_recommendations(
    user_id: int,
    limit: int = Query(
        10,
        ge=1,
        le=100
    ),
    minimum_rating: float = Query(
        4.0,
        ge=0.5,
        le=5.0
    )
):
    if user_id not in recommender.get_user_ids():
        raise HTTPException(
            status_code=404,
            detail="Utilisateur introuvable."
        )

    recommendations = recommender.recommend_for_user(
        user_id=user_id,
        number_of_recommendations=limit,
        minimum_rating=minimum_rating
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail=(
                "Aucune recommandation disponible pour "
                "cet utilisateur."
            )
        )

    return recommendations


@app.get(
    "/recommendations/genre",
    response_model=List[MovieRecommendation]
)
def genre_recommendations(
    genre: str = Query(
        ...,
        min_length=1
    ),
    limit: int = Query(
        10,
        ge=1,
        le=100
    ),
    minimum_ratings: int = Query(
        20,
        ge=1
    )
):
    recommendations = recommender.recommend_by_genre(
        genre=genre,
        number_of_recommendations=limit,
        minimum_ratings=minimum_ratings
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="Aucun film trouvé pour ce genre."
        )

    return recommendations


@app.get("/statistics")
def statistics():
    return recommender.get_statistics()

Démarre l’API :

bash
uvicorn api:app --reload

L’API sera disponible ici :

text
http://127.0.0.1:8000

La documentation automatique sera disponible ici :

text
http://127.0.0.1:8000/docs

Tester avec le navigateur :

text
http://127.0.0.1:8000/recommendations/similar?title=Toy%20Story%20(1995)&limit=5

Tester avec curl :

bash
curl "http://127.0.0.1:8000/recommendations/similar?title=Toy%20Story%20(1995)&limit=5"

10. Tests automatisés

Crée tests/test_recommender.py :

python
from src.recommender import MovieRecommender


def create_recommender():
    return MovieRecommender(
        data_directory="data/ml-latest-small"
    )


def test_dataset_is_loaded():
    recommender = create_recommender()

    assert len(recommender.movies) > 0
    assert len(recommender.ratings) > 0


def test_search_movies():
    recommender = create_recommender()

    results = recommender.search_movies(
        "Toy Story",
        limit=5
    )

    assert len(results) > 0
    assert "title" in results[0]


def test_similar_movies():
    recommender = create_recommender()

    results = recommender.recommend_similar(
        "Toy Story (1995)",
        number_of_recommendations=5
    )

    assert len(results) == 5
    assert "title" in results[0]
    assert "similarity" in results[0]


def test_unknown_movie():
    recommender = create_recommender()

    results = recommender.recommend_similar(
        "Film qui nexiste pas",
        number_of_recommendations=5
    )

    assert results == []


def test_user_recommendations():
    recommender = create_recommender()

    results = recommender.recommend_for_user(
        user_id=1,
        number_of_recommendations=5
    )

    assert len(results) <= 5

    if results:
        assert "title" in results[0]
        assert "score" in results[0]


def test_genre_recommendations():
    recommender = create_recommender()

    results = recommender.recommend_by_genre(
        genre="Action",
        number_of_recommendations=5
    )

    assert len(results) <= 5

    if results:
        assert "average_rating" in results[0]

Lance les tests :

bash
pytest -q

11. Évaluation simple

Crée src/evaluation.py :

python
from typing import List, Set

from src.recommender import MovieRecommender


def precision_at_k(
    recommended_movie_ids: List[int],
    relevant_movie_ids: Set[int],
    k: int
) -> float:
    """
    Calcule Precision@K.
    """
    if k <= 0:
        return 0.0

    top_k = recommended_movie_ids[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        movie_id in relevant_movie_ids
        for movie_id in top_k
    )

    return relevant_count / len(top_k)


def evaluate_user(
    recommender: MovieRecommender,
    user_id: int,
    k: int = 10,
    minimum_rating: float = 4.0
) -> float:
    """
    Évalue les recommandations pour un utilisateur.

    Cette version simple utilise les films bien notés comme
    éléments pertinents.
    """
    user_ratings = recommender.ratings[
        recommender.ratings["userId"] == user_id
    ]

    relevant_movie_ids = set(
        user_ratings[
            user_ratings["rating"] >= minimum_rating
        ]["movieId"].tolist()
    )

    recommendations = recommender.recommend_for_user(
        user_id=user_id,
        number_of_recommendations=k,
        minimum_rating=minimum_rating
    )

    recommended_movie_ids = [
        movie["movieId"]
        for movie in recommendations
    ]

    return precision_at_k(
        recommended_movie_ids,
        relevant_movie_ids,
        k
    )


if __name__ == "__main__":
    recommender = MovieRecommender()

    score = evaluate_user(
        recommender,
        user_id=1,
        k=10
    )

    print(f"Precision@10 : {score:.4f}")

Lance :

bash
python -m src.evaluation

Attention sur l’évaluation

Cette évaluation est pédagogique. Pour une vraie évaluation :

    Il faut séparer les notes d’entraînement et de test.

    Il faut cacher certaines notes de l’utilisateur.

    Le modèle doit recommander uniquement à partir des données d’entraînement.

    Il faut tester plusieurs utilisateurs.

    Il faut calculer la moyenne des scores.

12. Fichier README.md

Crée README.md :

text
# Movie Recommender

Système de recommandation de films basé sur MovieLens.

## Fonctionnalités

- Recommandation de films similaires.
- Recommandation personnalisée par utilisateur.
- Classement des films par genre.
- Recherche de films.
- Interface Streamlit.
- API FastAPI.
- Tests automatisés.

## Installation

```bash
git clone [https://github.com/TON_USERNAME/movie-recommender.git](https://github.com/TON_USERNAME/movie-recommender.git)
cd movie-recommender

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Dataset

Télécharger le dataset MovieLens Latest Small et placer les fichiers
dans :

```text
data/ml-latest-small/
```

Les fichiers nécessaires sont :

- movies.csv
- ratings.csv
- tags.csv
- links.csv

## Lancer Streamlit

```bash
streamlit run app.py
```

## Lancer l'API

```bash
uvicorn api:app --reload
```

Documentation :

```text
http://127.0.0.1:8000/docs
```

## Lancer les tests

```bash
pytest -q
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- FastAPI
- Pytest

## Méthode utilisée

Les genres et les tags sont transformés avec TF-IDF.
La similarité cosinus est ensuite utilisée pour trouver les films proches.

## Améliorations possibles

- Ajouter les affiches des films.
- Ajouter les genres comme filtres.
- Ajouter un système de connexion.
- Utiliser un filtrage collaboratif.
- Ajouter une base PostgreSQL.
- Ajouter une application React Native.
- Déployer l'API en ligne.

13. Fichier .gitignore

Crée .gitignore :

text
venv/
__pycache__/
*.pyc
.pytest_cache/
.streamlit/
.env
.DS_Store

Tu peux éviter de publier les fichiers du dataset sur GitHub si leur taille devient importante. Dans ce cas, ajoute :

text
data/ml-latest-small/

14. Publier sur GitHub

Initialise Git :

bash
git init
git add .
git commit -m "Initial movie recommendation system"

Crée un dépôt GitHub nommé :

text
movie-recommender

Puis exécute :

bash
git branch -M main
git remote add origin https://github.com/TON_USERNAME/movie-recommender.git
git push -u origin main

Remplace TON_USERNAME par ton nom d’utilisateur GitHub.
15. Problèmes fréquents
Erreur ModuleNotFoundError

Active l’environnement virtuel :

bash
source venv/bin/activate

Puis réinstalle les dépendances :

bash
pip install -r requirements.txt

Erreur FileNotFoundError

Vérifie que les fichiers sont bien présents :

bash
ls data/ml-latest-small

Tu dois voir :

text
links.csv
movies.csv
ratings.csv
tags.csv

Aucun film recommandé

Utilise le titre exact ou une partie du titre :

text
Toy Story

Le code accepte aussi :

text
Toy

Lancement Streamlit impossible

Utilise :

bash
python -m streamlit run app.py

L’API ne démarre pas

Utilise :

bash
python -m uvicorn api:app --reload

16. Amélioration suivante : affiches des films

Le fichier links.csv contient des identifiants permettant de relier les films à d’autres services. Pour une version professionnelle, tu peux :

    récupérer les affiches via une API publique ;

    stocker les URLs dans une base de données ;

    afficher les posters dans Streamlit ;

    ajouter une page détaillée pour chaque film.

Évite toutefois de télécharger massivement des images sans vérifier les conditions d’utilisation de l’API utilisée.
17. Amélioration avancée : filtrage collaboratif

La version actuelle compare le contenu des films. Une version plus avancée peut apprendre à partir des interactions utilisateurs.

Tu peux créer une matrice :

python
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

Cette matrice permet de comparer les utilisateurs selon leurs notes.

Une autre approche avancée est la factorisation matricielle :
R≈P×QT
R≈P×QT

où :

    RR est la matrice utilisateur-film ;

    PP représente les caractéristiques latentes des utilisateurs ;

    QQ représente les caractéristiques latentes des films.

Cela permet de découvrir des préférences cachées qui ne sont pas directement visibles dans les genres.