import pandas as pd
from neo4j import GraphDatabase
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
import logging
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def load_data():
    try:
        with driver.session() as session:
            # Load books
            books_query = """
            MATCH (b:Book)-[:HAS_GENRE]->(g:Genre)
            RETURN b.bookId AS bookId, b.title AS title, b.avgRating AS avgRating,
                   b.url AS url, collect(g.name) AS genres
            """
            books_result = session.run(books_query)
            books = []
            book_id_map = {}
            for idx, record in enumerate(books_result):
                books.append({
                    "bookId": record["bookId"],
                    "title": record["title"],
                    "avgRating": record["avgRating"],
                    "url": record["url"],
                    "genres": record["genres"]
                })
                book_id_map[record["bookId"]] = idx
            books_df = pd.DataFrame(books)

            # Load ratings
            ratings_query = """
            MATCH (u:User)-[r:RATED]->(b:Book)
            RETURN u.userId AS userId, b.bookId AS bookId, r.rating AS rating
            """
            ratings_result = session.run(ratings_query)
            ratings = []
            user_id_map = {}
            user_idx = 0
            for record in ratings_result:
                if record["userId"] not in user_id_map:
                    user_id_map[record["userId"]] = user_idx
                    user_idx += 1
                ratings.append({
                    "userId": record["userId"],
                    "bookId": record["bookId"],
                    "rating": record["rating"]
                })
            ratings_df = pd.DataFrame(ratings)
        logger.info(f"Loaded {len(books_df)} books and {len(ratings_df)} ratings from Neo4j")
        return books_df, ratings_df, book_id_map, user_id_map
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, None

def content_based_filtering(books_df):
    try:
        genres = books_df["genres"].explode().unique()
        genre_matrix = pd.DataFrame(0, index=books_df.index, columns=genres)
        for idx, row in books_df.iterrows():
            for genre in row["genres"]:
                genre_matrix.loc[idx, genre] = 1
        scaler = StandardScaler()
        genre_matrix_scaled = scaler.fit_transform(genre_matrix)
        nn = NearestNeighbors(n_neighbors=5, metric="cosine")
        nn.fit(genre_matrix_scaled)
        logger.info("Trained content-based filtering model")
        return nn, genre_matrix_scaled
    except Exception as e:
        logger.error(f"Content-based filtering error: {e}")
        return None, None

def collaborative_filtering(ratings_df, book_id_map, user_id_map):
    try:
        rating_matrix = pd.pivot_table(
            ratings_df,
            values="rating",
            index=[ratings_df["userId"].map(user_id_map)],
            columns=[ratings_df["bookId"].map(book_id_map)],
            fill_value=0
        )
        U, sigma, Vt = np.linalg.svd(rating_matrix, full_matrices=False)
        k = 50
        user_features = U[:, :k] @ np.diag(sigma[:k])
        book_features = Vt[:k, :]
        logger.info("Trained collaborative filtering model")
        return user_features, book_features, rating_matrix
    except Exception as e:
        logger.error(f"Collaborative filtering error: {e}")
        return None, None, None

def neural_collaborative_filtering(ratings_df, book_id_map, user_id_map):
    try:
        n_users = len(user_id_map)
        n_books = len(book_id_map)
        ratings_df["user_idx"] = ratings_df["userId"].map(user_id_map)
        ratings_df["book_idx"] = ratings_df["bookId"].map(book_id_map)
        ratings_df = ratings_df.dropna(subset=["user_idx", "book_idx"])
        ratings_df["user_idx"] = ratings_df["user_idx"].astype(int)
        ratings_df["book_idx"] = ratings_df["book_idx"].astype(int)

        # Ensure indices are within bounds
        if ratings_df["book_idx"].max() >= n_books or ratings_df["user_idx"].max() >= n_users:
            logger.warning("Invalid indices found in ratings data. Filtering...")
            ratings_df = ratings_df[
                (ratings_df["book_idx"] < n_books) & (ratings_df["user_idx"] < n_users)
            ]

        user_input = Input(shape=(1,), name="user")
        book_input = Input(shape=(1,), name="book")
        user_embedding = Embedding(n_users, 50, name="user_embedding")(user_input)
        book_embedding = Embedding(n_books, 50, name="book_embedding")(book_input)
        user_flat = Flatten()(user_embedding)
        book_flat = Flatten()(book_embedding)
        concat = Concatenate()([user_flat, book_flat])
        dense1 = Dense(128, activation="relu")(concat)
        dense2 = Dense(64, activation="relu")(dense1)
        output = Dense(1, activation="sigmoid")(dense2)

        model = Model(inputs=[user_input, book_input], outputs=output)
        model.compile(optimizer="adam", loss="mse")
        if not ratings_df.empty:
            model.fit(
                [ratings_df["user_idx"], ratings_df["book_idx"]],
                ratings_df["rating"] / 5.0,
                epochs=5,
                batch_size=32,
                verbose=1
            )
            logger.info("Trained neural collaborative filtering model")
        else:
            logger.warning("No valid ratings data for neural model")
            return None
        return model
    except Exception as e:
        logger.error(f"Neural collaborative filtering error: {e}")
        return None

def mood_based_filtering(books_df, mood):
    try:
        mood_genres = {
            "adventurous": ["Fantasy", "Science Fiction"],
            "romantic": ["Romance", "Fiction"]
        }
        selected_genres = mood_genres.get(mood, ["Fiction"])
        mood_books = books_df[books_df["genres"].apply(
            lambda x: any(genre in selected_genres for genre in x)
        )]
        logger.info(f"Selected {len(mood_books)} books for mood {mood}")
        return mood_books
    except Exception as e:
        logger.error(f"Mood-based filtering error: {e}")
        return pd.DataFrame()

def hybrid_recommendation(user_id, mood, books_df, ratings_df, book_id_map, user_id_map, nn, genre_matrix, user_features, book_features, neural_model):
    try:
        user_idx = user_id_map.get(user_id)
        if user_idx is None:
            logger.warning(f"User {user_id} not found")
            return []

        # Collaborative filtering
        collab_top = []
        collab_scores = []
        if user_features is not None and book_features is not None:
            collab_scores = user_features[user_idx] @ book_features
            collab_top = np.argsort(collab_scores)[-5:]

        # Content-based filtering
        content_top = []
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if not user_ratings.empty and nn is not None:
            book_idx = user_ratings["bookId"].map(book_id_map)
            book_idx = book_idx.dropna().astype(int)
            if len(book_idx) > 0:
                distances, indices = nn.kneighbors(genre_matrix[book_idx])
                content_top = indices.flatten()[:5]

        # Neural collaborative filtering
        neural_scores = []
        if neural_model:
            user_array = np.array([user_idx] * len(book_id_map))
            book_array = np.arange(len(book_id_map))
            neural_preds = neural_model.predict([user_array, book_array], verbose=0)
            neural_top = np.argsort(neural_preds.flatten())[-5:]
            neural_scores = neural_top.tolist()

        # Mood-based filtering
        mood_books = mood_based_filtering(books_df, mood)
        mood_book_indices = books_df[books_df["bookId"].isin(mood_books["bookId"])].index

        # Combine recommendations
        combined = set(collab_top).union(content_top, neural_scores, mood_book_indices)
        scores = {}
        for idx in combined:
            collab_score = float(collab_scores[idx]) if idx in collab_top and len(collab_scores) > idx else 0.0
            content_score = 1.0 if idx in content_top else 0.0
            neural_score = float(neural_preds[idx][0]) if idx in neural_scores and neural_model else 0.0
            mood_score = 1.0 if idx in mood_book_indices else 0.0
            scores[idx] = 0.4 * collab_score + 0.3 * content_score + 0.2 * neural_score + 0.1 * mood_score

        top_indices = sorted(scores, key=scores.get, reverse=True)[:5]
        recommendations = books_df.iloc[top_indices][["title", "url"]].to_dict("records")
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}, mood {mood}")
        return recommendations
    except Exception as e:
        logger.error(f"Hybrid recommendation error: {e}")
        return []

if __name__ == "__main__":
    try:
        books_df, ratings_df, book_id_map, user_id_map = load_data()
        if books_df is not None and ratings_df is not None:
            nn, genre_matrix = content_based_filtering(books_df)
            user_features, book_features, rating_matrix = collaborative_filtering(ratings_df, book_id_map, user_id_map)
            neural_model = neural_collaborative_filtering(ratings_df, book_id_map, user_id_map)
            recommendations = hybrid_recommendation(
                user_id="U1",
                mood="adventurous",
                books_df=books_df,
                ratings_df=ratings_df,
                book_id_map=book_id_map,
                user_id_map=user_id_map,
                nn=nn,
                genre_matrix=genre_matrix,
                user_features=user_features,
                book_features=book_features,
                neural_model=neural_model
            )
            print("Hybrid Recommendations for User U1 with mood adventurous")
            for rec in recommendations:
                print(f"- {rec['title']} (URL: {rec['url']})")
        else:
            logger.warning("No data loaded")
    finally:
        driver.close()
        logger.info("Neo4j driver closed")