import pandas as pd
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def load_data():
    try:
        with driver.session() as session:
            books_query = """
            MATCH (b:Book)-[:HAS_GENRE]->(g:Genre)
            RETURN b.bookId AS bookId, b.title AS title, b.authors AS authors,
                   b.avgRating AS avgRating, b.ratingsCount AS ratingsCount,
                   b.url AS url, b.description AS description, b.isbn AS isbn,
                   collect(g.name) AS genres
            """
            books_result = session.run(books_query)
            books = []
            for record in books_result:
                books.append({
                    "bookId": record["bookId"],
                    "title": record["title"],
                    "authors": record["authors"],
                    "avgRating": record["avgRating"],
                    "ratingsCount": record["ratingsCount"],
                    "url": record["url"],
                    "description": record["description"],
                    "isbn": record["isbn"],
                    "genres": record["genres"]
                })
            books_df = pd.DataFrame(books)

            ratings_query = """
            MATCH (u:User)-[r:RATED]->(b:Book)
            RETURN u.userId AS userId, b.bookId AS bookId, r.rating AS rating
            """
            ratings_result = session.run(ratings_query)
            ratings = []
            for record in ratings_result:
                ratings.append({
                    "userId": record["userId"],
                    "bookId": record["bookId"],
                    "rating": record["rating"]
                })
            ratings_df = pd.DataFrame(ratings)
        logger.info(f"Loaded {len(books_df)} books and {len(ratings_df)} ratings from Neo4j")
        return books_df, ratings_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None

def popularity_based_recommendation(books_df, top_n=50):
    try:
        C = books_df["avgRating"].mean()
        m = 150  # Minimum reviews, as per request
        qualified_books = books_df[books_df["ratingsCount"] >= m].copy()
        qualified_books["weightedRating"] = (
            qualified_books["ratingsCount"] / (qualified_books["ratingsCount"] + m) * qualified_books["avgRating"]
        ) + (m / (qualified_books["ratingsCount"] + m) * C)
        top_books = qualified_books.sort_values("weightedRating", ascending=False).head(top_n)
        logger.info(f"Generated {len(top_books)} popularity-based recommendations")
        return top_books[["title", "authors", "avgRating", "ratingsCount", "url", "genres", "isbn", "description"]]
    except Exception as e:
        logger.error(f"Popularity-based recommendation error: {e}")
        return pd.DataFrame()

def collaborative_filtering(ratings_df, books_df, book_title, top_n=4):
    try:
        user_counts = ratings_df["userId"].value_counts()
        book_counts = ratings_df["bookId"].value_counts()
        ratings_filtered = ratings_df[
            (ratings_df["userId"].isin(user_counts[user_counts > 10].index)) &  # Relaxed for synthetic data
            (ratings_df["bookId"].isin(book_counts[book_counts > 5].index))
        ]
        if ratings_filtered.empty:
            logger.warning("Not enough data for collaborative filtering")
            return []

        pivot_table = ratings_filtered.pivot_table(
            index="userId", columns="bookId", values="rating", fill_value=0
        )
        similarity_matrix = cosine_similarity(pivot_table.T)
        book_ids = pivot_table.columns
        input_book = books_df[books_df["title"].str.lower().str.contains(book_title.lower(), na=False)]
        if input_book.empty:
            logger.warning(f"Book {book_title} not found")
            return []
        book_id = input_book.iloc[0]["bookId"]
        if book_id not in book_ids:
            logger.warning(f"Book {book_title} not in ratings")
            return []
        book_idx = list(book_ids).index(book_id)
        similar_scores = similarity_matrix[book_idx]
        similar_indices = similar_scores.argsort()[::-1][1:top_n+1]
        similar_book_ids = [book_ids[i] for i in similar_indices]
        recommendations = books_df[books_df["bookId"].isin(similar_book_ids)][
            ["title", "authors", "avgRating", "url", "genres", "isbn", "description"]
        ]
        logger.info(f"Generated {len(recommendations)} collaborative filtering recommendations for {book_title}")
        return recommendations.to_dict("records")
    except Exception as e:
        logger.error(f"Collaborative filtering error: {e}")
        return []

def content_based_recommendation(books_df, book_title, top_n=4):
    try:
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(books_df["description"].fillna(""))
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        input_book = books_df[books_df["title"].str.lower().str.contains(book_title.lower(), na=False)]
        if input_book.empty:
            logger.warning(f"Book {book_title} not found")
            return []
        book_idx = input_book.index[0]
        similar_scores = cosine_sim[book_idx]
        similar_indices = similar_scores.argsort()[::-1][1:top_n+1]
        recommendations = books_df.iloc[similar_indices][
            ["title", "authors", "avgRating", "url", "genres", "isbn", "description"]
        ]
        logger.info(f"Generated {len(recommendations)} content-based recommendations for {book_title}")
        return recommendations.to_dict("records")
    except Exception as e:
        logger.error(f"Content-based recommendation error: {e}")
        return []

def get_eligible_collaborative_books(ratings_df, books_df, min_user_ratings=10, min_book_ratings=5):
    """
    Returns a list of book titles that have enough ratings to be used in collaborative filtering.
    """
    user_counts = ratings_df["userId"].value_counts()
    book_counts = ratings_df["bookId"].value_counts()
    ratings_filtered = ratings_df[
        (ratings_df["userId"].isin(user_counts[user_counts > min_user_ratings].index)) &
        (ratings_df["bookId"].isin(book_counts[book_counts > min_book_ratings].index))
    ]
    eligible_book_ids = set(ratings_filtered["bookId"].unique())
    eligible_books_df = books_df[books_df["bookId"].isin(eligible_book_ids)]
    return eligible_books_df["title"].dropna().unique().tolist()

if __name__ == "__main__":
    try:
        books_df, ratings_df = load_data()
        if books_df is not None and ratings_df is not None:
            top_books = popularity_based_recommendation(books_df, top_n=10)
            print("Top 10 Popular Books:")
            print(top_books[["title", "avgRating", "ratingsCount"]])

            collab_recs = collaborative_filtering(ratings_df, books_df, "Harry Potter")
            print("\nCollaborative Filtering Recommendations for 'Harry Potter':")
            for rec in collab_recs:
                print(f"- {rec['title']} (Rating: {rec['avgRating']}, Genres: {rec['genres']})")

            content_recs = content_based_recommendation(books_df, "Harry Potter")
            print("\nContent-Based Recommendations for 'Harry Potter':")
            for rec in content_recs:
                print(f"- {rec['title']} (Rating: {rec['avgRating']}, Genres: {rec['genres']})")
        else:
            logger.warning("No data loaded")
    finally:
        driver.close()
        logger.info("Neo4j driver closed")