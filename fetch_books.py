import pandas as pd
from neo4j import GraphDatabase
import logging
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import os
import csv
import random
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def create_driver(uri, user, password, retries=3, backoff_factor=2):
    for attempt in range(retries):
        try:
            driver = GraphDatabase.driver(
                uri, auth=(user, password), max_connection_lifetime=30,
                max_connection_pool_size=50, connection_timeout=15
            )
            with driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j")
            return driver
        except Exception as e:
            logger.warning(f"Neo4j connection attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff_factor ** attempt)
    logger.error("Failed to connect to Neo4j after retries")
    exit(1)

driver = create_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

def load_kaggle_books(file_path="data/books.csv"):
    try:
        df = pd.read_csv(file_path, quoting=csv.QUOTE_ALL, escapechar='\\')
        df.columns = [col.strip() for col in df.columns]
        required_columns = ["bookID", "title", "authors", "average_rating", "isbn", "language_code", "num_pages", "ratings_count", "publisher"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in CSV: {missing_columns}")
            return pd.DataFrame()
        df = df[required_columns].head(1000)
        df["bookId"] = df["bookID"].apply(lambda x: f"BOOK_{x}")
        df["avgRating"] = df["average_rating"].fillna(4.0)
        df["ratingsCount"] = df["ratings_count"].fillna(0)
        df["isbn"] = df["isbn"].str.strip()  # Clean ISBN for image fetching
        df["genres"] = df["title"].apply(
            lambda x: ["Fantasy"] if any(k in str(x).lower() for k in ["harry potter", "lord of the rings", "hobbit", "dragon", "wizard", "magic", "quest", "adventure", "epic"]) else
                      ["Science Fiction"] if any(k in str(x).lower() for k in ["dune", "foundation", "space", "alien", "galaxy", "star", "robot", "future", "tech"]) else
                      ["Mystery"] if any(k in str(x).lower() for k in ["sherlock", "mystery", "detective", "crime", "thriller"]) else
                      ["Romance"] if any(k in str(x).lower() for k in ["pride and prejudice", "love", "romance", "heart", "passion"]) else
                      ["Non-Fiction"] if any(k in str(x).lower() for k in ["biography", "history", "science", "business"]) else
                      ["Fiction"]
        )
        df["description"] = df.apply(
            lambda x: f"{x['title']} by {x['authors']}, a {', '.join(x['genres']).lower()} book with {x['num_pages']} pages.",
            axis=1
        )
        df["url"] = df["isbn"].apply(lambda x: f"https://www.goodreads.com/book/isbn/{x}")
        logger.info(f"Loaded {len(df)} books from Kaggle data")
        return df
    except Exception as e:
        logger.error(f"Error loading Kaggle data: {e}")
        return pd.DataFrame()

def import_books_to_neo4j(books_df):
    def create_book(tx, book, index):
        query = """
        MERGE (b:Book {bookId: $bookId})
        SET b.title = $title, b.authors = $authors, b.avgRating = $avgRating,
            b.ratingsCount = $ratingsCount, b.language = $language_code,
            b.numPages = $num_pages, b.publisher = $publisher,
            b.description = $description, b.url = $url, b.isbn = $isbn
        """
        tx.run(query, **book)
        for genre in book["genres"]:
            genre_query = """
            MATCH (b:Book {bookId: $bookId})
            MERGE (g:Genre {name: $genre})
            MERGE (b)-[:HAS_GENRE]->(g)
            """
            tx.run(genre_query, bookId=book["bookId"], genre=genre)
        if book["description"]:
            try:
                tokens = word_tokenize(book["description"].lower())
                keywords = [t for t in tokens if t.isalpha() and len(t) > 3][:5]
                for keyword in keywords:
                    keyword_query = """
                    MATCH (b:Book {bookId: $bookId})
                    MERGE (k:Keyword {name: $keyword})
                    MERGE (b)-[:HAS_KEYWORD]->(k)
                    """
                    tx.run(keyword_query, bookId=book["bookId"], keyword=keyword)
            except Exception as e:
                logger.warning(f"Skipping keywords for {book['title']} due to NLTK error: {e}")
        logger.info(f"Imported book {index + 1}/{len(books_df)}: {book['title']}")

    try:
        with driver.session() as session:
            for index, (_, book) in enumerate(books_df.iterrows()):
                book_data = book.to_dict()
                session.execute_write(create_book, book_data, index)
        logger.info("Imported books to Neo4j")
    except Exception as e:
        logger.error(f"Error importing books: {e}")
    finally:
        logger.info("Closing Neo4j session")

def create_synthetic_users(books_df):
    def create_user(tx, user_id, book_ids, ratings):
        tx.run("MERGE (u:User {userId: $userId})", userId=user_id)
        query = """
        UNWIND $pairs AS pair
        MATCH (u:User {userId: $userId})
        MATCH (b:Book {bookId: pair.bookId})
        MERGE (u)-[:RATED {rating: pair.rating}]->(b)
        """
        pairs = [{"bookId": book_id, "rating": rating} for book_id, rating in zip(book_ids, ratings)]
        tx.run(query, userId=user_id, pairs=pairs)
        logger.info(f"Created user {user_id}")

    try:
        with driver.session() as session:
            for i in range(1, 101):
                sample_books = books_df.sample(n=min(20, len(books_df)))
                book_ids = sample_books["bookId"].tolist()
                ratings = [random.randint(1, 5) for _ in range(len(book_ids))]
                session.execute_write(create_user, f"U{i}", book_ids, ratings)
        logger.info("Created synthetic users")
    except Exception as e:
        logger.error(f"Error creating users: {e}")
    finally:
        logger.info("Closing Neo4j session")

if __name__ == "__main__":
    try:
        books_df = load_kaggle_books()
        if not books_df.empty:
            print("Loaded Books:")
            print(books_df[["title", "authors", "avgRating", "genres", "ratingsCount", "isbn"]].head())
            import_books_to_neo4j(books_df)
            create_synthetic_users(books_df)
        else:
            logger.warning("No books loaded")
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    finally:
        driver.close()
        logger.info("Neo4j driver closed")