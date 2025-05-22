import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import requests
import base64
from recommender import load_data, popularity_based_recommendation, collaborative_filtering, content_based_recommendation

st.set_page_config(page_title="BookSphere", page_icon="📚", layout="wide")

load_dotenv()
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

def get_book_cover(isbn):
    try:
        url = f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"
        response = requests.get(url, timeout=5)
        if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
            img_data = base64.b64encode(response.content).decode("utf-8")
            return f"data:image/jpeg;base64,{img_data}"
    except Exception:
        pass
    return "https://via.placeholder.com/150x200.png?text=No+Cover"

books_df, ratings_df = load_data()
book_titles = books_df["title"].dropna().unique().tolist() if books_df is not None else []

# Sidebar
st.sidebar.title("BookSphere Options")
page = st.sidebar.radio("Navigate", ["Home", "Top 50 Books", "Recommendations"])

# Learning Mode as clickable expanders with no background
st.sidebar.subheader("Learn About Recommendations")
with st.sidebar.expander("Similar to You"):
    st.markdown('<span style="color:#ccc;">Recommends books based on what users with similar tastes liked. Great for discovering new favorites!</span>', unsafe_allow_html=True)
with st.sidebar.expander("Similar to This Book"):
    st.markdown('<span style="color:#ccc;">Recommends books similar to the one you pick, based on genres and descriptions.</span>', unsafe_allow_html=True)

if page == "Home":
    st.title("📚 BookSphere: Discover Your Next Read")
    st.markdown("Pick a book title to get personalized recommendations or explore our top picks!")

    # Top 10 Books
    st.subheader("Top 10 Popular Books")
    top_books = popularity_based_recommendation(books_df, top_n=10)
    cols = st.columns(5)
    for idx, book in top_books.iterrows():
        with cols[idx % 5]:
            cover = get_book_cover(book["isbn"])
            st.markdown(f"""
                <div style="background-color: #222; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <img src="{cover}" style="width:100%; height:200px; object-fit:cover; border-radius:5px;">
                    <h4 style="color:#fff;">{book["title"]}</h4>
                    <p style="color:#ccc;"><b>Author:</b> {book["authors"]}</p>
                    <p style="color:#ccc;"><b>Rating:</b> {book["avgRating"]} ({book["ratingsCount"]} reviews)</p>
                    <p style="color:#ccc;"><b>Genres:</b> {', '.join(book["genres"])}</p>
                    <a href="{book["url"]}" target="_blank" style="color:#4FC3F7;">View on Goodreads</a>
                </div>
            """, unsafe_allow_html=True)

    # Quick Recommendation with selectbox
    st.subheader("Get a Quick Recommendation")
    book_title = st.selectbox("Pick a book title", [""] + book_titles)
    rec_type = st.selectbox("Recommend by", ["Similar to You", "Similar to This Book"])
    if st.button("Recommend"):
        if book_title:
            if rec_type == "Similar to You":
                recommendations = collaborative_filtering(ratings_df, books_df, book_title)
            else:
                recommendations = content_based_recommendation(books_df, book_title)
            if recommendations:
                st.subheader(f"Recommendations for '{book_title}' ({rec_type})")
                cols = st.columns(4)
                for idx, book in enumerate(recommendations):
                    with cols[idx % 4]:
                        cover = get_book_cover(book["isbn"])
                        st.markdown(f"""
                            <div style="background-color: #222; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                                <img src="{cover}" style="width:100%; height:200px; object-fit:cover; border-radius:5px;">
                                <h4 style="color:#fff;">{book["title"]}</h4>
                                <p style="color:#ccc;"><b>Author:</b> {book["authors"]}</p>
                                <p style="color:#ccc;"><b>Rating:</b> {book["avgRating"]}</p>
                                <p style="color:#ccc;"><b>Genres:</b> {', '.join(book["genres"])}</p>
                                <p style="color:#ccc;">{book["description"]}</p>
                                <a href="{book["url"]}" target="_blank" style="color:#4FC3F7;">View on Goodreads</a>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try another book title!")
        else:
            st.warning("Please pick a book title.")

elif page == "Top 50 Books":
    st.title("📚 Top 50 Books on BookSphere")
    top_books = popularity_based_recommendation(books_df, top_n=50)
    for idx, book in top_books.iterrows():
        cover = get_book_cover(book["isbn"])
        st.markdown(f"""
            <div style="background-color: #222; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                <img src="{cover}" style="width:150px; height:200px; object-fit:cover; border-radius:5px; float:left; margin-right:15px;">
                <h4 style="color:#fff;">{book["title"]}</h4>
                <p style="color:#ccc;"><b>Author:</b> {book["authors"]}</p>
                <p style="color:#ccc;"><b>Rating:</b> {book["avgRating"]} ({book["ratingsCount"]} reviews)</p>
                <p style="color:#ccc;"><b>Genres:</b> {', '.join(book["genres"])}</p>
                <p style="color:#ccc;">{book["description"]}</p>
                <a href="{book["url"]}" target="_blank" style="color:#4FC3F7;">View on Goodreads</a>
            </div>
        """, unsafe_allow_html=True)

elif page == "Recommendations":
    st.title("📚 Personalized Book Recommendations")
    book_title = st.selectbox("Pick a book title", [""] + book_titles, key="rec_page")
    rec_type = st.selectbox("Recommend by", ["Similar to You", "Similar to This Book"], key="rec_type_page")
    if st.button("Get Recommendations"):
        if book_title:
            if rec_type == "Similar to You":
                recommendations = collaborative_filtering(ratings_df, books_df, book_title)
            else:
                recommendations = content_based_recommendation(books_df, book_title)
            if recommendations:
                st.subheader(f"Recommendations for '{book_title}' ({rec_type})")
                cols = st.columns(4)
                for idx, book in enumerate(recommendations):
                    with cols[idx % 4]:
                        cover = get_book_cover(book["isbn"])
                        st.markdown(f"""
                            <div style="background-color: #222; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                                <img src="{cover}" style="width:100%; height:200px; object-fit:cover; border-radius:5px;">
                                <h4 style="color:#fff;">{book["title"]}</h4>
                                <p style="color:#ccc;"><b>Author:</b> {book["authors"]}</p>
                                <p style="color:#ccc;"><b>Rating:</b> {book["avgRating"]}</p>
                                <p style="color:#ccc;"><b>Genres:</b> {', '.join(book["genres"])}</p>
                                <p style="color:#ccc;">{book["description"]}</p>
                                <a href="{book["url"]}" target="_blank" style="color:#4FC3F7;">View on Goodreads</a>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try another book title!")
        else:
            st.warning("Please pick a book title.")

driver.close()