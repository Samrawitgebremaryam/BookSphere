import streamlit as st
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

def get_recommendations(user_id, mood):
    with driver.session() as session:
        query = """
        MATCH (u:User {userId: $userId})-[:RATED]->(b:Book)
        MATCH (b)-[:HAS_GENRE]->(g:Genre)
        WHERE g.name IN $genres
        RETURN b.title, b.url, b.avgRating
        ORDER BY b.avgRating DESC
        LIMIT 5
        """
        genres = ["Fantasy", "Science Fiction"] if mood == "adventurous" else ["Romance", "Fiction"]
        result = session.run(query, userId=user_id, genres=genres)
        return [(record["b.title"], record["b.url"], record["b.avgRating"]) for record in result]

st.set_page_config(page_title="BookSphere", page_icon="📚")
st.title("BookSphere Recommender")
st.markdown("Discover books tailored to your mood!")
user_id = st.selectbox("Select User", [f"U{i}" for i in range(1, 101)])
mood = st.selectbox("Select Mood", ["adventurous", "romantic"])
if st.button("Get Recommendations"):
    st.subheader(f"Recommendations for {user_id} (Mood: {mood})")
    recommendations = get_recommendations(user_id, mood)
    if recommendations:
        for title, url, rating in recommendations:
            st.markdown(f"- **{title}** (Rating: {rating}) [Link]({url})")
    else:
        st.write("No recommendations found. Try another mood!")

driver.close()