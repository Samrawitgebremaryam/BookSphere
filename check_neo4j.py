from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
with driver.session() as session:
    session.run("RETURN 1")
print("Connected to Neo4j!")
driver.close()