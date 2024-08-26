from neo4j import GraphDatabase
import pandas as pd
import pathlib

# Connect to the Neo4j database
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "firstpaper"))

def keyword_lists(keywords):
    return keywords.split(',')

def drop_all_nodes(driver):
    with driver.session() as session:
        session.run("""
        MATCH (n)
        DETACH DELETE n""")

def create_keyword_nodes(driver, documents):
    with driver.session() as session:
        session.run("""
        UNWIND $documents AS document
        UNWIND document AS keyword
        MERGE (k:Keyword {name: keyword});
        """, documents=documents)

def create_cooccurrence_relationships(driver, documents):
    with driver.session() as session:
        session.run("""
        UNWIND $documents AS document
        UNWIND document AS keyword1
        UNWIND document AS keyword2
        WITH keyword1, keyword2
        WHERE keyword1 <> keyword2
        MATCH (k1:Keyword {name: keyword1}), (k2:Keyword {name: keyword2})
        MERGE (k1)-[r:COOCCURS_WITH]-(k2)
        ON CREATE SET r.weight = 1
        ON MATCH SET r.weight = r.weight + 1;
        """, documents=documents)

# Example documents
dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/keyword_tweets.csv"
dataset = pd.read_csv(dataset_path, sep=',')
keywords = dataset.iloc[:,8]
keywords = keywords.apply(keyword_lists)

drop_all_nodes(driver)

create_keyword_nodes(driver, keywords)
create_cooccurrence_relationships(driver, keywords)

