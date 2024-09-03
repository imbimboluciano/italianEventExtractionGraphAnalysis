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
        WITH (CASE WHEN keyword1 < keyword2 THEN keyword1 ELSE keyword2 END) AS kw1,
             (CASE WHEN keyword1 < keyword2 THEN keyword2 ELSE keyword1 END) AS kw2
        MATCH (k1:Keyword {name: kw1}), (k2:Keyword {name: kw2})
        MERGE (k1)-[r:COOCCURS_WITH]->(k2);
        """, documents=documents)

#dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "../dataset/keyword_english_tweets.csv"
dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute() / "dataset/keyword_italian_tweets.csv"
dataset = pd.read_csv(dataset_path, sep=';')
keywords = dataset['keyword'].astype(dtype='str')
keywords = keywords.apply(keyword_lists)
keywords = keywords[keywords.apply(lambda x: len(x) > 1)]

keywords = keywords.to_list()

drop_all_nodes(driver)

create_keyword_nodes(driver, keywords)
create_cooccurrence_relationships(driver, keywords)

