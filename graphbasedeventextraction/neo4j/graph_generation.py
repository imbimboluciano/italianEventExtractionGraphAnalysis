import spacy
from nltk.corpus import stopwords
from neo4j import GraphDatabase
import re, pathlib
import pandas as pd

# Load NLTK stop words
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load spaCy's small English model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Connect to Neo4j
uri = "bolt://localhost:7687"  # Adjust if needed
username = "neo4j"
password = "secondpaper"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Define number of surrounding words (k)
k = 2

def drop_all_nodes(tx):
    tx.run("""MATCH (n) DETACH DELETE n""")

def extract_entities_with_context(tweet):
    doc = nlp(tweet)
    entities_with_context = []
    
    # Identify named entities (NEs) and their positions
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC"}:
            start_index = ent.start
            end_index = ent.end
            
            # Extract context (k terms before and after)
            context_before = [token.text for token in doc[max(0, start_index-k):start_index] if token.text.lower() not in stop_words]
            context_after = [token.text for token in doc[end_index:min(len(doc), end_index+k)] if token.text.lower() not in stop_words]
            
            entities_with_context.append((ent.text,ent.label_, context_before, context_after))
    
    return entities_with_context


def create_neo4j_graph(tx, entities_with_context):
    for entity,label,context_before, context_after in entities_with_context:
        # Create a node for the entity (NE)
        tx.run("MERGE (n:Entity {name: $entity, label:$label})", entity=entity, label=label)
        
        # Create nodes and edges for the context terms before the NE
        for term in context_before:
            tx.run("MERGE (t:Term {name: $term})", term=term)
            tx.run("""
                MATCH (n:Entity {name: $entity}), (t:Term {name: $term})
                MERGE (n)-[r:CO_OCCUR]->(t)
                ON CREATE SET r.weight = 1
                ON MATCH SET r.weight = r.weight + 1
                """, entity=entity, term=term)
        
        # Create nodes and edges for the context terms after the NE
        for term in context_after:
            tx.run("MERGE (t:Term {name: $term})", term=term)
            tx.run("""
                MATCH (n:Entity {name: $entity}), (t:Term {name: $term})
                MERGE (n)-[r:CO_OCCUR]->(t)
                ON CREATE SET r.weight = 1
                ON MATCH SET r.weight = r.weight + 1
                """, entity=entity, term=term)

dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/cleaned_english_tweets.csv"
tweets = pd.read_csv(dataset_path, sep=';')
cleaned_tweet = tweets['cleaned_tweets']

# Clean tweets and process them
with driver.session() as session:
    session.execute_write(drop_all_nodes)
    for tweet in cleaned_tweet.to_list():
        entities_with_context = extract_entities_with_context(tweet)
        session.execute_write(create_neo4j_graph, entities_with_context)

# Close the Neo4j driver
driver.close()
