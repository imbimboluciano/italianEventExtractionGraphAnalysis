from neo4j import GraphDatabase
import spacy
import re, pathlib
import pandas as pd

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "secondpaper"))


def drop_all_projection(tx):
    tx.run("""
    CALL gds.graph.drop('graph_projection') 
    YIELD graphName;
    """)

def drop_all_nodes(tx):
    tx.run("""
        MATCH (n)
        DETACH DELETE n""")


def create_and_link_nodes(tx, tweet):
    query = """
    UNWIND $entities as entity
    MERGE (e:Entity {name: entity})
    WITH e
    UNWIND $context as ctx
    MERGE (c:Context {name: ctx})
    MERGE (e)-[r:MENTIONED_WITH]->(c)
    ON CREATE SET r.weight = 1
    ON MATCH SET r.weight = r.weight + 1
    """
    tx.run(query, entities=tweet['entities'], context=tweet['context'])



nlp = spacy.load("en_core_web_sm")

def preprocess_tweet(tweet):
    # Remove URLs, mentions, non-ASCII characters, etc.
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    return tweet

def extract_entities_and_context(tweet):
    doc = nlp(tweet)
    entities = [ent.text for ent in doc.ents]
    context = [token.text for token in doc if not token.is_stop and not token.is_punct and token.text not in entities]
    return {'entities': entities, 'context': context}

# Example
dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "../dataset/keyword_english_tweets.csv"
tweets = pd.read_csv(dataset_path)
tweets = tweets['full_text'].to_list()
processed_tweets = [extract_entities_and_context(preprocess_tweet(tweet)) for tweet in tweets]



def detect_events(tx):

    tx.run("""
    MATCH (source:Entity)-[r:MENTIONED_WITH]->(target:Context)
    RETURN gds.graph.project('graph_projection', source, target)
    """)

    # Run community detection to partition the graph
    tx.run("""
    CALL gds.louvain.stream('graph_projection')
    YIELD nodeId, communityId, intermediateCommunityIds
    RETURN gds.util.asNode(nodeId).name AS name, communityId
    ORDER BY name ASC
    """)

    # Run PageRank to rank nodes within the community
    tx.run("""
    CALL gds.pageRank.stream('graph_projection')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).name AS name, score
    ORDER BY score DESC, name ASC
    """)


with driver.session() as session:
    session.execute_write(drop_all_nodes)
    #session.execute_write(drop_all_projection)
    for processed_tweet in processed_tweets:
        session.execute_write(create_and_link_nodes,processed_tweet)
    #session.execute_write(detect_events)
