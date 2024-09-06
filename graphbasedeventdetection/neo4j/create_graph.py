import pandas as pd
import pathlib
import networkx as nx
from neo4j import GraphDatabase
from itertools import combinations


uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "firstpaper"))

def keyword_lists(keywords):
    return keywords.split(',')
    
def drop_all_nodes(tx):
    tx.run("""MATCH (n) DETACH DELETE n""")

def create_keyword_nodes(tx,documents):
    tx.run("""
    UNWIND $documents AS data
    MERGE (k:Keyword {name: data.keyword})
    ON CREATE SET k.frequency = data.frequency
    ON MATCH SET k.frequency = data.frequency;

    """, documents=documents)

def create_cooccurrence_relationships(tx, documents):
    tx.run("""
    UNWIND $documents AS pair
    MATCH (k1:Keyword {name: pair.k1})
    MATCH (k2:Keyword {name: pair.k2})
    MERGE (k1)-[:CO_OCCURS_WITH]->(k2);
    """, documents=documents)


def fetch_graph(tx):
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    """
    result = tx.run(query)
    graph = nx.Graph()  # or use nx.Graph() for undirected graphs

    for record in result:
        node_a = record['n']
        node_b = record['m']
        relationship = record['r']

        node_a_properties = dict(node_a)
        node_b_properties = dict(node_b)
        
        node_a_name = node_a_properties.pop('name', None)
        node_b_name = node_b_properties.pop('name', None)


        # Add nodes and edges to the NetworkX graph
        graph.add_node(node_a.id, name=node_a_name, **node_a_properties)
        graph.add_node(node_b.id, name=node_b_name, **node_b_properties)
        
        # Add edge with its properties
        graph.add_edge(node_a.id, node_b.id, **relationship)

    return graph 

def update_centrality(tx, centrality):
    for node, centrality_value in centrality.items():
        query = """
        MATCH (n:Keyword) 
        WHERE id(n) = $id
        SET n.vbc = $centrality
        """
        tx.run(query, id=node, centrality=centrality_value)

def update_neo4j_with_centrality(tx, node1, node2, centrality):
    query = """
    MATCH (n:Keyword)-[r:CO_OCCURS_WITH]-(m:Keyword)
    WHERE id(n) = $node1 AND id(m) = $node2
    SET r.betweenness_centrality = $centrality
    """
    tx.run(query, node1=node1, node2=node2, centrality=centrality)



""""dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute() / "dataset/keyword_english_tweets.csv"
#dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute() / "dataset/keyword_italian_tweets.csv"
dataset = pd.read_csv(dataset_path, sep=';')
dataset = dataset.head(20)
keywords_extracted = dataset['keyword'].astype(dtype='str')
keywords_extracted = keywords_extracted.apply(keyword_lists)
keywords_extracted = keywords_extracted[keywords_extracted.apply(lambda x: len(x) > 1)]

list_of_lists_of_keyword = keywords_extracted.to_list()"""

list_of_lists_of_keyword = [
    ["cheongju", "flood", "preparation","evacuation"],
    ["cheongju", "preparation", "disaster","flood"],
    ["text", "disaster", "cheongju"],
    ["text","musim river", "police"],
    ["police", "musim river", "bank"]
]


all_keywords = []

for list_of_keyword in list_of_lists_of_keyword:
    for keyword in list_of_keyword:
        all_keywords.append(keyword)

all_keywords = pd.Series(all_keywords)
keywords_and_frequencies = all_keywords.value_counts().to_dict()

list_of_maps = [{"keyword": k, "frequency": v} for k, v in keywords_and_frequencies.items()]


co_occurrences_set = set()

for keyword_list in list_of_lists_of_keyword:

    pairs = combinations(keyword_list, 2)
    for pair in pairs:
        co_occurrences_set.add(tuple(sorted(pair)))

co_occurrences_list = [{"k1": pair[0], "k2": pair[1]} for pair in co_occurrences_set if pair[0] != pair[1]]

with driver.session() as session:
    session.execute_write(drop_all_nodes)
    session.execute_write(create_keyword_nodes, list_of_maps)
    session.execute_write(create_cooccurrence_relationships, co_occurrences_list)
    G = session.execute_read(fetch_graph)

    vbc = nx.betweenness_centrality(G, weight='weight')
    
    ebc = nx.edge_betweenness_centrality(G)
    print(ebc)
   
    session.execute_write(update_centrality, vbc)

    for edge, centrality in ebc.items():
        node1, node2 = edge
        session.write_transaction(update_neo4j_with_centrality, node1, node2, centrality)



