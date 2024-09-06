import spacy
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from neo4j import GraphDatabase
import numpy as np
import nltk
import pathlib
import math

# Load NLTK stop words
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load spaCy's small English model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Connect to Neo4j
uri = "bolt://localhost:7687"
username = "neo4j"
password = "secondpaper"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Fetch nodes and edges from Neo4j
def fetch_data_from_neo4j():
    query_nodes = """
    MATCH (n)
    RETURN n.name AS name, labels(n) AS label
    """
    query_edges = """
    MATCH (n)-[r:CO_OCCUR]->(m)
    RETURN n.name AS source, m.name AS target, r.weight AS weight
    """
    
    nodes = []
    edges = []

    with driver.session() as session:
        # Fetch nodes
        result_nodes = session.run(query_nodes)
        for record in result_nodes:
            nodes.append((record["name"], {"label": record["label"]}))

        # Fetch edges
        result_edges = session.run(query_edges)
        for record in result_edges:
            edges.append((record["source"], record["target"], record["weight"]))

    return nodes, edges

def create_networkx_graph(nodes, edges):
    G = nx.DiGraph()  # Directed graph

    # Add nodes to the graph
    for node, attrs in nodes:
        G.add_node(node, **attrs)

    # Add edges to the graph
    for source, target, weight in edges:
        G.add_edge(source, target, weight=weight)

    return G

# Compute TF-IDF scores for the nodes
def compute_tfidf_scores(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    
    # Create a dictionary of terms and their corresponding TF-IDF scores
    terms = vectorizer.get_feature_names_out()
    tfidf_dict = dict(zip(terms, tfidf_scores))
    
    return tfidf_dict

# Compute PageRank with TF-IDF as penalization parameter
def compute_pagerank(G, tfidf_dict, d=0.85, tol=1.0e-4):
    n = G.number_of_nodes()
    pagerank = dict.fromkeys(G, 1.0 / n)  # initial value
    damping_value = (1.0 - d) / n

    while True:
        diff = 0  # convergence difference
        new_pagerank = dict.fromkeys(G, 0)
        for node in G:
            penalty = tfidf_dict.get(node, 1)  # Get TF-IDF for node, default to 1
            for nbr in G.neighbors(node):
                new_pagerank[nbr] += d * pagerank[node] / G.out_degree(node)
            new_pagerank[node] += damping_value * penalty * 1000  # Add damping factor with penalty
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in G)
        if diff < tol:
            break
        pagerank = new_pagerank

    return pagerank

# Fetch data from Neo4j
nodes, edges = fetch_data_from_neo4j()

# Create the NetworkX graph
G = create_networkx_graph(nodes, edges)

# Load the tweets dataset
dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/cleaned_english_tweets.csv"
dataset = pd.read_csv(dataset_path, sep=';')
cleaned_tweets = dataset['cleaned_tweets'].tolist()

# Compute TF-IDF scores based on tweet content
tfidf_dict = compute_tfidf_scores(cleaned_tweets)

# Compute PageRank with TF-IDF penalization
pagerank_scores = compute_pagerank(G, tfidf_dict)

# Print the PageRank scores
for node, score in pagerank_scores.items():
    print(f"Node: {node}, PageRank: {score}")

# Close the Neo4j driver
driver.close()


def graph_processing(G, pagerank_scores, alpha):
    E = []  # List of important events
    H = [v for v in G.nodes if pagerank_scores[v] >= alpha]  # Nodes with PageRank above the threshold

    print(H)
    while H:
        # Create a copy of the graph
        G_prime = G.copy()
        vi = H.pop()  # Pop the last node from the list H
        keywords = set()
        max_in_weight = 0
        highest_predecessor = None
        for predecessor in G.predecessors(vi):
            weight = G[predecessor][vi].get('weight', 0)
            if weight > max_in_weight:
                max_in_weight = weight
                highest_predecessor = predecessor

        if highest_predecessor is not None:
            keywords.add(highest_predecessor)
            G_prime.remove_edge(highest_predecessor, vi)

        keywords.add(vi)

        # Find the highest weighted successor
        max_out_weight = 0
        highest_successor = None
        for successor in G.successors(vi):
            weight = G[vi][successor].get('weight', 0)
            if weight > max_out_weight:
                max_out_weight = weight
                highest_successor = successor

        print()
        if highest_successor is not None:
            keywords.add(highest_successor)
            G_prime.remove_edge(vi, highest_successor)
    
        print(keywords)
        # Check if removing edges disconnects the graph
        if not nx.is_connected(G_prime.to_undirected()):
            # Find disconnected vertices
            disc_vertices = list(nx.connected_components(G_prime.to_undirected()))
            for vertices in disc_vertices:
                for vertex in vertices:
                    keywords.add(vertex)
        
        print(keywords)
        # Determine "who," "where," and "what" entities
        who = {v for v in keywords if G.nodes[vi].get('label') in ['PERSON', 'ORG']}
        where = {v for v in keywords if G.nodes[vi].get('label') in ['GPE','LOC']}
        what = keywords - who - where

        # Get the corresponding tweets based on keywords
        tweets = [tweet for tweet in cleaned_tweets if any(keyword in tweet for keyword in keywords)]
        print(dataset[dataset['cleaned_tweets'] == tweets]['created at'])
        #when = min(tweets, key=lambda x: x.date) if tweets else None

        when = None
        # Form the event tuple and add it to the list of events E
        event = (what, who, where, when)
        E.append(event)
    # Merge events based on common "what," "who," and "where"
    """"for i, e in enumerate(E):
        for j, e_prime in enumerate(E):
            if i != j:
                if e[0] & e_prime[0]:  # Common "what"
                    if e[1] & e_prime[1]:  # Common "who"
                        E[j] = merge_events(e, e_prime)
                    if e[2] & e_prime[2]:  # Common "where"
                        E[j] = merge_events(e, e_prime)

    # Discard events without "who" or "where"
    E = [e for e in E if e[1] or e[2]]"""

    return E

def merge_events(event1, event2):
    # Merge two events by combining their attributes
    what = event1[0] | event2[0]
    who = event1[1] | event2[1]
    where = event1[2] | event2[2]
    when = min(event1[3], event2[3]) if event1[3] and event2[3] else event1[3] or event2[3]
    return (what, who, where, when)

# Example usage
alpha = 0.5 # Define alpha threshold for PageRank score
events = graph_processing(G, pagerank_scores, alpha)

