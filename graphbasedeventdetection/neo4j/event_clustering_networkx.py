from neo4j import GraphDatabase
import networkx as nx
import math
from collections import defaultdict
from itertools import combinations
import pathlib
import pandas as pd

# Connect to Neo4j
uri = "bolt://localhost:7687"
username = "neo4j"
password = "firstpaper"
driver = GraphDatabase.driver(uri, auth=(username, password))

def keyword_lists(keywords):
    return keywords.split(',')

def keyword_string(keyword):
    return ','.join(keyword)

# Step 1: Fetch the graph from Neo4j
def fetch_graph(tx):
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    """
    result = tx.run(query)
    graph = nx.Graph()  # or use nx.Graph() for undirected graphs

    for record in result:
        print(record.data())
        node_a = record['n']
        node_b = record['m']
        name_a = record['n']['name']
        name_b = record['m']['name']
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


with driver.session() as session:
    G = session.execute_read(fetch_graph)

# Print node IDs, names, and other properties
for node in G.nodes(data=True):
    print(f"Node ID: {node[0]}, Name: {node[1].get('name')}, Properties: {node[1]}")

dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute() / "dataset/keyword_italian_tweets.csv"
tweets_df = pd.read_csv(dataset_path,sep=";")

tweets_df['keyword'] = tweets_df['keyword'].astype(dtype='str')
tweets_df['keyword'] = tweets_df['keyword'].apply(keyword_lists)
tweets_df = tweets_df[tweets_df['keyword'].apply(lambda x: len(x) > 1)]

tweets_df['keyword'] = tweets_df['keyword'].apply(keyword_string)
tweets_df['keyword'] = tweets_df['keyword'].fillna('').astype(str)


# Initialize data structures
retweets_t = defaultdict(int)
likes_t = defaultdict(int)
retweets_t_1 = defaultdict(int)
likes_t_1 = defaultdict(int)
co_occurrences = defaultdict(lambda: defaultdict(int))

# Populate the data structures
for i in range(len(tweets_df)):
    keywords = tweets_df['keyword'].iloc[i].split(',')
    retweet_count = tweets_df['retweet_count'].iloc[i]
    favorite_count = tweets_df['favorite_count'].iloc[i]
    retweet_count_previous = tweets_df['retweet_count_previous'].iloc[i]
    favorite_count_previous = tweets_df['favorite_count_previous'].iloc[i]
    
    for keyword in keywords:
        if keyword:  # Ensure the keyword is not empty
            # Update retweet and like counts
            retweets_t[keyword] += retweet_count
            likes_t[keyword] += favorite_count
            retweets_t_1[keyword] += retweet_count_previous
            likes_t_1[keyword] += favorite_count_previous

    # Update co-occurrences
    for combo in combinations(sorted(keywords), 2):
        co_occurrences[combo[0]][combo[1]] += 1
        co_occurrences[combo[1]][combo[0]] += 1

# Convert defaultdicts to regular dictionaries for easier handling
retweets_t = dict(retweets_t)
likes_t = dict(likes_t)
retweets_t_1 = dict(retweets_t_1)
likes_t_1 = dict(likes_t_1)
co_occurrences = {k: dict(v) for k, v in co_occurrences.items()}
print(co_occurrences)

print(retweets_t)

alpha = 0.9  # Example values, tune accordingly
beta = 0.5
mu = 0.8
lambda_threshold = 0.4  # Threshold value for simultaneous occurrence

# Replace None weights and compute new weights for each edge
"""for (u, v, data) in G.edges(data=True):
    rt_t = retweets_t.get(G.nodes[u].get('name'))
    ln_t = likes_t.get(G.nodes[u].get('name'))
    rt_t_1 = retweets_t_1.get(G.nodes[u].get('name'))
    ln_t_1 = likes_t_1.get(G.nodes[u].get('name'))
    print(rt_t)

    # Compute S_ij
    S_ij = (rt_t + ln_t) / (rt_t_1 + ln_t_1)
    print(S_ij)
    
    # Compute NS_ij
    NS_ij = mu * (beta ** S_ij - 1)
    
    # Compute F_ij
    CF_ij = co_occurrences.get((u, v), 1)
    F_ij = math.log(CF_ij)
    
    # Compute final weight
    w_ij = alpha * NS_ij + (1 - alpha) * F_ij
    
    # Assign the computed weight to the edge
    data['weight'] = w_ij"""

# Step 2: Compute Vertex Betweenness Centrality (VBC)
vbc = nx.betweenness_centrality(G, weight='weight')
topicVertex = max(vbc, key=vbc.get)

# Step 3: Identify nodes for removal based on distance
removal_set = set()

# Check the distance of each node from the topicVertex
for node in G.nodes:
    if node != topicVertex:
        try:
            # Using shortest path length to determine distance
            distance = nx.shortest_path_length(G, source=topicVertex, target=node)
            if distance > 2:
                removal_set.add(node)
        except nx.NetworkXNoPath:
            # If there's no path, it's considered "infinitely" far away
            removal_set.add(node)

print("Nodes to remove due to distance > 2:", removal_set)

# Compute Edge Betweenness Centrality (EBC)
ebc = nx.edge_betweenness_centrality(G, weight='weight')

# Calculate mean and standard deviation of EBC values
ebc_values = list(ebc.values())
mean_ebc = sum(ebc_values) / len(ebc_values)
std_dev_ebc = (sum((x - mean_ebc) ** 2 for x in ebc_values) / len(ebc_values)) ** 0.5

# Edge cutting and cluster identification
for edge, centrality in ebc.items():
    if centrality > (mean_ebc + 2 * std_dev_ebc):
        # If EBC is greater than mean + 2*std_dev, remove the edge
        G.remove_edge(*edge)
    else:
        # Check if nodes in removal set are still in the same cluster
        u, v = edge
        if u in removal_set and v in removal_set:
            if nx.has_path(G, u, v):
                removal_set.discard(u)
                removal_set.discard(v)

print("Final removal set:", removal_set)

def remove_nodes_from_neo4j(tx, removal_set):
    for node_id in removal_set:
        tx.run("MATCH (n:Keyword) WHERE id(n) = $node_id DETACH DELETE n", node_id=node_id)

with driver.session() as session:
    session.write_transaction(remove_nodes_from_neo4j, removal_set)



connected_components = list(nx.connected_components(G))  # This is for an undirected graph
# If G is directed, use nx.weakly_connected_components(G) or nx.strongly_connected_components(G)

print(f"Number of clusters: {len(connected_components)}" )
# Create subgraphs from each connected component
subgraphs = [G.subgraph(component).copy() for component in connected_components]


# Step 5: Verification Step


# Step 6: Update Neo4j with the new edge weights
"""def update_edge_weights(tx, source, target, weight):
    query = 
    MATCH (a:Keyword)-[r:COOCCURS_WITH]-(b:Keyword)
    WHERE id(a) = $source AND id(b) = $target
    SET r.weight = $weight
    
    tx.run(query, source=source, target=target, weight=weight)

def main():
    with driver.session() as session:
        for (u, v, data) in G.edges(data=True):
            session.execute_write(update_edge_weights, u, v, data['weight'])
    print("All edge weights updated successfully!")

main()"""
# Define the verify_and_cleanup function (given in your question)
def verify_and_cleanup(G, topicVertex, lambda_threshold):
    # 1-Hop Verification
    one_hop_neighbors = list(nx.single_source_shortest_path_length(G, topicVertex, cutoff=1).keys())
    for i, node1 in enumerate(one_hop_neighbors):
        for node2 in one_hop_neighbors[i+1:]:
            co_occurrence_count = co_occurrences.get((node1, node2), 0)
            if co_occurrence_count < lambda_threshold:
                G.remove_node(node2)
    
    # More than 2-Hop Verification
    for node in list(G.nodes):
        if nx.shortest_path_length(G, source=topicVertex, target=node) > 2:
            path = nx.shortest_path(G, source=topicVertex, target=node)
            for i in range(len(path) - 1):
                co_occurrence_count = co_occurrences.get((path[i], path[i + 1]), 0)
                if co_occurrence_count < lambda_threshold:
                    G.remove_node(node)
                    break

def remove_nodes(tx, nodes):
    query = """
    UNWIND $nodes AS node_id
    MATCH (n)
    WHERE id(n) = node_id
    DETACH DELETE n"""
    tx.run(query, nodes=list(nodes))

# Apply the verify_and_cleanup function to each subgraph
removed_nodes = set()

for subgraph in subgraphs:
    original_nodes = set(subgraph.nodes())
    
    vbc = nx.betweenness_centrality(subgraph, weight='weight')
    topicVertex = max(vbc, key=vbc.get)
    
    # Apply the verification and cleanup
    #verify_and_cleanup(subgraph, topicVertex, lambda_threshold)
    
    # Identify removed nodes
    removed = original_nodes - set(subgraph.nodes())
    removed_nodes.update(removed)
    with driver.session() as session:
        session.execute_write(remove_nodes, removed_nodes)


driver.close()
