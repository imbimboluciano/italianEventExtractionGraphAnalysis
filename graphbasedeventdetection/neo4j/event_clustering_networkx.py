import networkx as nx
import math
import pathlib
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from neo4j import GraphDatabase



uri = "bolt://localhost:7687"
username = "neo4j"
password = "firstpaper"
driver = GraphDatabase.driver(uri, auth=(username, password))

def keyword_lists(keywords):
    return keywords.split(',')

def in_same_cluster(G, node1, node2):
    components = list(nx.connected_components(G))
    for component in components:
        if node1 in component and node2 in component:
            return True
    return False

def fetch_graph(tx):
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    """
    result = tx.run(query)
    graph = nx.Graph()  

    for record in result:
        node_a = record['n']
        node_b = record['m']
        relationship = record['r']

        node_a_properties = dict(node_a)
        node_b_properties = dict(node_b)
        
        node_a_name = node_a_properties.pop('name', None)
        node_b_name = node_b_properties.pop('name', None)


        graph.add_node(node_a.id, name=node_a_name, **node_a_properties)
        graph.add_node(node_b.id, name=node_b_name, **node_b_properties)
        
        graph.add_edge(node_a.id, node_b.id, **relationship)

    return graph

def remove_edge(tx, node1, node2):
    query = """
    MATCH (n1:Keyword)-[r:CO_OCCURS_WITH]->(n2:Keyword)
    WHERE id(n1) = $node1 AND id(n2) = $node2
    DELETE r
    """
    tx.run(query, node1=node1, node2=node2)

def remove_nodes(tx, removal_set):
    for node_id in removal_set:
        tx.run("MATCH (n:Keyword) WHERE id(n) = $node_id DETACH DELETE n", node_id=node_id)


def verify_and_cleanup(G, topicVertex, lambda_threshold, co_occurrences):
    # 1-Hop Verification
    one_hop_neighbors = list(nx.single_source_shortest_path_length(G, topicVertex, cutoff=1).keys())
    print(f"From {topicVertex} to {one_hop_neighbors}")
    for node in one_hop_neighbors:
            if node != topicVertex:
                co_occurrence_count = co_occurrences.get((G.nodes[topicVertex]['name'], G.nodes[node]['name']), 0)
                print(co_occurrence_count)
                if co_occurrence_count < lambda_threshold:
                    print(f"Remove node {node}")
                    G.remove_node(node)
    
    # More than 2-Hop Verification
    for node in list(G.nodes):
        if nx.shortest_path_length(G, source=topicVertex, target=node) > 1:
            path = nx.shortest_path(G, source=topicVertex, target=node)
            print(path)
            for i in range(len(path) - 1):
                co_occurrence_count = co_occurrences.get((G.nodes[path[i]]['name'], G.nodes[path[i + 1]]['name']), 0)
                if co_occurrence_count < lambda_threshold:
                    G.remove_node(node)
                    break



with driver.session() as session:
    G = session.execute_read(fetch_graph)


#dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute() / "dataset/keyword_italian_tweets.csv"
dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute() / "dataset/keyword_english_tweets.csv"
tweets_df = pd.read_csv(dataset_path,sep=";")

tweets_df = tweets_df.head(40)
keywords_extracted = tweets_df['keyword'].astype(dtype='str')
keywords_extracted = keywords_extracted.apply(keyword_lists)
keywords_extracted = keywords_extracted[keywords_extracted.apply(lambda x: len(x) > 1)]

list_of_list_of_keyword = keywords_extracted.to_list()


removal_set = []

vbc = nx.betweenness_centrality(G, weight='weight')
topicVertex = max(vbc, key=vbc.get)

for node in G.nodes:
    if node != topicVertex:
        try:
            distance = nx.shortest_path_length(G, source=topicVertex, target=node) 
            #The length of the path is always 1 less than the number of nodes involved in the path since the length measures the number of edges followed.
            if distance > 1:
                removal_set.append(node)
        except nx.NetworkXNoPath:
            # If there's no path, it's considered "infinitely" far away
            removal_set.append(node)

print("Nodes to remove due to distance > 2:", removal_set)

ebc = nx.edge_betweenness_centrality(G, weight='weight')

ebc_values = list(ebc.values())
m = np.mean(ebc_values)
sigma = np.std(ebc_values)

threshold = m + 2 * sigma
lambda_threshold = 0.4

print(threshold)

for edge,centrality in ebc.items():
    if centrality > threshold:
        u, v = edge
        
        # Duplicate the two vertices connected by the edge ei (u, v)
        u_new = f'{u}_copy'
        v_new = f'{v}_copy'

        G.add_node(u_new)
        G.add_node(v_new)

        # Connect the new vertices to the original neighbors
        for neighbor in list(G.neighbors(u)):
            G.add_edge(u_new, neighbor)

        for neighbor in list(G.neighbors(v)):
            G.add_edge(v_new, neighbor)

        G.remove_edge(u, v)
        G.add_edge(u, v_new)
        G.add_edge(v, u_new)

        print(f"Cut edge: {u}-{v}, and created new clusters with {u_new} and {v_new}")
    else:
        for vertex in removal_set:
            if not in_same_cluster(G, topicVertex, vertex):
                removal_set.remove(vertex)


print("Node to delete:", removal_set)
G.remove_nodes_from(removal_set)




with driver.session() as session:
    for edge,centrality in ebc.items():  # ebc is the edge betweenness centrality dictionary
        if not G.has_edge(*edge): 
            session.execute_write(remove_edge, edge[0], edge[1])

    session.execute_write(remove_nodes,removal_set)




co_occurrences = defaultdict(int)

for keyword_list in list_of_list_of_keyword:
    for pair in combinations(sorted(keyword_list), 2):
        co_occurrences[pair] += 1


co_occurrences = dict(co_occurrences)

connected_components = list(nx.connected_components(G))  # This is for an undirected graph

print(f"Number of clusters: {len(connected_components)}" )


"""subgraphs = [G.subgraph(component).copy() for component in connected_components]

removed_nodes = set()

for subgraph in subgraphs:
    original_nodes = set(subgraph.nodes())
    
    vbc = nx.betweenness_centrality(subgraph, weight='weight')
    topicVertex = max(vbc, key=vbc.get)
    
    verify_and_cleanup(subgraph, topicVertex, lambda_threshold, co_occurrences)
    
    removed = original_nodes - set(subgraph.nodes())
    removed_nodes.update(removed)
    with driver.session() as session:
        session.execute_write(remove_nodes, removed_nodes)"""


"""# Initialize data structures
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
for (u, v, data) in G.edges(data=True):
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
    data['weight'] = w_ij








# Step 5: Verification Step


# Step 6: Update Neo4j with the new edge weights
def update_edge_weights(tx, source, target, weight):
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

main()


# Apply the verify_and_cleanup function to each subgraph



driver.close()"""
