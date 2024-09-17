from neo4j import GraphDatabase
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import pandas as pd
import numpy as np
import pathlib

uri = "bolt://localhost:7687"
username = "neo4j"
password = "secondpaper"
driver = GraphDatabase.driver(uri, auth=(username, password))

def remove_offset(date_string):
    date_string = str(datetime.strptime(date_string, "%a %b %d %H:%M:%S %z %Y")) # Only for italian
    if date_string.endswith('+00:00'):
        return date_string[:-6]  
    return date_string


def compute_community_detection(tx):
    query = "CALL gds.graph.drop('myGraph') YIELD graphName;"
    tx.run(query)

    query = "CALL gds.graph.project('myGraph', 'Term', 'POINTS_TO')"
    tx.run(query)

    query = """CALL gds.louvain.write('myGraph', { writeProperty: 'community' })
    YIELD communityCount, modularity, modularities"""
    tx.run(query)



def fetch_subgraphs(tx):
    query = "MATCH (n:Term) RETURN DISTINCT n.community as community"
    communities = tx.run(query).data()

    subgraphs = {}
    
    for record in communities:
        community_id = record["community"]

        query_nodes = f"""
        MATCH (n:Term)
        WHERE n.community = {community_id}
        RETURN n.name AS name, n.label AS label
        """
        query_edges = f"""
        MATCH (n:Term)-[r:POINTS_TO]-(m:Term)
        WHERE n.community = {community_id} AND m.community = {community_id}
        RETURN n.name AS source, m.name AS target, r.weight AS weight
        """
    
        nodes = []
        edges = []

        with driver.session() as session:
       
            result_nodes = session.run(query_nodes)
            for record in result_nodes:
                nodes.append((record["name"], {"label": record["label"]}))

           
            result_edges = session.run(query_edges)
            for record in result_edges:
                edges.append((record["source"], record["target"], record["weight"]))

        G = nx.DiGraph()  # Directed graph

        
        for node, attrs in nodes:
            G.add_node(node, **attrs)

       
        for source, target, weight in edges:
            G.add_edge(source, target, weight=weight)

        
            
        subgraphs[community_id] = G
    
    return subgraphs

def compute_tfidf_scores(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    
    terms = vectorizer.get_feature_names_out()
    tfidf_dict = dict(zip(terms, tfidf_scores))
    
    return tfidf_dict


def compute_pagerank(G, tfidf_dict, d=0.85, tol=1.0e-4):
    n = G.number_of_nodes()
    if n == 0:
        n = 1
    pagerank = dict.fromkeys(G, 1.0 / n)  
    damping_value = (1.0 - d) / n

    while True:
        diff = 0  
        new_pagerank = dict.fromkeys(G, 0)
        for node in G:
            penalty = tfidf_dict.get(node, 1)  
            for nbr in G.neighbors(node):
                new_pagerank[nbr] += d * pagerank[node] / G.out_degree(node)
            new_pagerank[node] += damping_value * penalty * 1000  # Add damping factor with penalty
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in G)
        if diff < tol:
            break
        pagerank = new_pagerank

    return pagerank


def graph_processing(G, pagerank_scores, alpha):
    E = [] 
    H = [v for v in G.nodes if pagerank_scores[v] >= alpha]  

    while H:
        G_prime = G.copy()
        vi = H.pop()  
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
            if G_prime.has_edge(vi, highest_predecessor):
                G_prime.remove_edge(highest_predecessor, vi)

        keywords.add(vi)

        
        max_out_weight = 0
        highest_successor = None
        for successor in G.successors(vi):
            weight = G[vi][successor].get('weight', 0)
            if weight > max_out_weight:
                max_out_weight = weight
                highest_successor = successor

        if highest_successor is not None:
            keywords.add(highest_successor)
            if G_prime.has_edge(vi, highest_successor):
                G_prime.remove_edge(vi, highest_successor)
    
        if not nx.is_connected(G_prime.to_undirected()):
            
            disc_vertices = list(nx.connected_components(G_prime.to_undirected()))
            for vertices in disc_vertices:
                for vertex in vertices:
                    keywords.add(vertex)
        
        
        who = {v for v in keywords if G.nodes[v].get('label') in ['PERSON', 'ORG']}
        where = {v for v in keywords if G.nodes[v].get('label') in ['GPE','LOC']}
        what = keywords - who - where


        dataset['cleaned_date'] = dataset['date'].apply(remove_offset)
        dataset['datetime'] = pd.to_datetime(dataset['cleaned_date'])
        filtered_df = dataset[dataset['cleaned_tweets'].apply(lambda x: any(keyword in x for keyword in keywords))]
        when = None
        if not filtered_df.empty:
            when = filtered_df.loc[filtered_df['datetime'].idxmin()]['datetime']

        event = {'what': what, 'who': who, 'where': where, 'when': when}
        E.append(event)

    return E

def merge_events(events):
    merged_events = events.copy()
    to_remove = set()

    for i, e in enumerate(merged_events):
        for j, e_prime in enumerate(merged_events[i+1:], start=i+1):
            if set(e['what']) & set(e_prime['what']):  
                if set(e['who']) & set(e_prime['who']) or set(e['where']) & set(e_prime['where']):
                   
                    merged_event = {
                        'what': list(set(e['what']) | set(e_prime['what'])),
                        'who': list(set(e['who']) | set(e_prime['who'])),
                        'where': list(set(e['where']) | set(e_prime['where'])),
                        'when': min(e['when'], e_prime['when'])  
                    }
                    merged_events[i] = merged_event
                    to_remove.add(j)

    
    merged_events = [e for i, e in enumerate(merged_events) if i not in to_remove]

    final_events = [e for e in merged_events if e['who'] or e['where']]

    return final_events


with driver.session() as session:
    session.execute_write(compute_community_detection)
    subgraphs = session.execute_read(fetch_subgraphs)


dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/cleaned_english_tweets.csv"
dataset_path = pathlib.Path(__file__).parent.parent.absolute() / "dataset/cleaned_italian_tweets.csv"
dataset = pd.read_csv(dataset_path, sep=';')
cleaned_tweets = dataset['cleaned_tweets'].tolist()

tfidf_dict = compute_tfidf_scores(cleaned_tweets)

all_events = []

for index,subgraph in subgraphs.items():

    pagerank_scores = compute_pagerank(subgraph, tfidf_dict)
    alpha = 0.5 #alpha threshold for PageRank score
    events = (graph_processing(subgraph, pagerank_scores, alpha))
    for e in events:
        all_events.append(e)


all_events = merge_events(all_events)

print(len(all_events))
for event in all_events:
    #print(f'What: {event[0]}, Who: {event[1]}, Where: {event[2]}, When: {event[3]}')
    print(event)