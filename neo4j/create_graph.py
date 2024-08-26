from neo4j import GraphDatabase

# Connect to the Neo4j database
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "your_password"))

def create_keyword_graph(driver, documents):
    with driver.session() as session:
        session.run("""
        UNWIND $documents AS document
        UNWIND document AS keyword
        MERGE (k:Keyword {name: keyword});
        
        UNWIND $documents AS document
        UNWIND document AS keyword1
        UNWIND document AS keyword2
        WITH keyword1, keyword2
        WHERE keyword1 <> keyword2
        MATCH (k1:Keyword {name: keyword1}), (k2:Keyword {name: keyword2})
        MERGE (k1)-[r:COOCCURS_WITH]->(k2)
        ON CREATE SET r.weight = 1
        ON MATCH SET r.weight = r.weight + 1;
        """, documents=documents)

# Example documents
documents = [
    ["keyword1", "keyword2", "keyword3"],
    ["keyword2", "keyword3", "keyword4"],
    ["keyword1", "keyword4"]
]

# Create the graph
create_keyword_graph(driver, documents)
