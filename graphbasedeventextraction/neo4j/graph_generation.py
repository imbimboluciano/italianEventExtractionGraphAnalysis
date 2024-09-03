from neo4j import GraphDatabase

class Neo4jEventGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def create_graph(self, tweets):
        with self.driver.session() as session:
            for tweet in tweets:
                session.write_transaction(self._create_and_link_nodes, tweet)
    
    @staticmethod
    def _create_and_link_nodes(tx, tweet):
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

