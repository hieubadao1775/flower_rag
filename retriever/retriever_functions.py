class MongoRetrieval:
    def __init__(self, db_client, embedding_model, k=10, db_name="mydb", collection_name="gemini_flower"):
        self.embedding_model = embedding_model
        self.k = k
        self.collection = db_client[db_name][collection_name]

    def retrieve(self, query):
        query_vector = self.embedding_model.encode(query)[0]
        pipline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embeddings",
                    "queryVector": query_vector,
                    "numCandidates": 400,
                    "limit": self.k
                },
            },
            {
                "$project": {
                    "_id": 0,
                    "img": 1,
                    "content": 1,
                    "price": 1,
                    "title": 1,
                }
            }

        ]

        return list(self.collection.aggregate(pipline))
