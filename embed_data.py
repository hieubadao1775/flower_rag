import pandas as pd
from databases import VectorDatabase
from embeddings import SentenceTransformerEmbedding, GeminiEmbedding
import json
from pymongo.operations import SearchIndexModel
import time

def embedding(rows, model):
    vectors = []
    for row in rows:
        vector = model.encode([row])
        vectors.append(vector[0] if isinstance(vector, list) else vector)

        time.sleep(3)
    return vectors

def embedd_data(raw_data_path="data/output.json", db_name="mydb", collection_name="gemini_flower", part=0):
    client = VectorDatabase().client
    db = client[db_name]
    collection = db[collection_name]

    with open(raw_data_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data).drop_duplicates()
    num_samples = len(df)
    # gemini: < 100 RPM -> split 3 part, 3 accounts
    if part == 0:
        df = df.iloc[:1 * (num_samples//3)]
        collection.delete_many({})
    elif part == 1:
        df = df.iloc[1 * (num_samples//3): 2 * (num_samples//3)]
    elif part == 2:
        df = df.iloc[2 * (num_samples//3):]

    embedding_model = GeminiEmbedding()
    df["embeddings"] = embedding(df["content"].tolist(), embedding_model)
    embedded_data = df.to_dict("records")

    collection.insert_many(embedded_data)

    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embeddings",
                    "numDimensions": 3072,
                    "similarity": "cosine"
                }
            ]
        },
        name="vector_index",
        type="vectorSearch"
    )

    collection.create_search_index(model=search_index_model)

if __name__ == "__main__":
    # embedd_data(part=0)
    # embedd_data(part=1)
    embedd_data(part=2)

