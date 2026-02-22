from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

class VectorDatabase:
    def __init__(self, db_type="mongo"):
        if db_type == "mongo":
            mong_uri = os.getenv("MONGO_URI")
            self.client = MongoClient(mong_uri)
        elif db_type == "supabase":
            ...
        elif db_type == "qdrant":
            ...
        elif db_type == "chroma":
            ...
