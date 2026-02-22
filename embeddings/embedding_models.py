class SentenceTransformerEmbedding:
    def __init__(self,
                model_name="/home/hieu/Documents/Self Study/Khóa học chuyên chatbot/DỰ ÁN GIỮA KHÓA 01: RAG TƯ VẤN BÁN HOA/models/embedding_model/qwen3-0,6b",
                local_files_only=True):
        from sentence_transformers import SentenceTransformer

        self.client = SentenceTransformer(model_name, local_files_only=local_files_only)
    def encode(self, docs: list[str]) -> list[list]:
        return self.client.encode(docs).tolist()

class GeminiEmbedding:
    def __init__(self, model_name="gemini-embedding-001"):
        from google import genai

        self.model_name = model_name
        self.client = genai.Client()
    def encode(self, docs: list[str]) -> list[list]:
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=docs
        )
        return [item.values for item in result.embeddings]

