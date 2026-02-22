from llms import DeepSeek, Gemini, OllamaHost
from embeddings import GeminiEmbedding
from retriever import MongoRetrieval
from databases import VectorDatabase
import re

def get_llm(llm_name="deepseek", temperature=0.7, top_p=0.9):
    if llm_name == "deepseek":
        return DeepSeek(temperature=temperature, top_p=top_p)
    elif llm_name == "gemini":
        return Gemini(temperature=temperature, top_p=top_p)
    elif llm_name == "ollama_host":
        return OllamaHost(temperature=temperature, top_p=top_p)
    elif llm_name == "unsloth_host":
        return UnslothHost(temperature=temperature, top_p=top_p)

def retrieval(query: str, db_type: str, top_k: int) -> list:
    db_client = VectorDatabase(db_type=db_type).client
    # embedding_model = SentenceTransformerEmbedding()
    embedding_model = GeminiEmbedding()
    if db_type == "mongo":
        retriever = MongoRetrieval(db_client=db_client,
                                   embedding_model=embedding_model,
                                   k=top_k)
        docs = retriever.retrieve(query)
        return docs
    elif db_type == "supabase":
        ...
    elif db_type == "chroma":
        ...
    elif db_type == "qdrant":
        ...

def extract_img(llm_answer: str) -> str:
    match = re.search(r'https\S+\.jpg', llm_answer)
    return match.group(0) if match else ""

def create_system_prompt(docs: list) -> str:
    context = []
    for doc in docs:
        url = doc["img"]
        content = doc["content"]

        text_match = re.search(r'>([^<]+)<', doc["price"]).group(1)
        price = int(re.sub(r'\D', '', text_match))
        title = doc["title"]

        context.append(f"""{'=' * 30}
    Tên sản phẩm: {title}.
    Link ảnh: {url}
    Mô tả: {content}
    Giá: {price} VNĐ
    """)

    system_prompt = f"""### NHIỆM VỤ ###
Bạn là trợ lý bán hàng chuyên nghiệp của tiệm hoa. Nhiệm vụ của bạn:
- Chỉ sử dụng thông tin trong phần DỮ LIỆU CỬA HÀNG để trả lời.
- Nếu CONTEXT không có thông tin, hãy lịch sự báo không biết và mời khách để lại SĐT.
- Trả lời ngắn gọn, đi thẳng vào vấn đề.
- Nếu có danh sách hoa, hãy dùng gạch đầu dòng.
- Nếu có link ảnh sản phẩm, hãy đặt URL đó trên một dòng riêng, KHÔNG kèm theo bất kỳ nhãn hay chữ mô tả nào (ví dụ KHÔNG viết "Ảnh:", "Link:", v.v.).

### DỮ LIỆU CỬA HÀNG ###
{context}"""
    return system_prompt

def chat(llm_name, temperature, top_p, db_type, top_k):
    llm = get_llm(llm_name=llm_name,
                  temperature=temperature,
                  top_p=top_p)

    history_chat = ["placeholder"]

    while True:
        query = input("User:\n")
        if query == "quit":
            break

        docs = retrieval(query, db_type, top_k)
        system_prompt = create_system_prompt(docs)
        history_chat[0] = {"role": "system", "content": system_prompt}
        history_chat.append({"role": "user", "content": query})

        response = llm.answer(history_chat)
        print("Assistant:\n", response)

        history_chat.append({"role": "assistant", "content": response})
