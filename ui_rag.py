import streamlit as st
from rag import get_llm, retrieval, create_system_prompt
import re

# ── Hàm chuyển link ảnh thành thẻ <img> HTML với kích thước cố định ─────────
IMG_WIDTH = 300   # ← thay đổi giá trị này để điều chỉnh kích thước ảnh (px)

def render_with_images(text: str) -> str:
    return re.sub(
        r'(https\S+\.jpg)',
        rf'<br><img src="\1" width="{IMG_WIDTH}" style="border-radius:8px; margin:4px 0;"><br>',
        text,
    )

# ── Cấu hình trang ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Tư vấn bán hoa 🌸", page_icon="🌸", layout="wide")
st.title("🌸 Chatbot Tư Vấn Bán Hoa")

# ── Sidebar: tùy chỉnh tham số & nút tạo cuộc trò chuyện mới ────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")

    llm_name = st.selectbox(
        "Mô hình LLM",
        ["deepseek", "gemini", "ollama_host", "unsloth_host"],
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    top_p       = st.slider("Top-p",        0.0, 1.0, 0.9, 0.05)
    top_k       = st.slider("Top-k (retrieval)", 1, 20, 5, 1)

    st.divider()

    # Nút tạo cuộc trò chuyện mới → xóa lịch sử cũ
    if st.button("🔄 Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Khởi tạo lịch sử trò chuyện trong session (biến tạm, không lưu DB) ──────
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user/assistant", "content": "..."}]

# ── Hiển thị lịch sử trò chuyện ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Ô nhập tin nhắn (giống ChatGPT) ─────────────────────────────────────────
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # Hiển thị tin nhắn người dùng ngay lập tức
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Truy xuất tài liệu liên quan & tạo system prompt
    with st.spinner("Đang tìm kiếm thông tin..."):
        docs          = retrieval(user_input, db_type="mongo", top_k=top_k)
        system_prompt = create_system_prompt(docs)

    # Xây dựng history gửi cho LLM (system prompt luôn ở đầu)
    history = [{"role": "system", "content": system_prompt}]
    history += st.session_state.messages   # thêm toàn bộ lịch sử người dùng/trợ lý

    # Gọi LLM và hiển thị phản hồi
    with st.chat_message("assistant"):
        with st.spinner("Đang trả lời..."):
            llm      = get_llm(llm_name=llm_name, temperature=temperature, top_p=top_p)
            response = llm.answer(history)
            response = render_with_images(response)
        st.markdown(response, unsafe_allow_html=True)

    # Lưu phản hồi vào lịch sử tạm
    st.session_state.messages.append({"role": "assistant", "content": response})

