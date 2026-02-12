import tempfile
from pathlib import Path

import streamlit as st

from main import AdvancedRAGEngine

st.set_page_config(page_title="Advanced RAG Studio", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 10%, #d6f5ff 0%, transparent 35%),
                    radial-gradient(circle at 80% 20%, #dff8e8 0%, transparent 35%),
                    linear-gradient(120deg, #f4f8ff 0%, #fdfcf8 50%, #f6fff9 100%);
        color: #111827;
    }

    .hero {
        padding: 1.25rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.78);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(17, 24, 39, 0.08);
        box-shadow: 0 14px 35px rgba(17, 24, 39, 0.08);
        animation: floatIn 0.55s ease;
    }

    .chip {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: #111827;
        color: #f9fafb;
    }

    .card {
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(17, 24, 39, 0.08);
        padding: 0.8rem 0.95rem;
        margin-bottom: 0.65rem;
        animation: slideUp 0.35s ease;
    }

    @keyframes floatIn {
      from {opacity: 0; transform: translateY(18px) scale(0.98);} 
      to {opacity: 1; transform: translateY(0) scale(1);} 
    }

    @keyframes slideUp {
      from {opacity: 0; transform: translateY(12px);} 
      to {opacity: 1; transform: translateY(0);} 
    }

    .stButton>button {
        border-radius: 11px;
        border: none;
        font-weight: 700;
        background: linear-gradient(135deg, #0f766e, #0891b2);
        color: #ffffff;
        box-shadow: 0 10px 24px rgba(8, 145, 178, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "engine" not in st.session_state:
    st.session_state.engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_loaded" not in st.session_state:
    st.session_state.kb_loaded = False

st.markdown(
    """
    <div class="hero">
        <span class="chip">ADVANCED RAG</span>
        <h1 style="margin: 0.5rem 0 0.25rem 0;">Document Q&A Studio</h1>
        <p style="margin: 0; color: #334155;">Upload TXT/PDF, build TF-IDF + FAISS index, and chat with grounded answers via Groq.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.8], gap="large")

with left:
    st.subheader("Knowledge Base Setup")

    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    uploaded_file = st.file_uploader("Upload a source file", type=["txt", "pdf"])

    c1, c2 = st.columns(2)
    with c1:
        chunk_size = st.slider("Chunk size (words)", min_value=80, max_value=350, value=180, step=10)
    with c2:
        chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=100, value=40, step=5)

    if st.button("Build Index", use_container_width=True):
        if not api_key:
            st.error("Please provide your Groq API key.")
        elif not uploaded_file:
            st.error("Please upload a TXT or PDF file.")
        else:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                engine = AdvancedRAGEngine(
                    groq_api_key=api_key,
                    chunk_size_words=chunk_size,
                    chunk_overlap_words=chunk_overlap,
                )
                total_chunks = engine.ingest_file(tmp_path)
                st.session_state.engine = engine
                st.session_state.kb_loaded = True
                st.session_state.messages = []
                st.success(f"Knowledge base ready. {total_chunks} chunks indexed.")
            except Exception as exc:
                st.error(f"Failed to build index: {exc}")

    if st.session_state.kb_loaded and st.session_state.engine:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"**Source:** {st.session_state.engine.source_name}")
        st.write(f"**Chunks:** {len(st.session_state.engine.chunks)}")
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.subheader("Chat")
    top_k = st.slider("Top-k retrieval", min_value=1, max_value=8, value=3)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("context"):
                with st.expander("Retrieved Context"):
                    for i, item in enumerate(msg["context"], start=1):
                        st.markdown(
                            f"<div class='card'><b>Chunk {i}</b> | score: {item.score:.4f}<br>{item.chunk[:650]}...</div>",
                            unsafe_allow_html=True,
                        )

    question = st.chat_input("Ask a question about your uploaded document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if not st.session_state.kb_loaded or not st.session_state.engine:
            answer = "Please build the index first from the setup panel."
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            with st.spinner("Generating grounded answer..."):
                try:
                    answer, docs = st.session_state.engine.answer(question, top_k=top_k)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "context": docs}
                    )
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        with st.expander("Retrieved Context"):
                            for i, item in enumerate(docs, start=1):
                                st.markdown(
                                    f"<div class='card'><b>Chunk {i}</b> | score: {item.score:.4f}<br>{item.chunk[:650]}...</div>",
                                    unsafe_allow_html=True,
                                )
                except Exception as exc:
                    err = f"Error while answering: {exc}"
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    with st.chat_message("assistant"):
                        st.error(err)
