import streamlit as st
import requests
import os

# Page Config & Custom Styling
st.set_page_config(
    page_title="Enterprise Document Intelligence RAG",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for modern UI cards and badges
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 25px;
    }
    .card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
    }
    .badge-strategy {
        background-color: #007bff;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)
# API Endpoint URL
API_URL = "http://127.0.0.1:8000"

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/brain.png", width=70)
    st.title("⚙️ RAG Engine Controls")
    st.write("Configure dynamic parameters for Chunking and Retrieval.")
    
    st.divider()
    
    # --- 1. Chunking Controls ---
    st.subheader("1. Ingestion Settings")
    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        options=["recursive", "semantic"],
        help="Recursive cuts text by character count. Semantic cuts by logical meaning using embeddings."
    )
    
    chunk_size = st.slider("Chunk Size", min_value=300, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    
    if st.button("🚀 Process & Index Documents", use_container_width=True):
        with st.spinner("Processing files with Docling & generating embeddings..."):
            try:
                payload = {
                    "chunking_strategy": chunking_strategy,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
                res = requests.post(f"{API_URL}/ingest", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"✅ {data['message']}")
                    st.metric("Total Docs", data['total_documents'])
                    st.metric("Chunks Stored", data['chunks_created'])
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server. Ensure it's running! Error: {e}")
                
    st.divider()
    
    # --- 2. Retrieval Controls ---
    st.subheader("2. Retrieval Settings")
    retrieval_strategy = st.selectbox(
        "Retrieval Algorithm",
        options=["hybrid", "cross_encoder", "vector", "bm25"],
        format_func=lambda x: {
            "hybrid": "🔀 Hybrid Search (BM25 + Vector)",
            "cross_encoder": "⚡ Cross-Encoder Reranking (Best Accuracy)",
            "vector": "🎯 Semantic Vector Search (MMR)",
            "bm25": "🔤 BM25 Keyword Search"
        }[x],
        help="Select how context chunks are pulled from ChromaDB."
    )
    
    top_k = st.slider("Top Chunks Retrieved (K)", min_value=1, max_value=10, value=5)

# Main Screen Layout
st.markdown('<div class="main-header">Enterprise Document Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions across your enterprise reports with verifiable source citations.</div>', unsafe_allow_html=True)

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Past Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Citation Sources"):
                for src in message["sources"]:
                    st.markdown(f"**Source:** `{src['source']}`")
                    st.caption(f"_{src['content_snippet']}_")

# User Query Input
if query := st.chat_input("Ask a question about your uploaded documents..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call RAG API
    with st.chat_message("assistant"):
        with st.spinner(f"Retrieving context via [{retrieval_strategy.upper()}] & generating answer..."):
            try:
                payload = {
                    "query": query,
                    "retrieval_strategy": retrieval_strategy,
                    "top_k": top_k
                }
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["answer"]
                    sources = result["sources"]
                    strategy_used = result["strategy_used"]

                    # Display Strategy Tag & Answer
                    st.markdown(f'<span class="badge-strategy">Strategy: {strategy_used}</span>', unsafe_allow_html=True)
                    st.markdown(f"\n{answer}")

                    # Render Citation Sources in expandable box
                    if sources:
                        with st.expander("📚 View Citation Sources"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"**Doc {idx}: `{src['source']}`**")
                                st.caption(f"\"{src['content_snippet']}\"")

                    # Save to chat session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [s.dict() for s in sources]
                    })
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Failed to reach API server: {e}")