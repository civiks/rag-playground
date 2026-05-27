"""Streamlit UI for the RAG pipeline. Run: `uv run streamlit run app.py`."""
from __future__ import annotations

import streamlit as st

from rag import answer
from retrieve import _get_client, _get_embedder


@st.cache_resource(show_spinner="Loading embedding model + vector DB")
def _warm() -> bool:
    _get_embedder()
    _get_client()
    return True


_warm()

CHUNKING_STRATEGIES = {
    "Naive (1200-char sliding)": "naive_1200",
    "Docling hybrid (structure-aware)": "docling_hybrid",
}

RETRIEVAL_STRATEGIES = {
    "Dense (cosine top-k)": "dense",
    "Hybrid (BM25 + dense, RRF)": "hybrid",
}

st.set_page_config(page_title="RAG Playground", layout="wide")

st.title("RAG Playground")
st.caption("Gemini · BGE embeddings · Qdrant · Phoenix traces on :6006")

with st.sidebar:
    st.header("Strategy")
    chunking_label = st.selectbox(
        "Chunking",
        list(CHUNKING_STRATEGIES.keys()),
        help="Naive splits by character count and shreds tables. Docling hybrid respects document structure.",
    )
    chunking = CHUNKING_STRATEGIES[chunking_label]

    retrieval_label = st.selectbox(
        "Retrieval",
        list(RETRIEVAL_STRATEGIES.keys()),
        help="Dense uses cosine on embeddings. Hybrid adds BM25 (keyword) and fuses via Reciprocal Rank Fusion.",
    )
    retrieval = RETRIEVAL_STRATEGIES[retrieval_label]

    st.divider()
    st.header("Knobs")
    k = st.slider("Top-k chunks to retrieve", min_value=1, max_value=15, value=5)

    st.divider()
    st.markdown(
        "**Phoenix UI:** [localhost:6006](http://localhost:6006)\n\n"
        "Open it to see traces, retrieved chunks, and Gemini calls."
    )
    st.divider()
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []

with st.form("ask", clear_on_submit=False):
    question = st.text_input(
        "Ask a question about the corpus",
        placeholder="e.g. What is multi-head attention and why is it useful?",
    )
    submitted = st.form_submit_button("Ask")

if "history" not in st.session_state:
    st.session_state.history = []

if submitted and question:
    with st.spinner("Retrieving + generating ..."):
        result = answer(question, k=k, collection=f"rag_{chunking}", strategy=retrieval)
    st.session_state.history.insert(0, {
        "chunking_label": chunking_label,
        "retrieval_label": retrieval_label,
        "k": k,
        "result": result,
    })


def _render(entry: dict) -> None:
    result = entry["result"]
    k = entry["k"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency", f"{result.latency_s:.2f}s")
    m2.metric("Input tokens", result.answer.input_tokens)
    m3.metric("Output tokens", result.answer.output_tokens)
    m4.metric("Chunks retrieved", len(result.hits))

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("Answer")
        st.write(result.answer.text)
        if result.answer.citations_used:
            st.success(f"Model cited chunks: {result.answer.citations_used}")
        else:
            st.warning("Model didn't cite any chunks — possible hallucination or refusal.")
        st.caption(
            f"Model: {result.answer.model} · "
            f"Chunking: {entry['chunking_label']} · "
            f"Retrieval: {entry['retrieval_label']} · "
            f"k={k}"
        )

    with col_b:
        st.subheader(f"Retrieved chunks (top {k})")
        for i, h in enumerate(result.hits, start=1):
            used = i in result.answer.citations_used
            badge = "USED" if used else "unused"
            with st.expander(f"[{i}] {badge} · {h.source} · chunk {h.chunk_idx} · score {h.score:.3f}"):
                st.text(h.text)

        max_score = max((h.score for h in result.hits), default=0.0)
        if max_score < 0.4:
            st.error(
                f"Top score is only {max_score:.3f} — retrieval likely failed. "
                "Try rephrasing using terms from the source."
            )


for idx, entry in enumerate(st.session_state.history):
    result = entry["result"]
    q_short = result.question if len(result.question) <= 70 else result.question[:67] + "..."
    header = (
        f"Q: {q_short}  ·  "
        f"{entry['chunking_label']} / {entry['retrieval_label']}  ·  "
        f"k={entry['k']}  ·  "
        f"{result.latency_s:.2f}s  ·  "
        f"{result.answer.input_tokens}+{result.answer.output_tokens} tok"
    )
    with st.expander(header, expanded=(idx == 0)):
        _render(entry)
