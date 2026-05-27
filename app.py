"""Streamlit UI for the RAG pipeline. Run: `uv run streamlit run app.py`."""
from __future__ import annotations

import streamlit as st

from rag import answer

st.set_page_config(page_title="RAG Playground", layout="wide")

st.title("RAG Playground")
st.caption("Gemini · BGE embeddings · Qdrant · Phoenix traces on :6006")

with st.sidebar:
    st.header("Knobs")
    k = st.slider("Top-k chunks to retrieve", min_value=1, max_value=15, value=5)
    st.divider()
    st.markdown(
        "**Phoenix UI:** [localhost:6006](http://localhost:6006)\n\n"
        "Open it to see traces with timings, retrieved chunks, and Gemini calls."
    )

question = st.text_input(
    "Ask a question about the corpus",
    placeholder="e.g. What is multi-head attention and why is it useful?",
)

if question:
    with st.spinner("Retrieving + generating ..."):
        result = answer(question, k=k)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("Answer")
        st.write(result.answer.text)
        if result.answer.citations_used:
            st.success(f"Model cited chunks: {result.answer.citations_used}")
        else:
            st.warning("Model didn't cite any chunks — possible hallucination or refusal.")
        st.caption(f"Model: {result.answer.model}")

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
