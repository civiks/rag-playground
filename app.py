"""Streamlit chat UI for the RAG pipeline. Run: `uv run streamlit run app.py`."""
from __future__ import annotations

import streamlit as st

from rag import answer_stream
from retrieve import _get_client, _get_embedder


@st.cache_resource(show_spinner="Loading embedding model + vector DB")
def _warm() -> bool:
    _get_embedder()
    _get_client()
    return True


_warm()

MODEL_OPTIONS = {
    "Gemini 2.5 Flash": "gemini:gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini:gemini-2.5-flash-lite",
    "Groq Llama 3.3 70B": "groq:llama-3.3-70b-versatile",
    "Groq Llama 3.1 8B (fast)": "groq:llama-3.1-8b-instant",
    "Ollama llama3.1:8b (local)": "ollama:llama3.1:8b",
}

CHUNKING_STRATEGIES = {
    "Naive (1200-char sliding)": "naive_1200",
    "Docling hybrid (structure-aware)": "docling_hybrid",
}

RETRIEVAL_STRATEGIES = {
    "Dense (cosine top-k)": "dense",
    "Hybrid (BM25 + dense, RRF)": "hybrid",
}

RERANK_OPTIONS = {
    "Off": False,
    "BGE reranker v2-m3": True,
}

REWRITE_OPTIONS = {
    "Off": "off",
    "HyDE (hypothetical answer)": "hyde",
    "Multi-query (3 paraphrases)": "multi",
}

st.set_page_config(page_title="RAG Playground", layout="wide")

st.title("RAG Playground")
st.caption("Gemini · Groq · Ollama · BGE embeddings · Qdrant · Phoenix traces on :6006")

with st.sidebar:
    st.header("Strategy")
    model_label = st.selectbox(
        "Model",
        list(MODEL_OPTIONS.keys()),
        help="Gemini needs GOOGLE_API_KEY. Groq needs GROQ_API_KEY (free at console.groq.com). Ollama needs `ollama serve` running locally with the model pulled.",
    )
    model = MODEL_OPTIONS[model_label]

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

    rerank_label = st.selectbox(
        "Reranking",
        list(RERANK_OPTIONS.keys()),
        help="Cross-encoder rescores (query, chunk) pairs jointly. Catches relevance failures BM25/dense miss.",
    )
    rerank = RERANK_OPTIONS[rerank_label]

    rewrite_label = st.selectbox(
        "Query rewriting",
        list(REWRITE_OPTIONS.keys()),
        help="HyDE: LLM writes a hypothetical answer and embeds that. Multi-query: 3 paraphrases, RRF-fused.",
    )
    rewrite = REWRITE_OPTIONS[rewrite_label]

    st.divider()
    st.header("Knobs")
    k = st.slider("Top-k chunks to retrieve", min_value=1, max_value=15, value=5)
    history_turns = st.slider(
        "Chat history turns to thread",
        min_value=0, max_value=8, value=4,
        help="How many prior (user, assistant) turns to fold into the prompt as conversation context. 0 = stateless.",
    )

    st.divider()
    st.markdown(
        "**Phoenix UI:** [localhost:6006](http://localhost:6006)\n\n"
        "Open it to see traces, retrieved chunks, and LLM calls."
    )
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


if "history" not in st.session_state:
    st.session_state.history = []


def _render_details(entry: dict) -> None:
    """One collapsible 'under the hood' panel per assistant response — config,
    rewritten queries (if any), and every retrieved chunk with its score + citation status.
    Flat layout because Streamlit can't reliably nest expanders.
    """
    meta = entry["meta"]
    citations = entry["citations"]
    k_used = entry["k"]
    n_cited = len(citations)
    n_hits = len(meta.hits)
    max_score = max((h.score for h in meta.hits), default=0.0)

    summary = (
        f"{n_hits} chunks retrieved · {n_cited} cited · "
        f"{entry['latency_s']:.2f}s · {entry['input_tokens']}+{entry['output_tokens']} tok"
    )
    with st.expander(summary):
        st.markdown(
            f"**Model:** `{entry['model_label']}`  &nbsp;·&nbsp;  "
            f"**Chunking:** `{entry['chunking_label']}`  &nbsp;·&nbsp;  "
            f"**Retrieval:** `{entry['retrieval_label']}`  &nbsp;·&nbsp;  "
            f"**Rerank:** `{entry['rerank_label']}`  &nbsp;·&nbsp;  "
            f"**Rewrite:** `{entry['rewrite_label']}`  &nbsp;·&nbsp;  "
            f"**k:** `{k_used}`"
        )

        if max_score < 0.4:
            st.error(
                f"Top score is only {max_score:.3f} — retrieval likely failed. "
                "Try rephrasing using terms from the source."
            )

        if meta.rewritten_queries:
            st.markdown(f"##### Rewritten queries ({meta.rewrite})")
            for rq in meta.rewritten_queries:
                st.markdown(f"- {rq}")

        st.markdown(f"##### Retrieved chunks (top {n_hits})")
        for i, h in enumerate(meta.hits, start=1):
            used = i in citations
            badge = "**CITED**" if used else "_unused_"
            st.markdown(
                f"---\n**[{i}]** {badge}  ·  `{h.source}` chunk `{h.chunk_idx}`  ·  score `{h.score:.3f}`"
            )
            st.text(h.text)


# Replay conversation history.
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer_text"])
        if entry["citations"]:
            st.success(f"Cited chunks: {entry['citations']}")
        else:
            st.info("Model didn't emit `[#]` markers — see the panel below for what was retrieved.")
        _render_details(entry)


if question := st.chat_input("Ask a question about the corpus"):
    # Thread the most-recent turns into the prompt as conversation context.
    history_pairs = (
        [(e["question"], e["answer_text"]) for e in st.session_state.history[-history_turns:]]
        if history_turns > 0
        else None
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        usage_out: dict = {}
        try:
            with st.spinner("Retrieving..."):
                gen = answer_stream(
                    question, k=k, collection=f"rag_{chunking}", strategy=retrieval,
                    rerank=rerank, rewrite=rewrite, model=model,
                    history=history_pairs, usage_out=usage_out,
                )
                meta = next(gen)              # retrieval/rerank/rewrite happen here
            answer_text = st.write_stream(gen)  # then stream LLM tokens
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        citations = usage_out.get("citations", [])
        if citations:
            st.success(f"Cited chunks: {citations}")
        else:
            st.info("Model didn't emit `[#]` markers — see the panel below for what was retrieved.")

        entry = {
            "question": question,
            "answer_text": answer_text,
            "model_label": model_label,
            "chunking_label": chunking_label,
            "retrieval_label": retrieval_label,
            "rerank_label": rerank_label,
            "rewrite_label": rewrite_label,
            "k": k,
            "meta": meta,
            "latency_s": usage_out.get("latency_s", 0.0),
            "input_tokens": usage_out.get("input_tokens", 0),
            "output_tokens": usage_out.get("output_tokens", 0),
            "citations": citations,
        }
        _render_details(entry)

    st.session_state.history.append(entry)
