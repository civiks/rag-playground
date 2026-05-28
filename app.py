"""Streamlit chat UI for the RAG pipeline. Run: `uv run streamlit run app.py`."""
from __future__ import annotations

import io
import os

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag import answer_stream
from retrieve import _get_client, _get_embedder


@st.cache_resource(show_spinner="Loading embedding model + vector DB")
def _warm() -> bool:
    _get_embedder()
    _get_client()
    return True


_warm()


def _ingest_upload(pdf_bytes: bytes, embedder) -> QdrantClient:
    """Parse an uploaded PDF, chunk it, embed it, and return an in-memory Qdrant client.

    Uses pypdf for parsing (no ML models) so hosted deployments don't need Docling.
    Simple sliding-window chunking matches the naive_1200 strategy.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Sliding-window chunks (1200 chars, 200 overlap) — same as naive_1200 strategy.
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + 1200
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - 200

    if not chunks:
        raise RuntimeError("Could not extract text from the uploaded PDF. Try a text-based PDF.")

    vectors = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

    client = QdrantClient(":memory:")
    client.create_collection(
        "upload_session",
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    client.upsert(
        "upload_session",
        points=[
            PointStruct(id=i, vector=vec.tolist(), payload={"text": chunk, "source": "upload", "chunk_idx": i})
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ],
    )
    return client

MODEL_OPTIONS = {
    "Gemini 2.5 Flash": "gemini:gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini:gemini-2.5-flash-lite",
    "Groq Llama 3.3 70B": "groq:llama-3.3-70b-versatile",
    "Groq Llama 3.1 8B (fast)": "groq:llama-3.1-8b-instant",
    "Ollama llama3.1:8b (local)": "ollama:llama3.1:8b",
}

MODE_OPTIONS = {
    "Manual (you pick)": "manual",
    "Auto (agent picks)": "auto",
}

CHUNKING_STRATEGIES = {
    "Naive (1200-char sliding)": "naive_1200",
    "Docling hybrid (structure-aware)": "docling_hybrid",
    "Contextual (hybrid + LLM preambles)": "contextual",
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

if st.session_state.get("upload_filename"):
    st.info(f"Chatting with: **{st.session_state['upload_filename']}** (uploaded) — remove the file in the sidebar to switch back to the pre-loaded corpus.")

with st.sidebar:
    st.header("API Keys")
    _env_gemini = os.environ.get("GOOGLE_API_KEY", "")
    _env_groq = os.environ.get("GROQ_API_KEY", "")
    gemini_key_input = st.text_input(
        "Gemini API key",
        value=_env_gemini,
        type="password",
        help="Free key from aistudio.google.com — never stored beyond this browser session.",
    )
    groq_key_input = st.text_input(
        "Groq API key (optional)",
        value=_env_groq,
        type="password",
        help="Free key from console.groq.com — needed only when a Groq model is selected.",
    )
    # Runtime keys: prefer sidebar input, fall back to env var.
    _gemini_api_key: str | None = gemini_key_input.strip() or None
    _groq_api_key: str | None = groq_key_input.strip() or None
    # Write Groq key back to env so the urllib-based Groq calls find it.
    if _groq_api_key:
        os.environ["GROQ_API_KEY"] = _groq_api_key

    st.divider()
    st.header("PDF")
    uploaded_file = st.file_uploader("Upload a PDF to chat with", type="pdf")
    if uploaded_file is not None:
        if st.session_state.get("upload_filename") != uploaded_file.name:
            with st.status(f"Ingesting {uploaded_file.name}…"):
                try:
                    session_client = _ingest_upload(uploaded_file.read(), _get_embedder())
                    st.session_state.upload_client = session_client
                    st.session_state.upload_filename = uploaded_file.name
                    st.session_state.history = []
                    st.success(f"Ready — {uploaded_file.name}")
                except Exception as e:
                    st.error(str(e))
                    st.session_state.pop("upload_client", None)
                    st.session_state.pop("upload_filename", None)
    elif "upload_filename" in st.session_state:
        # File was removed from the uploader — reset back to the pre-loaded corpus.
        st.session_state.pop("upload_client", None)
        st.session_state.pop("upload_filename", None)
        st.session_state.history = []

    st.divider()
    st.header("Strategy")
    model_label = st.selectbox(
        "Model",
        list(MODEL_OPTIONS.keys()),
        help="Gemini needs GOOGLE_API_KEY. Groq needs GROQ_API_KEY (free at console.groq.com). Ollama needs `ollama serve` running locally with the model pulled.",
    )
    model = MODEL_OPTIONS[model_label]

    mode_label = st.radio(
        "Mode",
        list(MODE_OPTIONS.keys()),
        help="Manual: you pick every knob. Auto: an agent classifies the question, picks retrieval / rerank / rewrite, gates on confidence, and self-critiques the answer.",
    )
    mode = MODE_OPTIONS[mode_label]
    auto = mode == "auto"

    chunking_label = st.selectbox(
        "Chunking",
        list(CHUNKING_STRATEGIES.keys()),
        help="Naive splits by character count and shreds tables. Docling hybrid respects document structure.",
    )
    chunking = CHUNKING_STRATEGIES[chunking_label]

    retrieval_label = st.selectbox(
        "Retrieval",
        list(RETRIEVAL_STRATEGIES.keys()),
        disabled=auto,
        help="Dense uses cosine on embeddings. Hybrid adds BM25 (keyword) and fuses via Reciprocal Rank Fusion. (Auto mode picks for you.)",
    )
    retrieval = RETRIEVAL_STRATEGIES[retrieval_label]

    rerank_label = st.selectbox(
        "Reranking",
        list(RERANK_OPTIONS.keys()),
        disabled=auto,
        help="Cross-encoder rescores (query, chunk) pairs jointly. Catches relevance failures BM25/dense miss. (Auto mode picks for you.)",
    )
    rerank = RERANK_OPTIONS[rerank_label]

    rewrite_label = st.selectbox(
        "Query rewriting",
        list(REWRITE_OPTIONS.keys()),
        disabled=auto,
        help="HyDE: LLM writes a hypothetical answer and embeds that. Multi-query: 3 paraphrases, RRF-fused. (Auto mode picks for you.)",
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

    auto_tag = " · **auto**" if entry.get("mode") == "auto" else ""
    summary = (
        f"{n_hits} chunks retrieved · {n_cited} cited · "
        f"{entry['latency_s']:.2f}s · {entry['input_tokens']}+{entry['output_tokens']} tok"
        f"{auto_tag}"
    )
    with st.expander(summary):
        st.markdown(
            f"**Model:** `{entry['model_label']}`  &nbsp;·&nbsp;  "
            f"**Chunking:** `{entry['chunking_label']}`  &nbsp;·&nbsp;  "
            f"**Retrieval:** `{meta.strategy}`  &nbsp;·&nbsp;  "
            f"**Rerank:** `{meta.rerank}`  &nbsp;·&nbsp;  "
            f"**Rewrite:** `{meta.rewrite}`  &nbsp;·&nbsp;  "
            f"**k:** `{k_used}`"
        )

        decision = entry.get("agent_decision")
        if decision is not None:
            st.markdown(
                f"**Agent decided** &nbsp; retrieval=`{decision.retrieval}` · "
                f"rerank=`{decision.rerank}` · rewrite=`{decision.rewrite}`"
            )
            if decision.reasoning:
                st.caption(f"_Reasoning:_ {decision.reasoning}")

        assessment = entry.get("agent_assessment")
        if assessment is not None:
            verdict = "OK" if assessment.ok else "weak"
            tail = " → retried with stronger strategy" if entry.get("agent_retried") else ""
            st.caption(f"_Assessment:_ {verdict} — {assessment.reason}{tail}")

        reflection = entry.get("agent_reflection")
        if reflection is not None:
            faithful = reflection.get("faithful") if isinstance(reflection, dict) else reflection.faithful
            critique = reflection.get("critique") if isinstance(reflection, dict) else reflection.critique
            verdict = "faithful" if faithful else "unfaithful"
            st.caption(f"_Self-critique:_ {verdict} — {critique}")

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
        tab_labels = [
            f"[{i}] {'✓' if i in citations else '·'}" for i in range(1, n_hits + 1)
        ]
        for tab, (i, h) in zip(st.tabs(tab_labels), enumerate(meta.hits, start=1)):
            with tab:
                used = i in citations
                badge = "**CITED**" if used else "_unused_"
                st.caption(f"{badge}  ·  `{h.source}` chunk `{h.chunk_idx}`  ·  score `{h.score:.3f}`")
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


if question := st.chat_input("Ask a question about your PDF"):
    # Thread the most-recent turns into the prompt as conversation context.
    history_pairs = (
        [(e["question"], e["answer_text"]) for e in st.session_state.history[-history_turns:]]
        if history_turns > 0
        else None
    )

    # When a PDF is uploaded, use its in-memory collection; otherwise use the sidebar selection.
    upload_client: QdrantClient | None = st.session_state.get("upload_client")
    active_collection = "upload_session" if upload_client else f"rag_{chunking}"
    active_client = upload_client  # None → retrieve.py uses the global (cloud/local) client

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        usage_out: dict = {}
        try:
            spinner_msg = "Agent thinking..." if auto else "Retrieving..."
            with st.spinner(spinner_msg):
                gen = answer_stream(
                    question, k=k, collection=active_collection, strategy=retrieval,
                    rerank=rerank, rewrite=rewrite, model=model,
                    history=history_pairs, usage_out=usage_out, mode=mode,
                    api_key=_gemini_api_key, client=active_client,
                )
                meta = next(gen)              # classify/retrieve/assess (and maybe retry) happen here
            answer_text = st.write_stream(gen)  # then stream LLM tokens; reflect runs after
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
            "mode": mode,
            "meta": meta,
            "latency_s": usage_out.get("latency_s", 0.0),
            "input_tokens": usage_out.get("input_tokens", 0),
            "output_tokens": usage_out.get("output_tokens", 0),
            "citations": citations,
            "agent_decision": meta.agent_decision,
            "agent_assessment": meta.agent_assessment,
            "agent_retried": meta.agent_retried,
            "agent_reflection": usage_out.get("agent_reflection"),
        }
        _render_details(entry)

    st.session_state.history.append(entry)
