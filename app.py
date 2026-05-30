from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ingest import chunks_naive, chunks_parent_child, chunks_semantic, parse_upload_docling, parse_upload_text
from rag import answer_stream
from retrieve import _get_client, _get_embedder, _get_reranker
from ui import (
    EXAMPLE_QUESTIONS_CORPUS, EXAMPLE_QUESTIONS_UPLOAD,
    render_citations, render_cited_sources, render_details, render_empty_state,
)


@st.cache_resource(show_spinner=False)
def _start_background_warmup() -> bool:
    def _warm() -> None:
        try:
            _get_embedder()
            _get_client()
            _get_reranker()
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True, name="rag-warmup").start()
    return True


_start_background_warmup()




@st.cache_resource(show_spinner=False, max_entries=4)
def _ingest_strategy(
    combined_hash: str,
    chunking: str,
    _files_data: tuple[tuple[str, bytes], ...],
    _status=None,
) -> tuple[QdrantClient, list[tuple[str, str]]]:
    embedder = _get_embedder()
    dim = embedder.get_embedding_dimension()
    client = QdrantClient(":memory:")

    def _step(msg: str) -> None:
        if _status is not None:
            _status.update(label=msg)

    pooled: list[tuple[str, str]] = []
    pooled_pc: list[tuple[str, dict]] = []
    skipped: list[tuple[str, str]] = []
    n_files = len(_files_data)

    for i, (name, data) in enumerate(_files_data, 1):
        prefix = f"[{i}/{n_files}] {name}"
        ext = Path(name).suffix.lower()
        try:
            if chunking == "docling_hybrid":
                _step(f"{prefix} — parsing layout with Docling")
                _, hybrid_chunks = parse_upload_docling(data, ext)
                pooled.extend((name, c) for c in hybrid_chunks)
            elif chunking == "parent_child":
                _step(f"{prefix} — extracting text")
                text = parse_upload_text(data, ext)
                pooled_pc.extend((name, d) for d in chunks_parent_child(text))
            elif chunking == "semantic":
                _step(f"{prefix} — extracting text")
                text = parse_upload_text(data, ext)
                _step(f"{prefix} — semantic split (per-sentence embeddings)")
                pooled.extend((name, c) for c in chunks_semantic(text, embedder))
            else:  # naive_1200
                _step(f"{prefix} — extracting text")
                text = parse_upload_text(data, ext)
                pooled.extend((name, c) for c in chunks_naive(text))
        except Exception as e:
            skipped.append((name, str(e)))
            _step(f"{prefix} — skipped ({e})")

    if not pooled and not pooled_pc:
        msg = "No files could be parsed:\n" + "\n".join(f"  • {n}: {e}" for n, e in skipped)
        raise RuntimeError(msg)

    col = f"upload_{chunking}"
    client.create_collection(col, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    if chunking == "parent_child":
        _step(f"Embedding {len(pooled_pc)} child chunks")
        vecs = embedder.encode([d["text"] for _, d in pooled_pc], normalize_embeddings=True, show_progress_bar=False)
        client.upsert(col, points=[
            PointStruct(id=i, vector=v.tolist(),
                        payload={"text": d["text"], "parent_text": d["parent_text"],
                                 "source": s, "chunk_idx": i})
            for i, ((s, d), v) in enumerate(zip(pooled_pc, vecs))
        ])
    else:
        _step(f"Embedding {len(pooled)} chunks")
        vecs = embedder.encode([c for _, c in pooled], normalize_embeddings=True, show_progress_bar=False)
        client.upsert(col, points=[
            PointStruct(id=i, vector=v.tolist(),
                        payload={"text": c, "source": s, "chunk_idx": i})
            for i, ((s, c), v) in enumerate(zip(pooled, vecs))
        ])

    return client, skipped

MODEL_OPTIONS = {
    "Gemini 2.5 Flash": "gemini:gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini:gemini-2.5-flash-lite",
    "Groq Llama 3.3 70B": "groq:llama-3.3-70b-versatile",
    "Groq Llama 3.1 8B (fast)": "groq:llama-3.1-8b-instant",
    "Ollama llama3.1:8b (local)": "ollama:llama3.1:8b",
}

MODE_OPTIONS = {
    "Manual": "manual",
    "Auto (agentic)": "auto",
}

CHUNKING_STRATEGIES = {
    "Naive (1200-char sliding)": "naive_1200",
    "Semantic (topic-shift splits)": "semantic",
    "Parent-child (300-char index, 1200-char context)": "parent_child",
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

upload_names: list[str] = st.session_state.get("upload_filenames", [])
if upload_names:
    if len(upload_names) == 1:
        target_label = f"**{upload_names[0]}**"
    else:
        target_label = f"**{len(upload_names)} uploaded documents** ({', '.join(upload_names)})"
    st.info(f"Chatting with: {target_label} — remove files in the sidebar to switch back to the pre-loaded corpus.")

with st.sidebar:
    st.header("API Keys")
    gemini_key_input = st.text_input(
        "Gemini API key",
        value="",
        type="password",
        help="Free key from aistudio.google.com — kept only in this browser session.",
    )
    groq_key_input = st.text_input(
        "Groq API key (optional)",
        value="",
        type="password",
        help="Free key from console.groq.com — needed only when a Groq model is selected.",
    )
    _gemini_api_key: str | None = gemini_key_input.strip() or None
    _groq_api_key: str | None = groq_key_input.strip() or None
    if _groq_api_key:
        os.environ["GROQ_API_KEY"] = _groq_api_key

    st.divider()
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload document(s) to chat with",
        type=["pdf", "docx", "pptx", "html", "htm", "md", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Max 2 MB per file. PDF / DOCX / PPTX / HTML parsed with Docling. MD / TXT read directly. PNG / JPG run through OCR. Drop multiple files to chat across them.",
    )
    if uploaded_files:
        files_data: tuple[tuple[str, bytes], ...] = tuple((f.name, f.read()) for f in uploaded_files)
        combined_hash = hashlib.sha1(
            b"||".join(name.encode() + b":" + data for name, data in files_data)
        ).hexdigest()
        if st.session_state.get("upload_combined_hash") != combined_hash:
            st.session_state.upload_files_data = files_data
            st.session_state.upload_filenames = [n for n, _ in files_data]
            st.session_state.upload_combined_hash = combined_hash
            st.session_state.upload_built_strategies = set()
            st.session_state.history = []
            n = len(files_data)
            st.success(f"Loaded {n} file{'s' if n != 1 else ''} — index builds when you ask.")
    elif "upload_combined_hash" in st.session_state:
        for k in ("upload_files_data", "upload_filenames", "upload_combined_hash", "upload_built_strategies"):
            st.session_state.pop(k, None)
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

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "")
    if endpoint and ("localhost" in endpoint or "127.0.0.1" in endpoint):
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



if not st.session_state.history and not st.session_state.get("pending_question"):
    render_empty_state()

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(
            render_citations(entry["answer_text"], entry["meta"].hits),
            unsafe_allow_html=True,
        )
        if entry["citations"]:
            render_cited_sources(entry["meta"].hits, entry["citations"])
        else:
            st.info("Model didn't emit `[#]` markers — see the panel below for what was retrieved.")
        render_details(entry)


def _is_substantive(q: str | None) -> bool:
    if not q:
        return False
    s = q.strip()
    letters = sum(c.isalpha() for c in s)
    return len(s) >= 3 and letters >= 2 and len(set(s.lower())) >= 3


question = st.chat_input("Ask a question about your documents")
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question and not _is_substantive(question):
    st.toast("Add a few more words.")
    question = None

if question:
    history_pairs = (
        [(e["question"], e["answer_text"]) for e in st.session_state.history[-history_turns:]]
        if history_turns > 0
        else None
    )

    files_data = st.session_state.get("upload_files_data")
    combined_hash = st.session_state.get("upload_combined_hash")
    active_chunking = chunking
    active_chunking_label = chunking_label
    if files_data and combined_hash:
        if chunking == "contextual":
            active_chunking = "docling_hybrid"
            active_chunking_label = next(
                (k for k, v in CHUNKING_STRATEGIES.items() if v == active_chunking),
                active_chunking,
            )
            st.info("Contextual chunking only works on the pre-indexed corpus — using **docling_hybrid** for this upload.")
        built: set = st.session_state.setdefault("upload_built_strategies", set())
        if active_chunking not in built:
            with st.status(f"Indexing for **{active_chunking}** (first time)…", expanded=True) as status:
                try:
                    active_client, skipped = _ingest_strategy(combined_hash, active_chunking, files_data, _status=status)
                    suffix = f" ({len(skipped)} skipped)" if skipped else ""
                    status.update(label=f"Indexed{suffix}", state="complete", expanded=False)
                    built.add(active_chunking)
                except Exception as e:
                    status.update(label=str(e), state="error")
                    st.stop()
            for name, err in skipped:
                st.warning(f"Skipped **{name}**: {err}")
        else:
            active_client, _ = _ingest_strategy(combined_hash, active_chunking, files_data)
        active_collection = f"upload_{active_chunking}"
    else:
        active_collection = f"rag_{chunking}"
        active_client = None

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        usage_out: dict = {}
        answer_slot = st.empty()
        try:
            spinner_msg = "Agent thinking..." if auto else "Retrieving..."
            with st.spinner(spinner_msg):
                gen = answer_stream(
                    question, k=k, collection=active_collection, strategy=retrieval,
                    rerank=rerank, rewrite=rewrite, model=model,
                    history=history_pairs, usage_out=usage_out, mode=mode,
                    api_key=_gemini_api_key, client=active_client,
                )
                meta = next(gen)
            answer_text = answer_slot.write_stream(gen)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        answer_slot.markdown(
            render_citations(answer_text, meta.hits),
            unsafe_allow_html=True,
        )

        citations = usage_out.get("citations", [])
        if citations:
            render_cited_sources(meta.hits, citations)
        else:
            st.info("Model didn't emit `[#]` markers — see the panel below for what was retrieved.")

        entry = {
            "question": question,
            "answer_text": answer_text,
            "model_label": model_label,
            "chunking_label": active_chunking_label,
            "k": k,
            "mode": mode,
            "meta": meta,
            "latency_s": usage_out.get("latency_s", 0.0),
            "retrieve_s": usage_out.get("retrieve_s", 0.0),
            "generate_s": usage_out.get("generate_s", 0.0),
            "input_tokens": usage_out.get("input_tokens", 0),
            "output_tokens": usage_out.get("output_tokens", 0),
            "citations": citations,
            "agent_decision": meta.agent_decision,
            "agent_assessment": meta.agent_assessment,
            "agent_retried": meta.agent_retried,
            "agent_reflection": usage_out.get("agent_reflection"),
        }
        render_details(entry)

    st.session_state.history.append(entry)
