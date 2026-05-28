"""Streamlit chat UI for the RAG pipeline. Run: `uv run streamlit run app.py`."""
from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import threading
from pathlib import Path

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ingest import chunks_naive, chunks_parent_child, chunks_semantic
from rag import answer_stream
from retrieve import _get_client, _get_embedder, _get_reranker


@st.cache_resource(show_spinner=False)
def _start_background_warmup() -> bool:
    """Load embedder, reranker, Qdrant client in a daemon thread so the UI renders
    immediately. By the time the user types a question, models are already in memory."""
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


_CITE_PATTERN = re.compile(r"\[(\d+)\]")

_CHIP_STYLE = (
    "color:#0a84ff;background:rgba(10,132,255,0.18);"
    "padding:0 6px;border-radius:4px;margin:0 1px;"
    "font-size:0.78em;font-weight:700;vertical-align:super;"
    "line-height:1.4;text-decoration:none;"
)


def _render_citations(text: str, hits) -> str:
    """Turn inline `[N]` markers into small raised blue chips."""
    def replace(m: re.Match) -> str:
        n = int(m.group(1))
        if not (1 <= n <= len(hits)):
            return m.group(0)
        return f'<sup style="{_CHIP_STYLE}">[{n}]</sup>'
    return _CITE_PATTERN.sub(replace, text)


def _render_cited_sources(hits, citations: list[int]) -> None:
    """Render a compact 'Cited sources' inline block below the answer."""
    if not citations:
        return
    for n in citations:
        if not (1 <= n <= len(hits)):
            continue
        h = hits[n - 1]
        snippet = h.text.strip().replace("\n", " ")
        if len(snippet) > 360:
            snippet = snippet[:357] + "…"
        st.markdown(
            f'<div style="border-left:3px solid #0a84ff;background:rgba(10,132,255,0.08);'
            f'padding:6px 10px;margin:4px 0;border-radius:0 4px 4px 0;'
            f'font-size:0.85em;color:inherit;">'
            f'<span style="color:#0a84ff;font-weight:700;">[{n}]</span> '
            f'<span style="opacity:0.65;font-size:0.85em;">'
            f'<code style="background:transparent;padding:0;">{_escape(h.source)}</code>'
            f' · chunk {h.chunk_idx} · score {h.score:.2f}</span>'
            f'<div style="margin-top:3px;color:inherit;opacity:0.92;">{_escape(snippet)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def _docling_parse_bytes(_data: bytes, ext: str) -> tuple[str, list[str]]:
    from ingest import _get_docling_data
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(_data)
        tmp_path = f.name
    return _get_docling_data(tmp_path, ext)


def _get_plain_text(_data: bytes, ext: str) -> str:
    ext = ext.lower()
    from ingest import PLAIN_TEXT_EXTS

    if ext in PLAIN_TEXT_EXTS:
        text = _data.decode("utf-8", errors="replace")
    elif ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(_data)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    else:
        markdown, _ = _docling_parse_bytes(_data, ext)
        text = markdown
    if not text.strip():
        raise RuntimeError(f"Could not extract text from {ext} file.")
    return text


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
                _, hybrid_chunks = _docling_parse_bytes(data, ext)
                pooled.extend((name, c) for c in hybrid_chunks)
            elif chunking == "parent_child":
                _step(f"{prefix} — extracting text")
                text = _get_plain_text(data, ext)
                pooled_pc.extend((name, d) for d in chunks_parent_child(text))
            elif chunking == "semantic":
                _step(f"{prefix} — extracting text")
                text = _get_plain_text(data, ext)
                _step(f"{prefix} — semantic split (per-sentence embeddings)")
                pooled.extend((name, c) for c in chunks_semantic(text, embedder))
            else:  # naive_1200
                _step(f"{prefix} — extracting text")
                text = _get_plain_text(data, ext)
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

EXAMPLE_QUESTIONS_CORPUS = [
    "What is multi-head attention?",
    "How does YOLO frame object detection differently?",
    "Compare the encoder and decoder stacks.",
]

EXAMPLE_QUESTIONS_UPLOAD = [
    "Summarise this document.",
    "What are the key takeaways?",
    "List the main sections.",
]

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
    # Inputs are intentionally blank — never pre-fill from os.environ, because on
    # Streamlit Cloud env vars come from the deploy's Secrets and would leak to every
    # visitor. Backend calls still read from os.environ, so `.env` (local) and properly
    # set secrets continue to work — they're just not surfaced in the UI.
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


def _render_empty_state() -> None:
    has_upload = bool(st.session_state.get("upload_filenames"))
    if has_upload:
        names = st.session_state["upload_filenames"]
        target = f"`{names[0]}`" if len(names) == 1 else f"your **{len(names)} documents**"
        suggestions = EXAMPLE_QUESTIONS_UPLOAD
    else:
        target = "the pre-loaded papers"
        suggestions = EXAMPLE_QUESTIONS_CORPUS

    st.markdown(
        f"<div style='text-align:center;padding:72px 0 20px;opacity:0.75;'>"
        f"<div style='font-size:1.05em;'>Ask anything about {target}.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(suggestions))
    for col, q in zip(cols, suggestions):
        with col:
            if st.button(q, use_container_width=True, key=f"suggest_{hash(q)}"):
                st.session_state["pending_question"] = q
                st.rerun()


if not st.session_state.history and not st.session_state.get("pending_question"):
    _render_empty_state()

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(
            _render_citations(entry["answer_text"], entry["meta"].hits),
            unsafe_allow_html=True,
        )
        if entry["citations"]:
            _render_cited_sources(entry["meta"].hits, entry["citations"])
        else:
            st.info("Model didn't emit `[#]` markers — see the panel below for what was retrieved.")
        _render_details(entry)


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
            _render_citations(answer_text, meta.hits),
            unsafe_allow_html=True,
        )

        citations = usage_out.get("citations", [])
        if citations:
            _render_cited_sources(meta.hits, citations)
        else:
            st.info("Model didn't emit `[#]` markers — see the panel below for what was retrieved.")

        entry = {
            "question": question,
            "answer_text": answer_text,
            "model_label": model_label,
            "chunking_label": active_chunking_label,
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
