from __future__ import annotations

import re

import streamlit as st

_CITE_PATTERN = re.compile(r"\[(\d+)\]")

_CHIP_STYLE = (
    "color:#0a84ff;background:rgba(10,132,255,0.18);"
    "padding:0 6px;border-radius:4px;margin:0 1px;"
    "font-size:0.78em;font-weight:700;vertical-align:super;"
    "line-height:1.4;text-decoration:none;"
)

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


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def render_citations(text: str, hits) -> str:
    def replace(m: re.Match) -> str:
        n = int(m.group(1))
        if not (1 <= n <= len(hits)):
            return m.group(0)
        return f'<sup style="{_CHIP_STYLE}">[{n}]</sup>'
    return _CITE_PATTERN.sub(replace, text)


def render_cited_sources(hits, citations: list[int]) -> None:
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


def render_details(entry: dict) -> None:
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
        st.caption(
            f"retrieve {entry.get('retrieve_s', 0.0):.2f}s  ·  "
            f"generate {entry.get('generate_s', 0.0):.2f}s"
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
                badge = "**CITED**" if i in citations else "_unused_"
                st.caption(f"{badge}  ·  `{h.source}` chunk `{h.chunk_idx}`  ·  score `{h.score:.3f}`")
                st.text(h.text)


def render_empty_state() -> None:
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
