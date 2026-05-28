from __future__ import annotations
import os

# Silence transformers' lazy-import deprecation noise.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Stop HF hub from showing download progress bars and doing a network check on every cache hit.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import time
from collections.abc import Iterator
from dataclasses import dataclass

from dotenv import load_dotenv
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry import trace

from generate import MODEL as DEFAULT_MODEL, Answer, generate, generate_stream, rewrite_to_hyde, rewrite_to_multi
from retrieve import (
    DEFAULT_COLLECTION,
    PREFETCH_N,
    RERANKER_MODEL,
    Hit,
    rerank_hits,
    retrieve,
    rrf_fuse,
)

load_dotenv()

_initialized = False
_tracer: trace.Tracer | None = None


def _init_phoenix() -> trace.Tracer:
    """Wire OTel to a Phoenix server (run `uv run phoenix serve` separately on :6006)."""
    global _initialized, _tracer
    if _initialized:
        return _tracer  # type: ignore[return-value]

    from phoenix.otel import register

    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    provider = register(project_name="rag-playground", auto_instrument=False)
    GoogleGenAIInstrumentor().instrument(tracer_provider=provider)

    _tracer = provider.get_tracer(__name__)
    _initialized = True
    return _tracer  # type: ignore[return-value]


@dataclass
class RagResult:
    question: str
    answer: Answer
    hits: list[Hit]
    latency_s: float
    collection: str
    strategy: str
    rerank: bool
    rewrite: str
    rewritten_queries: list[str]
    model: str


@dataclass
class RetrievalMeta:
    """First item yielded from answer_stream — everything known *before* the LLM streams."""
    question: str
    hits: list[Hit]
    collection: str
    strategy: str
    rerank: bool
    rewrite: str
    rewritten_queries: list[str]
    model: str


def _span_name(model: str, collection: str, strategy: str, rewrite: str, rerank: bool, k: int) -> str:
    chunking_tag = collection.removeprefix("rag_")
    rerank_tag = "rerank" if rerank else "norerank"
    provider, model_id = (model.split(":", 1) if ":" in model else ("gemini", model))
    model_tag = f"{provider}/{model_id.removeprefix('gemini-')}"
    return f"rag.pipeline [{model_tag} · {chunking_tag} · {strategy} · rewrite={rewrite} · {rerank_tag} · k={k}]"


def _set_root_attrs(span, question, k, collection, strategy, rerank, rewrite, model, history) -> None:
    span.set_attribute("openinference.span.kind", "CHAIN")
    span.set_attribute("input.value", question)
    span.set_attribute("retrieval.k", k)
    span.set_attribute("retrieval.collection", collection)
    span.set_attribute("retrieval.strategy", strategy)
    span.set_attribute("retrieval.rerank", rerank)
    span.set_attribute("retrieval.rewrite", rewrite)
    span.set_attribute("llm.model_name", model)
    if history:
        span.set_attribute("chat.history.turns", len(history))


def _run_retrieval(
    tracer: trace.Tracer,
    question: str,
    k: int,
    collection: str,
    strategy: str,
    rerank: bool,
    rewrite: str,
    model: str,
) -> tuple[list[Hit], list[str]]:
    """Shared rewrite + retrieve + rerank pipeline. Returns (hits, rewritten_queries)."""
    fetch_k = PREFETCH_N if rerank else k

    retrieval_queries: list[str]
    rewritten_queries: list[str] = []
    if rewrite == "hyde":
        with tracer.start_as_current_span("rag.rewrite") as rw_span:
            rw_span.set_attribute("openinference.span.kind", "CHAIN")
            rw_span.set_attribute("input.value", question)
            rw_span.set_attribute("rewrite.strategy", "hyde")
            hypothetical = rewrite_to_hyde(question, model=model)
            rewritten_queries = [hypothetical]
            rw_span.set_attribute("output.value", hypothetical)
        retrieval_queries = [hypothetical]
    elif rewrite == "multi":
        with tracer.start_as_current_span("rag.rewrite") as rw_span:
            rw_span.set_attribute("openinference.span.kind", "CHAIN")
            rw_span.set_attribute("input.value", question)
            rw_span.set_attribute("rewrite.strategy", "multi")
            paraphrases = rewrite_to_multi(question, n=3, model=model)
            rewritten_queries = paraphrases
            rw_span.set_attribute("output.value", "\n".join(paraphrases))
        retrieval_queries = [question, *paraphrases]
    else:
        retrieval_queries = [question]

    with tracer.start_as_current_span("rag.retrieve") as r_span:
        r_span.set_attribute("openinference.span.kind", "RETRIEVER")
        r_span.set_attribute("input.value", question)
        r_span.set_attribute("retrieval.collection", collection)
        r_span.set_attribute("retrieval.strategy", strategy)
        r_span.set_attribute("retrieval.query_count", len(retrieval_queries))
        if len(retrieval_queries) == 1:
            hits = retrieve(retrieval_queries[0], k=fetch_k, collection=collection, strategy=strategy)
        else:
            rankings = [
                retrieve(q, k=fetch_k, collection=collection, strategy=strategy)
                for q in retrieval_queries
            ]
            hits = rrf_fuse(rankings, k=fetch_k)
        r_span.set_attribute("retrieval.num_hits", len(hits))
        for i, h in enumerate(hits):
            p = f"retrieval.documents.{i}.document"
            r_span.set_attribute(f"{p}.id", f"{h.source}:{h.chunk_idx}")
            r_span.set_attribute(f"{p}.content", h.text)
            r_span.set_attribute(f"{p}.score", h.score)
            r_span.set_attribute(f"{p}.metadata", f'{{"source":"{h.source}","chunk_idx":{h.chunk_idx}}}')

    if rerank:
        pre_rerank = hits
        with tracer.start_as_current_span("rag.rerank") as rr_span:
            rr_span.set_attribute("openinference.span.kind", "RERANKER")
            rr_span.set_attribute("reranker.query", question)
            rr_span.set_attribute("reranker.model_name", RERANKER_MODEL)
            rr_span.set_attribute("reranker.top_k", k)
            rr_span.set_attribute("input.value", question)
            for i, h in enumerate(pre_rerank):
                p = f"reranker.input_documents.{i}.document"
                rr_span.set_attribute(f"{p}.id", f"{h.source}:{h.chunk_idx}")
                rr_span.set_attribute(f"{p}.content", h.text)
                rr_span.set_attribute(f"{p}.score", h.score)
            hits = rerank_hits(question, pre_rerank, k=k)
            for i, h in enumerate(hits):
                p = f"reranker.output_documents.{i}.document"
                rr_span.set_attribute(f"{p}.id", f"{h.source}:{h.chunk_idx}")
                rr_span.set_attribute(f"{p}.content", h.text)
                rr_span.set_attribute(f"{p}.score", h.score)
            rr_span.set_attribute(
                "output.value",
                ", ".join(f"{h.source}:{h.chunk_idx}" for h in hits),
            )

    return hits, rewritten_queries


def answer(
    question: str,
    k: int = 5,
    collection: str = DEFAULT_COLLECTION,
    strategy: str = "dense",
    rerank: bool = False,
    rewrite: str = "off",
    model: str = DEFAULT_MODEL,
    history: list[tuple[str, str]] | None = None,
) -> RagResult:
    tracer = _init_phoenix()
    start = time.perf_counter()
    with tracer.start_as_current_span(_span_name(model, collection, strategy, rewrite, rerank, k)) as span:
        _set_root_attrs(span, question, k, collection, strategy, rerank, rewrite, model, history)
        hits, rewritten_queries = _run_retrieval(
            tracer, question, k, collection, strategy, rerank, rewrite, model
        )

        with tracer.start_as_current_span("rag.generate") as g_span:
            g_span.set_attribute("openinference.span.kind", "CHAIN")
            g_span.set_attribute("input.value", question)
            ans = generate(question, hits, model=model, history=history)
            g_span.set_attribute("output.value", ans.text)
            g_span.set_attribute("llm.token_count.prompt", ans.input_tokens)
            g_span.set_attribute("llm.token_count.completion", ans.output_tokens)

        span.set_attribute("output.value", ans.text)
        span.set_attribute("citations.used", str(ans.citations_used))
        span.set_attribute("llm.token_count.prompt", ans.input_tokens)
        span.set_attribute("llm.token_count.completion", ans.output_tokens)

    latency_s = time.perf_counter() - start
    return RagResult(
        question=question, answer=ans, hits=hits,
        latency_s=latency_s, collection=collection, strategy=strategy, rerank=rerank,
        rewrite=rewrite, rewritten_queries=rewritten_queries, model=model,
    )


def answer_stream(
    question: str,
    k: int = 5,
    collection: str = DEFAULT_COLLECTION,
    strategy: str = "dense",
    rerank: bool = False,
    rewrite: str = "off",
    model: str = DEFAULT_MODEL,
    history: list[tuple[str, str]] | None = None,
    usage_out: dict | None = None,
) -> Iterator:
    """Streaming variant. Yields a RetrievalMeta first, then token-delta strings.

    Consumer pattern (chat UI):
        gen = answer_stream(..., usage_out=usage_out)
        meta = next(gen)                  # render chunks panel from meta
        full_text = st.write_stream(gen)  # stream tokens into the chat message
        # After: usage_out has input_tokens, output_tokens, latency_s,
        #        full_text, citations.
    """
    if usage_out is None:
        usage_out = {}
    tracer = _init_phoenix()
    start = time.perf_counter()
    with tracer.start_as_current_span(_span_name(model, collection, strategy, rewrite, rerank, k)) as span:
        _set_root_attrs(span, question, k, collection, strategy, rerank, rewrite, model, history)
        hits, rewritten_queries = _run_retrieval(
            tracer, question, k, collection, strategy, rerank, rewrite, model
        )

        yield RetrievalMeta(
            question=question, hits=hits, collection=collection,
            strategy=strategy, rerank=rerank, rewrite=rewrite,
            rewritten_queries=rewritten_queries, model=model,
        )

        with tracer.start_as_current_span("rag.generate") as g_span:
            g_span.set_attribute("openinference.span.kind", "CHAIN")
            g_span.set_attribute("input.value", question)
            full_text = ""
            for delta in generate_stream(question, hits, model=model, history=history, usage_out=usage_out):
                full_text += delta
                yield delta
            g_span.set_attribute("output.value", full_text)
            g_span.set_attribute("llm.token_count.prompt", usage_out.get("input_tokens", 0))
            g_span.set_attribute("llm.token_count.completion", usage_out.get("output_tokens", 0))

        citations = [i for i in range(1, len(hits) + 1) if f"[{i}]" in full_text]
        span.set_attribute("output.value", full_text)
        span.set_attribute("citations.used", str(citations))
        span.set_attribute("llm.token_count.prompt", usage_out.get("input_tokens", 0))
        span.set_attribute("llm.token_count.completion", usage_out.get("output_tokens", 0))

    usage_out["latency_s"] = time.perf_counter() - start
    usage_out["full_text"] = full_text
    usage_out["citations"] = citations


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is multi-head attention?"
    print(f"Q: {q}\n")
    result = answer(q)
    print(f"A: {result.answer.text}\n")
    print(f"Citations used: {result.answer.citations_used}")
    print("Phoenix UI: http://localhost:6006 (run `uv run phoenix serve` separately)")
