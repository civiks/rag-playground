from __future__ import annotations
import os

# Silence transformers' lazy-import deprecation noise.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Stop HF hub from showing download progress bars and doing a network check on every cache hit.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from generate import Answer, generate, rewrite_to_hyde, rewrite_to_multi
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


def answer(
    question: str,
    k: int = 5,
    collection: str = DEFAULT_COLLECTION,
    strategy: str = "dense",
    rerank: bool = False,
    rewrite: str = "off",
) -> RagResult:
    tracer = _init_phoenix()
    start = time.perf_counter()
    fetch_k = PREFETCH_N if rerank else k
    chunking_tag = collection.removeprefix("rag_")
    rerank_tag = "rerank" if rerank else "norerank"
    span_name = f"rag.pipeline [{chunking_tag} · {strategy} · rewrite={rewrite} · {rerank_tag} · k={k}]"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("input.value", question)
        span.set_attribute("retrieval.k", k)
        span.set_attribute("retrieval.collection", collection)
        span.set_attribute("retrieval.strategy", strategy)
        span.set_attribute("retrieval.rerank", rerank)
        span.set_attribute("retrieval.rewrite", rewrite)

        # 1. Optional query rewriting — LLM transforms the question before retrieval.
        retrieval_queries: list[str]
        rewritten_queries: list[str] = []
        if rewrite == "hyde":
            with tracer.start_as_current_span("rag.rewrite") as rw_span:
                rw_span.set_attribute("openinference.span.kind", "CHAIN")
                rw_span.set_attribute("input.value", question)
                rw_span.set_attribute("rewrite.strategy", "hyde")
                hypothetical = rewrite_to_hyde(question)
                rewritten_queries = [hypothetical]
                rw_span.set_attribute("output.value", hypothetical)
            retrieval_queries = [hypothetical]
        elif rewrite == "multi":
            with tracer.start_as_current_span("rag.rewrite") as rw_span:
                rw_span.set_attribute("openinference.span.kind", "CHAIN")
                rw_span.set_attribute("input.value", question)
                rw_span.set_attribute("rewrite.strategy", "multi")
                paraphrases = rewrite_to_multi(question, n=3)
                rewritten_queries = paraphrases
                rw_span.set_attribute("output.value", "\n".join(paraphrases))
            retrieval_queries = [question, *paraphrases]
        else:
            retrieval_queries = [question]

        # 2. Retrieve for each query; fuse if multiple.
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

        with tracer.start_as_current_span("rag.generate") as g_span:
            g_span.set_attribute("openinference.span.kind", "CHAIN")
            g_span.set_attribute("input.value", question)
            ans = generate(question, hits)
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
        rewrite=rewrite, rewritten_queries=rewritten_queries,
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is multi-head attention?"
    print(f"Q: {q}\n")
    result = answer(q)
    print(f"A: {result.answer.text}\n")
    print(f"Citations used: {result.answer.citations_used}")
    print("Phoenix UI: http://localhost:6006 (run `uv run phoenix serve` separately)")
