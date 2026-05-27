from __future__ import annotations
import os

# Silence transformers' lazy-import deprecation noise.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from generate import Answer, generate
from retrieve import DEFAULT_COLLECTION, Hit, retrieve

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


def answer(question: str, k: int = 5, collection: str = DEFAULT_COLLECTION) -> RagResult:
    tracer = _init_phoenix()
    start = time.perf_counter()
    with tracer.start_as_current_span("rag.pipeline") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("input.value", question)
        span.set_attribute("retrieval.k", k)
        span.set_attribute("retrieval.collection", collection)

        with tracer.start_as_current_span("rag.retrieve") as r_span:
            r_span.set_attribute("openinference.span.kind", "RETRIEVER")
            r_span.set_attribute("input.value", question)
            r_span.set_attribute("retrieval.collection", collection)
            hits = retrieve(question, k=k, collection=collection)
            r_span.set_attribute("retrieval.num_hits", len(hits))
            for i, h in enumerate(hits):
                p = f"retrieval.documents.{i}.document"
                r_span.set_attribute(f"{p}.id", f"{h.source}:{h.chunk_idx}")
                r_span.set_attribute(f"{p}.content", h.text)
                r_span.set_attribute(f"{p}.score", h.score)
                r_span.set_attribute(f"{p}.metadata", f'{{"source":"{h.source}","chunk_idx":{h.chunk_idx}}}')

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
        latency_s=latency_s, collection=collection,
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is multi-head attention?"
    print(f"Q: {q}\n")
    result = answer(q)
    print(f"A: {result.answer.text}\n")
    print(f"Citations used: {result.answer.citations_used}")
    print("Phoenix UI: http://localhost:6006 (run `uv run phoenix serve` separately)")
