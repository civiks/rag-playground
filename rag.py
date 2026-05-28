from __future__ import annotations
import os

# Silence transformers' lazy-import deprecation noise.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Stop HF hub from showing download progress bars and doing a network check on every cache hit.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry import trace

from agent import (
    Assessment,
    Reflection,
    Strategy,
    assess_retrieval,
    classify_query,
    reflect_on_answer,
    stronger_strategy,
)
from generate import MODEL as DEFAULT_MODEL, Answer, generate, generate_stream, rewrite_to_hyde, rewrite_to_multi
from retrieve import (
    DEFAULT_COLLECTION,
    PREFETCH_N,
    RERANKER_MODEL,
    Hit,
    QdrantClient,
    rerank_hits,
    retrieve,
    rrf_fuse,
)

load_dotenv()

_initialized = False
_tracer: trace.Tracer | None = None


def _init_phoenix() -> trace.Tracer:
    """Wire OTel to a Phoenix server (run `uv run phoenix serve` separately on :6006).

    Set RAG_NO_TRACE=1 to skip Phoenix entirely (used by eval.py).
    """
    global _initialized, _tracer
    if _initialized:
        return _tracer  # type: ignore[return-value]

    if os.environ.get("RAG_NO_TRACE"):
        _tracer = trace.get_tracer(__name__)
        _initialized = True
        return _tracer  # type: ignore[return-value]

    try:
        import contextlib, io
        from phoenix.otel import register

        os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            provider = register(project_name="rag-playground", auto_instrument=False)
        GoogleGenAIInstrumentor().instrument(tracer_provider=provider)
        _tracer = provider.get_tracer(__name__)
    except Exception:
        # Phoenix not running — fall back to a no-op tracer so spans are silently dropped.
        _tracer = trace.get_tracer(__name__)

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
    mode: str = "manual"
    agent_decision: Strategy | None = None
    agent_assessment: Assessment | None = None
    agent_reflection: Reflection | None = None
    agent_retried: bool = False


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
    mode: str = "manual"
    agent_decision: Strategy | None = None
    agent_assessment: Assessment | None = None
    agent_retried: bool = False


def _span_name(model: str, collection: str, strategy: str, rewrite: str, rerank: bool, k: int) -> str:
    chunking_tag = collection.removeprefix("rag_")
    rerank_tag = "rerank" if rerank else "norerank"
    provider, model_id = (model.split(":", 1) if ":" in model else ("gemini", model))
    model_tag = f"{provider}/{model_id.removeprefix('gemini-')}"
    return f"rag.pipeline [{model_tag} · {chunking_tag} · {strategy} · rewrite={rewrite} · {rerank_tag} · k={k}]"


def _set_root_attrs(span, question, k, collection, strategy, rerank, rewrite, model, history, mode) -> None:
    span.set_attribute("openinference.span.kind", "CHAIN")
    span.set_attribute("input.value", question)
    span.set_attribute("retrieval.k", k)
    span.set_attribute("retrieval.collection", collection)
    span.set_attribute("retrieval.strategy", strategy)
    span.set_attribute("retrieval.rerank", rerank)
    span.set_attribute("retrieval.rewrite", rewrite)
    span.set_attribute("llm.model_name", model)
    span.set_attribute("rag.mode", mode)
    if history:
        span.set_attribute("chat.history.turns", len(history))


def _agent_classify(tracer, question, model, api_key=None) -> Strategy:
    with tracer.start_as_current_span("agent.classify") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("input.value", question)
        decision = classify_query(question, model=model, api_key=api_key)
        span.set_attribute("output.value", json.dumps({
            "retrieval": decision.retrieval, "rerank": decision.rerank,
            "rewrite": decision.rewrite, "reasoning": decision.reasoning,
        }))
        return decision


def _agent_assess(tracer, hits, strategy_obj) -> Assessment:
    with tracer.start_as_current_span("agent.assess") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("input.value", f"n_hits={len(hits)} top={hits[0].score if hits else 'n/a'}")
        assessment = assess_retrieval(hits, strategy_obj)
        retry_summary = (
            f"{assessment.retry_strategy.retrieval}/rerank={assessment.retry_strategy.rerank}/rewrite={assessment.retry_strategy.rewrite}"
            if assessment.retry_strategy else "exhausted"
        )
        span.set_attribute("output.value", json.dumps({
            "ok": assessment.ok, "reason": assessment.reason, "retry": retry_summary,
        }))
        return assessment


def _agent_reflect(tracer, question, answer_text, hits, model, api_key=None) -> Reflection:
    with tracer.start_as_current_span("agent.reflect") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("input.value", answer_text)
        reflection = reflect_on_answer(question, answer_text, hits, model=model, api_key=api_key)
        span.set_attribute("output.value", json.dumps({
            "faithful": reflection.faithful, "critique": reflection.critique,
        }))
        return reflection


def _run_retrieval(
    tracer: trace.Tracer,
    question: str,
    k: int,
    collection: str,
    strategy: str,
    rerank: bool,
    rewrite: str,
    model: str,
    client: QdrantClient | None = None,
    api_key: str | None = None,
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
            hypothetical = rewrite_to_hyde(question, model=model, api_key=api_key)
            rewritten_queries = [hypothetical]
            rw_span.set_attribute("output.value", hypothetical)
        retrieval_queries = [hypothetical]
    elif rewrite == "multi":
        with tracer.start_as_current_span("rag.rewrite") as rw_span:
            rw_span.set_attribute("openinference.span.kind", "CHAIN")
            rw_span.set_attribute("input.value", question)
            rw_span.set_attribute("rewrite.strategy", "multi")
            paraphrases = rewrite_to_multi(question, n=3, model=model, api_key=api_key)
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
            hits = retrieve(retrieval_queries[0], k=fetch_k, collection=collection, strategy=strategy, client=client)
        else:
            rankings = [
                retrieve(q, k=fetch_k, collection=collection, strategy=strategy, client=client)
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
    mode: Literal["manual", "auto"] = "manual",
    api_key: str | None = None,
    client: QdrantClient | None = None,
) -> RagResult:
    tracer = _init_phoenix()
    start = time.perf_counter()

    agent_decision: Strategy | None = None
    agent_assessment: Assessment | None = None
    agent_reflection: Reflection | None = None
    agent_retried = False

    with tracer.start_as_current_span(_span_name(model, collection, strategy, rewrite, rerank, k)) as span:
        _set_root_attrs(span, question, k, collection, strategy, rerank, rewrite, model, history, mode)

        if mode == "auto":
            agent_decision = _agent_classify(tracer, question, model, api_key=api_key)
            strategy, rerank, rewrite = agent_decision.retrieval, agent_decision.rerank, agent_decision.rewrite

        hits, rewritten_queries = _run_retrieval(
            tracer, question, k, collection, strategy, rerank, rewrite, model,
            client=client, api_key=api_key,
        )

        if mode == "auto":
            current = Strategy(retrieval=strategy, rerank=rerank, rewrite=rewrite, reasoning="")
            agent_assessment = _agent_assess(tracer, hits, current)
            if (not agent_assessment.ok) and agent_assessment.retry_strategy is not None:
                rs = agent_assessment.retry_strategy
                strategy, rerank, rewrite = rs.retrieval, rs.rerank, rs.rewrite
                hits, rewritten_queries = _run_retrieval(
                    tracer, question, k, collection, strategy, rerank, rewrite, model,
                    client=client, api_key=api_key,
                )
                agent_retried = True

        with tracer.start_as_current_span("rag.generate") as g_span:
            g_span.set_attribute("openinference.span.kind", "CHAIN")
            g_span.set_attribute("input.value", question)
            ans = generate(question, hits, model=model, history=history, api_key=api_key)
            g_span.set_attribute("output.value", ans.text)
            g_span.set_attribute("llm.token_count.prompt", ans.input_tokens)
            g_span.set_attribute("llm.token_count.completion", ans.output_tokens)

        if mode == "auto" and not agent_retried:
            agent_reflection = _agent_reflect(tracer, question, ans.text, hits, model, api_key=api_key)
            if not agent_reflection.faithful:
                stronger = stronger_strategy(Strategy(retrieval=strategy, rerank=rerank, rewrite=rewrite, reasoning=""))
                if stronger is not None:
                    strategy, rerank, rewrite = stronger.retrieval, stronger.rerank, stronger.rewrite
                    hits, rewritten_queries = _run_retrieval(
                        tracer, question, k, collection, strategy, rerank, rewrite, model,
                        client=client, api_key=api_key,
                    )
                    with tracer.start_as_current_span("rag.generate.retry") as g_span:
                        g_span.set_attribute("openinference.span.kind", "CHAIN")
                        g_span.set_attribute("input.value", question)
                        ans = generate(question, hits, model=model, history=history, api_key=api_key)
                        g_span.set_attribute("output.value", ans.text)
                        g_span.set_attribute("llm.token_count.prompt", ans.input_tokens)
                        g_span.set_attribute("llm.token_count.completion", ans.output_tokens)
                    agent_retried = True

        span.set_attribute("output.value", ans.text)
        span.set_attribute("citations.used", str(ans.citations_used))
        span.set_attribute("llm.token_count.prompt", ans.input_tokens)
        span.set_attribute("llm.token_count.completion", ans.output_tokens)

    latency_s = time.perf_counter() - start
    return RagResult(
        question=question, answer=ans, hits=hits,
        latency_s=latency_s, collection=collection, strategy=strategy, rerank=rerank,
        rewrite=rewrite, rewritten_queries=rewritten_queries, model=model,
        mode=mode, agent_decision=agent_decision, agent_assessment=agent_assessment,
        agent_reflection=agent_reflection, agent_retried=agent_retried,
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
    mode: Literal["manual", "auto"] = "manual",
    api_key: str | None = None,
    client: QdrantClient | None = None,
) -> Iterator:
    """Streaming variant. Yields a RetrievalMeta first, then token-delta strings.

    Consumer pattern (chat UI):
        gen = answer_stream(..., usage_out=usage_out)
        meta = next(gen)                  # render chunks panel from meta
        full_text = st.write_stream(gen)  # stream tokens into the chat message
        # After: usage_out has input_tokens, output_tokens, latency_s,
        #        full_text, citations, agent_reflection (when mode="auto").

    mode="auto" runs classify + assess + (one retrieval retry if weak) up front,
    then streams. Reflect runs after the stream completes as advisory only — it
    can't trigger a regen here without restarting the stream, so the critique
    is logged to the trace + usage_out for the UI to surface. Use the
    non-streaming `answer()` if you want reflect-driven regeneration.
    """
    if usage_out is None:
        usage_out = {}
    tracer = _init_phoenix()
    start = time.perf_counter()

    agent_decision: Strategy | None = None
    agent_assessment: Assessment | None = None
    agent_retried = False

    with tracer.start_as_current_span(_span_name(model, collection, strategy, rewrite, rerank, k)) as span:
        _set_root_attrs(span, question, k, collection, strategy, rerank, rewrite, model, history, mode)

        if mode == "auto":
            agent_decision = _agent_classify(tracer, question, model, api_key=api_key)
            strategy, rerank, rewrite = agent_decision.retrieval, agent_decision.rerank, agent_decision.rewrite

        hits, rewritten_queries = _run_retrieval(
            tracer, question, k, collection, strategy, rerank, rewrite, model,
            client=client, api_key=api_key,
        )

        if mode == "auto":
            current = Strategy(retrieval=strategy, rerank=rerank, rewrite=rewrite, reasoning="")
            agent_assessment = _agent_assess(tracer, hits, current)
            if (not agent_assessment.ok) and agent_assessment.retry_strategy is not None:
                rs = agent_assessment.retry_strategy
                strategy, rerank, rewrite = rs.retrieval, rs.rerank, rs.rewrite
                hits, rewritten_queries = _run_retrieval(
                    tracer, question, k, collection, strategy, rerank, rewrite, model,
                    client=client, api_key=api_key,
                )
                agent_retried = True

        yield RetrievalMeta(
            question=question, hits=hits, collection=collection,
            strategy=strategy, rerank=rerank, rewrite=rewrite,
            rewritten_queries=rewritten_queries, model=model,
            mode=mode, agent_decision=agent_decision,
            agent_assessment=agent_assessment, agent_retried=agent_retried,
        )

        with tracer.start_as_current_span("rag.generate") as g_span:
            g_span.set_attribute("openinference.span.kind", "CHAIN")
            g_span.set_attribute("input.value", question)
            full_text = ""
            for delta in generate_stream(question, hits, model=model, history=history, usage_out=usage_out, api_key=api_key):
                full_text += delta
                yield delta
            g_span.set_attribute("output.value", full_text)
            g_span.set_attribute("llm.token_count.prompt", usage_out.get("input_tokens", 0))
            g_span.set_attribute("llm.token_count.completion", usage_out.get("output_tokens", 0))

        citations = [i for i in range(1, len(hits) + 1) if f"[{i}]" in full_text]

        if mode == "auto":
            reflection = _agent_reflect(tracer, question, full_text, hits, model, api_key=api_key)
            usage_out["agent_reflection"] = {
                "faithful": reflection.faithful, "critique": reflection.critique,
            }

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
