"""Agentic primitives — classify, assess, reflect.

Three small functions that decide *how* and *whether* the RAG pipeline runs.
None of them call retrieval or generation themselves; rag.py threads them in
`answer(mode="auto")`.

  classify_query     LLM picks (retrieval, rerank, rewrite) from question shape
  assess_retrieval   heuristic on top score — when to spend the one retry budget
  reflect_on_answer  LLM critique of the final answer's faithfulness to chunks

Each fails open — a parse error or API hiccup never blocks the pipeline; the
fallback is a sensible default and the failure surfaces in the Phoenix span.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from generate import _dispatch
from retrieve import Hit


@dataclass
class Strategy:
    retrieval: str   # "dense" | "hybrid"
    rerank: bool
    rewrite: str     # "off" | "hyde" | "multi"
    reasoning: str


@dataclass
class Assessment:
    ok: bool
    reason: str
    retry_strategy: Strategy | None  # None when no escalation is possible


@dataclass
class Reflection:
    faithful: bool
    critique: str


_DEFAULT_STRATEGY = Strategy(
    retrieval="hybrid", rerank=True, rewrite="off",
    reasoning="classifier output unparseable; using safe default",
)


CLASSIFY_PROMPT = """\
You are picking a retrieval strategy for a RAG system. Classify the question
and choose the cheapest combination of (retrieval, rerank, rewrite) that will
likely succeed. Escalate only when the question demands it.

retrieval:
- "dense": pure semantic embedding search. Cheap. Good when wording matches sources.
- "hybrid": dense + BM25 keyword search, fused. Catches rare technical jargon,
  equations, acronyms that semantic search misses.

rerank:
- true: cross-encoder rerank of top-20 down to top-k. Slower, much higher precision.
  Turn on for paraphrased questions or where precise matching matters.
- false: skip rerank. Use when question wording roughly matches source phrasing.

rewrite:
- "off": use the question as-is.
- "hyde": LLM writes a hypothetical answer first, embed *that*. Use when question
  phrasing differs strongly from how a source document would state the answer.
- "multi": generate 3 paraphrased queries and RRF-fuse rankings. Use for multi-hop
  synthesis where several aspects matter.

Respond ONLY with a single JSON object, no preface, no markdown fences:
{"retrieval": "...", "rerank": true|false, "rewrite": "...", "reasoning": "one short sentence"}

Question: %s
"""


REFLECT_PROMPT = """\
Judge whether the ASSISTANT answer is fully supported by the CONTEXT chunks.
- If every factual claim in the answer is grounded in the chunks → faithful=true.
- If the answer hallucinates, contradicts the chunks, or refuses despite evidence
  visible in the chunks → faithful=false with a one-sentence critique.

Respond ONLY with a single JSON object, no preface, no markdown fences:
{"faithful": true|false, "critique": "one short sentence"}

QUESTION: %s

CONTEXT:
%s

ASSISTANT ANSWER:
%s
"""


def _extract_json(text: str) -> dict:
    """Pick the first balanced {...} block. Tolerates ```json fences and prefaces."""
    s = text.strip()
    if s.startswith("```"):
        # Strip ```json ... ``` or ``` ... ``` fences.
        nl = s.find("\n")
        s = s[nl + 1:] if nl >= 0 else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    start = s.find("{")
    if start < 0:
        raise ValueError("no JSON object in output")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError("unbalanced JSON in output")


def classify_query(question: str, model: str, api_key: str | None = None) -> Strategy:
    prompt = CLASSIFY_PROMPT % question
    try:
        text, _, _ = _dispatch(prompt, model=model, temperature=0.0, api_key=api_key)
        obj = _extract_json(text)
        retrieval = obj.get("retrieval", "hybrid")
        if retrieval not in ("dense", "hybrid"):
            retrieval = "hybrid"
        rewrite = obj.get("rewrite", "off")
        if rewrite not in ("off", "hyde", "multi"):
            rewrite = "off"
        return Strategy(
            retrieval=retrieval,
            rerank=bool(obj.get("rerank", True)),
            rewrite=rewrite,
            reasoning=str(obj.get("reasoning", "")).strip(),
        )
    except Exception as e:
        return Strategy(
            retrieval=_DEFAULT_STRATEGY.retrieval,
            rerank=_DEFAULT_STRATEGY.rerank,
            rewrite=_DEFAULT_STRATEGY.rewrite,
            reasoning=f"classifier failed ({type(e).__name__}); using default",
        )


def stronger_strategy(current: Strategy) -> Strategy | None:
    """Progressive escalation: cheapest weak knob to strengthen first.

    dense → hybrid+rerank → +multi-query → exhausted (None).
    """
    if current.retrieval == "dense":
        return Strategy(
            retrieval="hybrid", rerank=True, rewrite=current.rewrite,
            reasoning="escalated: dense → hybrid + rerank",
        )
    if not current.rerank:
        return Strategy(
            retrieval=current.retrieval, rerank=True, rewrite=current.rewrite,
            reasoning="escalated: added rerank",
        )
    if current.rewrite == "off":
        return Strategy(
            retrieval=current.retrieval, rerank=current.rerank, rewrite="multi",
            reasoning="escalated: added multi-query rewrite",
        )
    return None


def assess_retrieval(
    hits: list[Hit],
    strategy: Strategy,
    threshold_dense: float = 0.4,
    threshold_rerank: float = 0.3,
) -> Assessment:
    """Heuristic gate on top score. Cheap — no LLM call.

    Score scales differ per strategy, so we threshold differently:
      - rerank ON  → cross-encoder sigmoid in [0, 1] (threshold 0.3)
      - dense, no rerank → cosine in [0, 1] (threshold 0.4)
      - hybrid, no rerank → RRF scores are tiny (~1/60), no clean threshold;
        trust the result unless hits is empty.
    """
    if not hits:
        return Assessment(
            ok=False, reason="no hits returned",
            retry_strategy=stronger_strategy(strategy),
        )
    top = hits[0].score
    if strategy.rerank and top < threshold_rerank:
        return Assessment(
            ok=False,
            reason=f"top rerank score {top:.3f} < {threshold_rerank}",
            retry_strategy=stronger_strategy(strategy),
        )
    if (not strategy.rerank) and strategy.retrieval == "dense" and top < threshold_dense:
        return Assessment(
            ok=False,
            reason=f"top dense score {top:.3f} < {threshold_dense}",
            retry_strategy=stronger_strategy(strategy),
        )
    return Assessment(ok=True, reason=f"top score {top:.3f} above threshold", retry_strategy=None)


def reflect_on_answer(question: str, answer_text: str, hits: list[Hit], model: str, api_key: str | None = None) -> Reflection:
    context = "\n\n---\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, start=1))
    prompt = REFLECT_PROMPT % (question, context, answer_text)
    try:
        text, _, _ = _dispatch(prompt, model=model, temperature=0.0, api_key=api_key)
        obj = _extract_json(text)
        return Reflection(
            faithful=bool(obj.get("faithful", True)),
            critique=str(obj.get("critique", "")).strip(),
        )
    except Exception as e:
        return Reflection(faithful=True, critique=f"reflect failed ({type(e).__name__})")
