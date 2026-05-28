"""Stuff retrieved chunks into a Gemini prompt and call generate_content."""
from __future__ import annotations

import os
from dataclasses import dataclass

from google import genai
from google.genai import types

from retrieve import Hit

MODEL = "gemini-2.5-flash"
# Rewrites don't need flagship reasoning
REWRITE_MODEL = "gemini-2.5-flash-lite"

SYSTEM = """You are a careful research assistant. Answer the user's question \
using ONLY the provided context. If the context does not contain the answer, \
say so plainly. Cite chunks by their [#] marker when you use them."""


@dataclass
class Answer:
    text: str
    model: str
    citations_used: list[int]
    input_tokens: int
    output_tokens: int


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env. "
                "Get one free at https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] (source: {h.source}, chunk {h.chunk_idx}, score {h.score:.3f})\n{h.text}")
    return "\n\n---\n\n".join(blocks)


# Cache rewrites within a process so re-asking the same question across
# different retrieval / rerank / k configs doesn't re-call Gemini.
_rewrite_cache: dict[tuple[str, str], list[str]] = {}


def rewrite_to_hyde(question: str) -> str:
    """Generate a hypothetical answer so its embedding matches chunk-shape, not question-shape.

    HyDE bet: questions and answers live in different regions of embedding space.
    A bi-encoder trained on (query, passage) pairs handles the gap; an off-the-shelf
    semantic encoder mostly doesn't. Embedding a plausible answer fixes that.
    """
    cache_key = (question, "hyde")
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key][0]
    client = _get_client()
    prompt = (
        "Write a single short paragraph (3-5 sentences) that would plausibly answer "
        "the following question. Write it in the register of a source document that "
        "would contain the answer — declarative, specific, no hedging or meta-commentary. "
        "Do NOT say you're hypothetical. Output only the paragraph.\n\n"
        f"Question: {question}"
    )
    resp = client.models.generate_content(
        model=REWRITE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3),
    )
    text = (resp.text or "").strip()
    _rewrite_cache[cache_key] = [text]
    return text


def rewrite_to_multi(question: str, n: int = 3) -> list[str]:
    """Generate N paraphrases of the question covering different lexical angles."""
    cache_key = (question, f"multi_{n}")
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key]
    client = _get_client()
    prompt = (
        f"Rewrite the following question in {n} different ways. Each rewrite should "
        "preserve meaning but use different vocabulary and phrasing — aim to cover "
        "different ways an answer might be phrased in a source document. Output one "
        "rewrite per line, no numbering, no preface, no quotes.\n\n"
        f"Question: {question}"
    )
    resp = client.models.generate_content(
        model=REWRITE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    lines = [ln.strip().lstrip("-*0123456789. )") for ln in (resp.text or "").split("\n") if ln.strip()]
    lines = lines[:n]
    _rewrite_cache[cache_key] = lines
    return lines


def generate(question: str, hits: list[Hit]) -> Answer:
    client = _get_client()
    context = _format_context(hits)
    prompt = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.2,
        ),
    )
    text = resp.text or ""
    citations = [i for i in range(1, len(hits) + 1) if f"[{i}]" in text]
    usage = getattr(resp, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = getattr(usage, "candidates_token_count", 0) or 0
    return Answer(
        text=text, model=MODEL, citations_used=citations,
        input_tokens=in_tok, output_tokens=out_tok,
    )
