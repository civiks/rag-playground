"""Stuff retrieved chunks into a Gemini prompt and call generate_content."""
from __future__ import annotations

import os
from dataclasses import dataclass

from google import genai
from google.genai import types

from retrieve import Hit

MODEL = "gemini-2.5-flash"

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
