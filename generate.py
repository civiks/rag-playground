"""Call an LLM with retrieved chunks as context — supports multiple providers.

Three providers wired up so we're never blocked by a single quota / outage:
  - gemini  : Google Generative AI API (SDK; default)
  - groq    : Groq cloud, OpenAI-compatible JSON over HTTP (free tier, very fast)
  - ollama  : Local server on :11434 (HTTP)

Model strings are namespaced "provider:model_id". The dispatch is keyed off the
prefix. rag.py threads one through the pipeline; app.py exposes the selector.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib import error as urlerr
from urllib import request as urlreq

from google import genai
from google.genai import types

from retrieve import Hit

# Default for both answers and rewrites. Override via the model= param.
MODEL = "gemini:gemini-2.5-flash"

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


_gemini_client: genai.Client | None = None


def _gemini_client_or_die() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env. "
                "Get one free at https://aistudio.google.com/apikey"
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _split_model(model: str) -> tuple[str, str]:
    # Bare model names (no "provider:") default to gemini for backward compat.
    if ":" not in model:
        return "gemini", model
    provider, model_id = model.split(":", 1)
    return provider, model_id


def _call_gemini(prompt: str, model_id: str, temperature: float, system: str | None) -> tuple[str, int, int]:
    client = _gemini_client_or_die()
    cfg = (
        types.GenerateContentConfig(temperature=temperature, system_instruction=system)
        if system
        else types.GenerateContentConfig(temperature=temperature)
    )
    resp = client.models.generate_content(model=model_id, contents=prompt, config=cfg)
    usage = getattr(resp, "usage_metadata", None)
    return (
        (resp.text or "").strip(),
        int(getattr(usage, "prompt_token_count", 0) or 0),
        int(getattr(usage, "candidates_token_count", 0) or 0),
    )


def _call_groq(prompt: str, model_id: str, temperature: float, system: str | None) -> tuple[str, int, int]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GROQ_API_KEY in .env. Get a free key at https://console.groq.com/keys"
        )
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps({
        "model": model_id, "messages": messages, "temperature": temperature,
    }).encode()
    req = urlreq.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        data = json.loads(urlreq.urlopen(req, timeout=60).read())
    except urlerr.HTTPError as e:
        raise RuntimeError(f"Groq call failed ({e.code}): {e.read().decode(errors='ignore')}") from e
    text = (data["choices"][0]["message"].get("content") or "").strip()
    usage = data.get("usage") or {}
    return (
        text,
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
    )


def _call_ollama(prompt: str, model_id: str, temperature: float, system: str | None) -> tuple[str, int, int]:
    body: dict = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        body["system"] = system
    req = urlreq.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        data = json.loads(urlreq.urlopen(req, timeout=180).read())
    except (urlerr.URLError, urlerr.HTTPError) as e:
        raise RuntimeError(
            f"Ollama call failed — is `ollama serve` running, and have you "
            f"`ollama pull {model_id}`? ({e})"
        ) from e
    return (
        (data.get("response") or "").strip(),
        int(data.get("prompt_eval_count", 0) or 0),
        int(data.get("eval_count", 0) or 0),
    )


def _dispatch(prompt: str, model: str, temperature: float, system: str | None = None) -> tuple[str, int, int]:
    provider, model_id = _split_model(model)
    if provider == "gemini":
        return _call_gemini(prompt, model_id, temperature, system)
    if provider == "groq":
        return _call_groq(prompt, model_id, temperature, system)
    if provider == "ollama":
        return _call_ollama(prompt, model_id, temperature, system)
    raise ValueError(f"Unknown LLM provider: {provider!r} (model={model!r})")


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] (source: {h.source}, chunk {h.chunk_idx}, score {h.score:.3f})\n{h.text}")
    return "\n\n---\n\n".join(blocks)


# Cache rewrites within a process: same question + same rewrite mode + same
# model won't re-call the provider when you flip retrieval / rerank / k.
# Model is part of the key so switching providers regenerates correctly.
_rewrite_cache: dict[tuple[str, str, str], list[str]] = {}


def rewrite_to_hyde(question: str, model: str = MODEL) -> str:
    """Generate a hypothetical answer so its embedding matches chunk-shape, not question-shape.

    HyDE bet: questions and answers live in different regions of embedding space.
    A bi-encoder trained on (query, passage) pairs handles the gap; an off-the-shelf
    semantic encoder mostly doesn't. Embedding a plausible answer fixes that.
    """
    cache_key = (question, "hyde", model)
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key][0]
    prompt = (
        "Write a single short paragraph (3-5 sentences) that would plausibly answer "
        "the following question. Write it in the register of a source document that "
        "would contain the answer — declarative, specific, no hedging or meta-commentary. "
        "Do NOT say you're hypothetical. Output only the paragraph.\n\n"
        f"Question: {question}"
    )
    text, _, _ = _dispatch(prompt, model=model, temperature=0.3)
    _rewrite_cache[cache_key] = [text]
    return text


def rewrite_to_multi(question: str, n: int = 3, model: str = MODEL) -> list[str]:
    """Generate N paraphrases of the question covering different lexical angles."""
    cache_key = (question, f"multi_{n}", model)
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key]
    prompt = (
        f"Rewrite the following question in {n} different ways. Each rewrite should "
        "preserve meaning but use different vocabulary and phrasing — aim to cover "
        "different ways an answer might be phrased in a source document. Output one "
        "rewrite per line, no numbering, no preface, no quotes.\n\n"
        f"Question: {question}"
    )
    text, _, _ = _dispatch(prompt, model=model, temperature=0.7)
    lines = [ln.strip().lstrip("-*0123456789. )") for ln in text.split("\n") if ln.strip()]
    lines = lines[:n]
    _rewrite_cache[cache_key] = lines
    return lines


def generate(question: str, hits: list[Hit], model: str = MODEL) -> Answer:
    context = _format_context(hits)
    prompt = f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"
    text, in_tok, out_tok = _dispatch(prompt, model=model, temperature=0.2, system=SYSTEM)
    citations = [i for i in range(1, len(hits) + 1) if f"[{i}]" in text]
    return Answer(
        text=text, model=model, citations_used=citations,
        input_tokens=in_tok, output_tokens=out_tok,
    )
