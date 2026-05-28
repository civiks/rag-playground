> A playground for experimenting with RAG techniques.

[Live demo](https://civiks-rag-playground.streamlit.app/) · bring your own Gemini key (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

## Run

Python 3.13 + [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
make install                # uv sync
cp .env.example .env        # add GOOGLE_API_KEY
make ingest                 # only if you drop new PDFs into corpus/
make phoenix &              # optional, trace UI on :6006
make app                    # http://localhost:8501
```

The Attention paper is pre-loaded and the Qdrant index ships with the repo, so `make app` works out of the box. `make help` lists the rest.

## Build log

1. **Naive RAG.** Docling → 1200-char sliding chunks (200 overlap) → `bge-small-en-v1.5` → Qdrant cosine top-5 → stuff into Gemini. Works for "What is multi-head attention?", fails on anything paraphrased or jargon-heavy.

2. **Eval harness.** Without numbers I can't tell whether a change helped. `evals/eval_questions.json` + RAGAS (faithfulness, answer relevancy, context precision). Gates every later change.

3. **Layout-aware chunking.** Fixed-size chunks split sentences mid-claim and lose section context. Swapped to Docling's `HybridChunker` — respects headings, keeps tables intact. New collection: `rag_docling_hybrid`.

4. **Hybrid retrieval.** Dense vectors miss rare technical terms ("scaled dot-product attention" tokenizes weirdly). Added BM25, fused with Reciprocal Rank Fusion (k=60). Both branches retrieve top-20, RRF picks top-k.

5. **Cross-encoder rerank.** RRF's top-k is still noisy. Retrieve 20, rerank with `bge-reranker-v2-m3`, take 5. MPS/CUDA aware.

6. **Query rewriting.** Question wording rarely matches source phrasing.
    - HyDE — LLM writes a hypothetical answer, embed *that*, retrieve against it.
    - Multi-query — three paraphrases, union their retrieved sets.

7. **Multi-provider LLM.** Gemini rate-limits hard on free tier. Added Groq (free, separate bucket) and Ollama (local). `provider:model_id` dispatch in `generate.py`, streaming for all three.

8. **Adaptive agent.** Manual mode makes the user pick a strategy. Auto mode: an LLM classifies the query, picks retrieval params, gates on retrieval confidence, self-critiques the answer (one retry max). Each decision is a Phoenix span.

9. **Contextual retrieval.** Anthropic's trick — each chunk gets an LLM-generated preamble describing where it sits in the doc, prepended *before* embedding. Cached by `sha1(chunk)`. Biggest single-step jump on hard questions. Collection: `rag_contextual`.

10. **Semantic + parent-child chunking.** Two more strategies:
     - Semantic — split where consecutive sentence embeddings drop in cosine (topic shift), bounded by min/max chars.
     - Parent-child — index small (~300 char) children, return the surrounding (~1200 char) parent to the LLM.

11. **PDF upload.** Drag a PDF into the sidebar, ingest into an in-memory Qdrant client, cache by `sha1(bytes)` so the same file is never re-parsed across sessions. OCR auto-detected per page via pdfplumber's text density.

## Stack

| | |
|---|---|
| LLM | Gemini 2.5 Flash · Groq Llama 3.x · Ollama |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, 384-dim) |
| Reranker | `BAAI/bge-reranker-v2-m3` (local) |
| Vector store | Qdrant (embedded, persistent) |
| PDF | Docling (layout) + pdfplumber (text-only + OCR check) |
| Tracing | Arize Phoenix / OpenTelemetry |
| UI | Streamlit |

No LangChain / LlamaIndex — orchestration is hand-written across 6 files.

## Repo layout

```
app.py            Streamlit UI
rag.py            orchestration: retrieve → rerank → generate → trace
retrieve.py       dense / BM25 / RRF / cross-encoder
generate.py       provider dispatch, streaming, query rewriting
ingest.py         PDF → chunks → embed → Qdrant (5 strategies)
agent.py          query classifier, retrieval gate, answer critic
corpus/           drop PDFs here, re-run `make ingest`
data/qdrant/      vector store (committed, pre-built for the Attention paper)
evals/            RAGAS harness
docs/             roadmap (TODO.md), example queries (queries.md)
```

## Provider keys

| | env var | source |
|---|---|---|
| Gemini | `GOOGLE_API_KEY` | aistudio.google.com/apikey (free) |
| Groq | `GROQ_API_KEY` | console.groq.com/keys (free) |
| Ollama | — | `brew install ollama && ollama pull llama3.1:8b && ollama serve` |

[docs/TODO.md](docs/TODO.md) · [docs/queries.md](docs/queries.md)
