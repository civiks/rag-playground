# rag-playground

A phased RAG system built step by step.

## Stack

Gemini · BGE embeddings (local) · Qdrant (local) · Docling · Arize Phoenix · Streamlit

## Run

```bash
uv sync
echo "GOOGLE_API_KEY=..." > .env       # get one at https://aistudio.google.com/apikey
uv run python ingest.py                 # parse PDFs in corpus/, build index
uv run phoenix serve &                  # trace UI on http://localhost:6006
uv run streamlit run app.py             # app on http://localhost:8501
```

## Phases

| # | Phase | Status |
|---|---|---|
| 1 | Naive RAG (fixed chunks, dense top-k, Gemini) | ✅ |
| 2 | RAGAS eval harness | |
| 3 | Structure-aware chunking | |
| 4 | Hybrid search (BM25 + dense) | |
| 5 | Cross-encoder reranking | |
| 6 | Query rewriting (HyDE, multi-query) | |
| 7 | Local LLM (Ollama) | |
| 8 | Contextual retrieval | |
