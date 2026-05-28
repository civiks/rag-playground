# rag-playground

A playground for experimenting with RAG techniques.

## Stack

LLM: Gemini · Groq · Ollama (selectable in UI) &nbsp;·&nbsp;
Embeddings: `BAAI/bge-small-en-v1.5` (local) &nbsp;·&nbsp;
Reranker: `BAAI/bge-reranker-v2-m3` (local) &nbsp;·&nbsp;
Vector store: Qdrant (local, embedded) &nbsp;·&nbsp;
PDF parsing: Docling &nbsp;·&nbsp;
Observability: Arize Phoenix &nbsp;·&nbsp;
UI: Streamlit

## Run

```bash
uv sync
cp .env.example .env                    # add keys for inference providers
uv run python ingest.py                 # parse PDFs in corpus/, build Qdrant collections
uv run phoenix serve &                  # trace UI on http://localhost:6006
uv run streamlit run app.py             # app on http://localhost:8501
```

### Provider keys

- **Gemini** — `GOOGLE_API_KEY` from https://aistudio.google.com/apikey
- **Groq** (free, separate quota bucket) — `GROQ_API_KEY` from https://console.groq.com/keys
- **Ollama** (local, no key) — `brew install ollama && ollama pull llama3.1:8b && ollama serve`

## What's in / what's next

See [TODO.md](./TODO.md).
