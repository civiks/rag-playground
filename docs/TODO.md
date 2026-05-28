# todo

- [x] naive rag: docling → 1200-char sliding chunks → bge embeddings → qdrant cosine top-k → gemini
- [x] structure-aware chunking: docling `HybridChunker`, `rag_docling_hybrid` collection
- [x] hybrid retrieval: bm25 + dense, fused via rrf (k=60)
- [x] cross-encoder rerank: `bge-reranker-v2-m3`, top-20 → top-k, MPS/CUDA aware
- [x] query rewriting: hyde + multi-query, both feed into hybrid+rerank
- [x] multi-provider llm: gemini / groq / ollama via `provider:model_id` dispatch
- [x] chat ui: `st.chat_message`, last N turns threaded, streaming for all 3 providers, under-the-hood panel per response
- [x] adaptive agent: classify query → gate retrieval → self-critique → one retry max, `agent.*` phoenix spans
- [x] ragas eval: `evals/eval_questions.json` + `evals/eval.py`, faithfulness / answer_relevancy / context_precision via ragas 0.3.x
- [x] contextual retrieval: llm-generated preamble per hybrid chunk, prepended pre-embed, sha1 cache, `rag_contextual` collection
- [x] semantic chunking: split where consecutive sentence embeddings drop in cosine (topic shift), bounded by min/max chars
- [x] parent-child chunking: index ~300-char children, return ~1200-char parents to the llm
- [x] pdf upload: drag-and-drop, ingest into in-memory qdrant, sha1(bytes) cache across sessions
- [x] ocr auto-detection: pdfplumber text density per page, only invoke docling OCR where needed
- [x] live deploy: streamlit community cloud, byo gemini key

### evals
- [ ] publish RAGAS numbers in README — run `make eval`, paste the comparison table
- [ ] expand `eval_questions.json` to 50+ qs, balance across failure modes (jargon, paraphrase, multi-hop, refusal, numerical)
- [ ] golden-set regression test in CI — fail PR if any metric drops > 3%
- [ ] open-source benchmarks: MultiHopRAG, FinQA, HotpotQA dev subsets

### retrieval frontier
- [ ] GraphRAG — entity/relation extraction at ingest, query over graph + vector hybrid for multi-hop questions
- [ ] RAPTOR — recursive tree of chunk summaries, retrieve at the right level of abstraction
- [ ] Self-RAG / Corrective-RAG — formal "should I retrieve?" gate + per-chunk relevance filter before reranking
- [ ] ColBERT late-interaction reranker — token-level matching, often beats cross-encoder on long contexts
- [ ] late chunking — embed the full doc with a long-context embedder, then chunk the resulting token embeddings
- [ ] metadata filtering — date / section / source-type filters applied before vector search

### production hardening
- [ ] clickable citations — answer span → highlight source chunk in a side panel
- [ ] semantic query cache — sha1(normalized query + config) → cached answer + sources, ttl-based
- [ ] cost/latency in UI — token count + $ per turn, latency breakdown per stage
- [ ] provider fallback chain — gemini 429 → groq → ollama, log the demotion as a span
- [ ] conversational query rewriting — reformulate the question using chat history before retrieval
- [ ] cost guardrails — soft warning > N tokens, hard refuse > M

### observability
- [ ] phoenix dashboards saved view — latency p50/p95, faithfulness over time, retrieval recall@k
- [ ] user feedback — thumbs up/down per answer, written to the trace, queryable in phoenix
- [ ] failure-mode autotagging — agent labels each turn (refusal / jargon-miss / hallucination / partial) for dashboard slicing

### infra / dx
- [ ] unit tests for chunking strategies (deterministic, no model calls)
- [ ] github actions: ruff + pytest + tiny eval on PR
- [ ] docker compose for local dev (app + phoenix in one `docker compose up`)
- [ ] type checking — pyright strict on `retrieve.py`, `agent.py`, `rag.py`

### nice-to-haves
- [ ] multimodal: index figures / tables / formulas separately (docling already extracts them)
- [ ] long-context mode: fewer, larger chunks for gemini-2.5-pro
- [ ] document deduplication at ingest (sha of chunk text, drop dups across PDFs)
- [ ] quantized embeddings (int8 BGE) — smaller index, faster cold start on hosted
