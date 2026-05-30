# todo

- [x] naive rag: docling → 1200-char sliding chunks → bge embeddings → qdrant cosine top-k → gemini
- [x] structure-aware chunking: docling `HybridChunker`, `rag_docling_hybrid` collection
- [x] hybrid retrieval: bm25 + dense, fused via rrf (k=60)
- [x] cross-encoder rerank: `bge-reranker-v2-m3`, top-20 → top-k
- [x] query rewriting: hyde + multi-query
- [x] multi-provider llm: gemini / groq / ollama via `provider:model_id` dispatch
- [x] chat ui: streaming for all 3 providers, under-the-hood panel, inline citation chips with cited-sources preview
- [x] adaptive RAG + CRAG-style correction + Self-RAG-style critique:
    - classify_query picks retrieval/rerank/rewrite by question shape (Adaptive RAG)
    - assess_retrieval gates on top score; retries with stronger strategy on weak retrieval (CRAG)
    - reflect_on_answer LLM-critiques final answer faithfulness (Self-RAG-style)
    - one retry max, every decision is an `agent.*` phoenix span
- [x] ragas eval: `evals/eval_questions.json` + `evals/eval.py` via ragas 0.3.x
- [x] contextual retrieval: llm-generated preamble per chunk, sha1 cache, `rag_contextual` collection
- [x] semantic chunking: topic-shift via cosine drop between consecutive sentence embeddings
- [x] parent-child chunking: index ~300-char children, return ~1200-char parents to the llm
- [x] pdf upload: drag-and-drop, in-memory qdrant, sha1(bytes) cache across sessions
- [x] ocr auto-detection: pdfplumber text density per page, only invoke docling OCR where needed
- [x] background model warmup: daemon thread loads embedder/reranker/qdrant on boot, thread-safe getters
- [x] live deploy: streamlit community cloud, byo gemini key

### evals
- [x] publish RAGAS numbers in README — comparison table across 4 configs
- [ ] expand `eval_questions.json` to 50+ qs, balance across failure modes (jargon, paraphrase, multi-hop, refusal, numerical)
- [ ] golden-set regression test in CI — fail PR if any metric drops > 3%
- [ ] open-source benchmarks: MultiHopRAG, FinQA, HotpotQA dev subsets

### retrieval frontier
- [ ] complete CRAG — web search fallback (DuckDuckGo, no key) for the "weak retrieval" branch
- [ ] RAPTOR — recursive cluster+summarize tree, 6th chunking strategy
- [ ] query decomposition — split multi-hop questions into sub-questions, retrieve per sub-question, RRF-fuse
- [ ] GraphRAG — entity/relation extraction, knowledge graph + community summaries, graph-aware retrieval
- [ ] Matryoshka embeddings — swap `bge-small` for an MRL model, ablate dims vs recall
- [ ] ColPali — embed PDF pages as images, no OCR/chunking pipeline
- [ ] LLM listwise reranker — RankGPT-style on top-10 after cross-encoder
- [ ] ColBERT late-interaction reranker — token-level matching
- [ ] late chunking — embed full doc, chunk the token embeddings
- [ ] metadata filtering — date / section / source-type filters before vector search
- [ ] semantic query cache — embed query, cosine-match against past Q→A pairs

### production hardening
- [ ] clickable citations — answer span scrolls to / highlights the source chunk
- [ ] cost / latency in UI — token count + $ per turn, latency breakdown per stage
- [ ] provider fallback chain — gemini 429 → groq → ollama, log the demotion as a span
- [ ] conversational query rewriting — reformulate question using chat history before retrieval
- [ ] cost guardrails — soft warning > N tokens, hard refuse > M

### observability
- [ ] phoenix dashboards saved view — latency p50/p95, faithfulness over time, retrieval recall@k
- [ ] user feedback — thumbs up/down per answer, written to the trace
- [ ] failure-mode autotagging — refusal / jargon-miss / hallucination / partial

### infra / dx
- [ ] unit tests for chunking strategies (deterministic, no model calls)
- [ ] github actions: ruff + pytest + tiny eval on PR
- [ ] docker compose for local dev (app + phoenix)
- [ ] type checking — pyright strict on `retrieve.py`, `agent.py`, `rag.py`

### nice-to-haves
- [ ] multimodal: index figures / tables / formulas separately (docling already extracts them)
- [ ] long-context mode: fewer, larger chunks for gemini-2.5-pro
- [ ] document deduplication at ingest (sha of chunk text, drop dups across PDFs)
- [ ] quantized embeddings (int8 BGE) — smaller index, faster cold start on hosted
