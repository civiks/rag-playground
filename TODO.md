# todo

- [x] naive rag: docling → 1200-char chunks → bge embeddings → qdrant cosine top-k → gemini
- [x] structure-aware chunking: docling `HybridChunker`, new `rag_docling_hybrid` collection
- [x] hybrid retrieval: bm25 + dense, fused via rrf (k=60)
- [x] cross-encoder rerank: `bge-reranker-v2-m3`, top-20 → top-k, MPS/CUDA aware
- [x] query rewriting: hyde + multi-query, both feed into hybrid+rerank
- [x] multi-provider llm: gemini / groq / ollama via `provider:model_id` dispatch
- [x] chat ui: `st.chat_message`, last N turns threaded, streaming for all 3 providers, under-the-hood panel per response

- [x] adaptive agent: `classify_query` → `assess_retrieval` → `reflect_on_answer`, one retry max, `agent.*` phoenix spans
- [x] ragas eval: `eval_questions.json` + `eval.py`, faithfulness / answer_relevancy / context_precision via ragas 0.3.x
- [x] contextual retrieval: llm-generated preamble per hybrid chunk, prepended pre-embed, cached by `sha1(chunk)`, `rag_contextual` collection

- [ ] shave off latency
- [ ] ingestion
    - [ ] drag and drop in st
    - [ ] scaling
