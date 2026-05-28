from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def _best_device() -> str:
    """MPS on Apple Silicon, CUDA on Linux/Windows GPUs, else CPU. Both BGE models
    run fine on MPS — unlike Docling's layout model, which hits float64 and crashes."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

QDRANT_PATH = Path(__file__).parent / "data" / "qdrant"
DEFAULT_COLLECTION = "rag_naive_1200"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RRF_K = 60
PREFETCH_N = 20  # how many to pull from each branch before fusing / before reranking


@dataclass
class Hit:
    text: str
    source: str
    chunk_idx: int
    score: float


_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_client: QdrantClient | None = None
_bm25_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device=_best_device())
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, device=_best_device())
    return _reranker


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def _tokenize(text: str) -> list[str]:
    # Strip Docling placeholders so the literal word `formula` inside
    # `<!-- formula-not-decoded -->` doesn't game BM25 on formula queries.
    text = text.replace("<!-- formula-not-decoded -->", "").replace("<!-- image -->", "")
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25(collection: str) -> tuple[BM25Okapi, list[dict]]:
    if collection in _bm25_cache:
        return _bm25_cache[collection]
    client = _get_client()
    payloads: list[dict] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection, limit=1000, offset=offset, with_payload=True, with_vectors=False
        )
        payloads.extend(r.payload for r in records)
        if offset is None:
            break
    corpus = [_tokenize(p["text"]) for p in payloads]
    bm25 = BM25Okapi(corpus)
    _bm25_cache[collection] = (bm25, payloads)
    return bm25, payloads


def _retrieve_dense(query: str, k: int, collection: str) -> list[Hit]:
    embedder = _get_embedder()
    client = _get_client()
    vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    res = client.query_points(collection_name=collection, query=vec, limit=k).points
    return [
        Hit(text=p.payload["text"], source=p.payload["source"], chunk_idx=p.payload["chunk_idx"], score=p.score)
        for p in res
    ]


def _retrieve_bm25(query: str, k: int, collection: str) -> list[Hit]:
    bm25, payloads = _get_bm25(collection)
    scores = bm25.get_scores(_tokenize(query))
    top = sorted(range(len(payloads)), key=lambda i: -scores[i])[:k]
    return [
        Hit(text=payloads[i]["text"], source=payloads[i]["source"], chunk_idx=payloads[i]["chunk_idx"], score=float(scores[i]))
        for i in top
    ]


def rrf_fuse(rankings: list[list[Hit]], k: int) -> list[Hit]:
    """Reciprocal Rank Fusion: 1 / (RRF_K + rank), summed across rankings."""
    scores: dict[tuple[str, int], float] = {}
    items: dict[tuple[str, int], Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.source, hit.chunk_idx)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            items.setdefault(key, hit)
    sorted_keys = sorted(scores, key=lambda x: -scores[x])[:k]
    out = []
    for key in sorted_keys:
        h = items[key]
        out.append(Hit(text=h.text, source=h.source, chunk_idx=h.chunk_idx, score=scores[key]))
    return out


def retrieve(query: str, k: int = 5, collection: str = DEFAULT_COLLECTION, strategy: str = "dense") -> list[Hit]:
    if strategy == "hybrid":
        dense = _retrieve_dense(query, k=PREFETCH_N, collection=collection)
        bm25 = _retrieve_bm25(query, k=PREFETCH_N, collection=collection)
        return rrf_fuse([dense, bm25], k=k)
    return _retrieve_dense(query, k=k, collection=collection)


def rerank_hits(query: str, hits: list[Hit], k: int) -> list[Hit]:
    """Cross-encoder rescores (query, chunk) pairs jointly; returns top-k by relevance.

    Unlike dense/BM25 which score query and chunk independently and then compare,
    a cross-encoder feeds both into a transformer together — so it can model
    fine-grained relevance ("does this chunk actually *answer* this question?").
    """
    reranker = _get_reranker()
    pairs = [(query, h.text) for h in hits]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, hits), key=lambda x: -x[0])[:k]
    return [
        Hit(text=h.text, source=h.source, chunk_idx=h.chunk_idx, score=float(s))
        for s, h in scored
    ]
