from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_PATH = Path(__file__).parent / "data" / "qdrant"
DEFAULT_COLLECTION = "rag_naive_1200"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class Hit:
    text: str
    source: str
    chunk_idx: int
    score: float


_embedder: SentenceTransformer | None = None
_client: QdrantClient | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def retrieve(query: str, k: int = 5, collection: str = DEFAULT_COLLECTION) -> list[Hit]:
    embedder = _get_embedder()
    client = _get_client()
    vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    res = client.query_points(collection_name=collection, query=vec, limit=k).points
    return [
        Hit(
            text=p.payload["text"],
            source=p.payload["source"],
            chunk_idx=p.payload["chunk_idx"],
            score=p.score,
        )
        for p in res
    ]
