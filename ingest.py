"""
Ingest pipeline: PDF -> Docling -> chunks (multiple strategies) -> embed -> Qdrant.

Builds one Qdrant collection per chunking strategy in a single run, so the UI
can swap strategies without re-ingesting.

Run: `uv run python ingest.py`
"""
from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import hashlib
import sys
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

CORPUS_DIR = Path(__file__).parent / "corpus"
QDRANT_PATH = Path(__file__).parent / "data" / "qdrant"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
HYBRID_MAX_TOKENS = 400

STRATEGIES = ["naive_1200", "docling_hybrid"]


def chunks_naive(markdown: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(markdown):
        end = start + CHUNK_CHARS
        chunks.append(markdown[start:end])
        if end >= len(markdown):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def chunks_hybrid(doc, chunker: HybridChunker) -> list[str]:
    return [c.text for c in chunker.chunk(doc)]


def stable_id(strategy: str, source: str, idx: int) -> int:
    h = hashlib.sha1(f"{strategy}:{source}:{idx}".encode()).hexdigest()
    return int(h[:15], 16)


def main() -> None:
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {CORPUS_DIR}. Drop some in and re-run.")
        sys.exit(1)
    print(f"Found {len(pdfs)} PDF(s) in corpus/")

    embedder = SentenceTransformer(EMBED_MODEL)
    dim = embedder.get_sentence_embedding_dimension()

    client = QdrantClient(path=str(QDRANT_PATH))
    for strategy in STRATEGIES:
        client.recreate_collection(
            collection_name=f"rag_{strategy}",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    hybrid_chunker = HybridChunker(tokenizer=EMBED_MODEL, max_tokens=HYBRID_MAX_TOKENS)

    for pdf in pdfs:
        print(f"\nParsing {pdf.name} ...")
        result = converter.convert(str(pdf))
        doc = result.document
        markdown = doc.export_to_markdown()

        per_strategy: dict[str, list[str]] = {
            "naive_1200": chunks_naive(markdown),
            "docling_hybrid": chunks_hybrid(doc, hybrid_chunker),
        }

        for strategy, chunks in per_strategy.items():
            collection = f"rag_{strategy}"
            print(f"  [{strategy}] {len(chunks)} chunks")
            vectors = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
            points = [
                PointStruct(
                    id=stable_id(strategy, pdf.name, i),
                    vector=vec.tolist(),
                    payload={"source": pdf.name, "chunk_idx": i, "text": chunk, "strategy": strategy},
                )
                for i, (chunk, vec) in enumerate(zip(chunks, vectors))
            ]
            client.upsert(collection_name=collection, points=points)

    print(f"\nPersistent store: {QDRANT_PATH}")
    print(f"Collections: {[f'rag_{s}' for s in STRATEGIES]}")


if __name__ == "__main__":
    main()
