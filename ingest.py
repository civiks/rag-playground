"""
Ingest pipeline: PDF -> Docling -> fixed-size chunks -> local embeddings -> Qdrant.

Run: `uv run python ingest.py`
"""
from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import hashlib
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

CORPUS_DIR = Path(__file__).parent / "corpus"
QDRANT_PATH = Path(__file__).parent / "data" / "qdrant"
COLLECTION = "rag_playground"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def stable_id(source: str, idx: int) -> int:
    """Qdrant point IDs must be int or UUID. Hash gives us deterministic ints across re-ingests."""
    h = hashlib.sha1(f"{source}:{idx}".encode()).hexdigest()
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
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Force CPU: Docling's layout model uses float64 ops that Apple MPS rejects.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    points: list[PointStruct] = []

    for pdf in pdfs:
        print(f"Parsing {pdf.name} ...")
        result = converter.convert(str(pdf))
        text = result.document.export_to_markdown()
        chunks = chunk_text(text)
        print(f"  -> {len(chunks)} chunks")

        vectors = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=stable_id(pdf.name, i),
                    vector=vec.tolist(),
                    payload={
                        "source": pdf.name,
                        "chunk_idx": i,
                        "text": chunk,
                    },
                )
            )

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"\nUpserted {len(points)} chunks into Qdrant collection '{COLLECTION}'.")
    print(f"Persistent store: {QDRANT_PATH}")


if __name__ == "__main__":
    main()
