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
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from dotenv import load_dotenv
from generate import _dispatch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()

CORPUS_DIR = Path(__file__).parent / "corpus"
QDRANT_PATH = Path(__file__).parent / "data" / "qdrant"
CONTEXTUAL_CACHE_PATH = Path(__file__).parent / "data" / "contextual_cache.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
HYBRID_MAX_TOKENS = 400
CONTEXTUAL_MODEL = "ollama:llama3.1:8b"

STRATEGIES = ["naive_1200", "docling_hybrid", "contextual"]


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


def _load_contextual_cache() -> dict[str, str]:
    if CONTEXTUAL_CACHE_PATH.exists():
        return json.loads(CONTEXTUAL_CACHE_PATH.read_text())
    return {}


def _save_contextual_cache(cache: dict[str, str]) -> None:
    CONTEXTUAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTEXTUAL_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _generate_preamble(chunk_text: str, doc_prefix: str, model: str) -> str:
    """One LLM call: given chunk + document excerpt, produce a 1-2 sentence summary."""
    prompt = (
        "Here is an excerpt from a document:\n\n"
        f"{doc_prefix}\n\n"
        "Here is a specific chunk from that document:\n\n"
        f"{chunk_text}\n\n"
        "Write 1-2 sentences describing what this chunk covers within the document. "
        "Be specific about its content. Output only the description, no preface."
    )
    text, _, _ = _dispatch(prompt, model=model, temperature=0.0)
    return text.strip()


def chunks_contextual(
    hybrid_chunks: list[str],
    doc_markdown: str,
    model: str = CONTEXTUAL_MODEL,
    workers: int = 4,
) -> list[str]:
    """Prepend an LLM-generated preamble to each chunk before embedding.

    Cached by sha1(chunk_text) — re-ingest skips already-processed chunks.
    Parallelises uncached chunks across `workers` threads; Ollama handles
    concurrent requests natively so this gives a near-linear speedup.
    """
    cache = _load_contextual_cache()
    doc_prefix = doc_markdown[:3000]

    uncached = [
        (i, chunk)
        for i, chunk in enumerate(hybrid_chunks)
        if hashlib.sha1(chunk.encode()).hexdigest() not in cache
    ]

    if uncached:
        print(f"      {len(uncached)} chunks to generate ({len(hybrid_chunks) - len(uncached)} cached)")
        done = 0

        def _gen(args):
            i, chunk = args
            key = hashlib.sha1(chunk.encode()).hexdigest()
            return key, _generate_preamble(chunk, doc_prefix, model)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_gen, item): item for item in uncached}
            for fut in as_completed(futures):
                key, preamble = fut.result()
                cache[key] = preamble
                done += 1
                print(f"      {done}/{len(uncached)}", end="\r", flush=True)

        print()
        _save_contextual_cache(cache)

    return [f"{cache[hashlib.sha1(c.encode()).hexdigest()]}\n\n{c}" for c in hybrid_chunks]


def stable_id(strategy: str, source: str, idx: int) -> int:
    h = hashlib.sha1(f"{strategy}:{source}:{idx}".encode()).hexdigest()
    return int(h[:15], 16)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contextual-model", default=CONTEXTUAL_MODEL,
        help="provider:model_id used to generate chunk preambles (default: %(default)s)",
    )
    args = parser.parse_args()

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {CORPUS_DIR}. Drop some in and re-run.")
        sys.exit(1)
    print(f"Found {len(pdfs)} PDF(s) in corpus/")

    embedder = SentenceTransformer(EMBED_MODEL)
    dim = embedder.get_sentence_embedding_dimension()

    active_strategies = STRATEGIES

    client = QdrantClient(path=str(QDRANT_PATH))
    for strategy in active_strategies:
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

        hybrid_chunks = chunks_hybrid(doc, hybrid_chunker)
        per_strategy: dict[str, list[str]] = {
            "naive_1200": chunks_naive(markdown),
            "docling_hybrid": hybrid_chunks,
        }
        if "contextual" in active_strategies:
            print(f"  [contextual] generating preambles via {args.contextual_model} ...")
            per_strategy["contextual"] = chunks_contextual(hybrid_chunks, markdown, args.contextual_model)

        for strategy, chunks in per_strategy.items():
            if strategy not in active_strategies:
                continue
            collection = f"rag_{strategy}"
            print(f"  [{strategy}] {len(chunks)} chunks → embedding")
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
    print(f"Collections: {[f'rag_{s}' for s in active_strategies]}")


if __name__ == "__main__":
    main()
