"""
Ingest pipeline: document -> Docling -> chunks (multiple strategies) -> embed -> Qdrant.

Supports PDF, DOCX, PPTX, HTML, MD, TXT, and images (PNG/JPG/JPEG) — Docling parses
the structured formats, plain text files take a fast bypass. Builds one Qdrant
collection per chunking strategy in a single run, so the UI can swap strategies
without re-ingesting.

Run: `uv run python ingest.py`
"""
from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

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
DOCLING_CACHE_DIR = Path(__file__).parent / "data" / "docling_cache"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
HYBRID_MAX_TOKENS = 400
CONTEXTUAL_MODEL = "ollama:llama3.1:8b"

STRATEGIES = ["naive_1200", "docling_hybrid", "contextual", "semantic", "parent_child"]
DOCLING_STRATEGIES = {"docling_hybrid", "contextual"}

INPUT_FORMAT_BY_EXT = {
    ".pdf":  InputFormat.PDF,
    ".docx": InputFormat.DOCX,
    ".pptx": InputFormat.PPTX,
    ".html": InputFormat.HTML, ".htm": InputFormat.HTML,
    ".png":  InputFormat.IMAGE, ".jpg": InputFormat.IMAGE, ".jpeg": InputFormat.IMAGE,
}
PLAIN_TEXT_EXTS = {".txt", ".md"}
SUPPORTED_EXTS = set(INPUT_FORMAT_BY_EXT) | PLAIN_TEXT_EXTS


def _pdf_needs_ocr(pdf_source) -> bool:
    """True if the PDF looks scanned: average extractable text per page is very low.

    pdfplumber is used as a cheap pre-check so Docling's OCR stack (RapidOCR model
    load + per-page inference) only runs when pages are actually image-based.
    Text-based pages typically yield 500–3000 chars each; scanned pages yield <100.
    """
    import pdfplumber
    with pdfplumber.open(pdf_source) as pdf:
        if not pdf.pages:
            return False
        total = sum(len(page.extract_text() or "") for page in pdf.pages)
        return (total / len(pdf.pages)) < 100


def _extract_text_fast(pdf_path: str) -> str:
    """Extract plain text with pdfplumber — no ML models, ~0.5s per PDF.
    Used for strategies that don't need layout structure (naive, semantic, parent_child).
    """
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _get_docling_data(doc_path: str, ext: str | None = None) -> tuple[str, list[str]]:
    """Parse any Docling-supported document and return (markdown, hybrid_chunks).
    Cached by file hash so Docling runs once per unique file."""
    if ext is None:
        ext = Path(doc_path).suffix.lower()
    file_hash = hashlib.sha1(Path(doc_path).read_bytes()).hexdigest()
    cache_file = DOCLING_CACHE_DIR / f"{file_hash}.json"

    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        print(f"  [docling] using cached parse ({file_hash[:8]})")
        return cached["markdown"], cached["hybrid_chunks"]

    print(f"  [docling] parsing {ext} (first time — will be cached after this)")
    if ext == ".pdf":
        opts = PdfPipelineOptions()
        opts.do_ocr = _pdf_needs_ocr(doc_path)
        opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    else:
        converter = DocumentConverter()

    result = converter.convert(doc_path)

    doc = result.document
    markdown = doc.export_to_markdown()
    hybrid_chunks = [c.text for c in HybridChunker(tokenizer=EMBED_MODEL, max_tokens=HYBRID_MAX_TOKENS).chunk(doc)]

    DOCLING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"markdown": markdown, "hybrid_chunks": hybrid_chunks}))
    return markdown, hybrid_chunks


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


def chunks_semantic(
    markdown: str,
    embedder: SentenceTransformer,
    threshold: float = 0.65,
    min_chars: int = 200,
    max_chars: int = 1500,
) -> list[str]:
    """Split where consecutive sentence embeddings diverge (topic shift).

    Avoids fixed-size cuts mid-argument. Threshold 0.65 = same topic;
    below that = likely a section boundary.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", markdown) if s.strip()]
    if not sentences:
        return [markdown]
    vecs = embedder.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    current_len = len(sentences[0])
    for i in range(1, len(sentences)):
        sim = float(np.dot(vecs[i - 1], vecs[i]))
        projected = current_len + 1 + len(sentences[i])
        if (sim < threshold and current_len >= min_chars) or projected > max_chars:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_len = len(sentences[i])
        else:
            current.append(sentences[i])
            current_len = projected
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunks_parent_child(
    markdown: str,
    parent_chars: int = CHUNK_CHARS,
    child_chars: int = 300,
    child_overlap: int = 50,
) -> list[dict]:
    """Index small child chunks for precise embedding; store parent for LLM context.

    Solves the precision-vs-context tradeoff: a 300-char child matches a specific
    claim precisely, but the LLM gets the surrounding 1200-char parent paragraph.
    """
    results: list[dict] = []
    p_start = 0
    p_idx = 0
    while p_start < len(markdown):
        parent = markdown[p_start : p_start + parent_chars]
        c_start = 0
        while c_start < len(parent):
            child = parent[c_start : c_start + child_chars]
            results.append({"text": child, "parent_text": parent, "parent_idx": p_idx})
            if c_start + child_chars >= len(parent):
                break
            c_start += child_chars - child_overlap
        if p_start + parent_chars >= len(markdown):
            break
        p_start += parent_chars - CHUNK_OVERLAP
        p_idx += 1
    return results


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

    docs = sorted(p for p in CORPUS_DIR.glob("*") if p.suffix.lower() in SUPPORTED_EXTS)
    if not docs:
        print(f"No supported documents in {CORPUS_DIR}. Supported: {sorted(SUPPORTED_EXTS)}")
        sys.exit(1)
    print(f"Found {len(docs)} document(s) in corpus/")

    embedder = SentenceTransformer(EMBED_MODEL)
    dim = embedder.get_embedding_dimension()

    active_strategies = STRATEGIES

    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url:
        client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))
        print(f"Using Qdrant Cloud: {qdrant_url}")
    else:
        client = QdrantClient(path=str(QDRANT_PATH))
    for strategy in active_strategies:
        client.recreate_collection(
            collection_name=f"rag_{strategy}",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    for doc_path in docs:
        print(f"\nIngesting {doc_path.name} ...")
        ext = doc_path.suffix.lower()
        has_layout = ext not in PLAIN_TEXT_EXTS

        markdown: str | None = None
        hybrid_chunks: list[str] | None = None
        if ext in PLAIN_TEXT_EXTS:
            fast_text = doc_path.read_text(encoding="utf-8", errors="replace")
        elif ext == ".pdf":
            fast_text = _extract_text_fast(str(doc_path))
        else:
            markdown, hybrid_chunks = _get_docling_data(str(doc_path), ext)
            fast_text = markdown

        per_strategy: dict[str, list[str]] = {}
        if "naive_1200" in active_strategies:
            per_strategy["naive_1200"] = chunks_naive(fast_text)
        if "semantic" in active_strategies:
            print(f"  [semantic] splitting on embedding similarity ...")
            per_strategy["semantic"] = chunks_semantic(fast_text, embedder)

        layout_strategies = DOCLING_STRATEGIES & set(active_strategies)
        if layout_strategies and has_layout:
            if hybrid_chunks is None:
                markdown, hybrid_chunks = _get_docling_data(str(doc_path), ext)
            if "docling_hybrid" in active_strategies:
                per_strategy["docling_hybrid"] = hybrid_chunks
            if "contextual" in active_strategies:
                print(f"  [contextual] generating preambles via {args.contextual_model} ...")
                per_strategy["contextual"] = chunks_contextual(hybrid_chunks, markdown, args.contextual_model)
        elif layout_strategies and not has_layout:
            print(f"  [skip] {sorted(layout_strategies)} — {ext} has no layout to extract")

        for strategy, chunks in per_strategy.items():
            collection = f"rag_{strategy}"
            print(f"  [{strategy}] {len(chunks)} chunks → embedding")
            vectors = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
            client.upsert(collection_name=collection, points=[
                PointStruct(
                    id=stable_id(strategy, doc_path.name, i),
                    vector=vec.tolist(),
                    payload={"source": doc_path.name, "chunk_idx": i, "text": chunk, "strategy": strategy},
                )
                for i, (chunk, vec) in enumerate(zip(chunks, vectors))
            ])

        if "parent_child" in active_strategies:
            pc_data = chunks_parent_child(fast_text)
            print(f"  [parent_child] {len(pc_data)} child chunks → embedding")
            vectors = embedder.encode([d["text"] for d in pc_data], show_progress_bar=False, normalize_embeddings=True)
            client.upsert(collection_name="rag_parent_child", points=[
                PointStruct(
                    id=stable_id("parent_child", doc_path.name, i),
                    vector=vec.tolist(),
                    payload={"source": doc_path.name, "chunk_idx": i, "text": d["text"],
                             "parent_text": d["parent_text"], "strategy": "parent_child"},
                )
                for i, (d, vec) in enumerate(zip(pc_data, vectors))
            ])

    print(f"\nPersistent store: {QDRANT_PATH}")
    print(f"Collections: {[f'rag_{s}' for s in active_strategies]}")


if __name__ == "__main__":
    main()
