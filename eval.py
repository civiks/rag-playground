"""RAGAS evaluation harness.

Runs eval_questions.json through the RAG pipeline and scores with RAGAS metrics:
  faithfulness, answer_relevancy, context_precision

Results are printed as a table and saved to data/eval/<config>.json.

Usage:
  uv run python eval.py
  uv run python eval.py --chunking contextual --retrieval hybrid --rerank --rewrite multi
  uv run python eval.py --chunking naive_1200 --retrieval dense --model groq:llama-3.3-70b-versatile
  uv run python eval.py --type jargon   # only run jargon questions
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os
os.environ["RAG_NO_TRACE"] = "1"  # eval runs don't need Phoenix; avoids connection noise

# ragas 0.3.x imports langchain_community.chat_models.vertexai at module level,
# but that module was removed in langchain-community 0.2+. Stub it out so the
# import succeeds — we never use VertexAI.
import sys as _sys
from types import ModuleType as _ModuleType
if "langchain_community.chat_models.vertexai" not in _sys.modules:
    _stub = _ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
    _sys.modules["langchain_community.chat_models.vertexai"] = _stub

try:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas import RunConfig
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _RAGAS_OK = True
    _RAGAS_ERR = ""
except ImportError as e:
    _RAGAS_OK = False
    _RAGAS_ERR = str(e)

from rag import answer

QUESTIONS_FILE = Path(__file__).parent / "eval_questions.json"
RESULTS_DIR = Path(__file__).parent / "data" / "eval"


def _build_eval_llm(model: str = "ollama:llama3.1:8b"):
    """Build RAGAS eval LLM from a provider:model_id string.

    Embeddings always use the local BGE model (already downloaded for retrieval)
    so no embedding API calls regardless of which LLM provider you pick.
    """
    provider, model_id = model.split(":", 1) if ":" in model else ("ollama", model)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY required for gemini eval LLM")
        llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=model_id, google_api_key=api_key, temperature=0))
    elif provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY required for groq eval LLM")
        llm = LangchainLLMWrapper(ChatGroq(model=model_id, groq_api_key=api_key, temperature=0))
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = LangchainLLMWrapper(ChatOllama(model=model_id, temperature=0))
    else:
        raise ValueError(f"Unknown eval LLM provider: {provider!r}")

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )
    return llm, embeddings


def run_pipeline(questions: list[dict], args: argparse.Namespace) -> list[dict]:
    results = []
    for i, q in enumerate(questions, start=1):
        question = q["question"]
        print(f"  [{i:>2}/{len(questions)}] {question[:65]}")
        t0 = time.perf_counter()
        try:
            result = answer(
                question,
                k=args.k,
                collection=f"rag_{args.chunking}",
                strategy=args.retrieval,
                rerank=args.rerank,
                rewrite=args.rewrite,
                model=args.model,
                mode=args.mode,
            )
            results.append({
                "type": q.get("type", ""),
                "question": question,
                "ground_truth": q.get("ground_truth", ""),
                "answer": result.answer.text,
                "contexts": [h.text for h in result.hits],
                "latency_s": round(time.perf_counter() - t0, 2),
                "error": None,
            })
        except Exception as e:
            print(f"      ERROR: {e}")
            results.append({
                "type": q.get("type", ""),
                "question": question,
                "ground_truth": q.get("ground_truth", ""),
                "answer": "",
                "contexts": [],
                "latency_s": round(time.perf_counter() - t0, 2),
                "error": str(e),
            })
    return results


def score_ragas(results: list[dict], llm, embeddings) -> dict:
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"] or None,
        )
        for r in results
        if not r["error"] and r["answer"]
    ]

    if not samples:
        return {}

    dataset = EvaluationDataset(samples=samples)
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
    run_config = RunConfig(max_workers=4, timeout=120)
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings, run_config=run_config)
    return dict(result)


def _print_table(results: list[dict], ragas_scores: dict, config_tag: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"Config: {config_tag}")
    print(f"{'─' * 72}")
    print(f"  {'#':>2}  {'Type':<12} {'Latency':>8}  Question")
    print(f"{'─' * 72}")
    for i, r in enumerate(results, start=1):
        err = " ERR" if r["error"] else ""
        print(f"  {i:>2}  {r['type']:<12} {r['latency_s']:>7.1f}s  {r['question'][:42]}{err}")

    if ragas_scores:
        print(f"\n{'─' * 40}")
        print("  RAGAS scores (averaged over successful samples)")
        print(f"{'─' * 40}")
        score_map = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
        }
        for key, label in score_map.items():
            val = ragas_scores.get(key)
            if val is not None:
                try:
                    print(f"  {label:<22} {float(val):.4f}")
                except (TypeError, ValueError):
                    print(f"  {label:<22} {val}")

    errors = [r for r in results if r["error"]]
    if errors:
        print(f"\n  {len(errors)} question(s) failed — see saved results for details")
    print(f"{'=' * 72}")


def _save(results: list[dict], ragas_scores: dict, config_tag: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{config_tag}.json"

    serializable_scores = {}
    for k, v in ragas_scores.items():
        try:
            serializable_scores[k] = float(v)
        except (TypeError, ValueError):
            serializable_scores[k] = str(v)

    out.write_text(json.dumps({
        "config_tag": config_tag,
        "ragas_scores": serializable_scores,
        "n_questions": len(results),
        "n_errors": sum(1 for r in results if r["error"]),
        "per_question": results,
    }, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS eval over eval_questions.json")
    parser.add_argument("--chunking", default="docling_hybrid",
                        choices=["naive_1200", "docling_hybrid", "contextual"])
    parser.add_argument("--retrieval", default="hybrid", choices=["dense", "hybrid"])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rewrite", default="off", choices=["off", "hyde", "multi"])
    parser.add_argument("--mode", default="manual", choices=["manual", "auto"])
    parser.add_argument("--model", default="gemini:gemini-2.5-flash-lite",
                        help="provider:model_id — Gemini, Groq, or Ollama")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--type", dest="filter_type", default=None,
                        choices=["direct", "jargon", "paraphrased", "multi_hop", "refusal"],
                        help="Only evaluate questions of this type")
    parser.add_argument("--eval-model", default="llama3.1:8b",
                        help="Ollama model ID used by RAGAS for scoring (separate from --model used for answers)")
    args = parser.parse_args()

    if not _RAGAS_OK:
        print(f"RAGAS not installed: {_RAGAS_ERR}")
        print("Install: uv add ragas langchain-google-genai")
        return

    questions = json.loads(QUESTIONS_FILE.read_text())
    if args.filter_type:
        questions = [q for q in questions if q.get("type") == args.filter_type]
        print(f"Filtered to {len(questions)} '{args.filter_type}' questions")

    config_tag = (
        f"{args.chunking}_{args.retrieval}"
        f"_rerank={args.rerank}"
        f"_rewrite={args.rewrite}"
        f"_mode={args.mode}"
    )
    print(f"Running eval: {config_tag}")
    print(f"Model: {args.model}  k={args.k}  n={len(questions)} questions\n")

    results = run_pipeline(questions, args)

    print(f"\nScoring with RAGAS (eval model: {args.eval_model}) ...")
    eval_llm, eval_embeddings = _build_eval_llm(args.eval_model)
    ragas_scores = score_ragas(results, eval_llm, eval_embeddings)

    _print_table(results, ragas_scores, config_tag)

    out = _save(results, ragas_scores, config_tag)
    print(f"Saved → {out.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
