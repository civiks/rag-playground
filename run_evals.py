"""Run the full eval matrix and print a comparison table.

Runs 4 configs from weakest to strongest, scores each with RAGAS,
then prints a side-by-side table showing how each technique improves the numbers.

Usage:
    uv run python run_evals.py                          # default: ollama llama3.2:3b
    uv run python run_evals.py --model ollama:llama3.1:8b --eval-model llama3.1:8b
    uv run python run_evals.py --skip-pipeline          # re-score from existing data/eval/ JSONs
"""
from __future__ import annotations

import argparse
import json
import os
import time
import types
from pathlib import Path

os.environ["RAG_NO_TRACE"] = "1"

# ---------------------------------------------------------------------------
# Configs — weakest to strongest. Each row is one eval run.
# ---------------------------------------------------------------------------
CONFIGS = [
    dict(
        label="naive · dense",
        chunking="naive_1200", retrieval="dense",
        rerank=False, rewrite="off", mode="manual",
    ),
    dict(
        label="hybrid · rerank",
        chunking="docling_hybrid", retrieval="hybrid",
        rerank=True, rewrite="off", mode="manual",
    ),
    dict(
        label="hybrid · rerank · multi-query",
        chunking="docling_hybrid", retrieval="hybrid",
        rerank=True, rewrite="multi", mode="manual",
    ),
    dict(
        label="contextual · hybrid · rerank · multi · auto",
        chunking="contextual", retrieval="hybrid",
        rerank=True, rewrite="multi", mode="auto",
    ),
]

RESULTS_DIR = Path(__file__).parent / "data" / "eval"
QUESTIONS_FILE = Path(__file__).parent / "eval_questions.json"


def _config_tag(cfg: dict) -> str:
    return (
        f"{cfg['chunking']}_{cfg['retrieval']}"
        f"_rerank={cfg['rerank']}"
        f"_rewrite={cfg['rewrite']}"
        f"_mode={cfg['mode']}"
    )


def _run_one(cfg: dict, questions: list[dict], model: str, eval_model: str, k: int) -> dict:
    from eval import _build_eval_llm, _save, run_pipeline, score_ragas

    args = types.SimpleNamespace(
        chunking=cfg["chunking"],
        retrieval=cfg["retrieval"],
        rerank=cfg["rerank"],
        rewrite=cfg["rewrite"],
        mode=cfg["mode"],
        model=model,
        k=k,
        filter_type=None,
    )
    tag = _config_tag(cfg)
    print(f"\n{'─' * 60}")
    print(f"  {cfg['label']}")
    print(f"{'─' * 60}")

    results = run_pipeline(questions, args)

    print(f"  Scoring with RAGAS ({eval_model}) …")
    llm, embeddings = _build_eval_llm(eval_model)
    scores = score_ragas(results, llm, embeddings)
    _save(results, scores, tag)
    return scores


def _load_existing(cfg: dict) -> dict | None:
    path = RESULTS_DIR / f"{_config_tag(cfg)}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data.get("ragas_scores", {})
    return None


def _print_table(rows: list[tuple[str, dict]]) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    headers = ["Config", "Faithful", "Ans. Rel.", "Ctx. Prec."]
    col_w = [max(len(h), max(len(label) for label, _ in rows)) for h in headers]
    col_w[0] = max(len(headers[0]), max(len(label) for label, _ in rows))
    for i, h in enumerate(headers[1:], 1):
        col_w[i] = max(len(h), 8)

    def row_str(cells):
        return "  ".join(str(c).ljust(col_w[i]) for i, c in enumerate(cells))

    print(f"\n{'=' * (sum(col_w) + 2 * len(col_w))}")
    print(row_str(headers))
    print("  ".join("─" * w for w in col_w))
    for label, scores in rows:
        cells = [label]
        for m in metrics:
            v = scores.get(m)
            try:
                cells.append(f"{float(v):.3f}" if v is not None else "—")
            except (TypeError, ValueError):
                cells.append("—")
        print(row_str(cells))
    print(f"{'=' * (sum(col_w) + 2 * len(col_w))}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ollama:llama3.2:3b",
                        help="LLM for answering questions (default: ollama:llama3.2:3b)")
    parser.add_argument("--eval-model", default="ollama:llama3.2:3b",
                        help="LLM for RAGAS scoring (default: ollama:llama3.2:3b)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip running the pipeline; re-score from existing data/eval/ JSONs only")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text())

    rows: list[tuple[str, dict]] = []
    t_total = time.perf_counter()

    for cfg in CONFIGS:
        if args.skip_pipeline:
            scores = _load_existing(cfg)
            if scores is None:
                print(f"  [skip] no saved results for '{cfg['label']}' — run without --skip-pipeline first")
                continue
        else:
            scores = _run_one(cfg, questions, args.model, args.eval_model, args.k)
        rows.append((cfg["label"], scores))

    print(f"\nTotal time: {time.perf_counter() - t_total:.0f}s")
    _print_table(rows)

    # Emit a markdown table for copy-paste into README.
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    print("Markdown (paste into README):")
    print("| Config | Faithfulness | Answer Relevancy | Context Precision |")
    print("|---|---|---|---|")
    for label, scores in rows:
        vals = []
        for m in metrics:
            v = scores.get(m)
            try:
                vals.append(f"{float(v):.3f}" if v is not None else "—")
            except (TypeError, ValueError):
                vals.append("—")
        print(f"| {label} | {' | '.join(vals)} |")


if __name__ == "__main__":
    main()
