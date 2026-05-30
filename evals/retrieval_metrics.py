"""Retrieval-only metrics: recall@k, MRR, nDCG@k.

No LLM, no new deps. Works across all chunking strategies via verbatim
span containment — gold_spans are short substrings that must appear in a
retrieved context for that context to count as a hit.
"""
from __future__ import annotations
import math
import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def span_hit(context: str, gold_spans: list[str]) -> bool:
    if not gold_spans:
        return False
    nc = _norm(context)
    return any(_norm(sp) in nc for sp in gold_spans)


def recall_at_k(contexts: list[str], gold_spans: list[str], k: int | None = None) -> float:
    pool = contexts[:k] if k else contexts
    return 1.0 if any(span_hit(c, gold_spans) for c in pool) else 0.0


def mrr(contexts: list[str], gold_spans: list[str]) -> float:
    for i, c in enumerate(contexts, start=1):
        if span_hit(c, gold_spans):
            return 1.0 / i
    return 0.0


def ndcg_at_k(contexts: list[str], gold_spans: list[str], k: int | None = None) -> float:
    pool = contexts[:k] if k else contexts
    hits = [1 if span_hit(c, gold_spans) else 0 for c in pool]
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    n_hits = sum(hits)
    if n_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_hits))
    return dcg / idcg


def score_retrieval(results: list[dict], k: int = 5) -> dict:
    """Aggregate recall@k, MRR, nDCG@k over results that have gold_spans."""
    by_type: dict[str, list] = {}
    all_r, all_m, all_n = [], [], []

    for r in results:
        gold = r.get("gold_spans")
        if gold is None or r.get("error"):
            continue
        if not gold:  # refusal questions — skip retrieval scoring
            continue
        contexts = r.get("contexts", [])
        rec = recall_at_k(contexts, gold, k)
        m = mrr(contexts, gold)
        n = ndcg_at_k(contexts, gold, k)
        all_r.append(rec); all_m.append(m); all_n.append(n)
        qtype = r.get("type", "unknown")
        by_type.setdefault(qtype, []).append((rec, m, n))

    def avg(lst): return sum(lst) / len(lst) if lst else None

    overall = {
        f"recall@{k}": avg(all_r),
        "mrr": avg(all_m),
        f"ndcg@{k}": avg(all_n),
    }
    per_type = {
        t: {
            f"recall@{k}": avg([x[0] for x in vs]),
            "mrr": avg([x[1] for x in vs]),
            f"ndcg@{k}": avg([x[2] for x in vs]),
        }
        for t, vs in by_type.items()
    }
    return {"overall": overall, "per_type": per_type, "n": len(all_r)}
