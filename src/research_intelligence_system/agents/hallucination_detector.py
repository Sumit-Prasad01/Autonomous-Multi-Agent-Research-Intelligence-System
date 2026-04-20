"""
hallucination_detector.py — NLI-based hallucination detection
Uses BGE cross-encoder (already loaded) to score summary faithfulness.
Measures how well each sentence in the summary is supported by source text.

Score: 0.0 = fully grounded (no hallucination)
       1.0 = fully hallucinated (no support in source)
"""
from __future__ import annotations

import asyncio
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hallucination")

# thresholds
_SUPPORT_THRESHOLD  = 0.0   # cross-encoder score above this = supported
_MIN_SENTENCE_LEN   = 20    # ignore very short sentences
_MAX_SENTENCES      = 15    # cap sentences to check (token budget)
_MAX_CHUNKS         = 10    # top chunks to check against


# ── Sentence splitter ─────────────────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    """Split summary into checkable sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [
        s.strip() for s in sentences
        if len(s.strip()) >= _MIN_SENTENCE_LEN
    ][:_MAX_SENTENCES]


# ── Core NLI scoring ──────────────────────────────────────────────────────────
def _score_sentence(
    sentence:   str,
    chunks:     List[str],
    model,
) -> Tuple[float, str]:
    """
    Score a single sentence against all source chunks.
    Returns (max_support_score, best_supporting_chunk).
    Higher score = better supported by source.
    """
    if not chunks:
        return 0.0, ""

    pairs  = [(sentence, chunk[:512]) for chunk in chunks[:_MAX_CHUNKS]]
    scores = model.predict(pairs)

    best_idx   = int(scores.argmax())
    best_score = float(scores[best_idx])
    best_chunk = chunks[best_idx][:200]

    return best_score, best_chunk


def _compute_hallucination_sync(
    summary: str,
    chunks:  List[str],
) -> Dict:
    """
    Sync hallucination computation — runs in threadpool.
    Returns detailed hallucination report.
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(
            "BAAI/bge-reranker-base",
            local_files_only=True,
        )
    except Exception as e:
        logger.warning(f"[HALLUCINATION] cross-encoder load failed: {e}")
        return _empty_result()

    sentences = _split_sentences(summary)
    if not sentences:
        return _empty_result()

    sentence_results = []
    hallucinated     = []
    supported        = []

    for sentence in sentences:
        score, best_chunk = _score_sentence(sentence, chunks, model)

        is_supported = score > _SUPPORT_THRESHOLD
        sentence_results.append({
            "sentence":       sentence,
            "support_score":  round(float(score), 4),
            "is_supported":   is_supported,
            "best_evidence":  best_chunk,
        })

        if is_supported:
            supported.append(sentence)
        else:
            hallucinated.append(sentence)

    total = len(sentences)
    hallucination_rate = len(hallucinated) / total if total > 0 else 0.0
    faithfulness_score = 1.0 - hallucination_rate

    logger.info(
        f"[HALLUCINATION] sentences={total} "
        f"supported={len(supported)} "
        f"hallucinated={len(hallucinated)} "
        f"rate={hallucination_rate:.2f}"
    )

    return {
        "hallucination_score":  round(hallucination_rate, 4),
        "faithfulness_score":   round(faithfulness_score, 4),
        "total_sentences":      total,
        "supported_count":      len(supported),
        "hallucinated_count":   len(hallucinated),
        "hallucinated_sentences": hallucinated[:5],   # top 5 for display
        "sentence_details":     sentence_results,
    }


def _empty_result() -> Dict:
    return {
        "hallucination_score":    0.0,
        "faithfulness_score":     1.0,
        "total_sentences":        0,
        "supported_count":        0,
        "hallucinated_count":     0,
        "hallucinated_sentences": [],
        "sentence_details":       [],
    }


# ── Public API ────────────────────────────────────────────────────────────────
async def compute_hallucination_score(
    summary: str,
    chunks:  List[str],
) -> Dict:
    """
    Async entry point — compute hallucination score for a summary.

    Args:
        summary: The refined summary from critic agent
        chunks:  Source paper text chunks from Qdrant

    Returns:
        {
          hallucination_score: float (0=grounded, 1=hallucinated),
          faithfulness_score:  float (0=hallucinated, 1=grounded),
          total_sentences:     int,
          supported_count:     int,
          hallucinated_count:  int,
          hallucinated_sentences: List[str],
          sentence_details:    List[dict]
        }
    """
    if not summary or not chunks:
        return _empty_result()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _POOL,
        _compute_hallucination_sync,
        summary,
        chunks,
    )


def hallucination_label(score: float) -> str:
    """Human-readable label for hallucination score."""
    if score < 0.1:  return "✅ Highly Faithful"
    if score < 0.3:  return "🟡 Mostly Faithful"
    if score < 0.5:  return "🟠 Partially Hallucinated"
    return                   "🔴 Highly Hallucinated"