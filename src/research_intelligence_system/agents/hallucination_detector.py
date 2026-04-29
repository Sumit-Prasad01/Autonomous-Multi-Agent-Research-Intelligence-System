"""
hallucination_detector.py — NLI-based faithfulness scoring
Uses BGE cross-encoder to score text against source chunks.

Supports:
  - Summary faithfulness (sentence-level)
  - Triple faithfulness (triple-level)
  - Gap evidence faithfulness (gap-level)

Score: 0.0 = fully grounded
       1.0 = fully hallucinated
"""
from __future__ import annotations

import asyncio
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hallucination")

# ── Thresholds ────────────────────────────────────────────────────────────────
_SUPPORT_THRESHOLD = 0.0    # cross-encoder score above this = supported
_TRIPLE_THRESHOLD  = -2.0   # lower threshold — triples are short, harder to match
_MIN_SENTENCE_LEN  = 20
_MAX_SENTENCES     = 15
_MAX_CHUNKS        = 10


# ── Singleton cross-encoder ───────────────────────────────────────────────────
_model      = None
_model_lock = threading.Lock()


def _get_model():
    """
    Load BGE cross-encoder once per process and reuse forever.

    The double-check locking pattern ensures thread safety:
    only one thread can create the model even under concurrent access.

    local_files_only is intentionally NOT set — sentence_transformers will
    use its disk cache (typically ~/.cache/huggingface/) on subsequent runs.
    The first run downloads the model (~500MB); all subsequent runs load
    from disk instantly. The HuggingFace API metadata ping happens only
    during the first CrossEncoder() construction call per process.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:   # second check under lock
                try:
                    import torch
                    from sentence_transformers import CrossEncoder
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    _model = CrossEncoder("BAAI/bge-reranker-base", device=device)
                    logger.info(f"[HALLUCINATION] cross-encoder loaded on {device}")
                except Exception as e:
                    logger.warning(f"[HALLUCINATION] cross-encoder load failed: {e}")
                    _model = None
    return _model


# ── Pre-warm: load model at module import time ────────────────────────────────
# This ensures the model is ready before the first paper analysis request,
# eliminating the 4s cold-load penalty that previously occurred mid-pipeline.
# The import of this module happens at server startup, so the 4s cost is
# paid once at boot — not once per paper.
def _prewarm() -> None:
    """Trigger model load in a background thread at import time."""
    def _load():
        _get_model()

    t = threading.Thread(target=_load, daemon=True, name="cross-encoder-prewarm")
    t.start()

_prewarm()


# Public alias for critic_agent and other callers
get_cross_encoder = _get_model


# ── Helpers ───────────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [
        s.strip() for s in sentences
        if len(s.strip()) >= _MIN_SENTENCE_LEN
    ][:_MAX_SENTENCES]


def _score_text_against_chunks(
    text:      str,
    chunks:    List[str],
    model,
    threshold: float = _SUPPORT_THRESHOLD,
) -> Tuple[float, str]:
    """
    Score a single text string against source chunks.
    Returns (best_score, best_supporting_chunk).
    Higher score = better supported by source.
    """
    if not chunks or not text.strip():
        return 0.0, ""

    pairs  = [(text[:512], chunk[:512]) for chunk in chunks[:_MAX_CHUNKS]]
    scores = model.predict(pairs)

    best_idx   = int(scores.argmax())
    best_score = float(scores[best_idx])
    best_chunk = chunks[best_idx][:200]

    return best_score, best_chunk


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


def _empty_triple_result() -> Dict:
    return {
        "filtered_triples":  [],
        "removed_count":     0,
        "kept_count":        0,
        "faithfulness_rate": 1.0,
    }


def _empty_gap_result() -> Dict:
    return {
        "scored_gaps":          [],
        "low_confidence_count": 0,
    }


# ── Summary faithfulness ──────────────────────────────────────────────────────
def _compute_hallucination_sync(
    summary: str,
    chunks:  List[str],
) -> Dict:
    """Sentence-level faithfulness scoring for summaries."""
    model = _get_model()
    if model is None:
        return _empty_result()

    sentences = _split_sentences(summary)
    if not sentences:
        return _empty_result()

    sentence_results = []
    hallucinated     = []
    supported        = []

    for sentence in sentences:
        score, best_chunk = _score_text_against_chunks(
            sentence, chunks, model, _SUPPORT_THRESHOLD
        )
        is_supported = score > _SUPPORT_THRESHOLD
        sentence_results.append({
            "sentence":      sentence,
            "support_score": round(float(score), 4),
            "is_supported":  is_supported,
            "best_evidence": best_chunk,
        })
        if is_supported:
            supported.append(sentence)
        else:
            hallucinated.append(sentence)

    total              = len(sentences)
    hallucination_rate = len(hallucinated) / total if total > 0 else 0.0
    faithfulness_score = 1.0 - hallucination_rate

    logger.info(
        f"[HALLUCINATION] sentences={total} "
        f"supported={len(supported)} "
        f"hallucinated={len(hallucinated)} "
        f"rate={hallucination_rate:.2f}"
    )

    return {
        "hallucination_score":    round(hallucination_rate, 4),
        "faithfulness_score":     round(faithfulness_score, 4),
        "total_sentences":        total,
        "supported_count":        len(supported),
        "hallucinated_count":     len(hallucinated),
        "hallucinated_sentences": hallucinated[:5],
        "sentence_details":       sentence_results,
    }


# ── Triple faithfulness ───────────────────────────────────────────────────────
def _filter_triples_sync(
    triples: List[Dict],
    chunks:  List[str],
) -> Dict:
    """Filter knowledge graph triples by faithfulness score."""
    model = _get_model()
    if model is None or not triples or not chunks:
        return {
            "filtered_triples":  triples,
            "removed_count":     0,
            "kept_count":        len(triples),
            "faithfulness_rate": 1.0,
        }

    kept    = []
    removed = []

    for triple in triples:
        subject  = triple.get("subject",  "")
        relation = triple.get("relation", "").replace("_", " ").lower()
        obj      = triple.get("object",   "")
        claim    = f"{subject} {relation} {obj}"

        score, _ = _score_text_against_chunks(
            claim, chunks, model, _TRIPLE_THRESHOLD
        )

        if score > _TRIPLE_THRESHOLD:
            triple["faithfulness_score"] = round(float(score), 4)
            kept.append(triple)
        else:
            removed.append(triple)
            logger.debug(f"[TRIPLE FILTER] removed '{claim}' score={score:.3f}")

    total             = len(triples)
    faithfulness_rate = len(kept) / total if total > 0 else 1.0

    logger.info(
        f"[TRIPLE FILTER] kept={len(kept)} "
        f"removed={len(removed)} "
        f"faithfulness={faithfulness_rate:.2f}"
    )

    return {
        "filtered_triples":  kept,
        "removed_count":     len(removed),
        "kept_count":        len(kept),
        "faithfulness_rate": round(faithfulness_rate, 4),
    }


# ── Gap evidence faithfulness ─────────────────────────────────────────────────
def _score_gaps_sync(
    gaps:   List[Dict],
    chunks: List[str],
) -> Dict:
    """Score gap supporting_evidence against source chunks."""
    model = _get_model()
    if model is None or not gaps or not chunks:
        return {"scored_gaps": gaps, "low_confidence_count": 0}

    scored_gaps    = []
    low_confidence = 0

    for gap in gaps:
        evidence = gap.get("supporting_evidence", "")

        if not evidence or len(evidence) < 20:
            gap["evidence_score"]    = 0.5
            gap["evidence_grounded"] = True
            scored_gaps.append(gap)
            continue

        score, _ = _score_text_against_chunks(
            evidence, chunks, model, _SUPPORT_THRESHOLD
        )

        is_grounded = score > _SUPPORT_THRESHOLD
        if not is_grounded:
            low_confidence += 1

        gap["evidence_score"]    = round(float(score), 4)
        gap["evidence_grounded"] = is_grounded
        scored_gaps.append(gap)

    logger.info(
        f"[GAP FILTER] scored={len(scored_gaps)} "
        f"low_confidence={low_confidence}"
    )

    return {
        "scored_gaps":          scored_gaps,
        "low_confidence_count": low_confidence,
    }


# ── Public async API ──────────────────────────────────────────────────────────
async def compute_hallucination_score(
    summary: str,
    chunks:  List[str],
) -> Dict:
    """Async — sentence-level hallucination score for a summary."""
    if not summary or not chunks:
        return _empty_result()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _POOL, _compute_hallucination_sync, summary, chunks,
    )


async def filter_triples_by_faithfulness(
    triples: List[Dict],
    chunks:  List[str],
) -> Dict:
    """Async — filter knowledge graph triples by source faithfulness."""
    if not triples or not chunks:
        return {
            "filtered_triples":  triples,
            "removed_count":     0,
            "kept_count":        len(triples),
            "faithfulness_rate": 1.0,
        }
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _POOL, _filter_triples_sync, triples, chunks,
    )


async def score_gap_evidence(
    gaps:   List[Dict],
    chunks: List[str],
) -> Dict:
    """Async — score gap supporting_evidence against source chunks."""
    if not gaps or not chunks:
        return {"scored_gaps": gaps, "low_confidence_count": 0}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _POOL, _score_gaps_sync, gaps, chunks,
    )


def hallucination_label(score: float) -> str:
    """Human-readable label for hallucination score."""
    if score < 0.1: return "✅ Highly Faithful"
    if score < 0.3: return "🟡 Mostly Faithful"
    if score < 0.5: return "🟠 Partially Hallucinated"
    return                  "🔴 Highly Hallucinated"