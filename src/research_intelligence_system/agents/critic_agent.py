"""
critic_agent.py — LangGraph-based critic/self-reflection agent (MIRROR framework)
Evaluates the comprehensive summary from two-stage summarizer.
Loops back to refine if quality score < 7.0 (max 2 retries).
Step 4: Hallucination feedback loop — if hallucination_score > 0.3,
        feeds hallucinated sentences back into refine node for targeted correction.

── THROTTLING ────────────────────────────────────────────────────────────────
Each LangGraph node calls sync_wait_for_groq() BEFORE llm.invoke() and
notify_groq_complete() AFTER. This registers every LLM call (up to 4-5 per
critic pipeline) individually in the token bucket — preventing the ~3000 token
under-count that caused cascade 429s on all subsequent stages.

The public evaluate() method does NOT call wait_for_groq — the nodes handle it.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.research_intelligence_system.core.groq_limiter import (
    sync_wait_for_groq,
    notify_groq_complete,
)
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class CriticState(TypedDict):
    paper_id:               str
    llm_id:                 str
    summaries:              Dict[str, str]
    entities:               Dict[str, Any]
    chunks:                 List[str]
    refined_summary:        str
    quality_score:          float
    missing_entities:       List[str]
    inconsistencies:        List[str]
    hallucination_score:    float
    hallucinated_sentences: List[str]
    critic_attempts:        int
    hallucination_checked:  bool
    error:                  str
    _feedback:              str


_QUALITY_THRESHOLD       = 7.0
_HALLUCINATION_THRESHOLD = 0.3
_MAX_ATTEMPTS            = 2


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0)


def _get_summary_to_evaluate(summaries: Dict[str, str]) -> str:
    if summaries.get("comprehensive"):
        return summaries["comprehensive"]
    if summaries.get("overall"):
        return summaries["overall"]
    return " ".join([
        summaries.get("abstract",    ""),
        summaries.get("methodology", ""),
        summaries.get("results",     ""),
        summaries.get("conclusion",  ""),
    ]).strip()


# ── Prompts ───────────────────────────────────────────────────────────────────
_CRITIC_PROMPT = """You are a scientific paper review critic.
Evaluate the comprehensive summary below against the extracted entities.

Comprehensive Summary to evaluate:
{summary}

Extracted entities from the paper:
Models:   {models}
Datasets: {datasets}
Metrics:  {metrics}
Methods:  {methods}

Evaluate and return ONLY valid JSON:
{{
  "quality_score": <float 0-10>,
  "is_complete": <true/false>,
  "missing_entities": ["entities mentioned in paper but missing from summary"],
  "inconsistencies": ["any factual inconsistencies found"],
  "feedback": "specific actionable feedback for improvement"
}}

Scoring guide:
9-10: Problem + method + results with numbers + contributions all covered
7-8:  Most content covered, minor gaps or missing metric values
5-6:  Major gaps — missing methodology or results section
0-4:  Significant content missing or factual errors

Return only JSON, no explanation."""


_REFINE_PROMPT = """You are an expert scientific paper summarizer.
The summary below needs improvement based on critic feedback.

Current summary:
{summary}

Critic feedback:
{feedback}

Missing entities that must be included:
{missing_entities}

Write an improved comprehensive 400-500 word summary that:
- Includes ALL missing entities listed above
- Fixes ALL inconsistencies mentioned
- Covers: Problem Statement, Proposed Approach, Key Results (with numbers),
  Main Contributions, Limitations
- Is written in academic style
- Mentions model names, dataset names, and metric values explicitly

Return ONLY the improved summary text, no explanation, no headers."""


_HALLUCINATION_REFINE_PROMPT = """You are an expert scientific paper summarizer.
The summary below contains sentences that are NOT supported by the source paper.
You must rewrite these sentences to be grounded in the actual paper content.

Current summary:
{summary}

Sentences that are NOT supported by the source paper (must be fixed or removed):
{hallucinated_sentences}

Source paper context (use this to ground your rewrite):
{source_context}

Rewrite the complete summary (400-500 words) ensuring:
- Every claim is directly supported by the source paper
- The hallucinated sentences above are either removed or rewritten with evidence
- All other content is preserved
- Academic writing style maintained

Return ONLY the rewritten summary text, no explanation, no headers."""


# ── Hallucination check (sync, runs inside critic pipeline) ──────────────────
def _check_hallucination_sync(
    summary: str,
    chunks:  List[str],
) -> Dict:
    """Run NLI-based hallucination check synchronously within critic pipeline."""
    if not chunks or not summary:
        return {"hallucination_score": 0.0, "hallucinated_sentences": []}

    try:
        # always use singleton from hallucination_detector — never create new CrossEncoder
        from src.research_intelligence_system.agents.hallucination_detector import get_cross_encoder
        model = get_cross_encoder()

        if model is None:
            return {"hallucination_score": 0.0, "hallucinated_sentences": []}

        sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 20][:15]

        hallucinated = []
        for sentence in sentences:
            pairs  = [(sentence, chunk[:512]) for chunk in chunks[:8]]
            scores = model.predict(pairs)
            if float(scores.max()) <= 0.0:
                hallucinated.append(sentence)

        total = len(sentences)
        rate  = len(hallucinated) / total if total > 0 else 0.0

        return {
            "hallucination_score":    round(rate, 4),
            "hallucinated_sentences": hallucinated[:5],
        }

    except Exception as e:
        logger.warning(f"[CRITIC] hallucination check failed: {e}")
        return {"hallucination_score": 0.0, "hallucinated_sentences": []}


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _critic_node(state: CriticState) -> CriticState:
    """Evaluate comprehensive summary quality and identify gaps."""
    logger.info(f"[CRITIC] paper_id={state['paper_id']} attempt={state['critic_attempts']+1}")

    summary = _get_summary_to_evaluate(state["summaries"])
    if not summary or len(summary) < 50:
        return {
            **state,
            "error":           "no summary to evaluate",
            "critic_attempts": state["critic_attempts"] + 1,
        }

    entities = state.get("entities", {})
    prompt   = _CRITIC_PROMPT.format(
        summary  = summary[:3000],
        models   = ", ".join(entities.get("models",   [])[:10]),
        datasets = ", ".join(entities.get("datasets", [])[:10]),
        metrics  = ", ".join(entities.get("metrics",  [])[:10]),
        methods  = ", ".join(entities.get("methods",  [])[:10]),
    )

    # ── throttle before LLM call ──────────────────────────────────────────────
    sync_wait_for_groq(state["llm_id"], "critic")
    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        notify_groq_complete()   # ← update _last_call to NOW

        raw = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in critic response")

        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', cleaned)

        try:
            evaluation = json.loads(cleaned)
        except json.JSONDecodeError:
            quality = re.search(r'"quality_score"\s*:\s*([\d.]+)', cleaned)
            evaluation = {
                "quality_score":    float(quality.group(1)) if quality else 5.0,
                "missing_entities": [],
                "inconsistencies":  [],
                "feedback":         "Parse error — accepted as-is",
            }

        quality_score    = float(evaluation.get("quality_score", 5.0))
        missing_entities = evaluation.get("missing_entities", [])
        inconsistencies  = evaluation.get("inconsistencies", [])
        feedback         = evaluation.get("feedback", "")

        logger.info(f"[CRITIC] quality_score={quality_score} missing={len(missing_entities)}")

        return {
            **state,
            "quality_score":    quality_score,
            "missing_entities": missing_entities,
            "inconsistencies":  inconsistencies,
            "refined_summary":  summary,
            "_feedback":        feedback,
            "error":            "",
        }

    except Exception as e:
        notify_groq_complete()   # always notify even on failure
        logger.warning(f"[CRITIC] evaluation failed: {e}")
        return {
            **state,
            "error":           str(e),
            "quality_score":   5.0,
            "refined_summary": _get_summary_to_evaluate(state["summaries"]),
            "critic_attempts": state["critic_attempts"] + 1,
        }


def _refine_node(state: CriticState) -> CriticState:
    """Refine summary based on critic quality feedback."""
    logger.info(f"[CRITIC] refining summary paper_id={state['paper_id']}")

    current_summary = (
        state.get("refined_summary")
        or _get_summary_to_evaluate(state["summaries"])
    )

    prompt = _REFINE_PROMPT.format(
        summary          = current_summary[:3000],
        feedback         = state.get("_feedback", "Improve completeness and include specific numbers"),
        missing_entities = ", ".join(state.get("missing_entities", [])[:15]),
    )

    # ── throttle before LLM call ──────────────────────────────────────────────
    sync_wait_for_groq(state["llm_id"], "critic_refine")
    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        notify_groq_complete()   # ← update _last_call to NOW

        refined = response.content.strip()
        logger.info(f"[CRITIC] refined summary generated ({len(refined)} chars)")
        return {
            **state,
            "refined_summary": refined,
            "critic_attempts": state["critic_attempts"] + 1,
            "error":           "",
        }

    except Exception as e:
        notify_groq_complete()   # always notify even on failure
        logger.warning(f"[CRITIC] refinement failed: {e}")
        return {
            **state,
            "error":           str(e),
            "critic_attempts": state["critic_attempts"] + 1,
        }


def _hallucination_check_node(state: CriticState) -> CriticState:
    """
    Step 4: Hallucination feedback loop.
    Uses singleton cross-encoder — no LLM call here, no throttling needed.
    """
    if state.get("hallucination_checked"):
        return state

    summary = state.get("refined_summary") or _get_summary_to_evaluate(state["summaries"])
    chunks  = state.get("chunks", [])

    logger.info(f"[CRITIC] hallucination check paper_id={state['paper_id']}")
    result = _check_hallucination_sync(summary, chunks)

    hall_score = result["hallucination_score"]
    hall_sents = result["hallucinated_sentences"]

    logger.info(
        f"[CRITIC] hallucination_score={hall_score:.2f} "
        f"hallucinated_sentences={len(hall_sents)}"
    )

    return {
        **state,
        "hallucination_score":    hall_score,
        "hallucinated_sentences": hall_sents,
        "hallucination_checked":  True,
        "error":                  "",
    }


def _hallucination_refine_node(state: CriticState) -> CriticState:
    """Targeted refinement for hallucinated sentences."""
    logger.info(f"[CRITIC] hallucination refinement paper_id={state['paper_id']}")

    current_summary    = state.get("refined_summary") or _get_summary_to_evaluate(state["summaries"])
    hallucinated_sents = state.get("hallucinated_sentences", [])
    chunks             = state.get("chunks", [])
    source_context     = " ".join(chunks[:5])[:2000]

    prompt = _HALLUCINATION_REFINE_PROMPT.format(
        summary                = current_summary[:3000],
        hallucinated_sentences = "\n".join([f"- {s}" for s in hallucinated_sents]),
        source_context         = source_context,
    )

    # ── throttle before LLM call ──────────────────────────────────────────────
    sync_wait_for_groq(state["llm_id"], "critic_refine")
    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        notify_groq_complete()   # ← update _last_call to NOW

        refined = response.content.strip()
        logger.info(f"[CRITIC] hallucination-refined summary ({len(refined)} chars)")
        return {
            **state,
            "refined_summary": refined,
            "error":           "",
        }

    except Exception as e:
        notify_groq_complete()   # always notify even on failure
        logger.warning(f"[CRITIC] hallucination refinement failed: {e}")
        return {**state, "error": str(e)}


def _accept_node(state: CriticState) -> CriticState:
    """Accept current summary as final."""
    logger.info(f"[CRITIC] accepted quality_score={state['quality_score']}")
    if not state.get("refined_summary"):
        return {
            **state,
            "refined_summary": _get_summary_to_evaluate(state["summaries"]),
        }
    return state


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_refine(state: CriticState) -> str:
    if state.get("error"):
        return "accept"

    score    = state.get("quality_score", 0.0)
    attempts = state.get("critic_attempts", 0)

    if score < _QUALITY_THRESHOLD and attempts < _MAX_ATTEMPTS:
        logger.info(f"[CRITIC] score={score} < threshold → refining")
        return "refine"

    logger.info(f"[CRITIC] score={score} → hallucination check")
    return "hallucination_check"


def _after_refine(state: CriticState) -> str:
    if state["critic_attempts"] < _MAX_ATTEMPTS:
        return "critic"
    return "hallucination_check"


def _should_hallucination_refine(state: CriticState) -> str:
    hall_score = state.get("hallucination_score", 0.0)
    hall_sents = state.get("hallucinated_sentences", [])

    if hall_score > _HALLUCINATION_THRESHOLD and hall_sents:
        logger.info(
            f"[CRITIC] hallucination_score={hall_score:.2f} > "
            f"threshold={_HALLUCINATION_THRESHOLD} → targeted refinement"
        )
        return "hallucination_refine"

    return "accept"


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph() -> Any:
    graph = StateGraph(CriticState)

    graph.add_node("critic",               _critic_node)
    graph.add_node("refine",               _refine_node)
    graph.add_node("hallucination_check",  _hallucination_check_node)
    graph.add_node("hallucination_refine", _hallucination_refine_node)
    graph.add_node("accept",               _accept_node)

    graph.set_entry_point("critic")

    graph.add_conditional_edges("critic", _should_refine, {
        "refine":              "refine",
        "hallucination_check": "hallucination_check",
        "accept":              "accept",
    })
    graph.add_conditional_edges("refine", _after_refine, {
        "critic":              "critic",
        "hallucination_check": "hallucination_check",
    })
    graph.add_conditional_edges("hallucination_check", _should_hallucination_refine, {
        "hallucination_refine": "hallucination_refine",
        "accept":               "accept",
    })
    graph.add_edge("hallucination_refine", "accept")
    graph.add_edge("accept", END)

    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
class CriticAgent:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def evaluate(
        self,
        paper_id:  str,
        summaries: Dict[str, str],
        entities:  Dict[str, Any],
        chunks:    Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate and refine the comprehensive summary.

        NOTE: wait_for_groq is NOT called here. Each LangGraph node calls
        sync_wait_for_groq() and notify_groq_complete() individually, so every
        LLM call (up to 4-5 per critic run) is correctly registered in the
        token bucket. The orchestrator must NOT call notify_groq_complete()
        after this method — the nodes handle all throttling internally.
        """
        import asyncio

        loop = asyncio.get_running_loop()

        initial_state: CriticState = {
            "paper_id":               paper_id,
            "llm_id":                 self.llm_id,
            "summaries":              summaries,
            "entities":               entities,
            "chunks":                 chunks or [],
            "refined_summary":        "",
            "quality_score":          0.0,
            "missing_entities":       [],
            "inconsistencies":        [],
            "hallucination_score":    0.0,
            "hallucinated_sentences": [],
            "critic_attempts":        0,
            "hallucination_checked":  False,
            "error":                  "",
            "_feedback":              "",
        }

        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        return {
            "refined_summary":        result.get("refined_summary", ""),
            "quality_score":          result.get("quality_score", 0.0),
            "missing_entities":       result.get("missing_entities", []),
            "inconsistencies":        result.get("inconsistencies", []),
            "hallucination_score":    result.get("hallucination_score", 0.0),
            "hallucinated_sentences": result.get("hallucinated_sentences", []),
            "critic_validated":       result.get("quality_score", 0.0) >= _QUALITY_THRESHOLD,
        }