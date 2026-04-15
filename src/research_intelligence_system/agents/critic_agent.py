"""
critic_agent.py — LangGraph-based critic/self-reflection agent (MIRROR framework)
Now evaluates the comprehensive summary from two-stage summarizer.
Loops back to refine if quality score < 7.0 (max 2 retries)
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class CriticState(TypedDict):
    paper_id:         str
    llm_id:           str
    summaries:        Dict[str, str]
    entities:         Dict[str, Any]
    refined_summary:  str
    quality_score:    float
    missing_entities: List[str]
    inconsistencies:  List[str]
    critic_attempts:  int
    error:            str
    _feedback:        str


_QUALITY_THRESHOLD = 7.0
_MAX_ATTEMPTS      = 2


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0)


def _get_summary_to_evaluate(summaries: Dict[str, str]) -> str:
    """
    Prefer the comprehensive summary from two-stage summarizer.
    Falls back to combining section summaries if not available.
    """
    # two-stage summarizer produces 'comprehensive' key
    if summaries.get("comprehensive"):
        return summaries["comprehensive"]
    if summaries.get("overall"):
        return summaries["overall"]
    # fallback — combine section summaries
    return " ".join([
        summaries.get("abstract", ""),
        summaries.get("methodology", ""),
        summaries.get("results", ""),
        summaries.get("conclusion", ""),
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
9-10: Problem + method + results with numbers + contributions all covered, no inconsistencies
7-8:  Most content covered, minor gaps or missing metric values
5-6:  Major gaps — missing methodology or results section
0-4:  Significant content missing or factual errors

Return only JSON, no explanation."""


_REFINE_PROMPT = """You are an expert scientific paper summarizer.
The summary below was evaluated by a critic and needs improvement.

Current summary:
{summary}

Critic feedback:
{feedback}

Missing entities that must be included:
{missing_entities}

Write an improved comprehensive 400-500 word summary that:
- Includes ALL missing entities listed above
- Fixes ALL inconsistencies mentioned
- Covers: Problem Statement, Proposed Approach, Key Results (with specific numbers), Main Contributions, Limitations
- Is written in academic style
- Mentions model names, dataset names, and metric values explicitly

Return ONLY the improved summary text, no explanation, no headers."""


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

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in critic response")
        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', cleaned)
        try:
            evaluation = json.loads(cleaned)
        except json.JSONDecodeError:
            # try extracting individual fields
            quality = re.search(r'"quality_score"\s*:\s*([\d.]+)', cleaned)
            evaluation = {
                "quality_score":    float(quality.group(1)) if quality else 5.0,
                "missing_entities": [],
                "inconsistencies":  [],
                "feedback":         "Parse error — accepted as-is",
            }

        evaluation       = json.loads(json_match.group())
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
            "refined_summary":  summary,   # will be refined if score < threshold
            "_feedback":        feedback,
            "error":            "",
        }

    except Exception as e:
        logger.warning(f"[CRITIC] evaluation failed: {e}")
        summary = _get_summary_to_evaluate(state["summaries"])
        return {
            **state,
            "error":           str(e),
            "quality_score":   5.0,
            "refined_summary": summary,
            "critic_attempts": state["critic_attempts"] + 1,
        }


def _refine_node(state: CriticState) -> CriticState:
    """Refine the comprehensive summary based on critic feedback."""
    logger.info(f"[CRITIC] refining summary paper_id={state['paper_id']}")

    # use current refined_summary or fall back to comprehensive
    current_summary = state.get("refined_summary") or _get_summary_to_evaluate(state["summaries"])

    prompt = _REFINE_PROMPT.format(
        summary          = current_summary[:3000],
        feedback         = state.get("_feedback", "Improve completeness and include specific numbers"),
        missing_entities = ", ".join(state.get("missing_entities", [])[:15]),
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        refined  = response.content.strip()

        logger.info(f"[CRITIC] refined summary generated ({len(refined)} chars)")
        return {
            **state,
            "refined_summary": refined,
            "critic_attempts": state["critic_attempts"] + 1,
            "error":           "",
        }

    except Exception as e:
        logger.warning(f"[CRITIC] refinement failed: {e}")
        return {
            **state,
            "error":           str(e),
            "critic_attempts": state["critic_attempts"] + 1,
        }


def _accept_node(state: CriticState) -> CriticState:
    """Accept current summary as final."""
    logger.info(f"[CRITIC] accepted quality_score={state['quality_score']}")
    if not state.get("refined_summary"):
        return {**state, "refined_summary": _get_summary_to_evaluate(state["summaries"])}
    return state


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_refine(state: CriticState) -> str:
    if state.get("error"):
        return "accept"

    score    = state.get("quality_score", 0.0)
    attempts = state.get("critic_attempts", 0)

    if score < _QUALITY_THRESHOLD and attempts < _MAX_ATTEMPTS:
        logger.info(f"[CRITIC] score={score} < threshold={_QUALITY_THRESHOLD} → refining")
        return "refine"

    logger.info(f"[CRITIC] score={score} → accepting")
    return "accept"


def _after_refine(state: CriticState) -> str:
    if state["critic_attempts"] < _MAX_ATTEMPTS:
        return "critic"
    return "accept"


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph() -> Any:
    graph = StateGraph(CriticState)

    graph.add_node("critic", _critic_node)
    graph.add_node("refine", _refine_node)
    graph.add_node("accept", _accept_node)

    graph.set_entry_point("critic")

    graph.add_conditional_edges("critic", _should_refine, {
        "refine": "refine",
        "accept": "accept",
    })
    graph.add_conditional_edges("refine", _after_refine, {
        "critic": "critic",
        "accept": "accept",
    })
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
    ) -> Dict[str, Any]:
        """
        Evaluate and optionally refine the comprehensive summary.
        Input summaries should include 'comprehensive' key from two-stage summarizer.
        Returns: refined_summary, quality_score, missing_entities, inconsistencies, critic_validated
        """
        import asyncio
        loop = asyncio.get_running_loop()

        initial_state: CriticState = {
            "paper_id":         paper_id,
            "llm_id":           self.llm_id,
            "summaries":        summaries,
            "entities":         entities,
            "refined_summary":  "",
            "quality_score":    0.0,
            "missing_entities": [],
            "inconsistencies":  [],
            "critic_attempts":  0,
            "error":            "",
            "_feedback":        "",
        }

        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        return {
            "refined_summary":  result.get("refined_summary", ""),
            "quality_score":    result.get("quality_score", 0.0),
            "missing_entities": result.get("missing_entities", []),
            "inconsistencies":  result.get("inconsistencies", []),
            "critic_validated": result.get("quality_score", 0.0) >= _QUALITY_THRESHOLD,
        }