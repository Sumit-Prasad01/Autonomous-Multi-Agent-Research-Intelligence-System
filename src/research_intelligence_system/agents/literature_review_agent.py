"""
literature_review_agent.py — LangGraph-based literature review generator
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from src.research_intelligence_system.core.groq_limiter import (
    sync_wait_for_groq,
    notify_groq_complete,
)
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class LitReviewState(TypedDict):
    chat_id:               str
    llm_id:                str
    paper_analyses:        List[Any]
    comparison:            Dict[str, Any]
    themes:                List[str]
    review_text:           str
    research_gaps_summary: str
    future_directions:     str
    overall_quality:       float
    retry_count:           int
    error:                 str


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0.3, max_tokens=2500)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _parse_json_safe(raw: str) -> Dict:
    """
    Robustly extract and parse JSON from an LLM response.
    Handles markdown fences, prose preamble, single-quote dicts.
    Does NOT collapse structural newlines.
    """
    raw = re.sub(r'```(?:json)?\s*', '', raw)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()

    # fast path
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in response")

    cleaned = json_match.group()
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
    cleaned = re.sub(r"(?<=[{,])\s*'([^']+)'\s*:", r' "\1":', cleaned)
    cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
    return json.loads(cleaned)


# ── Prompts ───────────────────────────────────────────────────────────────────

_THEME_EXTRACTION_PROMPT = """\
RESPOND WITH JSON ONLY. NO PREAMBLE. NO EXPLANATION. NO MARKDOWN.

You are a research literature analyst. Identify the main research themes \
across the papers listed below.

Papers:
{papers_summary}

Return ONLY this JSON structure — no other text:
{{
  "themes": [
    "Theme 1: brief description of first distinct research dimension",
    "Theme 2: brief description of second distinct research dimension",
    "Theme 3: brief description of third distinct research dimension",
    "Theme 4: brief description of fourth distinct research dimension"
  ]
}}

Rules:
- Generate exactly 4–5 themes.
- Each theme must be a DISTINCT research dimension with no overlap.
- Themes must reflect actual content from the papers above.
- First character of your response must be {{. Last character must be }}.\
"""


_REVIEW_PROMPT = """\
RESPOND WITH JSON ONLY. NO PREAMBLE. NO EXPLANATION. NO MARKDOWN.

You are an academic writer generating a structured literature review.

━━━ INPUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Papers analyzed:
{papers_text}

Research themes:
{themes}

Comparison context:
{comparison}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ OUTPUT RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

review_text:
  • Exactly 4 paragraphs separated by [PARA] — use [PARA] not actual line breaks.
  • Each paragraph: minimum 3 sentences, minimum 80 words.
  • Para 1 — Introduction and Background.
  • Para 2 — Thematic Analysis: compare how papers address the themes.
  • Para 3 — Comparative Analysis: strengths, limitations, specific results.
  • Para 4 — Future Directions: 4–5 specific, actionable research directions.
  • Minimum 300 words total.

research_gaps_summary: Exactly 3 complete sentences summarising key gaps.
future_directions: 4–5 complete sentences, each a specific research direction.
overall_quality: Float 1.0–10.0.

━━━ HALLUCINATION RULE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every specific claim must appear in the INPUT above.

━━━ STRICT OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
First character: {{   Last character: }}

{{
  "review_text": "Para 1 [PARA] Para 2 [PARA] Para 3 [PARA] Para 4",
  "research_gaps_summary": "Sentence 1. Sentence 2. Sentence 3.",
  "future_directions": "Direction 1. Direction 2. Direction 3. Direction 4. Direction 5.",
  "overall_quality": 7.5
}}\
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _extract_themes_node(state: LitReviewState) -> LitReviewState:
    logger.info(f"[LIT REVIEW] extracting themes chat_id={state['chat_id']}")

    analyses = state["paper_analyses"]
    if not analyses:
        return {**state, "themes": [], "error": "no papers to review"}

    # wait before LLM call
    sync_wait_for_groq(state["llm_id"], "lit_review_themes")

    papers_summary = "\n".join([
        f"Paper {i+1} ({a.filename or 'Unknown'}):\n"
        f"  Summary: {(a.refined_summary or '')[:300]}\n"
        f"  Methods: {', '.join((a.entities or {}).get('methods', [])[:5])}\n"
        f"  Tasks:   {', '.join((a.entities or {}).get('tasks',   [])[:5])}"
        for i, a in enumerate(analyses)
    ])

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(_THEME_EXTRACTION_PROMPT.format(
            papers_summary=papers_summary[:3000]
        ))
        notify_groq_complete()   # ← update _last_call to NOW (after LLM responded)

        result = _parse_json_safe(response.content.strip())
        themes = [_clean_text(t) for t in result.get("themes", []) if isinstance(t, str)]

        logger.info(f"[LIT REVIEW] extracted {len(themes)} themes")
        return {**state, "themes": themes, "error": ""}

    except Exception as e:
        notify_groq_complete()   # always update even on failure
        logger.warning(f"[LIT REVIEW] theme extraction failed: {e}")
        all_methods: List[str] = []
        for a in analyses:
            all_methods.extend((a.entities or {}).get("methods", [])[:2])
        themes = list(dict.fromkeys(all_methods))[:5]
        return {**state, "themes": themes, "error": ""}


def _generate_review_node(state: LitReviewState) -> LitReviewState:
    logger.info(f"[LIT REVIEW] generating review attempt={state['retry_count']+1}")

    # wait before LLM call — enforces gap since themes call completed
    sync_wait_for_groq(state["llm_id"], "lit_review_generate")

    analyses   = state["paper_analyses"]
    themes     = state.get("themes", [])
    comparison = state.get("comparison", {})

    papers_text = ""
    for i, a in enumerate(analyses, 1):
        entities  = a.entities or {}
        gaps      = a.research_gaps or []
        gaps_text = "; ".join([
            g["gap"] if isinstance(g, dict) else str(g) for g in gaps[:3]
        ])
        papers_text += (
            f"\nPaper {i}: {a.filename or f'Paper {i}'}\n"
            f"Summary:       {(a.refined_summary or '')[:400]}\n"
            f"Models:        {', '.join(entities.get('models',   [])[:6])}\n"
            f"Datasets:      {', '.join(entities.get('datasets', [])[:6])}\n"
            f"Metrics:       {', '.join(entities.get('metrics',  [])[:6])}\n"
            f"Methods:       {', '.join(entities.get('methods',  [])[:6])}\n"
            f"Quality Score: {a.quality_score or 0:.1f}/10\n"
            f"Key Gaps:      {gaps_text}\n---"
        )

    comp_text = ""
    if comparison:
        comp_text = (
            f"Evolution trends: {comparison.get('evolution_trends', '')[:300]}\n"
            f"Positioning: {comparison.get('positioning', '')[:300]}"
        )

    themes_text = "\n".join([
        f"- {t['gap'] if isinstance(t, dict) else t}" for t in themes
    ])

    prompt = _REVIEW_PROMPT.format(
        papers_text = papers_text[:4000],
        themes      = themes_text or "No themes extracted.",
        comparison  = comp_text   or "No comparison data available.",
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        notify_groq_complete()   # ← update _last_call immediately after LLM responds

        raw    = response.content.strip()
        result = _parse_json_safe(raw)

        review_text           = result.get("review_text",           "")
        research_gaps_summary = result.get("research_gaps_summary", "")
        future_directions     = result.get("future_directions",     "")
        overall_quality       = float(result.get("overall_quality", 7.0))

        review_text = review_text.replace("[PARA]", "\n\n")
        review_text           = _clean_text(review_text)
        research_gaps_summary = _clean_text(research_gaps_summary)
        future_directions     = _clean_text(future_directions)

        if len(review_text) < 800 and state["retry_count"] < 2:
            logger.warning(f"[LIT REVIEW] review too short ({len(review_text)} chars) — retrying")
            return {**state, "error": "review too short", "retry_count": state["retry_count"] + 1}

        logger.info(f"[LIT REVIEW] generated {len(review_text)} chars quality={overall_quality}")
        return {
            **state,
            "review_text":           review_text,
            "research_gaps_summary": research_gaps_summary,
            "future_directions":     future_directions,
            "overall_quality":       overall_quality,
            "error":                 "",
        }

    except Exception as e:
        notify_groq_complete()   # always update even on failure
        logger.warning(f"[LIT REVIEW] generation failed attempt {state['retry_count']+1}: {e}")
        return {**state, "error": str(e), "retry_count": state["retry_count"] + 1}


# ── Graph ─────────────────────────────────────────────────────────────────────

def _should_retry(state: LitReviewState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "generate_review"
    return END


def _build_graph():
    graph = StateGraph(LitReviewState)
    graph.add_node("extract_themes",  _extract_themes_node)
    graph.add_node("generate_review", _generate_review_node)
    graph.set_entry_point("extract_themes")
    graph.add_edge("extract_themes", "generate_review")
    graph.add_conditional_edges("generate_review", _should_retry, {
        "generate_review": "generate_review",
        END:               END,
    })
    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

class LiteratureReviewAgent:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def generate(
        self,
        chat_id:    str,
        analyses:   List[Any],
        comparison: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """
        NOTE: wait_for_groq is called INSIDE each node (sync_wait_for_groq),
        not here. notify_groq_complete() is called after each llm.invoke().
        Do NOT add wait_for_groq here — it would double-count the token budget.
        """
        loop = asyncio.get_running_loop()

        initial_state: LitReviewState = {
            "chat_id":               chat_id,
            "llm_id":                self.llm_id,
            "paper_analyses":        analyses,
            "comparison":            comparison,
            "themes":                [],
            "review_text":           "",
            "research_gaps_summary": "",
            "future_directions":     "",
            "overall_quality":       0.0,
            "retry_count":           0,
            "error":                 "",
        }

        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        return {
            "themes":                result.get("themes",                []),
            "review_text":           result.get("review_text",           ""),
            "research_gaps_summary": result.get("research_gaps_summary", ""),
            "future_directions":     result.get("future_directions",     ""),
            "overall_quality":       result.get("overall_quality",       0.0),
        }