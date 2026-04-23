"""
literature_review_agent.py — LangGraph-based literature review generator
Generates thematic literature review from all paper analyses + comparison results
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from src.research_intelligence_system.core.groq_limiter import wait_for_groq
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
    return ChatGroq(model=llm_id, temperature=0.3, max_tokens=2000)


# ── Text cleaner ──────────────────────────────────────────────────────────────
def _clean_text(text: str) -> str:
    """Fix hyphenated words split mid-word and clean whitespace."""
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _parse_json_safe(raw: str) -> Dict:
    """Extract and parse JSON from LLM response safely."""
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in response")
    cleaned = json_match.group()
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
    cleaned = re.sub(r'\n', ' ', cleaned)
    cleaned = re.sub(r'\r', ' ', cleaned)
    return json.loads(cleaned)


# ── Prompts ───────────────────────────────────────────────────────────────────
_THEME_EXTRACTION_PROMPT = """You are a research literature analyst.
Identify the main research themes across these papers.

Papers:
{papers_summary}

Return ONLY valid JSON:
{{
  "themes": [
    "Theme 1: brief description of first distinct research dimension",
    "Theme 2: brief description of second distinct research dimension",
    "Theme 3: brief description of third distinct research dimension",
    "Theme 4: brief description of fourth distinct research dimension"
  ]
}}

Rules:
- Generate 4-5 themes
- Each theme must be a DISTINCT research dimension — no overlap
- Themes should reflect actual content from the papers
- Return only JSON, no explanation"""


_REVIEW_PROMPT = """You are an academic writer generating a structured literature review.

Papers analyzed:
{papers_text}

Research themes: {themes}

Comparison context: {comparison}

Write a structured literature review with EXACTLY 4 paragraphs separated by [PARA].
Each paragraph must be 3+ sentences and 80+ words.

Paragraph 1 - Introduction and Background:
  Introduce the research area, its importance, and historical context.
  Mention specific model/method names from the papers.

Paragraph 2 - Thematic Analysis:
  Analyze the main themes across the papers.
  Compare how different papers approach the same problems.
  Reference specific methods, datasets, and results.

Paragraph 3 - Comparative Analysis:
  Compare the contributions, strengths, and limitations of each paper.
  Discuss how they relate to each other.
  Include specific performance numbers or findings if available.

Paragraph 4 - Future Directions:
  Propose 4-5 SPECIFIC future research directions with concrete methodology.
  These must be SOLUTIONS not restatements of problems.
  Each direction must suggest a different research avenue.
  Do NOT copy text from the Research Gaps field.

CRITICAL RULES:
- Return ONLY a JSON object
- Use [PARA] to separate paragraphs inside review_text — NOT actual newlines
- NO markdown, NO code blocks, NO preamble
- future_directions field: 4-5 complete sentences proposing specific research solutions
- research_gaps_summary: exactly 3 sentences summarizing key gaps
- Minimum 300 words total in review_text

Return ONLY this JSON structure:
{{"review_text": "Paragraph 1 text [PARA] Paragraph 2 text [PARA] Paragraph 3 text [PARA] Paragraph 4 text", "research_gaps_summary": "Gap sentence 1. Gap sentence 2. Gap sentence 3.", "future_directions": "Direction 1 sentence. Direction 2 sentence. Direction 3 sentence. Direction 4 sentence. Direction 5 sentence.", "overall_quality": 7.5}}"""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _extract_themes_node(state: LitReviewState) -> LitReviewState:
    """Extract common themes across all papers."""
    logger.info(f"[LIT REVIEW] extracting themes chat_id={state['chat_id']}")

    analyses = state["paper_analyses"]
    if not analyses:
        return {**state, "themes": [], "error": "no papers to review"}

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
        result = _parse_json_safe(response.content.strip())
        themes = result.get("themes", [])
        # clean hyphenation in themes
        themes = [_clean_text(t) for t in themes if isinstance(t, str)]

        logger.info(f"[LIT REVIEW] extracted {len(themes)} themes")
        return {**state, "themes": themes, "error": ""}

    except Exception as e:
        logger.warning(f"[LIT REVIEW] theme extraction failed: {e}")
        # fallback — use method names as themes
        all_methods = []
        for a in analyses:
            all_methods.extend((a.entities or {}).get("methods", [])[:2])
        themes = list(set(all_methods))[:5]
        return {**state, "themes": themes, "error": ""}


def _generate_review_node(state: LitReviewState) -> LitReviewState:
    """Generate full literature review text."""
    logger.info(f"[LIT REVIEW] generating review attempt={state['retry_count']+1}")

    analyses   = state["paper_analyses"]
    themes     = state.get("themes", [])
    comparison = state.get("comparison", {})

    # build detailed paper descriptions
    papers_text = ""
    for i, a in enumerate(analyses, 1):
        entities = a.entities or {}
        gaps     = a.research_gaps or []

        gaps_text = "; ".join([
            g["gap"] if isinstance(g, dict) else g
            for g in gaps[:3]
        ])

        papers_text += (
            f"\nPaper {i}: {a.filename or f'Paper {i}'}\n"
            f"Summary:       {(a.refined_summary or '')[:400]}\n"
            f"Models:        {', '.join(entities.get('models',   [])[:6])}\n"
            f"Datasets:      {', '.join(entities.get('datasets', [])[:6])}\n"
            f"Metrics:       {', '.join(entities.get('metrics',  [])[:6])}\n"
            f"Methods:       {', '.join(entities.get('methods',  [])[:6])}\n"
            f"Quality Score: {a.quality_score or 0:.1f}/10\n"
            f"Key Gaps:      {gaps_text}\n"
            f"---"
        )

    comp_text = ""
    if comparison:
        comp_text = (
            f"Evolution trends: {comparison.get('evolution_trends', '')[:300]}\n"
            f"Positioning: {comparison.get('positioning', '')[:300]}"
        )

    themes_text = "\n".join([
        f"- {t['gap'] if isinstance(t, dict) else t}"
        for t in themes
    ])

    prompt = _REVIEW_PROMPT.format(
        papers_text = papers_text[:4000],
        themes      = themes_text or "No themes extracted.",
        comparison  = comp_text or "No comparison data available.",
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()

        result = _parse_json_safe(raw)

        review_text           = result.get("review_text", "")
        research_gaps_summary = result.get("research_gaps_summary", "")
        future_directions     = result.get("future_directions", "")
        overall_quality       = float(result.get("overall_quality", 7.0))

        # convert [PARA] markers to actual paragraph breaks
        review_text = review_text.replace("[PARA]", "\n\n")

        # clean hyphenation throughout
        review_text           = _clean_text(review_text)
        research_gaps_summary = _clean_text(research_gaps_summary)
        future_directions     = _clean_text(future_directions)

        # enforce minimum length — retry if too short
        if len(review_text) < 800 and state["retry_count"] < 2:
            logger.warning(
                f"[LIT REVIEW] review too short ({len(review_text)} chars) — retrying"
            )
            return {
                **state,
                "error":       "review too short",
                "retry_count": state["retry_count"] + 1,
            }

        logger.info(
            f"[LIT REVIEW] generated {len(review_text)} chars "
            f"quality={overall_quality}"
        )

        return {
            **state,
            "review_text":           review_text,
            "research_gaps_summary": research_gaps_summary,
            "future_directions":     future_directions,
            "overall_quality":       overall_quality,
            "error":                 "",
        }

    except Exception as e:
        logger.warning(
            f"[LIT REVIEW] generation failed attempt {state['retry_count']+1}: {e}"
        )
        return {
            **state,
            "error":       str(e),
            "retry_count": state["retry_count"] + 1,
        }


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry(state: LitReviewState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "generate_review"
    return END


# ── Graph ─────────────────────────────────────────────────────────────────────
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
        Generate structured literature review from all paper analyses.
        Returns: {themes, review_text, research_gaps_summary,
                  future_directions, overall_quality}
        """
        import asyncio

        await wait_for_groq(self.llm_id, "lit_review_generate")

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
            "themes":                result.get("themes", []),
            "review_text":           result.get("review_text", ""),
            "research_gaps_summary": result.get("research_gaps_summary", ""),
            "future_directions":     result.get("future_directions", ""),
            "overall_quality":       result.get("overall_quality", 0.0),
        }