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
    return ChatGroq(model=llm_id, temperature=0.3)


# ── Prompts ───────────────────────────────────────────────────────────────────
_THEME_EXTRACTION_PROMPT = """You are a research literature analyst.
Identify the main themes across these research papers.

Papers:
{papers_summary}

Return ONLY valid JSON:
{{
  "themes": [
    "Theme 1: brief description",
    "Theme 2: brief description",
    "Theme 3: brief description"
  ]
}}

Return only JSON, no explanation."""


_REVIEW_PROMPT = """You are an expert academic writer specializing in literature reviews.
Write a comprehensive literature review based on the papers below.

Papers analyzed:
{papers_text}

Identified themes:
{themes}

Comparison results:
{comparison}

Write a structured literature review following this format:

1. **Introduction**: Briefly introduce the research area and scope
2. **Thematic Analysis**: Discuss each theme with relevant papers
3. **Comparative Analysis**: Highlight key differences and similarities
4. **Research Gaps**: Identify unexplored areas
5. **Future Directions**: Suggest promising research directions

Guidelines:
- Be academic and precise
- Cite papers by their filename/title
- 400-600 words total
- Connect ideas across papers

Return ONLY valid JSON:
{{
  "review_text": "Full literature review text here...",
  "research_gaps_summary": "2-3 sentence summary of key gaps identified",
  "future_directions": "2-3 sentence summary of future research directions",
  "overall_quality": <float 0-10, quality of the literature body reviewed>
}}

Return only JSON, no explanation."""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _extract_themes_node(state: LitReviewState) -> LitReviewState:
    """Extract common themes across all papers."""
    logger.info(f"[LIT REVIEW] extracting themes chat_id={state['chat_id']}")

    analyses = state["paper_analyses"]
    if not analyses:
        return {**state, "themes": [], "error": "no papers to review"}

    # build paper summaries for theme extraction
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
        raw = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in theme response")
        
        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x1f\x7f](?<![\n\t])', ' ', cleaned)
        result  = json.loads(cleaned)
        themes = result.get("themes", [])

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
        papers_text += f"""
Paper {i}: {a.filename or f'Paper {i}'}
Summary:          {(a.refined_summary or '')[:400]}
Models:           {', '.join(entities.get('models',   [])[:6])}
Datasets:         {', '.join(entities.get('datasets', [])[:6])}
Metrics:          {', '.join(entities.get('metrics',  [])[:6])}
Methods:          {', '.join(entities.get('methods',  [])[:6])}
Quality Score:    {a.quality_score or 0:.1f}/10
Research Gaps:    {'; '.join(gaps[:3])}
---"""

    # format comparison for prompt
    comp_text = ""
    if comparison:
        comp_text = f"Evolution trends: {comparison.get('evolution_trends', '')[:300]}\n"
        comp_text += f"Positioning: {comparison.get('positioning', '')[:300]}"

    prompt = _REVIEW_PROMPT.format(
        papers_text = papers_text[:4000],
        themes      = "\n".join([f"- {t}" for t in themes]),
        comparison  = comp_text or "No comparison data available.",
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in review response")
        
        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x1f\x7f](?<![\n\t])', ' ', cleaned)
        result  = json.loads(cleaned)

        review_text           = result.get("review_text", "")
        research_gaps_summary = result.get("research_gaps_summary", "")
        future_directions     = result.get("future_directions", "")
        overall_quality       = float(result.get("overall_quality", 7.0))

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
        logger.warning(f"[LIT REVIEW] generation failed attempt {state['retry_count']+1}: {e}")
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
        Generate literature review from all paper analyses.
        Returns: {themes, review_text, research_gaps_summary,
                  future_directions, overall_quality}
        """
        import asyncio
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