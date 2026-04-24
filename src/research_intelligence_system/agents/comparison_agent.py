"""
comparison_agent.py — LangGraph-based comparison agent
1 paper  → web-augmented comparison (arXiv + Tavily)
2+ papers → direct paper vs paper comparison
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class ComparisonState(TypedDict):
    chat_id:          str
    llm_id:           str
    paper_analyses:   List[Any]
    use_web:          bool
    web_papers:       List[Dict]
    comparison_table: Dict[str, Any]
    ranking:          List[str]
    evolution_trends: str
    positioning:      str
    web_papers_used:  List[Dict]
    retry_count:      int
    error:            str


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0.2)


# ── Title cleaner ─────────────────────────────────────────────────────────────
def _clean_title(filename: str) -> str:
    """Remove MD5 hash prefix and clean filename into readable title."""
    name = filename or "Uploaded Paper"
    # remove 32-char hex hash prefix + underscore
    name = re.sub(r'^[a-f0-9]{32}_', '', name)
    # remove .pdf extension
    name = re.sub(r'\.pdf$', '', name, flags=re.IGNORECASE)
    # replace underscores with spaces
    name = name.replace("_", " ").strip()
    return name or "Uploaded Paper"


# ── Prompts ───────────────────────────────────────────────────────────────────
_WEB_COMPARISON_PROMPT = """You are a research paper comparison expert.
Compare the uploaded paper against related papers found on the web.

Uploaded Paper:
Title:    {title}
Summary:  {summary}
Models:   {models}
Datasets: {datasets}
Metrics:  {metrics}
Methods:  {methods}

Related Papers Found Online:
{web_papers}

Return ONLY valid JSON:
{{
  "comparison_table": {{
    "headers": ["Paper", "Model", "Dataset", "Key Metric", "Score", "Year"],
    "rows": [
      ["Uploaded Paper", "model_name", "dataset_name", "metric_name", "score", "year"],
      ["Related Paper 1", "...", "...", "...", "...", "..."]
    ]
  }},
  "ranking": ["Paper names ranked best to worst by performance"],
  "evolution_trends": "How this research area has evolved over time",
  "positioning": "Where the uploaded paper sits in the research landscape",
  "web_papers_used": ["list of paper titles used in comparison"]
}}

Return only JSON, no explanation."""


_DIRECT_COMPARISON_PROMPT = """You are a research paper comparison expert.
Compare these {n} papers against each other.

Papers:
{papers}

Return ONLY valid JSON:
{{
  "comparison_table": {{
    "headers": ["Paper", "Model", "Dataset", "Key Metric", "Score", "Approach"],
    "rows": [
      ["Paper 1 title", "model", "dataset", "metric", "score", "approach"],
      ["Paper 2 title", "...", "...", "...", "...", "..."]
    ]
  }},
  "ranking": ["Paper titles ranked best to worst"],
  "evolution_trends": "How approaches evolved across these papers",
  "positioning": "Which paper makes the most significant contribution and why"
}}

Return only JSON, no explanation."""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _fetch_web_papers_node(state: ComparisonState) -> ComparisonState:
    if not state["use_web"]:
        return state

    logger.info("[COMPARISON] fetching web papers for single-paper comparison")
    analyses = state["paper_analyses"]
    if not analyses:
        return {**state, "error": "no paper analyses found"}

    paper    = analyses[0]
    entities = paper.entities or {}
    title    = _clean_title(paper.filename or "")

    web_papers = []

    # ── arXiv via search_by_entities (consistent query building) ──────────
    try:
        import asyncio
        from src.research_intelligence_system.tools.arxiv_service import ArxivService

        loop = asyncio.new_event_loop()
        arxiv_papers = loop.run_until_complete(
            ArxivService().search_by_entities(
                models   = entities.get("models",   []),
                datasets = entities.get("datasets", []),
                methods  = entities.get("methods",  []),
                tasks    = entities.get("tasks",    []),
                title    = title,
                max_results = 5,
            )
        )
        loop.close()
        web_papers.extend(arxiv_papers)
        logger.info(f"[COMPARISON] arXiv: {len(arxiv_papers)} papers")
    except Exception as e:
        logger.warning(f"[COMPARISON] arXiv failed: {e}")

    # ── Tavily web search ──────────────────────────────────────────────────
    try:
        from src.research_intelligence_system.tools.web_search import sync_web_search
        # use clean title + top 2 models only
        models_str   = " ".join(entities.get("models", [])[:2])
        tavily_query = f"research papers similar to {title} {models_str}".strip()
        web_text     = sync_web_search(tavily_query)
        if web_text:
            web_papers.append({
                "title":    "Web Search Results",
                "abstract": web_text[:800],
                "source":   "tavily",
            })
        logger.info(f"[WEB SEARCH] query={tavily_query!r:.60}")
    except Exception as e:
        logger.warning(f"[COMPARISON] Tavily failed: {e}")

    return {**state, "web_papers": web_papers, "error": ""}


def _compare_node(state: ComparisonState) -> ComparisonState:
    """Run comparison — web-augmented or direct."""
    logger.info(f"[COMPARISON] running {'web' if state['use_web'] else 'direct'} comparison")

    analyses = state["paper_analyses"]
    if not analyses:
        return {**state, "error": "no papers to compare",
                "retry_count": state["retry_count"] + 1}

    try:
        llm = _get_llm(state["llm_id"])

        if state["use_web"]:
            paper    = analyses[0]
            entities = paper.entities or {}
            title    = _clean_title(paper.filename or "")

            web_text = "\n".join([
                f"- {p.get('title', 'Unknown')} ({p.get('year', '')}): {p.get('abstract', '')[:300]}"
                for p in state.get("web_papers", [])
            ]) or "No related papers found online."

            prompt = _WEB_COMPARISON_PROMPT.format(
                title     = title,
                summary   = (paper.refined_summary or "")[:1000],
                models    = ", ".join(entities.get("models",   [])[:8]),
                datasets  = ", ".join(entities.get("datasets", [])[:8]),
                metrics   = ", ".join(entities.get("metrics",  [])[:8]),
                methods   = ", ".join(entities.get("methods",  [])[:8]),
                web_papers= web_text,
            )
        else:
            papers_text = ""
            for i, paper in enumerate(analyses, 1):
                entities = paper.entities or {}
                title    = _clean_title(paper.filename or f"Paper {i}")
                papers_text += f"""
Paper {i}: {title}
Summary:  {(paper.refined_summary or '')[:500]}
Models:   {', '.join(entities.get('models',   [])[:5])}
Datasets: {', '.join(entities.get('datasets', [])[:5])}
Metrics:  {', '.join(entities.get('metrics',  [])[:5])}
Methods:  {', '.join(entities.get('methods',  [])[:5])}
---"""

            prompt = _DIRECT_COMPARISON_PROMPT.format(
                n      = len(analyses),
                papers = papers_text,
            )

        response = llm.invoke(prompt)
        raw      = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in comparison response")

        # clean control characters before parsing
        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
        result  = json.loads(cleaned)

        logger.info(f"[COMPARISON] done — {len(result.get('ranking', []))} papers ranked")

        return {
            **state,
            "comparison_table": result.get("comparison_table", {}),
            "ranking":          result.get("ranking", []),
            "evolution_trends": result.get("evolution_trends", ""),
            "positioning":      result.get("positioning", ""),
            "web_papers_used":  state.get("web_papers", []),
            "error":            "",
        }

    except Exception as e:
        logger.warning(f"[COMPARISON] failed attempt {state['retry_count']+1}: {e}")
        return {
            **state,
            "error":       str(e),
            "retry_count": state["retry_count"] + 1,
        }


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry(state: ComparisonState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "compare"
    return END


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph():
    graph = StateGraph(ComparisonState)

    graph.add_node("fetch_web", _fetch_web_papers_node)
    graph.add_node("compare",   _compare_node)

    graph.set_entry_point("fetch_web")
    graph.add_edge("fetch_web", "compare")
    graph.add_conditional_edges("compare", _should_retry, {
        "compare": "compare",
        END:       END,
    })

    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
class ComparisonAgent:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def compare(
        self,
        chat_id:        str,
        paper_analyses: List[Any],
        use_web:        bool = False,
    ) -> Dict[str, Any]:
        import asyncio
        loop = asyncio.get_running_loop()

        initial_state: ComparisonState = {
            "chat_id":          chat_id,
            "llm_id":           self.llm_id,
            "paper_analyses":   paper_analyses,
            "use_web":          use_web,
            "web_papers":       [],
            "comparison_table": {},
            "ranking":          [],
            "evolution_trends": "",
            "positioning":      "",
            "web_papers_used":  [],
            "retry_count":      0,
            "error":            "",
        }

        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        return {
            "comparison_table": result.get("comparison_table", {}),
            "ranking":          result.get("ranking", []),
            "evolution_trends": result.get("evolution_trends", ""),
            "positioning":      result.get("positioning", ""),
            "web_papers_used":  result.get("web_papers_used", []),
        }