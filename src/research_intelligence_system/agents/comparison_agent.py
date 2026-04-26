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
_WEB_COMPARISON_PROMPT = """[SYSTEM] You are a scientific literature analyst. \
Your task is to compare an uploaded research paper against related papers retrieved from the web. \
You produce structured JSON only. Every claim in your output must be traceable to the provided text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UPLOADED PAPER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title   : {title}
Summary : {summary}
Models  : {models}
Datasets: {datasets}
Metrics : {metrics}
Methods : {methods}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RELATED PAPERS (retrieved from arXiv / web)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{web_papers}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# In _WEB_COMPARISON_PROMPT, inside the comparison_table instruction block, add:

PHANTOM ROW RULE: 
  • Every row must correspond to a named research paper.
    Do not create rows for datasets, benchmarks, or metric values that are
    not themselves paper titles. If the uploaded paper reports a result
    (e.g. "85.43% Top-1 on ImageNet"), that value belongs in the uploaded
    paper's row — not in a separate row named after the dataset.
comparison_table:
  • One row per paper (uploaded paper first, then related papers).
  • Model: use the primary model/algorithm name. Use "N/A" if not stated.
  • Dataset: use the primary evaluation dataset. Use "N/A" if not stated.
  • Key Metric: use the metric name that best characterises the paper's main result.
  • Score: copy the EXACT value from the source text. Use "N/A" if not explicitly stated — do NOT invent or estimate.
  • Year: use publication year. Use "N/A" if unknown.
  • PHANTOM ROW RULE: every row must correspond to a named research paper.
    Do not create rows for datasets, benchmarks, or metric values that are
    not themselves paper titles. Results from the uploaded paper belong in
    the uploaded paper's row only — never in a separate row.
  • HALLUCINATION RULE: every cell value must appear verbatim in the provided text above.
    If you cannot find it, write "N/A".

ranking:
  • Order papers from strongest to weakest contribution.
  • If the primary metric is the same across papers, rank by that metric score descending.
  • If metrics differ, rank by recency (newest first).
  • If only one paper has a score, it ranks first; others follow by year.
  • State your ranking criterion in a brief prefix, e.g. "Ranked by Top-1 Accuracy on ImageNet:"

evolution_trends:
  • 2–3 sentences. Describe how the methods or results in these papers represent a progression
    or shift in the research area. Ground claims in the paper titles/abstracts provided.

positioning:
  • 2–3 sentences. Explain where the uploaded paper sits relative to the retrieved work:
    ahead, behind, orthogonal, or complementary. Be specific about WHY.

web_papers_used:
  • List only papers that contributed at least one cell value in the comparison_table.
  • Format: {{"title": "paper title", "url": "url or arxiv_id if available"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — JSON ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "comparison_table": {{
    "headers": ["Paper", "Model", "Dataset", "Key Metric", "Score", "Year"],
    "rows": [
      ["{title}", "model_name", "dataset_name", "metric_name", "score_or_N/A", "year_or_N/A"],
      ["Related Paper Title", "...", "...", "...", "...", "..."]
    ]
  }},
  "ranking": "Ranked by <criterion>: 1. Paper A, 2. Paper B, ...",
  "evolution_trends": "2-3 sentence trend analysis grounded in the provided abstracts.",
  "positioning": "2-3 sentence positioning of the uploaded paper relative to retrieved work.",
  "web_papers_used": [
    {{"title": "paper title", "url": "url or N/A"}}
  ]
}}"""


_DIRECT_COMPARISON_PROMPT = """[SYSTEM] You are a scientific literature analyst. \
Your task is to compare {n} uploaded research papers directly against each other. \
You produce structured JSON only. Every claim must be traceable to the provided paper summaries below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAPERS TO COMPARE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{papers}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
comparison_table:
  • One row per paper, in the order provided above.
  • Model: primary model or algorithm name. "N/A" if absent.
  • Dataset: primary evaluation dataset. "N/A" if absent.
  • Key Metric: the metric that best characterises each paper's main claim.
    If papers use different metrics, choose the most comparable one; note discrepancy in positioning.
  • Score: EXACT value from the provided summary text. "N/A" if not explicitly stated. Do NOT estimate.
  • Key Innovation: one clause naming what is technically new (e.g. "dynamic sparse attention",
    "graph-based gap detection", "cross-encoder reranking"). Not the topic — the mechanism.
  • HALLUCINATION RULE: if a value is not in the provided text, write "N/A".

ranking:
  • If all papers share the same primary metric → rank by that score descending.
  • If metrics differ → rank by recency (newest first), note "metrics incomparable".
  • Format: "Ranked by <criterion>: 1. <Paper Title>, 2. <Paper Title>, ..."

evolution_trends:
  • 2–3 sentences. Identify the methodological direction these papers collectively represent.
    Is the field moving toward efficiency? Scale? Interpretability? Ground this in the summaries.

positioning:
  • 2–3 sentences. Identify which paper makes the most novel or impactful contribution and explain
    the specific technical reason. If papers are in different sub-areas, say so explicitly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — JSON ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "comparison_table": {{
    "headers": ["Paper", "Model", "Dataset", "Key Metric", "Score", "Key Innovation"],
    "rows": [
      ["Paper 1 Title", "model", "dataset", "metric", "score_or_N/A", "one-clause innovation"],
      ["Paper 2 Title", "...", "...", "...", "...", "..."]
    ]
  }},
  "ranking": "Ranked by <criterion>: 1. Paper Title, 2. Paper Title, ...",
  "evolution_trends": "2-3 sentence methodological trend analysis.",
  "positioning": "2-3 sentence assessment of the most significant contribution and why."
}}"""


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