"""
comparison_agent.py — LangGraph-based comparison agent
1 paper  → web-augmented comparison (arXiv + Tavily)
2+ papers → direct paper vs paper comparison

── QUALITY DESIGN ────────────────────────────────────────────────────────────
Three problems caused poor comparison table quality:

1. Uploaded paper row showed Score=N/A, Year=N/A:
   The prompt only received summary[:1000] and entity lists. If the summary
   didn't explicitly say "41.0 BLEU" and the year wasn't in those fields,
   the LLM had no data to fill those cells and correctly wrote N/A.
   Fix: explicitly pass year, key_results, and a longer summary (1500 chars).
   key_results is built from the refined_summary by extracting numeric patterns.

2. Retrieved papers were irrelevant (5 dropout papers for a Transformer paper):
   Caused by arXiv query containing "dropout" — fixed in arxiv_service.py.
   But as a second layer of defence, the prompt now instructs the LLM to skip
   rows for papers that are clearly from a different research area.

3. Tavily query had duplicate terms ("BERT BERT Transformer"):
   Already fixed — title_words deduplication is preserved here.
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
class ComparisonState(TypedDict):
    chat_id:          str
    llm_id:           str
    paper_analyses:   List[Any]
    use_web:          bool
    web_papers:       List[Dict]
    comparison_table: Dict[str, Any]
    ranking:          str
    evolution_trends: str
    positioning:      str
    web_papers_used:  List[Dict]
    retry_count:      int
    error:            str


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0.2)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _clean_title(filename: str) -> str:
    name = filename or "Uploaded Paper"
    name = re.sub(r'^[a-f0-9]{32}_', '', name)
    name = re.sub(r'\.pdf$', '', name, flags=re.IGNORECASE)
    name = name.replace("_", " ").strip()
    return name or "Uploaded Paper"


def _extract_key_results(summary: str, metrics: List[str], datasets: List[str]) -> str:
    """
    Pull numeric result sentences from the summary to give the LLM explicit
    score data for filling the Score column.

    Strategy: find sentences that contain both a number and a metric/dataset name.
    Returns a concise string like "41.0 BLEU on WMT 2014 EN-FR; 28.4 BLEU on EN-DE"
    """
    if not summary:
        return ""

    # Build a set of known metric/dataset keywords for matching
    known_terms = set()
    for m in metrics[:5]:
        known_terms.update(m.lower().split())
    for d in datasets[:5]:
        known_terms.update(d.lower().split())
    # remove stopwords from the match set
    _sw = {"on", "the", "a", "an", "of", "for", "in", "with", "and", "or", "to"}
    known_terms -= _sw

    # Split summary into sentences
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())

    # A sentence is a "result sentence" if it has a number AND a known term
    result_sentences = []
    for sent in sentences:
        has_number = bool(re.search(r'\b\d+\.?\d*\b', sent))
        sent_lower = sent.lower()
        has_metric = any(kw in sent_lower for kw in known_terms if len(kw) >= 3)
        if has_number and has_metric and len(sent) < 300:
            result_sentences.append(sent.strip())

    if not result_sentences:
        return ""

    # Return up to 4 result sentences, joined with semicolons for compact display
    return " | ".join(result_sentences[:4])


# ── Prompts ───────────────────────────────────────────────────────────────────
_WEB_COMPARISON_PROMPT = """\
[SYSTEM] You are a scientific literature analyst producing a structured comparison \
of a research paper against related works. Output is JSON only. \
Every cell value must be traceable to the provided text — do not invent or estimate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UPLOADED PAPER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title      : {title}
Year       : {year}
Summary    : {summary}
Key Results: {key_results}
Models     : {models}
Datasets   : {datasets}
Metrics    : {metrics}
Methods    : {methods}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RELATED PAPERS (retrieved from arXiv / web)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{web_papers}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
comparison_table:
  • One row per paper: uploaded paper FIRST, then only RELEVANT related papers.
  • RELEVANCE FILTER: skip any retrieved paper that is clearly from a different
    research area than the uploaded paper. For example, if the uploaded paper is
    about Transformer architecture for machine translation, skip papers that are
    purely about dropout regularization, image segmentation, or reinforcement
    learning — they are not meaningful comparisons.
  • Paper column: exact paper title.
  • Model: primary model or algorithm name. "N/A" if not stated in the provided text.
  • Dataset: primary evaluation dataset. "N/A" if not stated.
  • Key Metric: metric name that best characterises the paper's main result.
  • Score: EXACT numeric value from the provided text.
    For the uploaded paper: look in Key Results above first, then the Summary.
    For retrieved papers: look in their abstracts below.
    Use "N/A" ONLY if no numeric score appears anywhere in the provided text.
    Do NOT estimate, round, or invent values.
  • Year: for the uploaded paper use Year={year}. For retrieved papers use
    their publication year. "N/A" only if genuinely unknown.
  • PHANTOM ROW RULE: only create rows for named research papers. No rows
    for datasets, metrics, or benchmark names.

ranking:
  • Include only papers that appear in the comparison_table.
  • If papers share the same metric → rank by score descending.
  • If metrics differ → rank by recency (newest first).
  • Prefix: "Ranked by <criterion>: 1. Paper, 2. Paper, ..."

evolution_trends:
  • 2–3 sentences on the methodological progression these papers represent.
  • Cite specific paper titles and years.

positioning:
  • 2–3 sentences. Where does the uploaded paper sit relative to the relevant
    retrieved works — ahead, behind, orthogonal, or complementary? WHY specifically?

web_papers_used:
  • Only papers that appear as rows in comparison_table.
  • Format: {{"title": "paper title", "url": "url or N/A"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — JSON ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "comparison_table": {{
    "headers": ["Paper", "Model", "Dataset", "Key Metric", "Score", "Year"],
    "rows": [
      ["{title}", "primary_model", "primary_dataset", "primary_metric", "score_from_key_results", "{year}"],
      ["Relevant Related Paper", "...", "...", "...", "...", "..."]
    ]
  }},
  "ranking": "Ranked by <criterion>: 1. Paper A, 2. Paper B, ...",
  "evolution_trends": "2-3 sentence trend analysis citing specific papers.",
  "positioning": "2-3 sentence positioning with specific technical reason.",
  "web_papers_used": [
    {{"title": "paper title", "url": "url or N/A"}}
  ]
}}\
"""


_DIRECT_COMPARISON_PROMPT = """\
[SYSTEM] You are a scientific literature analyst producing a direct comparison \
of {n} research papers. Output is JSON only. \
Every cell value must be traceable to the provided paper summaries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAPERS TO COMPARE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{papers}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
comparison_table:
  • One row per paper, in the order provided.
  • Model: primary model or algorithm name. "N/A" if absent from the summary.
  • Dataset: primary evaluation dataset. "N/A" if absent.
  • Key Metric: metric that best characterises the paper's main claim.
  • Score: EXACT numeric value from the summary. "N/A" only if not present.
  • Key Innovation: one clause describing the specific technical mechanism
    (e.g. "scaled dot-product self-attention without recurrence").
    Not the topic — the mechanism. "N/A" if not determinable.
  • HALLUCINATION RULE: if a value is not in the provided summaries, write "N/A".

ranking:
  • Same primary metric across papers → rank by score descending.
  • Different metrics → rank by recency, note "metrics incomparable".
  • Format: "Ranked by <criterion>: 1. Paper, 2. Paper, ..."

evolution_trends:
  • 2–3 sentences on the methodological direction these papers collectively represent.

positioning:
  • 2–3 sentences. Which paper makes the most novel or impactful contribution?
    Give a specific technical reason, not a general statement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — JSON ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "comparison_table": {{
    "headers": ["Paper", "Model", "Dataset", "Key Metric", "Score", "Key Innovation"],
    "rows": [
      ["Paper 1 Title", "model", "dataset", "metric", "score_or_N/A", "one-clause mechanism"],
      ["Paper 2 Title", "...", "...", "...", "...", "..."]
    ]
  }},
  "ranking": "Ranked by <criterion>: 1. Paper Title, 2. Paper Title, ...",
  "evolution_trends": "2-3 sentence methodological trend analysis.",
  "positioning": "2-3 sentence assessment with specific technical reason."
}}\
"""


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

    # ── arXiv via search_by_entities ─────────────────────────────────────────
    try:
        import asyncio
        from src.research_intelligence_system.tools.arxiv_service import ArxivService

        loop = asyncio.new_event_loop()
        arxiv_papers = loop.run_until_complete(
            ArxivService().search_by_entities(
                models      = entities.get("models",   []),
                datasets    = entities.get("datasets", []),
                methods     = entities.get("methods",  []),
                tasks       = entities.get("tasks",    []),
                title       = title,
                max_results = 5,
            )
        )
        loop.close()
        web_papers.extend(arxiv_papers)
        logger.info(f"[COMPARISON] arXiv: {len(arxiv_papers)} papers")
    except Exception as e:
        logger.warning(f"[COMPARISON] arXiv failed: {e}")

    # ── Tavily web search ─────────────────────────────────────────────────────
    try:
        from src.research_intelligence_system.tools.web_search import sync_web_search

        # Deduplicate: skip model names already in the title
        title_words   = set(title.lower().split())
        unique_models = [
            m for m in entities.get("models", [])[:2]
            if m.lower() not in title_words and m.lower() not in title.lower()
        ]
        models_str   = " ".join(unique_models[:2])
        tavily_query = f"research papers similar to {title} {models_str}".strip()

        web_text = sync_web_search(tavily_query)
        if web_text:
            web_papers.append({
                "title":    "Web Search Results",
                "abstract": web_text[:800],
                "source":   "tavily",
            })
        logger.info(f"[WEB SEARCH] query={tavily_query!r:.80}")
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

    if state["use_web"]:
        paper    = analyses[0]
        entities = paper.entities or {}
        title    = _clean_title(paper.filename or "")

        # ── Extract year ──────────────────────────────────────────────────────
        year = str(entities.get("year", "") or "").strip() or "N/A"

        # ── Extract key numeric results from the refined summary ──────────────
        refined_summary = paper.refined_summary or ""
        key_results = _extract_key_results(
            summary  = refined_summary,
            metrics  = entities.get("metrics",  []),
            datasets = entities.get("datasets", []),
        )
        if not key_results:
            key_results = "See summary above — extract numeric scores from Key Metric + Dataset mentions."

        # ── Build web papers text ─────────────────────────────────────────────
        web_text = "\n".join([
            f"- {p.get('title', 'Unknown')} ({p.get('year', 'N/A')}): {p.get('abstract', '')[:300]}"
            for p in state.get("web_papers", [])
        ]) or "No related papers retrieved."

        prompt = _WEB_COMPARISON_PROMPT.format(
            title       = title,
            year        = year,
            summary     = refined_summary[:1500],   # ↑ was 1000
            key_results = key_results,
            models      = ", ".join(entities.get("models",   [])[:8]),
            datasets    = ", ".join(entities.get("datasets", [])[:8]),
            metrics     = ", ".join(entities.get("metrics",  [])[:8]),
            methods     = ", ".join(entities.get("methods",  [])[:8]),
            web_papers  = web_text,
        )

    else:
        papers_text = ""
        for i, paper in enumerate(analyses, 1):
            entities = paper.entities or {}
            title    = _clean_title(paper.filename or f"Paper {i}")
            year     = str(entities.get("year", "") or "").strip() or "N/A"

            key_results = _extract_key_results(
                summary  = paper.refined_summary or "",
                metrics  = entities.get("metrics",  []),
                datasets = entities.get("datasets", []),
            )

            papers_text += (
                f"\nPaper {i}: {title} ({year})\n"
                f"Summary:     {(paper.refined_summary or '')[:600]}\n"
                f"Key Results: {key_results or 'not explicitly stated'}\n"
                f"Models:      {', '.join(entities.get('models',   [])[:5])}\n"
                f"Datasets:    {', '.join(entities.get('datasets', [])[:5])}\n"
                f"Metrics:     {', '.join(entities.get('metrics',  [])[:5])}\n"
                f"Methods:     {', '.join(entities.get('methods',  [])[:5])}\n---"
            )

        prompt = _DIRECT_COMPARISON_PROMPT.format(
            n      = len(analyses),
            papers = papers_text,
        )

    # ── throttle before LLM call ──────────────────────────────────────────────
    sync_wait_for_groq(state["llm_id"], "comparison")
    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        notify_groq_complete()

        raw = response.content.strip()

        # Strip markdown fences
        raw = re.sub(r'```(?:json)?\s*', '', raw)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()

        # Try direct parse first
        result = None
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            pass

        if result is None:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON in comparison response")
            cleaned = json_match.group()
            cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
            result  = json.loads(cleaned)

        ranking_str  = result.get("ranking", "")
        papers_count = ranking_str.count(",") + 1 if ranking_str else 0
        logger.info(f"[COMPARISON] done ✅ {papers_count} papers ranked")

        return {
            **state,
            "comparison_table": result.get("comparison_table", {}),
            "ranking":          result.get("ranking", ""),
            "evolution_trends": result.get("evolution_trends", ""),
            "positioning":      result.get("positioning", ""),
            "web_papers_used":  result.get("web_papers_used", state.get("web_papers", [])),
            "error":            "",
        }

    except Exception as e:
        notify_groq_complete()
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
        """
        _compare_node is self-throttling (sync_wait_for_groq + notify_groq_complete).
        The orchestrator must NOT call notify_groq_complete() after this.
        """
        import asyncio
        loop = asyncio.get_running_loop()

        initial_state: ComparisonState = {
            "chat_id":          chat_id,
            "llm_id":           self.llm_id,
            "paper_analyses":   paper_analyses,
            "use_web":          use_web,
            "web_papers":       [],
            "comparison_table": {},
            "ranking":          "",
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
            "ranking":          result.get("ranking",          ""),
            "evolution_trends": result.get("evolution_trends", ""),
            "positioning":      result.get("positioning",      ""),
            "web_papers_used":  result.get("web_papers_used",  []),
        }