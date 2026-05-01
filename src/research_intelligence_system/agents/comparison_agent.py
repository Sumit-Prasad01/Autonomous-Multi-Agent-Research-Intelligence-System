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


# Broad patterns that identify benchmark score sentences without needing an entity list.
# These fire even when extraction finds metrics=[].
_SCORE_PATTERNS = re.compile(
    r'\b\d+\.?\d*\s*(?:BLEU|bleu)\b'
    r'|\bBLEU\s+(?:score\s+)?(?:of\s+)?\d+\.?\d*'
    r'|\b\d+\.?\d*\s*(?:top-1|top-5|mAP|IoU|WER|CER|ROUGE|BERTScore|perplexity)\b'
    r'|\bscore\s+of\s+\d+\.?\d*'
    r'|\b\d+\.?\d*\s*(?:%|percent)\s+(?:accuracy|precision|recall|F1)'
    r'|\bF1\s+(?:score\s+)?(?:of\s+)?\d+\.?\d*'
    r'|\baccuracy\s+(?:of\s+)?\d+\.?\d*',
    re.IGNORECASE,
)


def _extract_key_results(
    summary:  str,
    metrics:  List[str],
    datasets: List[str],
    triples:  Optional[List[Dict]] = None,
) -> str:
    """
    Extract key numeric results from multiple sources, in priority order.

    Layer 1 — Broad pattern scan (no entity list needed):
      Finds sentences containing known benchmark score patterns: BLEU, F1, accuracy,
      top-1, etc. Works even when metrics=[] from extraction.

    Layer 2 — Entity-aware scan (existing logic):
      Finds sentences containing both a number AND a known metric/dataset keyword.
      Supplements Layer 1 when entity lists are populated.

    Layer 3 — ACHIEVES triples fallback:
      When Layers 1+2 find nothing, mines ACHIEVES triples from the knowledge graph.
      These were independently extracted from the paper text and contain
      model→result relationships even when the summary doesn't have clean sentences.

    Returns a pipe-separated string of up to 4 result statements, or "" if none found.
    """
    result_sentences: List[str] = []
    seen_lower: set = set()

    def _add(s: str) -> None:
        s = s.strip()
        key = s.lower()[:80]
        if s and key not in seen_lower and len(result_sentences) < 4:
            seen_lower.add(key)
            result_sentences.append(s)

    if summary:
        sentences = re.split(r'(?<=[.!?])\s+', summary.strip())

        # Layer 1: broad score pattern scan
        for sent in sentences:
            if _SCORE_PATTERNS.search(sent) and len(sent) < 300:
                _add(sent)

        # Layer 2: entity-aware scan
        known_terms: set = set()
        for m in metrics[:5]:
            known_terms.update(m.lower().split())
        for d in datasets[:5]:
            known_terms.update(d.lower().split())
        _sw = {"on", "the", "a", "an", "of", "for", "in", "with", "and", "or", "to"}
        known_terms -= _sw

        if known_terms:
            for sent in sentences:
                has_number = bool(re.search(r'\b\d+\.?\d*\b', sent))
                has_term   = any(kw in sent.lower() for kw in known_terms if len(kw) >= 3)
                if has_number and has_term and len(sent) < 300:
                    _add(sent)

    # Layer 3: ACHIEVES triples fallback (fires when Layers 1+2 find nothing)
    if not result_sentences and triples:
        achieves = [t for t in triples if t.get("relation") == "ACHIEVES"]
        for t in achieves[:4]:
            subj = t.get("subject", "")
            obj  = t.get("object", "")
            if subj and obj:
                _add(f"{subj} achieves {obj}")

    if not result_sentences:
        return ""

    return " | ".join(result_sentences)


def _extract_year(entities: Dict, summary: str) -> str:
    """
    Extract the paper's publication year from entities or summary text.

    Stage 1: entities["year"] if non-empty and looks like a valid year.
    Stage 2: regex scan of summary for citation-context year patterns.
             Explicitly avoids false positives from dataset year tokens
             like "WMT 2014" or "ImageNet 2012".

    Returns the year as a string, or "N/A" if not found.
    """
    # Stage 1: trust extraction when it populated the year field
    year_from_entities = str(entities.get("year", "") or "").strip()
    if re.match(r'^(?:19|20)\d{2}$', year_from_entities):
        return year_from_entities

    if not summary:
        return "N/A"

    # Build dataset-year exclusion set to avoid "WMT 2014" → "2014" false positives
    dataset_years: set = set()
    for d in entities.get("datasets", []):
        m = re.search(r'\b(?:19|20)\d{2}\b', d)
        if m:
            dataset_years.add(m.group())

    # Citation-context patterns ordered by specificity
    citation_patterns = [
        r'\((?:[^)]*?)(\b(?:19|20)\d{2}\b)(?:[^)]*?)\)',    # (Vaswani et al., 2017)
        r'et\s+al[.,][,\s]+(\b(?:19|20)\d{2}\b)',            # et al., 2017
        r'et\s+al[.,]\s+(\b(?:19|20)\d{2}\b)',               # et al. 2017
        r'published\s+(?:in\s+)?(\b(?:19|20)\d{2}\b)',       # published in 2017
        r'introduced\s+in\s+(\b(?:19|20)\d{2}\b)',           # introduced in 2017
        r'proposed\s+(?:in\s+)?(\b(?:19|20)\d{2}\b)',        # proposed in 2017
        r'(?:^|[,.\s])in\s+(\b(?:20\d{2}|19[89]\d)\b)(?:[,.\s]|$)',  # "in 2017,"
    ]

    for pattern in citation_patterns:
        for m in re.finditer(pattern, summary, re.IGNORECASE):
            year = m.group(1)
            if year not in dataset_years:
                return year

    return "N/A"


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
  • One row per paper: uploaded paper FIRST, then ALL retrieved related papers.
  • INCLUSION RULE: include all retrieved papers that are in the same broad
    research area as the uploaded paper — even if they are not directly
    on the same benchmark. Only exclude papers that are completely unrelated
    (e.g. a medical imaging paper when the uploaded paper is about NLP).
    Papers in adjacent areas (e.g. LSTM variants when the uploaded paper is
    about Transformers) should be included, not skipped.
  • Paper column: exact paper title.
  • Model: primary model or algorithm name. "N/A" if not stated.
  • Dataset: primary evaluation dataset. "N/A" if not stated.
  • Key Metric: metric name that best characterises the paper's main result.
  • Score: EXACT numeric value from the provided text.
    For the uploaded paper: look in Key Results above first, then the Summary.
    For retrieved papers: look in their abstracts.
    Use "N/A" ONLY if no numeric score appears anywhere in the provided text.
    Do NOT estimate, round, or invent values.
  • Year: for the uploaded paper use Year={year}. For retrieved papers use
    their publication year from the abstracts. "N/A" only if genuinely unknown.
  • PHANTOM ROW RULE: only create rows for named research papers.

ranking:
  • Include only papers in the comparison_table.
  • If papers share the same primary metric → rank by score descending.
  • If metrics differ → rank by recency (newest first).
  • Prefix: "Ranked by <criterion>: 1. Paper, 2. Paper, ..."

evolution_trends:
  • 2–3 sentences on the methodological progression these papers represent.
  • Cite specific paper titles and years from the provided text.

positioning:
  • 2–3 sentences. Where does the uploaded paper sit relative to the retrieved
    works — ahead, behind, orthogonal, or complementary? State the specific
    technical reason.

web_papers_used:
  • All papers that appear as rows in comparison_table (except the uploaded paper).
  • Format: {{"title": "paper title", "url": "url or N/A"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — JSON ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "comparison_table": {{
    "headers": ["Paper", "Model", "Dataset", "Key Metric", "Score", "Year"],
    "rows": [
      ["{title}", "primary_model", "primary_dataset", "primary_metric", "score_from_key_results", "{year}"],
      ["Related Paper Title", "...", "...", "...", "...", "..."]
    ]
  }},
  "ranking": "Ranked by <criterion>: 1. Paper A, 2. Paper B, ...",
  "evolution_trends": "2-3 sentence trend analysis citing specific papers and years.",
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
  • Key Innovation: one clause describing the specific technical mechanism.
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
        paper           = analyses[0]
        entities        = paper.entities or {}
        title           = _clean_title(paper.filename or "")
        refined_summary = paper.refined_summary or ""

        # Pull paper triples for the ACHIEVES fallback in _extract_key_results
        triples = list(getattr(paper, "triples", None) or [])

        # Year: entities["year"] → then regex scan summary
        year = _extract_year(entities, refined_summary)

        # Key results: three-layer extraction
        key_results = _extract_key_results(
            summary  = refined_summary,
            metrics  = entities.get("metrics",  []),
            datasets = entities.get("datasets", []),
            triples  = triples,
        )

        if not key_results:
            key_results = (
                "Numeric scores not found in summary. "
                "Look for benchmark results in the Summary field above."
            )

        web_text = "\n".join([
            f"- {p.get('title', 'Unknown')} ({p.get('year', 'N/A')}): "
            f"{p.get('abstract', '')[:300]}"
            for p in state.get("web_papers", [])
        ]) or "No related papers retrieved."

        prompt = _WEB_COMPARISON_PROMPT.format(
            title       = title,
            year        = year,
            summary     = refined_summary[:1500],
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
            entities        = paper.entities or {}
            title           = _clean_title(paper.filename or f"Paper {i}")
            refined_summary = paper.refined_summary or ""
            triples         = list(getattr(paper, "triples", None) or [])

            year        = _extract_year(entities, refined_summary)
            key_results = _extract_key_results(
                summary  = refined_summary,
                metrics  = entities.get("metrics",  []),
                datasets = entities.get("datasets", []),
                triples  = triples,
            )

            papers_text += (
                f"\nPaper {i}: {title} ({year})\n"
                f"Summary:     {refined_summary[:600]}\n"
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
        raw = re.sub(r'```(?:json)?\s*', '', raw)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()

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
        The orchestrator must NOT call notify_groq_complete() after this method.
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