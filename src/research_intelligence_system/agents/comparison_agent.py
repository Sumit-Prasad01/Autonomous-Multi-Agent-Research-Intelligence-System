"""
comparison_agent.py — LangGraph-based comparison agent
1 paper  → web-augmented comparison (arXiv + Tavily)
2+ papers → direct paper vs paper comparison
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.research_intelligence_system.core.groq_limiter import (
    notify_groq_complete,
    sync_wait_for_groq,
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


# ── Score extraction ──────────────────────────────────────────────────────────
# Compiled at module level (one-time cost).
# Fires on benchmark score sentences even when metrics=[] from extraction.
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
      Matches BLEU, F1, accuracy, top-1 etc. Works even when metrics=[].

    Layer 2 — Entity-aware scan:
      Sentences with both a number AND a known metric/dataset keyword.

    Layer 3 — ACHIEVES triples fallback:
      When Layers 1+2 find nothing, mines ACHIEVES triples from KG.

    Returns pipe-separated string of up to 4 result statements, or "".
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

        # Layer 1
        for sent in sentences:
            if _SCORE_PATTERNS.search(sent) and len(sent) < 300:
                _add(sent)

        # Layer 2
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

    # Layer 3
    if not result_sentences and triples:
        for t in [t for t in triples if t.get("relation") == "ACHIEVES"][:4]:
            subj, obj = t.get("subject", ""), t.get("object", "")
            if subj and obj:
                _add(f"{subj} achieves {obj}")

    return " | ".join(result_sentences)


def _extract_year(entities: Dict, summary: str) -> str:
    """
    Extract the paper's publication year from entities or summary text.

    Stage 1: entities["year"] if it's a valid 4-digit year.
    Stage 2: citation-context regex patterns (ordered by specificity),
             with 3 new patterns added for papers that don't use traditional
             citation format.
    Stage 3 (new): last-resort fallback — most frequent year in the summary
             that does not match a dataset year (e.g. "WMT 2014").

    Returns year string, or "N/A" if not found.
    """
    # Stage 1
    year_from_entities = str(entities.get("year", "") or "").strip()
    if re.match(r'^(?:19|20)\d{2}$', year_from_entities):
        return year_from_entities

    if not summary:
        return "N/A"

    # Build exclusion set from dataset names to avoid "WMT 2014" → "2014"
    dataset_years: set = set()
    for d in entities.get("datasets", []):
        m = re.search(r'\b(?:19|20)\d{2}\b', d)
        if m:
            dataset_years.add(m.group())

    # Stage 2: citation-context patterns, ordered by specificity
    citation_patterns = [
        r'\((?:[^)]*?)(\b(?:19|20)\d{2}\b)(?:[^)]*?)\)',       # (Vaswani et al., 2017)
        r'et\s+al[.,][,\s]+(\b(?:19|20)\d{2}\b)',               # et al., 2017
        r'et\s+al[.,]\s+(\b(?:19|20)\d{2}\b)',                  # et al. 2017
        r'published\s+(?:in\s+)?(\b(?:19|20)\d{2}\b)',          # published in 2017
        r'introduced\s+in\s+(\b(?:19|20)\d{2}\b)',              # introduced in 2017
        r'proposed\s+(?:in\s+)?(\b(?:19|20)\d{2}\b)',           # proposed in 2017
        r'(?:^|[,.\s])in\s+(\b(?:20\d{2}|19[89]\d)\b)(?:[,.\s]|$)',  # "in 2017,"
        # New patterns for papers that don't use traditional citation style
        r'arXiv:\d{4}\.\d{4,5}[v\d]*\s*\((\b20\d{2}\b)\)',    # arXiv:2402.17764 (2024)
        r'\b(20\d{2})\b[,\s]+(?:Microsoft|Google|Meta|Apple|OpenAI|DeepMind|Anthropic)',
        r'(?:paper|work|model|system)\s+(?:from|of)\s+(\b20\d{2}\b)',
    ]

    for pattern in citation_patterns:
        for m in re.finditer(pattern, summary, re.IGNORECASE):
            year = m.group(1)
            if year not in dataset_years:
                return year

    # Stage 3: last-resort — most frequent year in summary not matching dataset year
    all_years = re.findall(r'\b(20\d{2}|19\d{2})\b', summary)
    valid = [y for y in all_years if y not in dataset_years]
    if valid:
        return Counter(valid).most_common(1)[0][0]

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
RELATED PAPERS (from arXiv — create one table row per paper below)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{web_papers}
{tavily_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
comparison_table:
  • One row per paper: uploaded paper FIRST, then related papers from arXiv above.
  • INCLUSION RULE: include all arXiv papers in the same broad research area.
    EXCEPTION 1 — skip papers that are PURELY a dataset or benchmark corpus
      with no model proposed (titles like "A corpus of X sentences for Y task"
      or "WinoWhat: A Parallel Corpus of..." have no Model/Score to compare).
    EXCEPTION 2 — the ADDITIONAL WEB CONTEXT block is background only.
      Do NOT create a table row for it under any circumstances.
  • Paper column: exact paper title as shown in RELATED PAPERS.
  • Model: primary model or algorithm name. "N/A" if not stated.
  • Dataset: primary evaluation dataset. "N/A" if not stated in the abstract.
  • Key Metric: metric name that best characterises the paper's main result.
  • Score: EXACT numeric value from the provided text.
    For the uploaded paper: look in Key Results above first, then the Summary.
    For retrieved papers: look only in their abstracts below.
    Use "N/A" ONLY if no numeric score appears in the provided text.
    Do NOT estimate, round, or invent values.
  • Year: for the uploaded paper use Year={year}. For retrieved papers use
    the year shown in parentheses after the title. "N/A" if genuinely unknown.
  • PHANTOM ROW RULE: only create rows for named research papers listed
    in RELATED PAPERS above. Never create rows for datasets, benchmarks,
    or the web context block.

ranking:
  • If papers share the same primary metric → rank by score descending.
  • If metrics differ → rank by recency (newest first).
  • Prefix: "Ranked by <criterion>: 1. Paper, 2. Paper, ..."

evolution_trends:
  • 2–3 sentences on the methodological progression these papers represent.
  • Cite specific paper titles and years.

positioning:
  • 2–3 sentences. Where does the uploaded paper sit relative to the retrieved
    works — ahead, behind, orthogonal, or complementary? State the specific
    technical reason.

web_papers_used:
  • All papers that appear as rows in comparison_table (except uploaded paper).
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
  • Model: primary model or algorithm name. "N/A" if absent.
  • Dataset: primary evaluation dataset. "N/A" if absent.
  • Key Metric: metric that best characterises the paper's main claim.
  • Score: EXACT numeric value from the summary. "N/A" only if not present.
  • Key Innovation: one clause describing the specific technical mechanism.
    Not the topic — the mechanism. "N/A" if not determinable.
  • HALLUCINATION RULE: if a value is not in the provided summaries, write "N/A".

ranking:
  • Same primary metric → rank by score descending.
  • Different metrics → rank by recency, note "metrics incomparable".
  • Format: "Ranked by <criterion>: 1. Paper, 2. Paper, ..."

evolution_trends:
  • 2–3 sentences on the methodological direction these papers represent.

positioning:
  • 2–3 sentences. Which paper makes the most novel contribution?
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
    web_papers: List[Dict] = []

    # ── arXiv via search_by_entities ─────────────────────────────────────────
    try:
        import asyncio
        from src.research_intelligence_system.tools.arxiv_service import ArxivService

        loop         = asyncio.new_event_loop()
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

    # ── Tavily — stored as source="tavily", separated in _compare_node ────────
    try:
        from src.research_intelligence_system.tools.web_search import sync_web_search

        title_words   = set(title.lower().split())
        unique_models = [
            m for m in entities.get("models", [])[:2]
            if m.lower() not in title_words and m.lower() not in title.lower()
        ]
        tavily_query = f"research papers similar to {title} {' '.join(unique_models[:2])}".strip()
        web_text     = sync_web_search(tavily_query)
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
        triples         = list(getattr(paper, "triples", None) or [])

        year        = _extract_year(entities, refined_summary)
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

        # ── Separate arXiv papers from Tavily context ─────────────────────────
        # arXiv entries go into the RELATED PAPERS block as structured paper list.
        # Tavily entry goes into a separate ADDITIONAL WEB CONTEXT block explicitly
        # labelled to prevent the LLM from creating a table row for it.
        all_web     = state.get("web_papers", [])
        arxiv_only  = [p for p in all_web if p.get("source") == "arxiv"]
        tavily_blob = next((p for p in all_web if p.get("source") == "tavily"), None)

        web_papers_text = "\n".join([
            f"- {p.get('title', 'Unknown')} ({p.get('year', 'N/A')}): "
            f"{p.get('abstract', '')[:300]}"
            for p in arxiv_only
        ]) or "No arXiv papers retrieved."

        tavily_context = (
            "\n\nADDITIONAL WEB CONTEXT "
            "(background information only — do NOT create a table row for this):\n"
            + tavily_blob["abstract"][:600]
        ) if tavily_blob else ""

        prompt = _WEB_COMPARISON_PROMPT.format(
            title          = title,
            year           = year,
            summary        = refined_summary[:1500],
            key_results    = key_results,
            models         = ", ".join(entities.get("models",   [])[:8]),
            datasets       = ", ".join(entities.get("datasets", [])[:8]),
            metrics        = ", ".join(entities.get("metrics",  [])[:8]),
            methods        = ", ".join(entities.get("methods",  [])[:8]),
            web_papers     = web_papers_text,
            tavily_context = tavily_context,
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
        raw = re.sub(r'```\s*$',         '', raw, flags=re.MULTILINE)
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
            "ranking":          result.get("ranking",          ""),
            "evolution_trends": result.get("evolution_trends", ""),
            "positioning":      result.get("positioning",      ""),
            "web_papers_used":  result.get("web_papers_used",  state.get("web_papers", [])),
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