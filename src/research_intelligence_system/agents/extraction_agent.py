from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class ExtractionState(TypedDict):
    paper_id:    str
    llm_id:      str
    sections:    Dict[str, str]
    entities:    Dict[str, Any]
    retry_count: int
    error:       str


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0)


# ── Math / formula pre-filter ─────────────────────────────────────────────────
_MATH_PATTERN = re.compile(
    r'(\\[a-zA-Z]+\{.*?\}|'   # LaTeX commands \frac{}, \sum{}
    r'\$.*?\$|'                 # inline math $...$
    r'[A-Za-z]\s*=\s*[\d.]+|'  # simple assignments x = 0.5
    r'(?:\d+\s*[\+\-\*/]\s*){2,})',  # repeated arithmetic
    re.DOTALL,
)

def _clean_section(text: str, max_chars: int) -> str:
    """Remove math noise and truncate."""
    text = _MATH_PATTERN.sub(' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()[:max_chars]


# ── Regex fallback ────────────────────────────────────────────────────────────
def _regex_fallback(sections: Dict[str, str]) -> Dict[str, Any]:
    """
    Last-resort extraction when LLM returns no JSON.
    Pulls capitalized multi-word phrases near domain keywords.
    """
    full_text = " ".join(sections.values())

    def _find_near(keyword: str) -> List[str]:
        pattern = rf'{keyword}[s]?\s+(?:called\s+|named\s+|dubbed\s+)?([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){{0,2}})'
        return list(set(re.findall(pattern, full_text)))[:5]

    return {
        "models":          _find_near("model") + _find_near("network") + _find_near("algorithm"),
        "datasets":        _find_near("dataset") + _find_near("corpus") + _find_near("benchmark"),
        "metrics":         _find_near("metric") + _find_near("score") + _find_near("accuracy"),
        "methods":         _find_near("method") + _find_near("approach") + _find_near("technique"),
        "tasks":           _find_near("task") + _find_near("problem"),
        "hyperparameters": {},
        "authors":         [],
        "year":            "",
    }


# ── Prompt ────────────────────────────────────────────────────────────────────
_EXTRACTION_PROMPT = """You are a scientific paper entity extractor.
Extract ALL named entities from the paper sections below.
This works for ANY scientific domain — not just ML/NLP.

Return ONLY valid JSON with this exact structure:
{{
  "models": ["model, algorithm, or framework names used or proposed"],
  "datasets": ["dataset, corpus, benchmark, or data source names"],
  "metrics": ["evaluation metrics, measurements, or performance indicators"],
  "methods": ["methods, techniques, algorithms, or analytical approaches"],
  "tasks": ["research tasks, objectives, or problems addressed"],
  "hyperparameters": {{"param_name": "value"}},
  "authors": ["author names if explicitly mentioned"],
  "year": "publication year if mentioned, else empty string"
}}

If no entities exist for a field, return an empty list [].
Include domain-specific terms — e.g. for biology: genes, proteins, assays.
For finance: indices, instruments, models. For NLP: model architectures, corpora.

Paper sections:
ABSTRACT: {abstract}

INTRODUCTION: {introduction}

METHODOLOGY: {methodology}

RESULTS: {results}

CONCLUSION: {conclusion}

CRITICAL: Response must start with {{ and end with }}.
No text before or after. No markdown. No explanation. Pure JSON only."""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _extract_node(state: ExtractionState) -> ExtractionState:
    """Call LLM to extract entities from paper sections."""
    logger.info(f"[EXTRACTION] paper_id={state['paper_id']} attempt={state['retry_count']+1}")

    sections = state["sections"]

    # clean + truncate each section
    prompt = _EXTRACTION_PROMPT.format(
        abstract     = _clean_section(sections.get("abstract",     ""), 1500),
        introduction = _clean_section(sections.get("introduction", ""), 1000),
        methodology  = _clean_section(sections.get("methodology",  ""), 1500),
        results      = _clean_section(sections.get("results",      ""), 1500),
        conclusion   = _clean_section(sections.get("conclusion",   ""),  800),
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")

        cleaned  = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', json_match.group())
        entities = json.loads(cleaned)

        count = sum(len(v) if isinstance(v, list) else 1 for v in entities.values())
        logger.info(f"[EXTRACTION] extracted {count} entities")

        return {**state, "entities": entities, "error": ""}

    except Exception as e:
        logger.warning(f"[EXTRACTION] failed attempt {state['retry_count']+1}: {e}")
        return {**state, "error": str(e), "retry_count": state["retry_count"] + 1}


def _validate_node(state: ExtractionState) -> ExtractionState:
    """
    Validate + clean extracted entities.
    Falls back to regex extraction if entities are empty after retries.
    """
    entities = state.get("entities", {})
    required = ["models", "datasets", "metrics", "methods", "tasks"]

    # if completely empty after all retries — use regex fallback
    if not entities or all(not entities.get(k) for k in required):
        logger.warning(f"[EXTRACTION] LLM failed — using regex fallback paper_id={state['paper_id']}")
        entities = _regex_fallback(state["sections"])

    # ensure all required keys exist and are lists
    for key in required:
        if key not in entities:
            entities[key] = []
        if not isinstance(entities[key], list):
            entities[key] = [str(entities[key])]
        # clean empty strings
        entities[key] = [e.strip() for e in entities[key] if str(e).strip()]
        # deduplicate case-insensitive
        seen = set()
        deduped = []
        for e in entities[key]:
            if e.lower() not in seen:
                seen.add(e.lower())
                deduped.append(e)
        entities[key] = deduped

    if "hyperparameters" not in entities:
        entities["hyperparameters"] = {}
    if "authors" not in entities:
        entities["authors"] = []
    if "year" not in entities:
        entities["year"] = ""

    logger.info(
        f"[EXTRACTION] validated ✓ "
        f"models={len(entities['models'])} "
        f"datasets={len(entities['datasets'])} "
        f"metrics={len(entities['metrics'])}"
    )
    return {**state, "entities": entities}


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry(state: ExtractionState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "extract"
    return "validate"


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph() -> Any:
    graph = StateGraph(ExtractionState)

    graph.add_node("extract",  _extract_node)
    graph.add_node("validate", _validate_node)

    graph.set_entry_point("extract")
    graph.add_conditional_edges("extract", _should_retry, {
        "extract":  "extract",
        "validate": "validate",
    })
    graph.add_edge("validate", END)

    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
class ExtractionAgent:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def extract(
        self,
        paper_id: str,
        sections: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Run entity extraction pipeline.
        Domain-agnostic — works for ML, biology, finance, physics etc.
        Falls back to regex extraction if LLM fails twice.
        Returns entities dict.
        """
        import asyncio
        loop = asyncio.get_running_loop()

        initial_state: ExtractionState = {
            "paper_id":    paper_id,
            "llm_id":      self.llm_id,
            "sections":    sections,
            "entities":    {},
            "retry_count": 0,
            "error":       "",
        }

        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        entities = result.get("entities", {})
        if not entities:
            logger.warning(f"[EXTRACTION] empty result paper_id={paper_id}")
            return {
                "models": [], "datasets": [], "metrics": [],
                "methods": [], "tasks": [], "hyperparameters": {},
                "authors": [], "year": "",
            }

        return entities


def get_sections_from_chunks(chunks) -> Dict[str, str]:
    """
    Group parsed chunks by section into a dict.
    Input: list of LangChain Documents with metadata.section
    """
    sections: Dict[str, List[str]] = {
        "abstract": [], "introduction": [], "methodology": [],
        "results":  [], "conclusion":   [], "body":        [],
    }
    for chunk in chunks:
        section = chunk.metadata.get("section", "body")
        sections.setdefault(section, []).append(chunk.page_content)

    return {k: " ".join(v)[:3000] for k, v in sections.items() if v}