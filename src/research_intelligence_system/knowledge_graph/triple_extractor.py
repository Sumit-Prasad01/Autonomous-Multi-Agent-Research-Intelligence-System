"""
triple_extractor.py — LangGraph-based knowledge triple extraction agent
Extracts (subject, relation, object) triples from paper text
These triples are stored in both Postgres and Neo4j
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
class TripleState(TypedDict):
    paper_id:    str
    llm_id:      str
    sections:    Dict[str, str]
    entities:    Dict[str, Any]
    triples:     List[Dict[str, str]]
    retry_count: int
    error:       str


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0, max_tokens=2000)


# ── Text cleaning ─────────────────────────────────────────────────────────────
_MATH_NOISE = re.compile(
    r'(\\[a-zA-Z]+\{[^}]*\}|'      # LaTeX: \frac{}, \sum{}
    r'\$[^$]*\$|'                   # inline math $...$
    r'\\\([^)]*\\\)|'               # \( ... \)
    r'(?:\d+\s*[\+\-\*/]\s*){3,}|' # repeated arithmetic
    r'(\d+\s*){5,}|'               # long number sequences
    r'[\+\-\*/=<>]{3,}|'           # symbol runs
    r'exp\s*\([^)]*\)\s*[\.\,]?)',  # exp(...) patterns
    re.DOTALL,
)

def _clean_text_for_triples(text: str) -> str:
    """Remove math formulas, LaTeX, and noise before triple extraction."""
    text = _MATH_NOISE.sub(' ', text)
    text = re.sub(r'(\.\s*){3,}', ' ', text)       # ellipsis sequences
    text = re.sub(r'\s{2,}', ' ', text)             # collapse whitespace
    text = re.sub(r'\n{2,}', '\n', text)            # collapse newlines
    # remove lines that are >50% digits/symbols (formula-heavy lines)
    lines = []
    for line in text.split('\n'):
        if len(line) > 10:
            non_alpha = sum(1 for c in line if not c.isalpha() and not c.isspace())
            if non_alpha / len(line) < 0.65:
                lines.append(line)
    return '\n'.join(lines).strip()


# ── Prompt ────────────────────────────────────────────────────────────────────
_TRIPLE_PROMPT = """You are a knowledge graph builder for scientific papers.
Extract knowledge triples (subject, relation, object) from the paper below.
Works for ALL scientific domains — ML, biology, physics, finance, etc.

Relation types to use:
- TRAINED_ON       : model/method trained on dataset
- EVALUATED_ON     : model evaluated on dataset/benchmark
- ACHIEVES         : model achieves metric/result
- USES             : model/method uses technique/component
- IMPROVES_OVER    : method improves over baseline
- PROPOSED_BY      : method proposed by authors
- APPLIED_TO       : method applied to task/domain
- COMPARED_WITH    : method compared with another
- BASED_ON         : method based on architecture/theory
- REPLACES         : method replaces older approach
- OUTPERFORMS      : method outperforms another on benchmark
- BOUNDED_BY       : method performance bounded by theoretical limit
- EXTENDS          : method extends or generalizes another
- VALIDATED_ON     : theoretical result validated on dataset

Known entities from this paper:
Models/Algorithms: {models}
Datasets/Corpora:  {datasets}
Metrics:           {metrics}
Methods:           {methods}

Paper text:
{text}

CRITICAL: Response must start with [ and end with ]. Pure JSON array only.
No text before or after. No markdown.

Return ONLY this format:
[
  {{"subject": "entity name", "relation": "RELATION_TYPE", "object": "entity name", "confidence": 0.9}},
  ...
]

Rules:
- Use EXACT entity names from known entities list when possible
- If specific name unclear use descriptive name like "proposed method" or "baseline model"
- Skip any triple where subject or object is a math formula, symbol, or number
- confidence: 0.7-1.0 based on how explicitly stated in paper
- Extract 10-40 triples maximum
- For theory papers: include OUTPERFORMS, BOUNDED_BY, EXTENDS relations
- No duplicate triples"""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _extract_triples_node(state: TripleState) -> TripleState:
    """Extract knowledge triples using LLM."""
    logger.info(f"[TRIPLES] paper_id={state['paper_id']} attempt={state['retry_count']+1}")

    sections = state["sections"]
    entities = state.get("entities", {})

    # combine most informative sections — trim aggressively
    raw_text = " ".join([
        sections.get("abstract",     "")[:1500],
        sections.get("introduction", "")[:2000],  # ADD THIS — clean entity text
        sections.get("results",      "")[:3500],  # INCREASE — comparisons live here
        sections.get("methodology",  "")[:1500],   # REDUCE — mostly math
        sections.get("conclusion",   "")[:1000],
    ]).strip()

    text = _clean_text_for_triples(raw_text)[:3500]

    if len(text) < 80:
        logger.warning(f"[TRIPLES] text too noisy after cleaning paper_id={state['paper_id']}")
        return {**state, "triples": [], "error": "text too noisy after cleaning"}

    prompt = _TRIPLE_PROMPT.format(
        models   = ", ".join(entities.get("models",   [])[:12]) or "none identified",
        datasets = ", ".join(entities.get("datasets", [])[:12]) or "none identified",
        metrics  = ", ".join(entities.get("metrics",  [])[:12]) or "none identified",
        methods  = ", ".join(entities.get("methods",  [])[:12]) or "none identified",
        text     = text,
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()
        logger.info(f"[TRIPLES DEBUG] raw={raw[:300]}")

        # find JSON array — handle cases where LLM adds text before/after
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON array found in response")

        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
        triples = json.loads(cleaned)
        logger.info(f"[TRIPLES] extracted {len(triples)} triples")

        return {**state, "triples": triples, "error": ""}

    except Exception as e:
        logger.warning(f"[TRIPLES] extraction failed attempt {state['retry_count']+1}: {e}")
        return {
            **state,
            "error":       str(e),
            "retry_count": state["retry_count"] + 1,
        }


def _validate_triples_node(state: TripleState) -> TripleState:
    """Validate, clean, and deduplicate extracted triples."""
    triples = state.get("triples", [])
    valid   = []

    # regex to detect math/formula strings
    _is_formula = re.compile(r'^[\d\s\+\-\*/=<>\.\\{}()\[\]]+$')

    for t in triples:
        if not isinstance(t, dict):
            continue

        subject  = str(t.get("subject",  "")).strip()
        relation = str(t.get("relation", "")).strip().upper()
        obj      = str(t.get("object",   "")).strip()

        # skip incomplete
        if not subject or not relation or not obj:
            continue
        if len(subject) < 2 or len(obj) < 2:
            continue
        # skip self-referential
        if subject.lower() == obj.lower():
            continue
        # skip formula subjects/objects
        if _is_formula.match(subject) or _is_formula.match(obj):
            continue
        # skip if subject or object is just digits
        if subject.replace(' ', '').isdigit() or obj.replace(' ', '').isdigit():
            continue
        # skip unknown relation types
        valid_relations = {
            "TRAINED_ON", "EVALUATED_ON", "ACHIEVES", "USES",
            "IMPROVES_OVER", "PROPOSED_BY", "APPLIED_TO",
            "COMPARED_WITH", "BASED_ON", "REPLACES",
        }
        if relation not in valid_relations:
            continue

        confidence = float(t.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))

        valid.append({
            "subject":    subject,
            "relation":   relation,
            "object":     obj,
            "confidence": confidence,
        })

    # deduplicate — case-insensitive on subject + object
    seen   = set()
    unique = []
    for t in valid:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    # sort by confidence descending
    unique.sort(key=lambda x: x["confidence"], reverse=True)

    logger.info(
        f"[TRIPLES] validated: {len(unique)} unique triples "
        f"from {len(triples)} raw"
    )
    return {**state, "triples": unique}


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry(state: TripleState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "extract"
    return "validate"


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph() -> Any:
    graph = StateGraph(TripleState)

    graph.add_node("extract",  _extract_triples_node)
    graph.add_node("validate", _validate_triples_node)

    graph.set_entry_point("extract")
    graph.add_conditional_edges("extract", _should_retry, {
        "extract":  "extract",
        "validate": "validate",
    })
    graph.add_edge("validate", END)

    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
class TripleExtractor:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def extract(
        self,
        paper_id: str,
        sections: Dict[str, str],
        entities: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Run triple extraction pipeline.
        Returns list of {subject, relation, object, confidence} dicts.
        Filters out math formulas and invalid triples automatically.
        """
        import asyncio
        loop = asyncio.get_running_loop()

        initial_state: TripleState = {
            "paper_id":    paper_id,
            "llm_id":      self.llm_id,
            "sections":    sections,
            "entities":    entities,
            "triples":     [],
            "retry_count": 0,
            "error":       "",
        }

        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        triples = result.get("triples", [])
        if not triples:
            logger.warning(f"[TRIPLES] no triples extracted for paper_id={paper_id}")

        return triples