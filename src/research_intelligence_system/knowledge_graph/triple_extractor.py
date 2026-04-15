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
    return ChatGroq(model=llm_id, temperature=0)


# ── Prompts ───────────────────────────────────────────────────────────────────
_TRIPLE_PROMPT = """You are a knowledge graph builder for scientific papers.
Extract knowledge triples (subject, relation, object) from the paper below.

Focus on these relation types:
- TRAINED_ON       : model trained on dataset
- EVALUATED_ON     : model evaluated on dataset/benchmark
- ACHIEVES         : model achieves metric/score
- USES             : model/method uses technique/component
- IMPROVES_OVER    : model improves over baseline
- PROPOSED_BY      : method proposed by authors
- APPLIED_TO       : method applied to task
- COMPARED_WITH    : model compared with another model
- BASED_ON         : method based on architecture
- REPLACES         : method replaces older approach

Known entities:
Models:   {models}
Datasets: {datasets}
Metrics:  {metrics}
Methods:  {methods}

Paper text:
{text}

Return ONLY valid JSON array:
[
  {{"subject": "entity name", "relation": "RELATION_TYPE", "object": "entity name", "confidence": 0.9}},
  ...
]

Rules:
- Use EXACT entity names from the known entities list when possible
- confidence: 0.7-1.0 based on how explicitly stated in paper
- Extract at least 5 triples, maximum 30
- Return only the JSON array, no explanation."""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _clean_text_for_triples(text: str) -> str:
    """Remove math formulas and noise before triple extraction."""
    # remove repeated math patterns
    text = re.sub(r'(\d+\s*[\+\-\*/]\s*exp\s*\([^)]*\)\s*[\.\,]?\s*){2,}', ' ', text)
    # remove sequences of isolated numbers/symbols
    text = re.sub(r'(\s+\d+\s+){3,}', ' ', text)
    # remove ellipsis sequences
    text = re.sub(r'(\.\s*){3,}', ' ', text)
    return text.strip()


def _extract_triples_node(state: TripleState) -> TripleState:
    """Extract knowledge triples using LLM."""
    logger.info(f"[TRIPLES] paper_id={state['paper_id']} attempt={state['retry_count']+1}")

    sections = state["sections"]
    entities = state.get("entities", {})

    # combine most informative sections
    text = " ".join([
        sections.get("abstract", "")[:1000],
        sections.get("methodology", "")[:1500],
        sections.get("results", "")[:1500],
        sections.get("conclusion", "")[:500],
    ]).strip()

    text = _clean_text_for_triples(text)
    if len(text) < 100:
        return {**state, "triples": [], "error": "text too noisy after cleaning"}

    prompt = _TRIPLE_PROMPT.format(
        models   = ", ".join(entities.get("models",   [])[:15]),
        datasets = ", ".join(entities.get("datasets", [])[:15]),
        metrics  = ", ".join(entities.get("metrics",  [])[:15]),
        methods  = ", ".join(entities.get("methods",  [])[:15]),
        text     = text,
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()
        logger.info(f"[TRIPLES DEBUG] raw={raw[:300]}")

        # extract JSON array from response
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON array found in response")

        triples = json.loads(json_match.group())
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
    """Validate and clean extracted triples."""
    triples = state.get("triples", [])
    valid   = []

    for t in triples:
        if not isinstance(t, dict):
            continue

        subject  = str(t.get("subject", "")).strip()
        relation = str(t.get("relation", "")).strip().upper()
        obj      = str(t.get("object",  "")).strip()

        # skip empty or too short
        if not subject or not relation or not obj:
            continue
        if len(subject) < 2 or len(obj) < 2:
            continue
        # skip self-referential
        if subject.lower() == obj.lower():
            continue

        confidence = float(t.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))

        valid.append({
            "subject":    subject,
            "relation":   relation,
            "object":     obj,
            "confidence": confidence,
        })

    # deduplicate
    seen   = set()
    unique = []
    for t in valid:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    logger.info(f"[TRIPLES] validated: {len(unique)} unique triples from {len(triples)} raw")
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
        Run triple extraction graph.
        Returns list of {subject, relation, object, confidence} dicts.
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