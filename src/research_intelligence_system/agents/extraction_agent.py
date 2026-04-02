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
    sections:    Dict[str, str]    # {abstract, introduction, methodology, results, conclusion}
    entities:    Dict[str, Any]    # output
    retry_count: int
    error:       str


# ── LLM loader ────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0)


# ── Prompts ───────────────────────────────────────────────────────────────────
_EXTRACTION_PROMPT = """You are a scientific paper entity extractor.
Extract ALL entities from the paper sections below.

Return ONLY valid JSON with this exact structure:
{{
  "models": ["list of ML model names"],
  "datasets": ["list of dataset names"],
  "metrics": ["list of evaluation metrics"],
  "methods": ["list of methods/techniques"],
  "tasks": ["list of NLP/ML tasks"],
  "hyperparameters": {{"param_name": "value"}},
  "authors": ["list of author names if mentioned"],
  "year": "publication year if mentioned"
}}

Paper sections:
ABSTRACT: {abstract}

INTRODUCTION: {introduction}

METHODOLOGY: {methodology}

RESULTS: {results}

CONCLUSION: {conclusion}

Return only the JSON object, no explanation."""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _extract_node(state: ExtractionState) -> ExtractionState:
    """Main extraction node — calls LLM to extract entities."""
    logger.info(f"[EXTRACTION] paper_id={state['paper_id']} attempt={state['retry_count']+1}")

    sections = state["sections"]
    prompt   = _EXTRACTION_PROMPT.format(
        abstract     = sections.get("abstract", "")[:2000],
        introduction = sections.get("introduction", "")[:1500],
        methodology  = sections.get("methodology", "")[:2000],
        results      = sections.get("results", "")[:2000],
        conclusion   = sections.get("conclusion", "")[:1000],
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()

        # extract JSON from response
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")

        entities = json.loads(json_match.group())
        logger.info(f"[EXTRACTION] extracted {sum(len(v) if isinstance(v, list) else 1 for v in entities.values())} entities")

        return {**state, "entities": entities, "error": ""}

    except Exception as e:
        logger.warning(f"[EXTRACTION] failed attempt {state['retry_count']+1}: {e}")
        return {**state, "error": str(e), "retry_count": state["retry_count"] + 1}


def _validate_node(state: ExtractionState) -> ExtractionState:
    """Validate extracted entities — ensure required keys exist."""
    entities = state.get("entities", {})
    required = ["models", "datasets", "metrics", "methods", "tasks"]

    for key in required:
        if key not in entities:
            entities[key] = []
        if not isinstance(entities[key], list):
            entities[key] = [str(entities[key])]

    if "hyperparameters" not in entities:
        entities["hyperparameters"] = {}

    # deduplicate
    for key in required:
        entities[key] = list(set(entities[key]))

    logger.info(f"[EXTRACTION] validated — models={len(entities['models'])} datasets={len(entities['datasets'])} metrics={len(entities['metrics'])}")
    return {**state, "entities": entities}


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry(state: ExtractionState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "extract"      # retry
    return "validate"         # proceed


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
        Run entity extraction graph.
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

        # run sync LangGraph in threadpool
        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        entities = result.get("entities", {})
        if not entities:
            logger.warning(f"[EXTRACTION] empty result for paper_id={paper_id}")
            return {"models": [], "datasets": [], "metrics": [],
                    "methods": [], "tasks": [], "hyperparameters": {}}

        return entities


def get_sections_from_chunks(chunks) -> Dict[str, str]:
    """
    Helper — group parsed chunks by section into a dict.
    Input: list of LangChain Documents with metadata.section
    """
    sections: Dict[str, List[str]] = {
        "abstract": [], "introduction": [], "methodology": [],
        "results":  [], "conclusion":   [], "body": [],
    }
    for chunk in chunks:
        section = chunk.metadata.get("section", "body")
        if section in sections:
            sections[section].append(chunk.page_content)
        else:
            sections["body"].append(chunk.page_content)

    return {k: " ".join(v)[:3000] for k, v in sections.items() if v}