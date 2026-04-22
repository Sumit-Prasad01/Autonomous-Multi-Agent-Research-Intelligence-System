"""
gap_detection_agent.py — LangGraph-based research gap detection
Novelty: gaps computed as MISSING EDGES in knowledge graph (not LLM hallucination)
Flow: Neo4j cartesian product → find missing combos → LLM interprets gaps
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
class GapState(TypedDict):
    chat_id:           str
    paper_id:          str
    llm_id:            str
    entities:          Dict[str, Any]
    similar_papers:    List[Dict]
    missing_edges:     List[Dict]
    research_gaps:     List[Dict]
    future_directions: List[str]
    novelty_score:     float
    retry_count:       int
    error:             str


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0.2, max_tokens=3000)


# ── Text cleaner ──────────────────────────────────────────────────────────────
def _clean_gap_text(text: str) -> str:
    """Fix hyphenated words split mid-word (e.g. 'Convolu- tional' → 'Convolutional')."""
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Prompt ────────────────────────────────────────────────────────────────────
_GAP_INTERPRETATION_PROMPT = """You are a research gap analyst.
Given missing combinations in a knowledge graph, identify and RANK research gaps.

Paper entities:
Models/Algorithms: {models}
Datasets/Corpora:  {datasets}
Tasks/Problems:    {tasks}

Missing combinations (structural gaps from knowledge graph):
{missing_edges}

Similar papers in the field:
{similar_papers}

Return ONLY valid JSON with NO newlines inside string values:
{{
  "research_gaps": [
    {{
      "gap": "1-2 sentence description explaining WHAT is missing AND WHY it matters scientifically",
      "novelty_score": <float 0-10>,
      "supporting_evidence": "Specific reason this gap exists, referencing similar papers if available",
      "suggested_experiment": "Concrete experiment with specific dataset, metric, and comparison method"
    }}
  ],
  "future_directions": [
    "Direction 1: specific research proposal with methodology",
    "Direction 2: specific research proposal with methodology",
    "Direction 3: specific research proposal with methodology",
    "Direction 4: specific research proposal with methodology",
    "Direction 5: specific research proposal with methodology"
  ],
  "overall_novelty_score": <float 0-10>
}}

Novelty scoring guide:
9-10: Completely unexplored — no similar paper addresses this
7-8:  Partially explored — some papers touch on it but not directly
5-6:  Known gap — mentioned in literature but not solved
0-4:  Well explored — many papers address this

STRICT RULES:
- Return 3-8 gaps ranked by novelty_score DESCENDING
- NEVER use the phrase "has not been applied to" more than once across all gaps
- NEVER use the same sentence structure twice — each gap must be uniquely worded
- Each gap must explain WHY the absence matters scientifically, not just WHAT is missing
- suggested_experiment must name a specific dataset, metric, and comparison baseline
- Generate EXACTLY 4-5 future_directions
- future_directions must propose SOLUTIONS not restate problems
- Each future direction must be a different research avenue
- Do NOT copy text from gap descriptions into future_directions
- Return only JSON, no explanation, no markdown"""


# ── Structural gap computation ────────────────────────────────────────────────
def _compute_missing_edges_from_neo4j(chat_id: str, entities: Dict) -> List[Dict]:
    """
    Core novelty: compute gaps as missing edges in knowledge graph.
    Cartesian product of (models × datasets/tasks) → find which combos don't exist.
    """
    try:
        from src.research_intelligence_system.knowledge_graph.neo4j_service import Neo4jService
        neo4j = Neo4jService()

        models   = neo4j.get_nodes_by_type(chat_id, "Model")
        datasets = neo4j.get_nodes_by_type(chat_id, "Dataset")
        tasks    = neo4j.get_nodes_by_type(chat_id, "Task")

        # fallback to extracted entities if graph is empty
        if not models:   models   = entities.get("models",   [])
        if not datasets: datasets = entities.get("datasets", [])
        if not tasks:    tasks    = entities.get("tasks",    [])

        existing_model_dataset = set(
            (e["model"], e["dataset"])
            for e in neo4j.get_edges_by_type(chat_id, "TRAINED_ON")
            + neo4j.get_edges_by_type(chat_id, "EVALUATED_ON")
        )
        existing_model_task = set(
            (e.get("model", ""), e.get("task", e.get("dataset", "")))
            for e in neo4j.get_edges_by_type(chat_id, "APPLIED_TO")
        )
        neo4j.close()

        missing = []

        for model in models[:10]:
            for dataset in datasets[:10]:
                if (model, dataset) not in existing_model_dataset:
                    missing.append({
                        "type":    "model_dataset",
                        "subject": model,
                        "object":  dataset,
                        "gap":     f"{model} evaluated on {dataset}",
                    })

        for model in models[:10]:
            for task in tasks[:10]:
                if (model, task) not in existing_model_task:
                    missing.append({
                        "type":    "model_task",
                        "subject": model,
                        "object":  task,
                        "gap":     f"{model} applied to {task}",
                    })

        logger.info(f"[GAP] structural missing edges: {len(missing)}")
        return missing[:20]   # cap at 15 for faster LLM call

    except Exception as e:
        logger.warning(f"[GAP] Neo4j computation failed, using entity-based fallback: {e}")
        return _compute_missing_edges_from_entities(entities)


def _compute_missing_edges_from_entities(entities: Dict) -> List[Dict]:
    """Fallback: compute missing combos from extracted entities only."""
    models   = entities.get("models",   [])[:8]
    datasets = entities.get("datasets", [])[:8]
    tasks    = entities.get("tasks",    [])[:8]
    missing  = []

    if len(models) > 1 and datasets:
        for model in models[1:]:
            for dataset in datasets:
                missing.append({
                    "type":    "model_dataset",
                    "subject": model,
                    "object":  dataset,
                    "gap":     f"{model} evaluated on {dataset}",
                })

    if models and tasks:
        for model in models[:3]:
            for task in tasks[1:]:
                missing.append({
                    "type":    "model_task",
                    "subject": model,
                    "object":  task,
                    "gap":     f"{model} applied to {task}",
                })

    return missing[:15]


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _compute_gaps_node(state: GapState) -> GapState:
    """Structurally compute missing edges from knowledge graph."""
    logger.info(f"[GAP] computing missing edges paper_id={state['paper_id']}")
    missing_edges = _compute_missing_edges_from_neo4j(
        state["chat_id"], state["entities"]
    )
    return {**state, "missing_edges": missing_edges, "error": ""}


def _interpret_gaps_node(state: GapState) -> GapState:
    """LLM interprets missing edges as ranked, scored research gaps."""
    logger.info(f"[GAP] interpreting {len(state['missing_edges'])} missing edges")

    missing_edges  = state.get("missing_edges", [])
    entities       = state.get("entities", {})
    similar_papers = state.get("similar_papers", [])

    if not missing_edges and not entities:
        return {
            **state,
            "research_gaps":     [],
            "future_directions": ["Further evaluation on diverse benchmarks recommended."],
            "novelty_score":     5.0,
        }

    missing_text = "\n".join([
        f"- {e['gap']}" for e in missing_edges[:15]
    ]) or "No structural gaps found — using entity analysis."

    similar_text = "\n".join([
        f"- {p.get('title', '')} ({p.get('year', '')})"
        for p in similar_papers[:5]
    ]) or "No similar papers found."

    prompt = _GAP_INTERPRETATION_PROMPT.format(
        models         = ", ".join(entities.get("models",   [])[:10]) or "none identified",
        datasets       = ", ".join(entities.get("datasets", [])[:10]) or "none identified",
        tasks          = ", ".join(entities.get("tasks",    [])[:10]) or "none identified",
        missing_edges  = missing_text,
        similar_papers = similar_text,
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in gap interpretation response")

        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)

        # ── JSON parse with error recovery ────────────────────────────────────
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"[GAP] JSON truncated — attempting partial recovery")
            # attempt to extract research_gaps array
            gaps_match  = re.search(r'"research_gaps"\s*:\s*(\[.*?\])\s*[,}}]',
                                     cleaned, re.DOTALL)
            score_match = re.search(r'"overall_novelty_score"\s*:\s*([\d.]+)', cleaned)
            dirs_match  = re.search(r'"future_directions"\s*:\s*(\[.*?\])\s*[,}}]',
                                     cleaned, re.DOTALL)
            result = {
                "research_gaps": (
                    json.loads(gaps_match.group(1)) if gaps_match else []
                ),
                "future_directions": (
                    json.loads(dirs_match.group(1)) if dirs_match else []
                ),
                "overall_novelty_score": (
                    float(score_match.group(1)) if score_match else 5.0
                ),
            }

        raw_gaps          = result.get("research_gaps", [])
        future_directions = result.get("future_directions", [])
        novelty_score     = float(result.get("overall_novelty_score", 5.0))

        # build structured gaps + clean hyphenation
        structured_gaps = []
        for g in raw_gaps:
            if isinstance(g, dict):
                structured_gaps.append({
                    "gap":                  _clean_gap_text(str(g.get("gap", ""))),
                    "novelty_score":        float(g.get("novelty_score", 5.0)),
                    "supporting_evidence":  _clean_gap_text(str(g.get("supporting_evidence", ""))),
                    "suggested_experiment": _clean_gap_text(str(g.get("suggested_experiment", ""))),
                })
            elif isinstance(g, str):
                structured_gaps.append({
                    "gap":                  _clean_gap_text(g),
                    "novelty_score":        5.0,
                    "supporting_evidence":  "",
                    "suggested_experiment": "",
                })

        # sort by novelty_score descending
        structured_gaps.sort(key=lambda x: x["novelty_score"], reverse=True)

        # clean future directions
        future_directions = [
            _clean_gap_text(d) for d in future_directions
            if isinstance(d, str) and d.strip()
        ]

        if len(future_directions) < 4:
            logger.warning(
                f"[GAP] only {len(future_directions)} future directions — "
                f"LLM returned fewer than expected"
            )

        logger.info(
            f"[GAP] found {len(structured_gaps)} ranked gaps "
            f"overall_novelty={novelty_score}"
        )

        return {
            **state,
            "research_gaps":     structured_gaps,
            "future_directions": future_directions,
            "novelty_score":     novelty_score,
            "error":             "",
        }

    except Exception as e:
        logger.warning(f"[GAP] LLM interpretation failed attempt {state['retry_count']+1}: {e}")
        return {
            **state,
            "error":       str(e),
            "retry_count": state["retry_count"] + 1,
        }


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry(state: GapState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "interpret"
    return END


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph():
    graph = StateGraph(GapState)

    graph.add_node("compute",   _compute_gaps_node)
    graph.add_node("interpret", _interpret_gaps_node)

    graph.set_entry_point("compute")
    graph.add_edge("compute", "interpret")
    graph.add_conditional_edges("interpret", _should_retry, {
        "interpret": "interpret",
        END:         END,
    })

    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
class GapDetectionAgent:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def detect(
        self,
        chat_id:        str,
        paper_id:       str,
        entities:       Dict[str, Any],
        similar_papers: List[Dict] = [],
    ) -> Dict[str, Any]:
        """
        Run gap detection pipeline:
        1. Structurally compute missing edges from Neo4j
        2. LLM interprets missing edges as ranked research gaps
        Returns: {missing_edges, research_gaps, future_directions, novelty_score}
        """
        import asyncio
        loop = asyncio.get_running_loop()

        initial_state: GapState = {
            "chat_id":           chat_id,
            "paper_id":          paper_id,
            "llm_id":            self.llm_id,
            "entities":          entities,
            "similar_papers":    similar_papers,
            "missing_edges":     [],
            "research_gaps":     [],
            "future_directions": [],
            "novelty_score":     0.0,
            "retry_count":       0,
            "error":             "",
        }

        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        return {
            "missing_edges":     result.get("missing_edges", []),
            "research_gaps":     result.get("research_gaps", []),
            "future_directions": result.get("future_directions", []),
            "novelty_score":     result.get("novelty_score", 0.0),
        }