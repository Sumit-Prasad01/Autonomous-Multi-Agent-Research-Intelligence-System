"""
cross_paper_gap_detection.py — Cross-paper gap detection
Finds research gaps ACROSS multiple papers in a chat.
Novelty: gaps computed as missing edges between entities from DIFFERENT papers.

This extends the single-paper gap detection by:
1. Collecting all entities across all papers in a chat
2. Finding missing combinations that span paper boundaries
3. LLM interprets cross-paper gaps — these are stronger research contributions
   because they show the field collectively hasn't explored certain combinations

Example:
  Paper 1: BERT + SQuAD (QA task)
  Paper 2: GPT-3 + WebText (language modeling)
  Cross-paper gap: BERT not applied to language modeling
                   GPT-3 not evaluated on SQuAD
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── Prompt ────────────────────────────────────────────────────────────────────
_CROSS_PAPER_GAP_PROMPT = """You are a research gap analyst specializing in cross-paper analysis.
Given entities from MULTIPLE research papers in the same field, identify gaps that span across papers.

Papers analyzed:
{papers_summary}

All entities across papers:
Models/Algorithms: {all_models}
Datasets/Corpora:  {all_datasets}
Tasks/Problems:    {all_tasks}

Missing cross-paper combinations (entities from different papers never combined):
{missing_combinations}

Return ONLY valid JSON:
{{
  "cross_paper_gaps": [
    {{
      "gap": "Clear description mentioning specific entities from different papers",
      "novelty_score": <float 0-10>,
      "paper_1": "which paper has entity 1",
      "paper_2": "which paper has entity 2",
      "supporting_evidence": "why this cross-paper combination is unexplored",
      "suggested_experiment": "specific experiment combining elements from both papers"
    }}
  ],
  "field_level_insight": "1-2 sentences about what the cross-paper gaps reveal about the field",
  "overall_novelty_score": <float 0-10>
}}

Rules:
- Return 3-6 cross-paper gaps ranked by novelty_score descending
- Each gap must involve entities from at LEAST 2 different papers
- Be specific — name actual models, datasets, tasks from the papers
- suggested_experiment must be a concrete runnable study
- Return only JSON, no markdown, no explanation"""


def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0.2, max_tokens=2000)


def _clean_text(text: str) -> str:
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


# ── Core computation ──────────────────────────────────────────────────────────
def _get_all_entities_from_neo4j(chat_id: str) -> Dict[str, List[str]]:
    """Get all entities across all papers in this chat from Neo4j."""
    try:
        from src.research_intelligence_system.knowledge_graph.neo4j_service import Neo4jService
        neo4j = Neo4jService()

        models   = neo4j.get_nodes_by_type(chat_id, "Model")
        datasets = neo4j.get_nodes_by_type(chat_id, "Dataset")
        tasks    = neo4j.get_nodes_by_type(chat_id, "Task")
        methods  = neo4j.get_nodes_by_type(chat_id, "Method")
        neo4j.close()

        return {
            "models":   list(set(models)),
            "datasets": list(set(datasets)),
            "tasks":    list(set(tasks)),
            "methods":  list(set(methods)),
        }
    except Exception as e:
        logger.warning(f"[CROSS GAP] Neo4j entity fetch failed: {e}")
        return {"models": [], "datasets": [], "tasks": [], "methods": []}


def _get_existing_edges(chat_id: str) -> set:
    """Get all existing (subject, object) pairs from Neo4j for this chat."""
    try:
        from src.research_intelligence_system.knowledge_graph.neo4j_service import Neo4jService
        neo4j = Neo4jService()

        existing = set()
        for rel in ["TRAINED_ON", "EVALUATED_ON", "APPLIED_TO", "ACHIEVES", "USES"]:
            edges = neo4j.get_edges_by_type(chat_id, rel)
            for e in edges:
                model   = e.get("model", "")
                dataset = e.get("dataset", e.get("task", ""))
                if model and dataset:
                    existing.add((model.lower(), dataset.lower()))

        neo4j.close()
        return existing

    except Exception as e:
        logger.warning(f"[CROSS GAP] Neo4j edge fetch failed: {e}")
        return set()


def _compute_cross_paper_missing(
    all_entities:    Dict[str, List[str]],
    existing_edges:  set,
    paper_entities:  List[Dict],
) -> List[Dict]:
    """
    Compute missing combinations across papers.
    A cross-paper gap exists when:
    - Entity A comes from paper 1
    - Entity B comes from paper 2
    - No edge exists between A and B in the graph
    """
    missing = []

    # build entity → paper mapping
    entity_to_paper: Dict[str, str] = {}
    for i, paper in enumerate(paper_entities):
        paper_name = paper.get("filename", f"Paper {i+1}")
        entities   = paper.get("entities", {})
        for model in entities.get("models", []):
            entity_to_paper[model.lower()] = paper_name
        for dataset in entities.get("datasets", []):
            entity_to_paper[dataset.lower()] = paper_name
        for task in entities.get("tasks", []):
            entity_to_paper[task.lower()] = paper_name

    models   = all_entities.get("models",   [])[:10]
    datasets = all_entities.get("datasets", [])[:10]
    tasks    = all_entities.get("tasks",    [])[:10]

    # cross-paper model × dataset gaps
    for model in models:
        for dataset in datasets:
            if (model.lower(), dataset.lower()) not in existing_edges:
                model_paper   = entity_to_paper.get(model.lower(),   "unknown paper")
                dataset_paper = entity_to_paper.get(dataset.lower(), "unknown paper")

                # only include if entities come from DIFFERENT papers
                if model_paper != dataset_paper and model_paper != "unknown paper":
                    missing.append({
                        "type":         "cross_model_dataset",
                        "entity_1":     model,
                        "entity_2":     dataset,
                        "paper_1":      model_paper,
                        "paper_2":      dataset_paper,
                        "gap":          f"{model} from {model_paper} not evaluated on {dataset} from {dataset_paper}",
                    })

    # cross-paper model × task gaps
    for model in models:
        for task in tasks:
            if (model.lower(), task.lower()) not in existing_edges:
                model_paper = entity_to_paper.get(model.lower(), "unknown paper")
                task_paper  = entity_to_paper.get(task.lower(),  "unknown paper")

                if model_paper != task_paper and model_paper != "unknown paper":
                    missing.append({
                        "type":     "cross_model_task",
                        "entity_1": model,
                        "entity_2": task,
                        "paper_1":  model_paper,
                        "paper_2":  task_paper,
                        "gap":      f"{model} from {model_paper} not applied to {task} studied in {task_paper}",
                    })

    logger.info(f"[CROSS GAP] found {len(missing)} cross-paper missing combinations")
    return missing[:15]


# ── Public API ────────────────────────────────────────────────────────────────
async def detect_cross_paper_gaps(
    chat_id:        str,
    paper_analyses: List[Any],   # SQLAlchemy PaperAnalysis objects
    llm_id:         str = "llama-3.3-70b-versatile",
) -> Dict[str, Any]:
    """
    Detect research gaps across multiple papers in a chat.
    Only meaningful when chat has 2+ papers.

    Returns:
        {
          cross_paper_gaps: List[Dict] — ranked gaps with novelty scores
          field_level_insight: str — what gaps reveal about the field
          overall_novelty_score: float
          missing_combinations: List[Dict] — raw structural gaps
        }
    """
    if len(paper_analyses) < 2:
        logger.info(f"[CROSS GAP] only {len(paper_analyses)} paper(s) — skipping cross-paper detection")
        return {
            "cross_paper_gaps":    [],
            "field_level_insight": "",
            "overall_novelty_score": 0.0,
            "missing_combinations":  [],
        }

    logger.info(f"[CROSS GAP] detecting gaps across {len(paper_analyses)} papers chat_id={chat_id}")

    import asyncio
    loop = asyncio.get_running_loop()

    # fetch all entities + edges from Neo4j in threadpool
    all_entities, existing_edges = await asyncio.gather(
        loop.run_in_executor(None, _get_all_entities_from_neo4j, chat_id),
        loop.run_in_executor(None, _get_existing_edges, chat_id),
    )

    # build paper entity list for cross-paper mapping
    paper_entity_list = [
        {
            "filename": a.filename or f"Paper {i+1}",
            "entities": a.entities or {},
        }
        for i, a in enumerate(paper_analyses)
    ]

    # compute missing combinations
    missing = _compute_cross_paper_missing(
        all_entities, existing_edges, paper_entity_list
    )

    if not missing:
        logger.info("[CROSS GAP] no cross-paper missing combinations found")
        return {
            "cross_paper_gaps":     [],
            "field_level_insight":  "",
            "overall_novelty_score": 0.0,
            "missing_combinations":  [],
        }

    # build papers summary for prompt
    import re as _re
    def _clean_name(fn: str) -> str:
        fn = _re.sub(r'^[a-f0-9]{32}_', '', fn or "")
        return _re.sub(r'\.pdf$', '', fn, flags=_re.IGNORECASE).replace("_", " ").strip()

    papers_summary = "\n".join([
        f"Paper {i+1} ({_clean_name(a.filename)}): "
        f"models={', '.join((a.entities or {}).get('models', [])[:4])} | "
        f"datasets={', '.join((a.entities or {}).get('datasets', [])[:4])} | "
        f"tasks={', '.join((a.entities or {}).get('tasks', [])[:4])}"
        for i, a in enumerate(paper_analyses)
    ])

    missing_text = "\n".join([f"- {m['gap']}" for m in missing[:15]])

    prompt = _CROSS_PAPER_GAP_PROMPT.format(
        papers_summary      = papers_summary,
        all_models          = ", ".join(all_entities.get("models",   [])[:12]) or "none",
        all_datasets        = ", ".join(all_entities.get("datasets", [])[:12]) or "none",
        all_tasks           = ", ".join(all_entities.get("tasks",    [])[:12]) or "none",
        missing_combinations= missing_text,
    )

    try:
        llm      = _get_llm(llm_id)
        response = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
        raw      = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in cross-paper gap response")

        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            score_match = re.search(r'"overall_novelty_score"\s*:\s*([\d.]+)', cleaned)
            result = {
                "cross_paper_gaps":     [],
                "field_level_insight":  "",
                "overall_novelty_score": float(score_match.group(1)) if score_match else 5.0,
            }

        # clean and structure gaps
        raw_gaps = result.get("cross_paper_gaps", [])
        gaps = []
        for g in raw_gaps:
            if isinstance(g, dict):
                gaps.append({
                    "gap":                  _clean_text(str(g.get("gap", ""))),
                    "novelty_score":        float(g.get("novelty_score", 5.0)),
                    "paper_1":              str(g.get("paper_1", "")),
                    "paper_2":              str(g.get("paper_2", "")),
                    "supporting_evidence":  _clean_text(str(g.get("supporting_evidence", ""))),
                    "suggested_experiment": _clean_text(str(g.get("suggested_experiment", ""))),
                    "is_cross_paper":       True,   # flag for UI display
                })

        gaps.sort(key=lambda x: x["novelty_score"], reverse=True)

        field_insight    = _clean_text(result.get("field_level_insight", ""))
        novelty_score    = float(result.get("overall_novelty_score", 5.0))

        logger.info(
            f"[CROSS GAP] found {len(gaps)} cross-paper gaps "
            f"novelty={novelty_score}"
        )

        return {
            "cross_paper_gaps":     gaps,
            "field_level_insight":  field_insight,
            "overall_novelty_score": novelty_score,
            "missing_combinations":  missing,
        }

    except Exception as e:
        logger.warning(f"[CROSS GAP] LLM interpretation failed: {e}")
        return {
            "cross_paper_gaps":     [],
            "field_level_insight":  "",
            "overall_novelty_score": 0.0,
            "missing_combinations":  missing,
        }