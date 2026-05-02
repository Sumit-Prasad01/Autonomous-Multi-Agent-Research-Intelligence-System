"""
orchestrator_agent.py — LangGraph-based orchestrator
Runs all agents in correct order for each paper, then cross-paper agents.
Per-paper agents run in parallel across multiple papers.

── THROTTLING ARCHITECTURE ───────────────────────────────────────────────────
Agents fall into two categories:

SELF-THROTTLING (handle wait + notify internally per node):
  CriticAgent           — each node calls sync_wait_for_groq + notify_groq_complete
  ComparisonAgent       — _compare_node calls sync_wait_for_groq + notify_groq_complete
  LiteratureReviewAgent — each node calls sync_wait_for_groq + notify_groq_complete

PUBLIC-METHOD THROTTLING (call wait_for_groq in public method, ONE call):
  ExtractionAgent    — wait_for_groq in .extract()
  SummarizerAgent    — wait_for_groq in .summarize()
  TripleExtractor    — wait_for_groq in .extract()
  GapDetectionAgent  — wait_for_groq in .detect()

For public-method agents, the orchestrator MUST call notify_groq_complete()
after the await returns — in BOTH the success and except paths.
For self-throttling agents, the orchestrator must NOT call notify_groq_complete().

── PIPELINE STAGES ───────────────────────────────────────────────────────────
Per-paper (parallel across papers):
  Stage 3:  Entity Extraction
  Stage 4:  Two-stage BART+LLM Summarization
  Stage 5:  Critic Self-reflection
  Stage 5b: Hallucination Detection (cross-encoder, no LLM)
  Stage 6:  Triple Extraction → Neo4j
  Stage 6b: Triple Faithfulness Filter (cross-encoder, no LLM)
  Stage 7:  Knowledge Graph Construction
  Stage 8:  Similar Papers (arXiv)
  Stage 9:  Research Gap Detection
  Stage 9b: Gap Evidence Scoring (cross-encoder, no LLM)
  Stage 9c: Graph Evolution Snapshot  ← Contribution 2

Cross-paper (sequential):
  Stage 10: Comparison
  Stage 11: Cross-paper Gap Detection (multi-paper only)
  Stage 12: Literature Review
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from src.research_intelligence_system.agents.comparison_agent import ComparisonAgent
from src.research_intelligence_system.agents.critic_agent import CriticAgent
from src.research_intelligence_system.agents.extraction_agent import ExtractionAgent
from src.research_intelligence_system.agents.gap_detection_agent import GapDetectionAgent
from src.research_intelligence_system.agents.hallucination_detector import (
    compute_hallucination_score,
    filter_triples_by_faithfulness,
    score_gap_evidence,
)
from src.research_intelligence_system.agents.literature_review_agent import LiteratureReviewAgent
from src.research_intelligence_system.agents.summarizer_agent import SummarizerAgent
from src.research_intelligence_system.constants import COLLECTION_NAME
from src.research_intelligence_system.core.groq_limiter import notify_groq_complete
from src.research_intelligence_system.database.paper_repository import (
    get_paper_analyses,
    get_paper_analysis,
    save_comparison,
    save_critic_output,
    save_entities,
    save_gaps,
    save_hallucination,
    save_literature_review,
    save_similar_papers,
    save_summaries,
    save_triples,
    set_analysis_status,
    update_paper_analysis,
)
from src.research_intelligence_system.knowledge_graph.graph_builder import GraphBuilder
from src.research_intelligence_system.knowledge_graph.graph_evolution_tracker import (
    GraphEvolutionTracker,
)
from src.research_intelligence_system.knowledge_graph.triple_extractor import TripleExtractor
from src.research_intelligence_system.rag.vector_store import _store
from src.research_intelligence_system.tools.arxiv_service import ArxivService
from src.research_intelligence_system.utils.logger import get_logger
from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
_ARXIV_DELAY  = 5    # seconds between arXiv calls to avoid 429
_QDRANT_LIMIT = 200  # max chunks to fetch per paper


# ── State ─────────────────────────────────────────────────────────────────────
class OrchestratorState(TypedDict):
    chat_id:          str
    paper_ids:        List[str]
    llm_id:           str
    paper_count:      int
    results:          Dict[str, Any]
    comparison:       Dict[str, Any]
    lit_review:       Dict[str, Any]
    cross_paper_gaps: Dict[str, Any]
    errors:           List[str]
    current_step:     str


# ── Helpers ───────────────────────────────────────────────────────────────────
async def _fetch_sections(chat_id: str) -> Dict[str, str]:
    """Fetch paper chunks from Qdrant and group by section."""
    results, _ = _store.client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(
            key="metadata.chat_id",
            match=MatchValue(value=chat_id),
        )]),
        limit=_QDRANT_LIMIT,
        with_payload=True,
        with_vectors=False,
    )

    sections: Dict[str, List[str]] = {
        "abstract":     [],
        "introduction": [],
        "methodology":  [],
        "results":      [],
        "conclusion":   [],
        "body":         [],
    }

    for point in results:
        payload = point.payload or {}
        section = payload.get("metadata", {}).get("section", "body")
        content = payload.get("page_content", "")
        if content:
            sections.setdefault(section, []).append(content)

    return {k: " ".join(v)[:3000] for k, v in sections.items() if v}


async def _fetch_similar_papers(
    entities: Dict[str, Any],
    filename: str,
) -> List[Dict]:
    """Fetch similar papers via entity-based arXiv query."""
    try:
        similar = await ArxivService().search_by_entities(
            models      = entities.get("models",   []),
            datasets    = entities.get("datasets", []),
            methods     = entities.get("methods",  []),
            tasks       = entities.get("tasks",    []),
            title       = entities.get("title", filename or ""),
            max_results = 5,
        )
        await asyncio.sleep(_ARXIV_DELAY)
        return similar
    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] arXiv search failed (non-fatal): {e}")
        return []


# ── Per-paper pipeline ────────────────────────────────────────────────────────
async def _run_single_paper(
    chat_id:  str,
    paper_id: str,
    llm_id:   str,
    db:       AsyncSession,
) -> Dict[str, Any]:
    """
    Full per-paper agent pipeline (Stages 3–9c).

    Throttling contract:
      PUBLIC-METHOD agents: orchestrator calls notify_groq_complete() after await,
                            in BOTH success and except paths.
      SELF-THROTTLING agents: orchestrator never calls notify_groq_complete().
    """
    t0 = time.monotonic()
    logger.info(f"[ORCHESTRATOR] starting paper_id={paper_id}")
    await set_analysis_status(db, paper_id, "running")

    entities   = {}
    gap_result: Optional[Dict] = None

    try:
        paper = await get_paper_analysis(db, paper_id)
        if not paper:
            raise ValueError(f"PaperAnalysis not found: {paper_id}")

        sections_text = await _fetch_sections(chat_id)
        if not sections_text:
            raise ValueError(f"No sections found in Qdrant for chat_id={chat_id}")

        chunks = list(sections_text.values())

        # ── Stage 3: Entity Extraction ────────────────────────────────────────
        logger.info(f"[ORCHESTRATOR] extraction paper_id={paper_id}")
        t = time.monotonic()
        try:
            entities = await ExtractionAgent(llm_id=llm_id).extract(
                paper_id, sections_text
            )
            notify_groq_complete()
        except Exception as e:
            notify_groq_complete()
            raise RuntimeError(f"Extraction failed: {e}") from e
        await save_entities(db, paper_id, entities)
        logger.info(f"[ORCHESTRATOR] extraction done ({time.monotonic()-t:.1f}s) "
                    f"entities={sum(len(v) for v in entities.values() if isinstance(v,list))}")

        # ── Stage 4: Two-stage Summarization ─────────────────────────────────
        logger.info(f"[ORCHESTRATOR] summarization paper_id={paper_id}")
        t = time.monotonic()
        try:
            summaries = await SummarizerAgent(llm_id=llm_id).summarize(
                paper_id, sections_text, entities
            )
            notify_groq_complete()
        except Exception as e:
            notify_groq_complete()
            raise RuntimeError(f"Summarization failed: {e}") from e
        await save_summaries(db, paper_id, summaries)
        logger.info(f"[ORCHESTRATOR] summarization done ({time.monotonic()-t:.1f}s)")

        # ── Stage 5: Critic Self-reflection ───────────────────────────────────
        # SELF-THROTTLING — do NOT call notify_groq_complete() here.
        logger.info(f"[ORCHESTRATOR] critic paper_id={paper_id}")
        t = time.monotonic()
        critic_summaries = {
            **summaries,
            "overall": summaries.get("comprehensive", summaries.get("overall", "")),
        }
        critic_result = await CriticAgent(llm_id=llm_id).evaluate(
            paper_id, critic_summaries, entities, chunks=chunks,
        )
        await save_critic_output(
            db, paper_id,
            refined_summary  = critic_result["refined_summary"],
            quality_score    = critic_result["quality_score"],
            missing_entities = critic_result["missing_entities"],
            critic_validated = critic_result["critic_validated"],
        )
        logger.info(f"[ORCHESTRATOR] critic done ({time.monotonic()-t:.1f}s) "
                    f"score={critic_result['quality_score']}")

        # ── Stage 5b: Hallucination Detection ────────────────────────────────
        # Cross-encoder only — no LLM, no throttling.
        logger.info(f"[ORCHESTRATOR] hallucination detection paper_id={paper_id}")
        t = time.monotonic()
        try:
            hall_result = await compute_hallucination_score(
                summary = critic_result["refined_summary"],
                chunks  = chunks,
            )
            await save_hallucination(
                db, paper_id,
                # Fix: use hall_result fields, NOT critic_result fields
                hallucination_score    = hall_result["hallucination_score"],
                faithfulness_score     = hall_result["faithfulness_score"],
                hallucinated_sentences = hall_result["hallucinated_sentences"],
            )
            logger.info(
                f"[HALLUCINATION] sentences={hall_result.get('total_sentences',0)} "
                f"supported={hall_result.get('supported_count',0)} "
                f"hallucinated={hall_result.get('hallucinated_count',0)} "
                f"rate={hall_result['hallucination_score']:.2f} "
                f"({time.monotonic()-t:.1f}s)"
            )
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] hallucination detection failed (non-fatal): {e}")

        # ── Stage 6: Triple Extraction ────────────────────────────────────────
        logger.info(f"[ORCHESTRATOR] triples paper_id={paper_id}")
        t = time.monotonic()
        try:
            triples = await TripleExtractor(llm_id=llm_id).extract(
                paper_id, sections_text, entities
            )
            notify_groq_complete()
        except Exception as e:
            notify_groq_complete()
            logger.warning(f"[ORCHESTRATOR] triple extraction failed (non-fatal): {e}")
            triples = []

        # Save initial triples (rows in KnowledgeTriple table)
        if triples:
            await save_triples(db, paper_id, chat_id, triples)
        logger.info(f"[ORCHESTRATOR] triples done ({time.monotonic()-t:.1f}s) "
                    f"count={len(triples)}")

        # ── Stage 6b: Triple Faithfulness Filter ─────────────────────────────
        # Cross-encoder only — no LLM, no throttling.
        # After filtering, only UPDATE the JSON column — do NOT re-insert rows
        # (that would create duplicates in the KnowledgeTriple table).
        logger.info(f"[ORCHESTRATOR] triple faithfulness filter paper_id={paper_id}")
        t = time.monotonic()
        if triples:
            try:
                triple_result = await filter_triples_by_faithfulness(triples, chunks)
                triples       = triple_result["filtered_triples"]
                logger.info(
                    f"[TRIPLE FILTER] kept={triple_result['kept_count']} "
                    f"removed={triple_result['removed_count']} "
                    f"faithfulness={triple_result.get('faithfulness_score',1.0):.2f} "
                    f"({time.monotonic()-t:.1f}s)"
                )
                # Update only the JSON column — rows already saved in Stage 6
                await update_paper_analysis(db, paper_id, triples=triples)
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] triple filter failed (non-fatal): {e}")

        # ── Stage 7: Knowledge Graph Construction ────────────────────────────
        logger.info(f"[ORCHESTRATOR] building graph paper_id={paper_id}")
        t = time.monotonic()
        try:
            await GraphBuilder().build(
                chat_id, paper_id, entities, triples,
                filename=paper.filename or "",
            )
            logger.info(f"[ORCHESTRATOR] graph built ({time.monotonic()-t:.1f}s)")
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] graph build failed (non-fatal): {e}")

        # ── Stage 8: Similar Papers (arXiv) ──────────────────────────────────
        logger.info(f"[ORCHESTRATOR] fetching similar papers paper_id={paper_id}")
        t = time.monotonic()
        similar = await _fetch_similar_papers(entities, paper.filename or "")
        await save_similar_papers(db, paper_id, similar)
        logger.info(f"[ORCHESTRATOR] similar papers done ({time.monotonic()-t:.1f}s) "
                    f"count={len(similar)}")

        # ── Stage 9: Research Gap Detection ──────────────────────────────────
        logger.info(f"[ORCHESTRATOR] gap detection paper_id={paper_id}")
        t = time.monotonic()
        try:
            gap_result = await GapDetectionAgent(llm_id=llm_id).detect(
                chat_id        = chat_id,
                paper_id       = paper_id,
                entities       = entities,
                similar_papers = similar,
            )
            notify_groq_complete()
            await save_gaps(
                db, paper_id,
                research_gaps     = gap_result["research_gaps"],
                missing_edges     = gap_result["missing_edges"],
                future_directions = gap_result["future_directions"],
                novelty_score     = gap_result["novelty_score"],
            )
            logger.info(
                f"[ORCHESTRATOR] gap detection done ({time.monotonic()-t:.1f}s) "
                f"gaps={len(gap_result['research_gaps'])} "
                f"novelty={gap_result['novelty_score']}"
            )
        except Exception as e:
            notify_groq_complete()   # always notify even on failure
            logger.warning(f"[ORCHESTRATOR] gap detection failed (non-fatal): {e}")
            gap_result = None

        # ── Stage 9b: Gap Evidence Scoring ───────────────────────────────────
        # Cross-encoder only — no LLM, no throttling.
        if gap_result:
            logger.info(f"[ORCHESTRATOR] gap evidence scoring paper_id={paper_id}")
            t = time.monotonic()
            try:
                gap_scored             = await score_gap_evidence(
                    gap_result["research_gaps"], chunks
                )
                gap_result["research_gaps"] = gap_scored["scored_gaps"]
                logger.info(
                    f"[GAP FILTER] scored={len(gap_scored['scored_gaps'])} "
                    f"low_confidence={gap_scored['low_confidence_count']} "
                    f"({time.monotonic()-t:.1f}s)"
                )
                await save_gaps(
                    db, paper_id,
                    research_gaps     = gap_result["research_gaps"],
                    missing_edges     = gap_result["missing_edges"],
                    future_directions = gap_result["future_directions"],
                    novelty_score     = gap_result["novelty_score"],
                )
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] gap scoring failed (non-fatal): {e}")

        # ── Stage 9c: Graph Evolution Snapshot ───────────────────────────────
        # Contribution 2: records graph state for KG evolution tracking.
        # Always non-fatal — must never block pipeline completion.
        logger.info(f"[ORCHESTRATOR] graph evolution snapshot paper_id={paper_id}")
        try:
            paper_year   = int(entities.get("year") or 0)
            current_gaps = gap_result["research_gaps"] if gap_result else []
            await GraphEvolutionTracker().snapshot(
                db         = db,
                chat_id    = chat_id,
                paper_id   = paper_id,
                paper_year = paper_year,
                gaps       = current_gaps,
            )
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] graph evolution snapshot failed (non-fatal): {e}")

        # ── Done ──────────────────────────────────────────────────────────────
        await set_analysis_status(db, paper_id, "complete")
        elapsed = time.monotonic() - t0
        logger.info(f"[ORCHESTRATOR] paper_id={paper_id} complete ({elapsed:.1f}s total)")

        return {
            "paper_id":      paper_id,
            "entities":      entities,
            "summaries":     summaries,
            "critic_result": critic_result,
            "triples":       triples,
            "gap_result":    gap_result,
            "status":        "complete",
        }

    except Exception as e:
        logger.error(f"[ORCHESTRATOR] paper_id={paper_id} failed: {e}", exc_info=True)
        await set_analysis_status(db, paper_id, "failed", str(e))
        return {"paper_id": paper_id, "status": "failed", "error": str(e)}


# ── Orchestrator nodes ────────────────────────────────────────────────────────
def _detect_task_node(state: OrchestratorState) -> OrchestratorState:
    count = len(state["paper_ids"])
    task  = "single_paper_with_web" if count == 1 else "multi_paper_comparison"
    logger.info(f"[ORCHESTRATOR] task={task} papers={count}")
    return {**state, "paper_count": count, "current_step": task}


async def _run_per_paper_node(
    state: OrchestratorState,
    db:    AsyncSession,
) -> OrchestratorState:
    """Run all per-paper pipelines in parallel."""
    logger.info(f"[ORCHESTRATOR] running {state['paper_count']} papers in parallel")

    results_list = await asyncio.gather(*[
        _run_single_paper(
            chat_id  = state["chat_id"],
            paper_id = pid,
            llm_id   = state["llm_id"],
            db       = db,
        )
        for pid in state["paper_ids"]
    ], return_exceptions=True)

    results: Dict[str, Any] = {}
    errors  = list(state.get("errors", []))

    for r in results_list:
        if isinstance(r, Exception):
            errors.append(str(r))
        elif isinstance(r, dict) and r.get("status") != "failed":
            results[r["paper_id"]] = r
        elif isinstance(r, dict):
            errors.append(r.get("error", "unknown failure"))

    return {**state, "results": results, "errors": errors}


async def _run_comparison_node(
    state: OrchestratorState,
    db:    AsyncSession,
) -> OrchestratorState:
    """
    Cross-paper comparison (Stage 10).
    ComparisonAgent is SELF-THROTTLING — do NOT call notify_groq_complete() here.
    """
    logger.info(f"[ORCHESTRATOR] comparison type={state['current_step']}")
    t = time.monotonic()

    try:
        analyses   = await get_paper_analyses(db, state["chat_id"])
        comparison = await ComparisonAgent(llm_id=state["llm_id"]).compare(
            chat_id        = state["chat_id"],
            paper_analyses = analyses,
            use_web        = len(analyses) == 1,
        )
        await save_comparison(
            db,
            chat_id          = state["chat_id"],
            paper_ids        = [str(a.id) for a in analyses],
            comparison_type  = "web_augmented" if len(analyses) == 1 else "direct",
            comparison_table = comparison.get("comparison_table", {}),
            ranking          = comparison.get("ranking", ""),
            evolution_trends = comparison.get("evolution_trends", ""),
            positioning      = comparison.get("positioning", ""),
            web_papers_used  = comparison.get("web_papers_used", []),
        )
        logger.info(f"[ORCHESTRATOR] comparison done ({time.monotonic()-t:.1f}s)")
        return {**state, "comparison": comparison}

    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] comparison failed (non-fatal): {e}")
        return {**state, "comparison": {}}


async def _run_cross_paper_gaps_node(
    state: OrchestratorState,
    db:    AsyncSession,
) -> OrchestratorState:
    """Cross-paper gap detection (Stage 11) — multi-paper only."""
    if state["paper_count"] < 2:
        return {**state, "cross_paper_gaps": {}}

    logger.info("[ORCHESTRATOR] cross-paper gap detection")
    t = time.monotonic()

    try:
        from src.research_intelligence_system.agents.cross_paper_gap_detection import (
            detect_cross_paper_gaps,
        )
        analyses = await get_paper_analyses(db, state["chat_id"])
        result   = await detect_cross_paper_gaps(
            chat_id        = state["chat_id"],
            paper_analyses = analyses,
            llm_id         = state["llm_id"],
        )
        logger.info(
            f"[CROSS GAP] {len(result.get('cross_paper_gaps', []))} gaps found "
            f"({time.monotonic()-t:.1f}s)"
        )
        return {**state, "cross_paper_gaps": result}

    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] cross-paper gaps failed (non-fatal): {e}")
        return {**state, "cross_paper_gaps": {}}


async def _run_literature_review_node(
    state: OrchestratorState,
    db:    AsyncSession,
) -> OrchestratorState:
    """
    Literature review (Stage 12).
    LiteratureReviewAgent is SELF-THROTTLING — do NOT call notify_groq_complete() here.
    """
    logger.info("[ORCHESTRATOR] literature review")
    t = time.monotonic()

    try:
        analyses   = await get_paper_analyses(db, state["chat_id"])
        lit_review = await LiteratureReviewAgent(llm_id=state["llm_id"]).generate(
            chat_id    = state["chat_id"],
            analyses   = analyses,
            comparison = state.get("comparison", {}),
        )
        await save_literature_review(
            db,
            chat_id               = state["chat_id"],
            paper_ids             = [str(a.id) for a in analyses],
            themes                = lit_review.get("themes",                []),
            review_text           = lit_review.get("review_text",           ""),
            research_gaps_summary = lit_review.get("research_gaps_summary", ""),
            future_directions     = lit_review.get("future_directions",     ""),
            overall_quality       = lit_review.get("overall_quality",       0.0),
            cross_paper_gaps      = state["cross_paper_gaps"].get("cross_paper_gaps",     []),
            field_level_insight   = state["cross_paper_gaps"].get("field_level_insight",  ""),
            cross_paper_novelty   = state["cross_paper_gaps"].get("overall_novelty_score", 0.0),
        )
        logger.info(f"[ORCHESTRATOR] literature review done ({time.monotonic()-t:.1f}s) "
                    f"quality={lit_review.get('overall_quality', 0.0)}")
        return {**state, "lit_review": lit_review}

    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] literature review failed (non-fatal): {e}")
        return {**state, "lit_review": {}}


# ── Public API ────────────────────────────────────────────────────────────────
class OrchestratorAgent:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def run_full_analysis(
        self,
        chat_id:   str,
        paper_ids: List[str],
        db:        AsyncSession,
    ) -> Dict[str, Any]:
        """
        Full analysis pipeline for one or more papers.
        Returns results dict on completion — never raises.
        """
        t0 = time.monotonic()
        logger.info(
            f"[ORCHESTRATOR] full analysis "
            f"chat_id={chat_id} papers={len(paper_ids)}"
        )

        state: OrchestratorState = {
            "chat_id":          chat_id,
            "paper_ids":        paper_ids,
            "llm_id":           self.llm_id,
            "paper_count":      len(paper_ids),
            "results":          {},
            "comparison":       {},
            "lit_review":       {},
            "cross_paper_gaps": {},
            "errors":           [],
            "current_step":     "init",
        }

        state = _detect_task_node(state)
        state = await _run_per_paper_node(state, db)
        state = await _run_comparison_node(state, db)
        state = await _run_cross_paper_gaps_node(state, db)
        state = await _run_literature_review_node(state, db)

        logger.info(
            f"[ORCHESTRATOR] all done chat_id={chat_id} "
            f"errors={state['errors']} "
            f"total={time.monotonic()-t0:.1f}s"
        )

        return {
            "results":    state["results"],
            "comparison": state["comparison"],
            "lit_review": state["lit_review"],
            "errors":     state["errors"],
        }