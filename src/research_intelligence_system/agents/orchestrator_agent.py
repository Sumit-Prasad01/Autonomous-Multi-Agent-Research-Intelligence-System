"""
orchestrator_agent.py — LangGraph-based orchestrator
Runs all agents in correct order for each paper, then cross-paper agents.
Per-paper agents run in parallel across multiple papers.

── THROTTLING ARCHITECTURE ───────────────────────────────────────────────────
Agents fall into two categories:

SELF-THROTTLING (handle wait + notify internally per node):
  CriticAgent      — each node calls sync_wait_for_groq + notify_groq_complete
  ComparisonAgent  — _compare_node calls sync_wait_for_groq + notify_groq_complete
  LiteratureReviewAgent — each node calls sync_wait_for_groq + notify_groq_complete

PUBLIC-METHOD THROTTLING (call wait_for_groq in public method, ONE call):
  ExtractionAgent  — wait_for_groq in .extract()
  SummarizerAgent  — wait_for_groq in .summarize()
  TripleExtractor  — wait_for_groq in .extract()
  GapDetectionAgent — wait_for_groq in .detect()

For public-method agents, the orchestrator MUST call notify_groq_complete()
after the await returns. This updates _last_call from "before LLM" to "after LLM".
For self-throttling agents, the orchestrator must NOT call notify_groq_complete().
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from src.research_intelligence_system.agents.comparison_agent import ComparisonAgent
from src.research_intelligence_system.agents.critic_agent import CriticAgent
from src.research_intelligence_system.agents.extraction_agent import ExtractionAgent
from src.research_intelligence_system.agents.gap_detection_agent import GapDetectionAgent
from src.research_intelligence_system.agents.literature_review_agent import LiteratureReviewAgent
from src.research_intelligence_system.agents.summarizer_agent import SummarizerAgent
from src.research_intelligence_system.constants import COLLECTION_NAME
from src.research_intelligence_system.core.groq_limiter import notify_groq_complete
from src.research_intelligence_system.database.paper_repository import (
    get_paper_analyses, get_paper_analysis,
    save_comparison, save_critic_output, save_entities,
    save_gaps, save_literature_review, save_similar_papers,
    save_summaries, save_triples, set_analysis_status,
)
from src.research_intelligence_system.knowledge_graph.graph_builder import GraphBuilder
from src.research_intelligence_system.knowledge_graph.triple_extractor import TripleExtractor
from src.research_intelligence_system.rag.vector_store import _store
from src.research_intelligence_system.tools.arxiv_service import ArxivService
from src.research_intelligence_system.utils.logger import get_logger
from qdrant_client.models import FieldCondition, Filter, MatchValue
from src.research_intelligence_system.agents.hallucination_detector import (
    compute_hallucination_score,
    filter_triples_by_faithfulness,
    score_gap_evidence,
)
from src.research_intelligence_system.database.paper_repository import save_hallucination

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
_ARXIV_DELAY  = 5    # seconds between arXiv calls to avoid 429
_QDRANT_LIMIT = 200  # max chunks to fetch per paper


# ── State ─────────────────────────────────────────────────────────────────────
class OrchestratorState(TypedDict):
    chat_id:      str
    paper_ids:    List[str]
    llm_id:       str
    paper_count:  int
    results:      Dict[str, Any]
    comparison:   Dict[str, Any]
    lit_review:   Dict[str, Any]
    errors:       List[str]
    current_step: str


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
        "abstract": [], "introduction": [], "methodology": [],
        "results":  [], "conclusion":   [], "body":        [],
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
    """Fetch similar papers using entity-based query."""
    try:
        arxiv   = ArxivService()
        similar = await arxiv.search_by_entities(
            models   = entities.get("models",   []),
            datasets = entities.get("datasets", []),
            methods  = entities.get("methods",  []),
            tasks    = entities.get("tasks",    []),
            title    = entities.get("title", filename or ""),
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
    Full per-paper agent pipeline.
    notify_groq_complete() is called after every public-method-throttled agent.
    Self-throttling agents (critic) do NOT get an orchestrator-level notify.
    """
    logger.info(f"[ORCHESTRATOR] starting paper_id={paper_id}")
    await set_analysis_status(db, paper_id, "running")

    try:
        paper = await get_paper_analysis(db, paper_id)
        if not paper:
            raise ValueError(f"PaperAnalysis not found: {paper_id}")

        sections_text = await _fetch_sections(chat_id)
        if not sections_text:
            raise ValueError(f"No sections found in Qdrant for chat_id={chat_id}")

        # ── Stage 3: Entity Extraction ────────────────────────────────────────
        # ExtractionAgent throttles via public method (wait_for_groq in .extract())
        logger.info(f"[ORCHESTRATOR] extraction paper_id={paper_id}")
        entities = await ExtractionAgent(llm_id=llm_id).extract(paper_id, sections_text)
        notify_groq_complete()   # ← update _last_call after LLM responded
        await save_entities(db, paper_id, entities)

        # ── Stage 4: Two-stage Summarization ─────────────────────────────────
        # SummarizerAgent throttles via public method
        logger.info(f"[ORCHESTRATOR] summarization paper_id={paper_id}")
        summaries = await SummarizerAgent(llm_id=llm_id).summarize(
            paper_id, sections_text, entities
        )
        notify_groq_complete()   # ← update _last_call after LLM responded
        await save_summaries(db, paper_id, summaries)

        # ── Stage 5: Critic Self-reflection ──────────────────────────────────
        # CriticAgent is SELF-THROTTLING — nodes call sync_wait + notify internally.
        # DO NOT call notify_groq_complete() here.
        logger.info(f"[ORCHESTRATOR] critic paper_id={paper_id}")
        critic_summaries = {
            **summaries,
            "overall": summaries.get("comprehensive", summaries.get("overall", "")),
        }
        critic_result = await CriticAgent(llm_id=llm_id).evaluate(
            paper_id, critic_summaries, entities,
            chunks=list(sections_text.values()),
        )
        await save_critic_output(
            db, paper_id,
            refined_summary  = critic_result["refined_summary"],
            quality_score    = critic_result["quality_score"],
            missing_entities = critic_result["missing_entities"],
            critic_validated = critic_result["critic_validated"],
        )

        # ── Stage 5b: Hallucination Detection ────────────────────────────────
        # Uses cross-encoder (CPU/GPU) — no LLM call, no throttling needed.
        logger.info(f"[ORCHESTRATOR] hallucination detection paper_id={paper_id}")
        try:
            chunks      = list(sections_text.values())
            hall_result = await compute_hallucination_score(
                summary = critic_result["refined_summary"],
                chunks  = chunks,
            )
            await save_hallucination(
                db, paper_id,
                hallucination_score    = critic_result["hallucination_score"],
                faithfulness_score     = 1.0 - critic_result["hallucination_score"],
                hallucinated_sentences = critic_result["hallucinated_sentences"],
            )
            logger.info(
                f"[HALLUCINATION] score={hall_result['hallucination_score']:.2f} "
                f"faithfulness={hall_result['faithfulness_score']:.2f}"
            )
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] hallucination detection failed (non-fatal): {e}")

        # ── Stage 6: Triple Extraction ────────────────────────────────────────
        # TripleExtractor throttles via public method
        logger.info(f"[ORCHESTRATOR] triples paper_id={paper_id}")
        triples = await TripleExtractor(llm_id=llm_id).extract(
            paper_id, sections_text, entities
        )
        notify_groq_complete()   # ← update _last_call after LLM responded
        await save_triples(db, paper_id, chat_id, triples)

        # ── Stage 6b: Triple Faithfulness Filter ─────────────────────────────
        # Cross-encoder only — no LLM call, no throttling needed.
        logger.info(f"[ORCHESTRATOR] triple faithfulness filter paper_id={paper_id}")
        try:
            chunks        = list(sections_text.values())
            triple_result = await filter_triples_by_faithfulness(triples, chunks)
            triples       = triple_result["filtered_triples"]
            logger.info(
                f"[TRIPLE FILTER] kept={triple_result['kept_count']} "
                f"removed={triple_result['removed_count']}"
            )
            await save_triples(db, paper_id, chat_id, triples)
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] triple filter failed (non-fatal): {e}")

        # ── Stage 7: Neo4j Graph Construction ────────────────────────────────
        logger.info(f"[ORCHESTRATOR] building graph paper_id={paper_id}")
        try:
            await GraphBuilder().build(
                chat_id, paper_id, entities, triples,
                filename=paper.filename or "",
            )
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] graph build failed (non-fatal): {e}")

        # ── Stage 8: Similar Papers (arXiv) ──────────────────────────────────
        # HTTP I/O only — no LLM call, no throttling needed.
        logger.info(f"[ORCHESTRATOR] fetching similar papers paper_id={paper_id}")
        similar = await _fetch_similar_papers(entities, paper.filename or "")
        await save_similar_papers(db, paper_id, similar)

        # ── Stage 9: Research Gap Detection ──────────────────────────────────
        # GapDetectionAgent throttles via public method
        logger.info(f"[ORCHESTRATOR] gap detection paper_id={paper_id}")
        gap_result = None
        try:
            gap_result = await GapDetectionAgent(llm_id=llm_id).detect(
                chat_id       = chat_id,
                paper_id      = paper_id,
                entities      = entities,
                similar_papers = similar,
            )
            notify_groq_complete()   # ← update _last_call after LLM responded
            await save_gaps(
                db, paper_id,
                research_gaps     = gap_result["research_gaps"],
                missing_edges     = gap_result["missing_edges"],
                future_directions = gap_result["future_directions"],
                novelty_score     = gap_result["novelty_score"],
            )
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] gap detection failed (non-fatal): {e}")

        # ── Stage 9b: Gap Evidence Faithfulness ──────────────────────────────
        # Cross-encoder only — no LLM call, no throttling needed.
        if gap_result:
            logger.info(f"[ORCHESTRATOR] gap evidence scoring paper_id={paper_id}")
            try:
                chunks     = list(sections_text.values())
                gap_scored = await score_gap_evidence(
                    gap_result["research_gaps"], chunks
                )
                gap_result["research_gaps"] = gap_scored["scored_gaps"]
                logger.info(
                    f"[GAP FILTER] scored={len(gap_scored['scored_gaps'])} "
                    f"low_confidence={gap_scored['low_confidence_count']}"
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

        await set_analysis_status(db, paper_id, "complete")
        logger.info(f"[ORCHESTRATOR] paper_id={paper_id} complete")

        return {
            "paper_id":      paper_id,
            "entities":      entities,
            "summaries":     summaries,
            "critic_result": critic_result,
            "triples":       triples,
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

    results = {}
    errors  = list(state.get("errors", []))
    for r in results_list:
        if isinstance(r, Exception):
            errors.append(str(r))
        elif r.get("status") != "failed":
            results[r["paper_id"]] = r

    return {**state, "results": results, "errors": errors}


async def _run_comparison_node(
    state: OrchestratorState,
    db:    AsyncSession,
) -> OrchestratorState:
    """
    Cross-paper comparison.
    ComparisonAgent is SELF-THROTTLING (_compare_node handles wait + notify).
    DO NOT call notify_groq_complete() here.
    The asyncio.sleep(3) is also removed — the limiter's 4s gap handles spacing.
    """
    logger.info(f"[ORCHESTRATOR] comparison type={state['current_step']}")

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
        return {**state, "comparison": comparison}

    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] comparison failed (non-fatal): {e}")
        return {**state, "comparison": {}}


async def _run_cross_paper_gaps_node(
    state: OrchestratorState,
    db:    AsyncSession,
) -> OrchestratorState:
    """Detect gaps across all papers in this chat (multi-paper only)."""
    if state["paper_count"] < 2:
        return {**state, "cross_paper_gaps": {}}

    logger.info("[ORCHESTRATOR] cross-paper gap detection")
    try:
        from src.research_intelligence_system.agents.cross_paper_gap_detection import (
            detect_cross_paper_gaps
        )
        analyses = await get_paper_analyses(db, state["chat_id"])
        result   = await detect_cross_paper_gaps(
            chat_id        = state["chat_id"],
            paper_analyses = analyses,
            llm_id         = state["llm_id"],
        )
        logger.info(f"[CROSS GAP] {len(result.get('cross_paper_gaps', []))} gaps found")
        return {**state, "cross_paper_gaps": result}
    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] cross-paper gaps failed (non-fatal): {e}")
        return {**state, "cross_paper_gaps": {}}


async def _run_literature_review_node(
    state: OrchestratorState,
    db:    AsyncSession,
) -> OrchestratorState:
    """
    Generate literature review.
    LiteratureReviewAgent is SELF-THROTTLING (nodes call sync_wait + notify).
    DO NOT call notify_groq_complete() here.
    """
    logger.info("[ORCHESTRATOR] literature review")
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
            themes                = lit_review.get("themes", []),
            review_text           = lit_review.get("review_text", ""),
            research_gaps_summary = lit_review.get("research_gaps_summary", ""),
            future_directions     = lit_review.get("future_directions", ""),
            overall_quality       = lit_review.get("overall_quality", 0.0),
            cross_paper_gaps      = state.get("cross_paper_gaps", {}).get("cross_paper_gaps", []),
            field_level_insight   = state.get("cross_paper_gaps", {}).get("field_level_insight", ""),
            cross_paper_novelty   = state.get("cross_paper_gaps", {}).get("overall_novelty_score", 0.0),
        )
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
        Full analysis pipeline:
          0. Detect task type
          3-9. Per-paper agents (parallel across papers)
          10. Cross-paper comparison
          11. Cross-paper gap detection (multi-paper only)
          12. Literature review
        """
        logger.info(
            f"[ORCHESTRATOR] full analysis "
            f"chat_id={chat_id} papers={len(paper_ids)}"
        )

        state: OrchestratorState = {
            "chat_id":      chat_id,
            "paper_ids":    paper_ids,
            "llm_id":       self.llm_id,
            "paper_count":  len(paper_ids),
            "results":      {},
            "comparison":   {},
            "lit_review":   {},
            "errors":       [],
            "current_step": "init",
        }

        state = _detect_task_node(state)
        state = await _run_per_paper_node(state, db)
        state = await _run_comparison_node(state, db)
        state = await _run_cross_paper_gaps_node(state, db)
        state = await _run_literature_review_node(state, db)

        logger.info(
            f"[ORCHESTRATOR] all done "
            f"chat_id={chat_id} errors={state['errors']}"
        )

        return {
            "results":    state["results"],
            "comparison": state["comparison"],
            "lit_review": state["lit_review"],
            "errors":     state["errors"],
        }