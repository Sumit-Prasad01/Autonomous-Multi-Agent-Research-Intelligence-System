"""
orchestrator_agent.py — LangGraph-based orchestrator
Runs all agents in correct order for each paper, then cross-paper agents.
Per-paper agents run in parallel across multiple papers.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from src.research_intelligence_system.agents.extraction_agent import (
    ExtractionAgent, get_sections_from_chunks
)
from src.research_intelligence_system.agents.summarizer_agent import SummarizerAgent
from src.research_intelligence_system.agents.critic_agent import CriticAgent
from src.research_intelligence_system.knowledge_graph.triple_extractor import TripleExtractor
from src.research_intelligence_system.database.paper_repository import (
    get_paper_analysis, save_entities, save_summaries,
    save_critic_output, save_triples, save_similar_papers,
    save_gaps, set_analysis_status,
)
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class OrchestratorState(TypedDict):
    chat_id:       str
    paper_ids:     List[str]
    llm_id:        str
    paper_count:   int
    results:       Dict[str, Any]   # paper_id → per-paper results
    comparison:    Dict[str, Any]
    lit_review:    Dict[str, Any]
    errors:        List[str]
    current_step:  str


# ── Per-paper pipeline ────────────────────────────────────────────────────────
async def _run_single_paper(
    chat_id:  str,
    paper_id: str,
    llm_id:   str,
    db:       AsyncSession,
) -> Dict[str, Any]:
    """
    Run full agent pipeline for a single paper:
    extraction → summarization → critic → triples
    """
    logger.info(f"[ORCHESTRATOR] starting paper_id={paper_id}")
    await set_analysis_status(db, paper_id, "running")

    try:
        # ── fetch paper from DB ───────────────────────────────────────────────
        paper = await get_paper_analysis(db, paper_id)
        if not paper:
            raise ValueError(f"PaperAnalysis not found: {paper_id}")

        # ── fetch chunks from Qdrant to get sections ──────────────────────────
        from src.research_intelligence_system.rag.vector_store import _store
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from src.research_intelligence_system.constants import COLLECTION_NAME

        results, _ = _store.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(
                key="metadata.chat_id",
                match=MatchValue(value=chat_id),
            )]),
            limit=200,
            with_payload=True,
            with_vectors=False,
        )

        # build sections dict from Qdrant payloads
        sections: Dict[str, List[str]] = {
            "abstract": [], "introduction": [], "methodology": [],
            "results":  [], "conclusion":   [], "body": [],
        }
        for point in results:
            payload = point.payload or {}
            section = payload.get("metadata", {}).get("section", "body")
            content = payload.get("page_content", "")
            if content:
                sections.setdefault(section, []).append(content)

        sections_text = {k: " ".join(v)[:3000] for k, v in sections.items() if v}

        if not sections_text:
            raise ValueError(f"No sections found in Qdrant for chat_id={chat_id}")

        # ── Stage 3: Entity Extraction ────────────────────────────────────────
        logger.info(f"[ORCHESTRATOR] extraction paper_id={paper_id}")
        extraction_agent = ExtractionAgent(llm_id=llm_id)
        entities = await extraction_agent.extract(paper_id, sections_text)
        await save_entities(db, paper_id, entities)

        # ── Stage 4: Summarization (two-stage: BART + LLM synthesis) ───────────
        logger.info(f"[ORCHESTRATOR] summarization paper_id={paper_id}")
        summarizer_agent = SummarizerAgent(llm_id=llm_id)
        summaries = await summarizer_agent.summarize(paper_id, sections_text, entities)
        await save_summaries(db, paper_id, summaries)

        # ── Stage 5: Critic ───────────────────────────────────────────────────
        logger.info(f"[ORCHESTRATOR] critic paper_id={paper_id}")
        critic_agent = CriticAgent(llm_id=llm_id)
        # pass comprehensive summary to critic for better evaluation
        critic_summaries = {**summaries, "overall": summaries.get("comprehensive", summaries.get("overall", ""))}
        critic_result = await critic_agent.evaluate(paper_id, critic_summaries, entities)
        await save_critic_output(
            db, paper_id,
            refined_summary  = critic_result["refined_summary"],
            quality_score    = critic_result["quality_score"],
            missing_entities = critic_result["missing_entities"],
            critic_validated = critic_result["critic_validated"],
        )

        # ── Stage 6: Triple Extraction ────────────────────────────────────────
        logger.info(f"[ORCHESTRATOR] triples paper_id={paper_id}")
        triple_extractor = TripleExtractor(llm_id=llm_id)
        triples = await triple_extractor.extract(paper_id, sections_text, entities)
        await save_triples(db, paper_id, chat_id, triples)

        # ── Stage 7: Neo4j Graph ──────────────────────────────────────────────
        logger.info(f"[ORCHESTRATOR] building graph paper_id={paper_id}")
        try:
            from src.research_intelligence_system.knowledge_graph.graph_builder import GraphBuilder
            graph_builder = GraphBuilder()
            await graph_builder.build(chat_id, paper_id, entities, triples)
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] graph build failed (non-fatal): {e}")

        # ── Stage 8: Similar Papers (arXiv) ──────────────────────────────────
        logger.info(f"[ORCHESTRATOR] fetching similar papers paper_id={paper_id}")
        try:
            from src.research_intelligence_system.tools.arxiv_service import ArxivService
            arxiv   = ArxivService()
            title   = entities.get("title", "")
            methods = entities.get("methods", [])[:3]
            query   = f"{title} {' '.join(methods)}".strip()
            similar = await arxiv.search(query, max_results=5)
            await save_similar_papers(db, paper_id, similar)
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] arXiv search failed (non-fatal): {e}")
            similar = []

        # ── Stage 9: Gap Detection ────────────────────────────────────────────
        logger.info(f"[ORCHESTRATOR] gap detection paper_id={paper_id}")
        try:
            from src.research_intelligence_system.agents.gap_detection_agent import GapDetectionAgent
            gap_agent = GapDetectionAgent(llm_id=llm_id)
            gap_result = await gap_agent.detect(
                chat_id=chat_id,
                paper_id=paper_id,
                entities=entities,
                similar_papers=similar,
            )
            await save_gaps(
                db, paper_id,
                research_gaps     = gap_result["research_gaps"],
                missing_edges     = gap_result["missing_edges"],
                future_directions = gap_result["future_directions"],
                novelty_score     = gap_result["novelty_score"],
            )
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] gap detection failed (non-fatal): {e}")

        await set_analysis_status(db, paper_id, "complete")
        logger.info(f"[ORCHESTRATOR] paper_id={paper_id} complete")

        return {
            "paper_id":       paper_id,
            "entities":       entities,
            "summaries":      summaries,
            "critic_result":  critic_result,
            "triples":        triples,
            "status":         "complete",
        }

    except Exception as e:
        logger.error(f"[ORCHESTRATOR] paper_id={paper_id} failed: {e}", exc_info=True)
        await set_analysis_status(db, paper_id, "failed", str(e))
        return {"paper_id": paper_id, "status": "failed", "error": str(e)}


# ── Orchestrator LangGraph nodes ──────────────────────────────────────────────
def _detect_task_node(state: OrchestratorState) -> OrchestratorState:
    """Stage 0 — detect task type based on paper count."""
    paper_count = len(state["paper_ids"])
    task        = "full_analysis"

    if paper_count == 1:
        task = "single_paper_with_web"
    elif paper_count >= 2:
        task = "multi_paper_comparison"

    logger.info(f"[ORCHESTRATOR] task={task} papers={paper_count}")
    return {
        **state,
        "paper_count":  paper_count,
        "current_step": task,
    }


async def _run_per_paper_node(state: OrchestratorState, db: AsyncSession) -> OrchestratorState:
    """Run all per-paper agents in parallel across all papers."""
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
        else:
            results[r["paper_id"]] = r

    return {**state, "results": results, "errors": errors}


async def _run_comparison_node(state: OrchestratorState, db: AsyncSession) -> OrchestratorState:
    """Stage 11 — comparison agent (web-augmented or direct)."""
    logger.info(f"[ORCHESTRATOR] comparison type={state['current_step']}")
    try:
        from src.research_intelligence_system.agents.comparison_agent import ComparisonAgent
        from src.research_intelligence_system.database.paper_repository import (
            get_paper_analyses, save_comparison
        )

        analyses   = await get_paper_analyses(db, state["chat_id"])
        comp_agent = ComparisonAgent(llm_id=state["llm_id"])
        comparison = await comp_agent.compare(
            chat_id       = state["chat_id"],
            paper_analyses= analyses,
            use_web       = len(analyses) == 1,
        )
        await save_comparison(
            db,
            chat_id          = state["chat_id"],
            paper_ids        = [str(a.id) for a in analyses],
            comparison_type  = "web_augmented" if len(analyses) == 1 else "direct",
            comparison_table = comparison.get("comparison_table", {}),
            ranking          = comparison.get("ranking", []),
            evolution_trends = comparison.get("evolution_trends", ""),
            positioning      = comparison.get("positioning", ""),
            web_papers_used  = comparison.get("web_papers_used", []),
        )
        return {**state, "comparison": comparison}

    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] comparison failed (non-fatal): {e}")
        return {**state, "comparison": {}}


async def _run_literature_review_node(state: OrchestratorState, db: AsyncSession) -> OrchestratorState:
    """Stage 12 — literature review agent."""
    logger.info("[ORCHESTRATOR] literature review")
    try:
        from src.research_intelligence_system.agents.literature_review_agent import LiteratureReviewAgent
        from src.research_intelligence_system.database.paper_repository import (
            get_paper_analyses, save_literature_review
        )

        analyses   = await get_paper_analyses(db, state["chat_id"])
        lit_agent  = LiteratureReviewAgent(llm_id=state["llm_id"])
        lit_review = await lit_agent.generate(
            chat_id   = state["chat_id"],
            analyses  = analyses,
            comparison= state.get("comparison", {}),
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
        Full pipeline:
        1. Detect task
        2. Run per-paper agents in parallel
        3. Run comparison agent
        4. Run literature review agent
        5. Return all results
        """
        logger.info(f"[ORCHESTRATOR] full analysis chat_id={chat_id} papers={len(paper_ids)}")

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

        # Stage 0 — detect task
        state = _detect_task_node(state)

        # Stage 3-9 — per-paper agents (parallel)
        state = await _run_per_paper_node(state, db)

        # Stage 11 — comparison
        state = await _run_comparison_node(state, db)

        # Stage 12 — literature review
        state = await _run_literature_review_node(state, db)

        logger.info(f"[ORCHESTRATOR] all done chat_id={chat_id} errors={state['errors']}")

        return {
            "results":    state["results"],
            "comparison": state["comparison"],
            "lit_review": state["lit_review"],
            "errors":     state["errors"],
        }