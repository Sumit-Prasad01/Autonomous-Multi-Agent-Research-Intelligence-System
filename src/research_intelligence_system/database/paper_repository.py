from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.research_intelligence_system.database.models import (
    KnowledgeTriple, LiteratureReview, PaperAnalysis, PaperComparison
)
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── PaperAnalysis ─────────────────────────────────────────────────────────────
async def get_paper_analyses(
    db: AsyncSession, chat_id: str
) -> List[PaperAnalysis]:
    result = await db.execute(
        select(PaperAnalysis)
        .where(PaperAnalysis.chat_id == uuid.UUID(chat_id))
        .order_by(PaperAnalysis.created_at)
    )
    return list(result.scalars().all())


async def get_paper_analysis(
    db: AsyncSession, paper_id: str
) -> Optional[PaperAnalysis]:
    return await db.scalar(
        select(PaperAnalysis)
        .where(PaperAnalysis.id == uuid.UUID(paper_id))
    )


async def update_paper_analysis(
    db: AsyncSession,
    paper_id: str,
    **kwargs,
) -> None:
    """Update any fields on PaperAnalysis by keyword args."""
    await db.execute(
        update(PaperAnalysis)
        .where(PaperAnalysis.id == uuid.UUID(paper_id))
        .values(**kwargs)
    )
    await db.commit()


async def set_analysis_status(
    db: AsyncSession,
    paper_id: str,
    status: str,
    error: str = "",
) -> None:
    await update_paper_analysis(
        db, paper_id,
        analysis_status=status,
        analysis_error=error,
    )


async def save_entities(
    db: AsyncSession, paper_id: str, entities: dict
) -> None:
    await update_paper_analysis(db, paper_id, entities=entities)


async def save_summaries(
    db: AsyncSession, paper_id: str, summaries: dict
) -> None:
    await update_paper_analysis(db, paper_id, summaries=summaries)


async def save_critic_output(
    db: AsyncSession,
    paper_id: str,
    refined_summary: str,
    quality_score: float,
    missing_entities: list,
    critic_validated: bool,
) -> None:
    await update_paper_analysis(
        db, paper_id,
        refined_summary=refined_summary,
        quality_score=quality_score,
        missing_entities=missing_entities,
        critic_validated=critic_validated,
    )


async def save_triples(
    db: AsyncSession, paper_id: str, chat_id: str, triples: list
) -> None:
    """Save triples to PaperAnalysis JSON + KnowledgeTriple rows."""
    # update JSON column
    await update_paper_analysis(db, paper_id, triples=triples)

    # save individual rows for querying
    for t in triples:
        triple = KnowledgeTriple(
            paper_id=uuid.UUID(paper_id),
            chat_id=uuid.UUID(chat_id),
            subject=t.get("subject", ""),
            relation=t.get("relation", ""),
            object=t.get("object", ""),
            confidence=t.get("confidence", 1.0),
        )
        db.add(triple)
    await db.commit()


async def save_similar_papers(
    db: AsyncSession, paper_id: str, similar_papers: list
) -> None:
    await update_paper_analysis(db, paper_id, similar_papers=similar_papers)


async def save_gaps(
    db: AsyncSession,
    paper_id: str,
    research_gaps: list,
    missing_edges: list,
    future_directions: list,
    novelty_score: float,
) -> None:
    await update_paper_analysis(
        db, paper_id,
        research_gaps=research_gaps,
        missing_edges=missing_edges,
        future_directions=future_directions,
        novelty_score=novelty_score,
    )


# ── KnowledgeTriple ───────────────────────────────────────────────────────────
async def get_triples_for_chat(
    db: AsyncSession, chat_id: str
) -> List[KnowledgeTriple]:
    result = await db.execute(
        select(KnowledgeTriple)
        .where(KnowledgeTriple.chat_id == uuid.UUID(chat_id))
    )
    return list(result.scalars().all())


# ── PaperComparison ───────────────────────────────────────────────────────────
async def save_comparison(
    db: AsyncSession,
    chat_id: str,
    paper_ids: list,
    comparison_type: str,
    comparison_table: dict,
    ranking: list,
    evolution_trends: str,
    positioning: str,
    web_papers_used: list,
) -> PaperComparison:
    comp = PaperComparison(
        chat_id=uuid.UUID(chat_id),
        paper_ids=paper_ids,
        comparison_type=comparison_type,
        comparison_table=comparison_table,
        ranking=ranking,
        evolution_trends=evolution_trends,
        positioning=positioning,
        web_papers_used=web_papers_used,
    )
    db.add(comp)
    await db.commit()
    await db.refresh(comp)
    return comp


async def get_comparison(
    db: AsyncSession, chat_id: str
) -> Optional[PaperComparison]:
    return await db.scalar(
        select(PaperComparison)
        .where(PaperComparison.chat_id == uuid.UUID(chat_id))
        .order_by(PaperComparison.created_at.desc())
    )


# ── LiteratureReview ──────────────────────────────────────────────────────────
async def save_literature_review(
    db: AsyncSession,
    chat_id: str,
    paper_ids: list,
    themes: list,
    review_text: str,
    research_gaps_summary: str,
    future_directions: str,
    overall_quality: float,
) -> LiteratureReview:
    review = LiteratureReview(
        chat_id=uuid.UUID(chat_id),
        paper_ids=paper_ids,
        themes=themes,
        review_text=review_text,
        research_gaps_summary=research_gaps_summary,
        future_directions=future_directions,
        overall_quality=overall_quality,
    )
    db.add(review)
    await db.commit()
    await db.refresh(review)
    return review


async def get_literature_review(
    db: AsyncSession, chat_id: str
) -> Optional[LiteratureReview]:
    return await db.scalar(
        select(LiteratureReview)
        .where(LiteratureReview.chat_id == uuid.UUID(chat_id))
        .order_by(LiteratureReview.created_at.desc())
    )