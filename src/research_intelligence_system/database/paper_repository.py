"""
paper_repository.py
────────────────────────────────────────────────────────────────────────────────
Async-first, cache-backed repository for all multi-agent analysis tables:

  • PaperAnalysis
  • KnowledgeTriple
  • PaperComparison
  • LiteratureReview

Architecture
────────────
  • All DB calls use AsyncSession (same engine / session factory as the rest of
    the project — no new connections opened here).
  • Redis is used as a read-through cache:
      GET  → check Redis first → fallback to Postgres → populate cache
      SET  → write Postgres first → invalidate / repopulate cache
      DEL  → delete from Postgres → purge cache
  • Cache keys follow the project convention:  <namespace>:<identifier>
  • TTLs are defined as module-level constants so they are easy to tune.
  • All public methods are async.  Internal helpers are prefixed with _.
  • Singleton instances are exported at the bottom (same pattern as
    redis_service.py / chat_repository.py).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.research_intelligence_system.database.models import (
    KnowledgeTriple,
    LiteratureReview,
    PaperAnalysis,
    PaperComparison,
)
from src.research_intelligence_system.services.redis_service import get_redis
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

# ── Cache TTLs (seconds) ──────────────────────────────────────────────────────
_TTL_PAPER        = 3_600       # single PaperAnalysis row          — 1 h
_TTL_PAPER_LIST   = 300         # list of papers per chat            — 5 min
_TTL_TRIPLE_LIST  = 3_600       # triples per paper / chat           — 1 h
_TTL_COMPARISON   = 600         # latest comparison per chat         — 10 min
_TTL_LIT_REVIEW   = 600         # latest lit-review per chat         — 10 min

# ── Cache key builders ────────────────────────────────────────────────────────
def _ck_paper(paper_id: str)            -> str: return f"paper:{paper_id}"
def _ck_paper_hash(file_hash: str)      -> str: return f"paper_hash:{file_hash}"
def _ck_papers_chat(chat_id: str)       -> str: return f"papers_chat:{chat_id}"
def _ck_papers_chat_done(chat_id: str)  -> str: return f"papers_chat_done:{chat_id}"
def _ck_triples_paper(paper_id: str)   -> str: return f"triples_paper:{paper_id}"
def _ck_triples_chat(chat_id: str)     -> str: return f"triples_chat:{chat_id}"
def _ck_comparison(chat_id: str)       -> str: return f"comparison_latest:{chat_id}"
def _ck_lit_review(chat_id: str)       -> str: return f"lit_review_latest:{chat_id}"


# ── Serialisation helpers ─────────────────────────────────────────────────────
def _model_to_dict(obj: Any) -> dict:
    """
    Convert a SQLAlchemy model instance to a plain dict.
    Drops SQLAlchemy internals (_sa_*).  UUIDs → str for JSON compat.
    """
    out: dict = {}
    for k, v in obj.__dict__.items():
        if k.startswith("_"):
            continue
        out[k] = str(v) if isinstance(v, uuid.UUID) else v
    return out


def _to_json(obj: Any) -> str:
    return json.dumps(_model_to_dict(obj), default=str)


# ══════════════════════════════════════════════════════════════════════════════
# PaperRepository
# ══════════════════════════════════════════════════════════════════════════════

class PaperRepository:
    """
    Async CRUD + Redis cache for PaperAnalysis.

    Write path  →  Postgres first, then cache invalidated / repopulated.
    Read path   →  Redis first, Postgres on miss, result cached.
    """

    # ── internal cache helpers ────────────────────────────────────────────────

    async def _cache_set(self, paper: PaperAnalysis) -> None:
        try:
            r = await get_redis()
            data = _to_json(paper)
            async with r.pipeline(transaction=False) as pipe:
                await pipe.setex(_ck_paper(str(paper.id)), _TTL_PAPER, data)
                await pipe.setex(_ck_paper_hash(paper.file_hash), _TTL_PAPER, data)
                await pipe.execute()
        except Exception as e:
            logger.warning(f"paper cache set failed: {e}")

    async def _cache_invalidate(self, paper: PaperAnalysis) -> None:
        """Blow away all keys that could hold stale data for this paper."""
        try:
            r = await get_redis()
            async with r.pipeline(transaction=False) as pipe:
                await pipe.delete(_ck_paper(str(paper.id)))
                await pipe.delete(_ck_paper_hash(paper.file_hash))
                await pipe.delete(_ck_papers_chat(str(paper.chat_id)))
                await pipe.delete(_ck_papers_chat_done(str(paper.chat_id)))
                await pipe.execute()
        except Exception as e:
            logger.warning(f"paper cache invalidate failed: {e}")

    # ── Create ────────────────────────────────────────────────────────────────

    async def create(
        self,
        db: AsyncSession,
        *,
        chat_id: str,
        filename: str,
        file_hash: str,
    ) -> PaperAnalysis:
        """
        Insert a new PaperAnalysis row with status='pending'.
        Called immediately after PDF ingestion is queued.
        """
        paper = PaperAnalysis(
            id=uuid.uuid4(),
            chat_id=uuid.UUID(chat_id),
            filename=filename,
            file_hash=file_hash,
            analysis_status="pending",
        )
        db.add(paper)
        await db.commit()
        await db.refresh(paper)

        await self._cache_set(paper)
        # Invalidate list caches so next get_by_chat() sees this new row.
        try:
            r = await get_redis()
            async with r.pipeline(transaction=False) as pipe:
                await pipe.delete(_ck_papers_chat(chat_id))
                await pipe.delete(_ck_papers_chat_done(chat_id))
                await pipe.execute()
        except Exception as e:
            logger.warning(f"list cache invalidate on create failed: {e}")

        return paper

    # ── Read — single paper ───────────────────────────────────────────────────

    async def get_by_id(
        self,
        db: AsyncSession,
        paper_id: str,
    ) -> PaperAnalysis | None:
        # 1. Cache hit — still go to DB to get a live ORM object.
        #    We use the cache only as an existence check + TTL guard here;
        #    agents mutate the returned object so it must be a real ORM instance.
        try:
            r = await get_redis()
            raw = await r.get(_ck_paper(paper_id))
            if raw:
                return await self._fetch_db(db, paper_id)
        except Exception as e:
            logger.warning(f"cache read failed, falling through to DB: {e}")

        # 2. DB — populate cache on miss.
        return await self._fetch_db(db, paper_id)

    async def _fetch_db(
        self,
        db: AsyncSession,
        paper_id: str,
    ) -> PaperAnalysis | None:
        result = await db.execute(
            select(PaperAnalysis).where(PaperAnalysis.id == uuid.UUID(paper_id))
        )
        paper = result.scalar_one_or_none()
        if paper:
            await self._cache_set(paper)
        return paper

    async def get_by_hash(
        self,
        db: AsyncSession,
        file_hash: str,
    ) -> PaperAnalysis | None:
        """
        Return the most-recent analysis for an MD5 hash (deduplication).
        Checked before PDF parsing so the same file is never re-embedded.
        """
        try:
            r = await get_redis()
            raw = await r.get(_ck_paper_hash(file_hash))
            if raw:
                data = json.loads(raw)
                return await self._fetch_db(db, data["id"])
        except Exception as e:
            logger.warning(f"hash cache read failed: {e}")

        result = await db.execute(
            select(PaperAnalysis)
            .where(PaperAnalysis.file_hash == file_hash)
            .order_by(PaperAnalysis.created_at.desc())
        )
        paper = result.scalars().first()
        if paper:
            await self._cache_set(paper)
        return paper

    # ── Read — lists ──────────────────────────────────────────────────────────

    async def get_by_chat(
        self,
        db: AsyncSession,
        chat_id: str,
    ) -> list[PaperAnalysis]:
        """All papers in a chat, newest first."""
        try:
            r = await get_redis()
            raw = await r.get(_ck_papers_chat(chat_id))
            if raw:
                ids: list[str] = json.loads(raw)
                papers = []
                for pid in ids:
                    p = await self.get_by_id(db, pid)
                    if p:
                        papers.append(p)
                return papers
        except Exception as e:
            logger.warning(f"papers_chat cache read failed: {e}")

        result = await db.execute(
            select(PaperAnalysis)
            .where(PaperAnalysis.chat_id == uuid.UUID(chat_id))
            .order_by(PaperAnalysis.created_at.desc())
        )
        papers = list(result.scalars().all())

        try:
            r = await get_redis()
            ids = [str(p.id) for p in papers]
            await r.setex(_ck_papers_chat(chat_id), _TTL_PAPER_LIST, json.dumps(ids))
            for p in papers:
                await self._cache_set(p)
        except Exception as e:
            logger.warning(f"papers_chat cache write failed: {e}")

        return papers

    async def get_complete_by_chat(
        self,
        db: AsyncSession,
        chat_id: str,
    ) -> list[PaperAnalysis]:
        """
        Only fully analysed papers (status='complete').
        Used by comparison / lit-review agents.
        Short TTL (_TTL_PAPER_LIST = 5 min) — stale data here is expensive.
        """
        try:
            r = await get_redis()
            raw = await r.get(_ck_papers_chat_done(chat_id))
            if raw:
                ids: list[str] = json.loads(raw)
                papers = []
                for pid in ids:
                    p = await self.get_by_id(db, pid)
                    if p:
                        papers.append(p)
                return papers
        except Exception as e:
            logger.warning(f"papers_chat_done cache read failed: {e}")

        result = await db.execute(
            select(PaperAnalysis)
            .where(
                PaperAnalysis.chat_id == uuid.UUID(chat_id),
                PaperAnalysis.analysis_status == "complete",
            )
            .order_by(PaperAnalysis.created_at.desc())
        )
        papers = list(result.scalars().all())

        try:
            r = await get_redis()
            ids = [str(p.id) for p in papers]
            await r.setex(
                _ck_papers_chat_done(chat_id), _TTL_PAPER_LIST, json.dumps(ids)
            )
        except Exception as e:
            logger.warning(f"papers_chat_done cache write failed: {e}")

        return papers

    # ── Status helpers ────────────────────────────────────────────────────────

    async def set_status(
        self,
        db: AsyncSession,
        paper_id: str,
        status: str,
        error: str = "",
    ) -> PaperAnalysis | None:
        paper = await self.get_by_id(db, paper_id)
        if not paper:
            return None
        paper.analysis_status = status
        paper.analysis_error = error
        await db.commit()
        await db.refresh(paper)
        await self._cache_invalidate(paper)
        await self._cache_set(paper)
        return paper

    async def mark_running(self, db: AsyncSession, paper_id: str) -> PaperAnalysis | None:
        return await self.set_status(db, paper_id, "running")

    async def mark_complete(self, db: AsyncSession, paper_id: str) -> PaperAnalysis | None:
        return await self.set_status(db, paper_id, "complete")

    async def mark_failed(
        self, db: AsyncSession, paper_id: str, error: str
    ) -> PaperAnalysis | None:
        return await self.set_status(db, paper_id, "failed", error=error)

    # ── Per-stage field writers ───────────────────────────────────────────────
    # Each agent calls its own method — parallel agents never overwrite each
    # other's columns.  All delegate to the internal _patch() helper.

    async def _patch(
        self,
        db: AsyncSession,
        paper_id: str,
        fields: dict[str, Any],
    ) -> PaperAnalysis | None:
        """
        Apply `fields` to a PaperAnalysis row, commit, refresh, repopulate cache.
        All per-stage save_* methods delegate here.
        """
        paper = await self.get_by_id(db, paper_id)
        if not paper:
            return None
        for k, v in fields.items():
            if hasattr(paper, k):
                setattr(paper, k, v)
        await db.commit()
        await db.refresh(paper)
        await self._cache_invalidate(paper)
        await self._cache_set(paper)
        return paper

    async def save_entities(
        self,
        db: AsyncSession,
        paper_id: str,
        entities: dict[str, Any],
    ) -> PaperAnalysis | None:
        """Stage 3 — ExtractionAgent."""
        return await self._patch(db, paper_id, {"entities": entities})

    async def save_summaries(
        self,
        db: AsyncSession,
        paper_id: str,
        summaries: dict[str, str],
    ) -> PaperAnalysis | None:
        """Stage 4 — SummarizerAgent."""
        return await self._patch(db, paper_id, {"summaries": summaries})

    async def save_critic_results(
        self,
        db: AsyncSession,
        paper_id: str,
        *,
        quality_score: float,
        refined_summary: str,
        missing_entities: list[str],
        critic_validated: bool,
    ) -> PaperAnalysis | None:
        """Stage 5 — CriticAgent."""
        return await self._patch(
            db,
            paper_id,
            {
                "quality_score":    quality_score,
                "refined_summary":  refined_summary,
                "missing_entities": missing_entities,
                "critic_validated": critic_validated,
            },
        )

    async def save_triples(
        self,
        db: AsyncSession,
        paper_id: str,
        triples: list[tuple[str, str, str]],
    ) -> PaperAnalysis | None:
        """Stage 6 — TripleExtractor (JSON mirror on PaperAnalysis)."""
        return await self._patch(db, paper_id, {"triples": triples})

    async def save_similar_papers(
        self,
        db: AsyncSession,
        paper_id: str,
        similar_papers: list[dict[str, Any]],
    ) -> PaperAnalysis | None:
        """Stage 8 — ArxivService."""
        return await self._patch(db, paper_id, {"similar_papers": similar_papers})

    async def save_gaps(
        self,
        db: AsyncSession,
        paper_id: str,
        *,
        research_gaps: list[str],
        missing_edges: list[dict[str, Any]],
    ) -> PaperAnalysis | None:
        """Stage 9 — GapDetectionAgent."""
        return await self._patch(
            db,
            paper_id,
            {"research_gaps": research_gaps, "missing_edges": missing_edges},
        )

    async def save_future_directions(
        self,
        db: AsyncSession,
        paper_id: str,
        *,
        future_directions: list[str],
        novelty_score: float,
    ) -> PaperAnalysis | None:
        """Stage 10 — AdvancementAgent."""
        return await self._patch(
            db,
            paper_id,
            {"future_directions": future_directions, "novelty_score": novelty_score},
        )

    async def update_fields(
        self,
        db: AsyncSession,
        paper_id: str,
        fields: dict[str, Any],
    ) -> PaperAnalysis | None:
        """
        Generic multi-field update — orchestrator uses this at pipeline end
        to flush any remaining fields in a single commit.
        """
        return await self._patch(db, paper_id, fields)

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete(self, db: AsyncSession, paper_id: str) -> bool:
        paper = await self.get_by_id(db, paper_id)
        if not paper:
            return False
        await self._cache_invalidate(paper)
        await db.delete(paper)
        await db.commit()
        return True


# ══════════════════════════════════════════════════════════════════════════════
# TripleRepository
# ══════════════════════════════════════════════════════════════════════════════

class TripleRepository:
    """
    Postgres mirror of Neo4j triples.
    graph_builder writes here after writing to Neo4j — keeps both stores in sync.
    Cache is write-invalidate (triples are bulk-inserted once, then read-many).
    """

    async def _invalidate(self, paper_id: str, chat_id: str) -> None:
        try:
            r = await get_redis()
            async with r.pipeline(transaction=False) as pipe:
                await pipe.delete(_ck_triples_paper(paper_id))
                await pipe.delete(_ck_triples_chat(chat_id))
                await pipe.execute()
        except Exception as e:
            logger.warning(f"triple cache invalidate failed: {e}")

    async def bulk_create(
        self,
        db: AsyncSession,
        *,
        paper_id: str,
        chat_id: str,
        triples: list[dict[str, Any]],
    ) -> list[KnowledgeTriple]:
        """
        Insert many triples in one commit.
        Each dict must contain: subject, relation, object
        Optional: confidence (float, default 1.0)
        """
        rows: list[KnowledgeTriple] = []
        for t in triples:
            row = KnowledgeTriple(
                id=uuid.uuid4(),
                paper_id=uuid.UUID(paper_id),
                chat_id=uuid.UUID(chat_id),
                subject=t["subject"],
                relation=t["relation"],
                object=t["object"],
                confidence=t.get("confidence", 1.0),
            )
            db.add(row)
            rows.append(row)
        await db.commit()
        await self._invalidate(paper_id, chat_id)
        return rows

    async def get_by_paper(
        self,
        db: AsyncSession,
        paper_id: str,
    ) -> list[KnowledgeTriple]:
        try:
            r = await get_redis()
            raw = await r.get(_ck_triples_paper(paper_id))
            if raw:
                ids: list[str] = json.loads(raw)
                result = await db.execute(
                    select(KnowledgeTriple).where(
                        KnowledgeTriple.id.in_([uuid.UUID(i) for i in ids])
                    )
                )
                return list(result.scalars().all())
        except Exception as e:
            logger.warning(f"triples_paper cache read failed: {e}")

        result = await db.execute(
            select(KnowledgeTriple).where(
                KnowledgeTriple.paper_id == uuid.UUID(paper_id)
            )
        )
        rows = list(result.scalars().all())

        try:
            r = await get_redis()
            ids = [str(row.id) for row in rows]
            await r.setex(
                _ck_triples_paper(paper_id), _TTL_TRIPLE_LIST, json.dumps(ids)
            )
        except Exception as e:
            logger.warning(f"triples_paper cache write failed: {e}")

        return rows

    async def get_by_chat(
        self,
        db: AsyncSession,
        chat_id: str,
    ) -> list[KnowledgeTriple]:
        """
        All triples across all papers in a chat.
        Injected as graph context into the GraphRAG QA prompt.
        """
        try:
            r = await get_redis()
            raw = await r.get(_ck_triples_chat(chat_id))
            if raw:
                ids: list[str] = json.loads(raw)
                result = await db.execute(
                    select(KnowledgeTriple).where(
                        KnowledgeTriple.id.in_([uuid.UUID(i) for i in ids])
                    )
                )
                return list(result.scalars().all())
        except Exception as e:
            logger.warning(f"triples_chat cache read failed: {e}")

        result = await db.execute(
            select(KnowledgeTriple).where(
                KnowledgeTriple.chat_id == uuid.UUID(chat_id)
            )
        )
        rows = list(result.scalars().all())

        try:
            r = await get_redis()
            ids = [str(row.id) for row in rows]
            await r.setex(
                _ck_triples_chat(chat_id), _TTL_TRIPLE_LIST, json.dumps(ids)
            )
        except Exception as e:
            logger.warning(f"triples_chat cache write failed: {e}")

        return rows

    async def delete_by_paper(self, db: AsyncSession, paper_id: str) -> int:
        result = await db.execute(
            select(KnowledgeTriple).where(
                KnowledgeTriple.paper_id == uuid.UUID(paper_id)
            )
        )
        rows = list(result.scalars().all())
        chat_id = str(rows[0].chat_id) if rows else None
        for row in rows:
            await db.delete(row)
        await db.commit()
        if chat_id:
            await self._invalidate(paper_id, chat_id)
        return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ComparisonRepository
# ══════════════════════════════════════════════════════════════════════════════

class ComparisonRepository:

    async def create(
        self,
        db: AsyncSession,
        *,
        chat_id: str,
        paper_ids: list[str],
        comparison_type: str = "direct",
        comparison_table: dict[str, Any] | None = None,
        ranking: list[Any] | None = None,
        evolution_trends: str = "",
        positioning: str = "",
        web_papers_used: list[Any] | None = None,
    ) -> PaperComparison:
        comparison = PaperComparison(
            id=uuid.uuid4(),
            chat_id=uuid.UUID(chat_id),
            paper_ids=paper_ids,
            comparison_type=comparison_type,
            comparison_table=comparison_table or {},
            ranking=ranking or [],
            evolution_trends=evolution_trends,
            positioning=positioning,
            web_papers_used=web_papers_used or [],
        )
        db.add(comparison)
        await db.commit()
        await db.refresh(comparison)

        try:
            r = await get_redis()
            await r.setex(
                _ck_comparison(chat_id), _TTL_COMPARISON, _to_json(comparison)
            )
        except Exception as e:
            logger.warning(f"comparison cache write failed: {e}")

        return comparison

    async def get_latest_by_chat(
        self,
        db: AsyncSession,
        chat_id: str,
    ) -> PaperComparison | None:
        try:
            r = await get_redis()
            raw = await r.get(_ck_comparison(chat_id))
            if raw:
                data = json.loads(raw)
                result = await db.execute(
                    select(PaperComparison).where(
                        PaperComparison.id == uuid.UUID(data["id"])
                    )
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.warning(f"comparison cache read failed: {e}")

        result = await db.execute(
            select(PaperComparison)
            .where(PaperComparison.chat_id == uuid.UUID(chat_id))
            .order_by(PaperComparison.created_at.desc())
        )
        comparison = result.scalars().first()

        if comparison:
            try:
                r = await get_redis()
                await r.setex(
                    _ck_comparison(chat_id), _TTL_COMPARISON, _to_json(comparison)
                )
            except Exception as e:
                logger.warning(f"comparison cache write on miss failed: {e}")

        return comparison

    async def get_all_by_chat(
        self,
        db: AsyncSession,
        chat_id: str,
    ) -> list[PaperComparison]:
        # Not cached — full history is only needed for the history tab, not hot path.
        result = await db.execute(
            select(PaperComparison)
            .where(PaperComparison.chat_id == uuid.UUID(chat_id))
            .order_by(PaperComparison.created_at.desc())
        )
        return list(result.scalars().all())


# ══════════════════════════════════════════════════════════════════════════════
# LiteratureReviewRepository
# ══════════════════════════════════════════════════════════════════════════════

class LiteratureReviewRepository:

    async def create(
        self,
        db: AsyncSession,
        *,
        chat_id: str,
        paper_ids: list[str],
        themes: list[str] | None = None,
        review_text: str = "",
        research_gaps_summary: str = "",
        future_directions: str = "",
        overall_quality: float = 0.0,
    ) -> LiteratureReview:
        review = LiteratureReview(
            id=uuid.uuid4(),
            chat_id=uuid.UUID(chat_id),
            paper_ids=paper_ids,
            themes=themes or [],
            review_text=review_text,
            research_gaps_summary=research_gaps_summary,
            future_directions=future_directions,
            overall_quality=overall_quality,
        )
        db.add(review)
        await db.commit()
        await db.refresh(review)

        try:
            r = await get_redis()
            await r.setex(_ck_lit_review(chat_id), _TTL_LIT_REVIEW, _to_json(review))
        except Exception as e:
            logger.warning(f"lit_review cache write failed: {e}")

        return review

    async def get_latest_by_chat(
        self,
        db: AsyncSession,
        chat_id: str,
    ) -> LiteratureReview | None:
        try:
            r = await get_redis()
            raw = await r.get(_ck_lit_review(chat_id))
            if raw:
                data = json.loads(raw)
                result = await db.execute(
                    select(LiteratureReview).where(
                        LiteratureReview.id == uuid.UUID(data["id"])
                    )
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.warning(f"lit_review cache read failed: {e}")

        result = await db.execute(
            select(LiteratureReview)
            .where(LiteratureReview.chat_id == uuid.UUID(chat_id))
            .order_by(LiteratureReview.created_at.desc())
        )
        review = result.scalars().first()

        if review:
            try:
                r = await get_redis()
                await r.setex(
                    _ck_lit_review(chat_id), _TTL_LIT_REVIEW, _to_json(review)
                )
            except Exception as e:
                logger.warning(f"lit_review cache write on miss failed: {e}")

        return review

    async def update(
        self,
        db: AsyncSession,
        review_id: str,
        fields: dict[str, Any],
    ) -> LiteratureReview | None:
        result = await db.execute(
            select(LiteratureReview).where(
                LiteratureReview.id == uuid.UUID(review_id)
            )
        )
        review = result.scalar_one_or_none()
        if not review:
            return None
        for k, v in fields.items():
            if hasattr(review, k):
                setattr(review, k, v)
        await db.commit()
        await db.refresh(review)

        try:
            r = await get_redis()
            await r.setex(
                _ck_lit_review(str(review.chat_id)), _TTL_LIT_REVIEW, _to_json(review)
            )
        except Exception as e:
            logger.warning(f"lit_review cache update failed: {e}")

        return review


# ══════════════════════════════════════════════════════════════════════════════
# Singletons — import these in agents / services / router
# ══════════════════════════════════════════════════════════════════════════════

paper_repo      = PaperRepository()
triple_repo     = TripleRepository()
comparison_repo = ComparisonRepository()
lit_review_repo = LiteratureReviewRepository()