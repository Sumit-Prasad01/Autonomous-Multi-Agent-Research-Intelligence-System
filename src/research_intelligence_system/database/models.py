from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean, Column, DateTime, Float,
    ForeignKey, Index, Integer, JSON,
    String, Text, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Existing tables ───────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    username   = Column(String(100), unique=True, nullable=False)
    password   = Column(String(255), nullable=False)
    is_active  = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")


class Chat(Base):
    __tablename__ = "chats"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id      = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    title        = Column(String(200), default="New Chat")
    llm_id       = Column(String(100), default="llama-3.1-8b-instant")
    allow_search = Column(Boolean, default=False)
    is_deleted   = Column(Boolean, default=False, nullable=False)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at   = Column(DateTime(timezone=True), server_default=func.now(),
                          onupdate=func.now())

    user          = relationship("User", back_populates="chats")
    messages      = relationship("Message", back_populates="chat",
                                 cascade="all, delete-orphan",
                                 order_by="Message.created_at")
    paper_analyses = relationship("PaperAnalysis", back_populates="chat",
                                  cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_chats_user_deleted", "user_id", "is_deleted"),
    )


class Message(Base):
    __tablename__ = "messages"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id    = Column(UUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    role       = Column(String(20), nullable=False)
    content    = Column(Text, nullable=False)
    source     = Column(String(50), default="")
    confidence = Column(String(10), default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chat = relationship("Chat", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_chat_created", "chat_id", "created_at"),
    )


# ── New tables for multi-agent system ────────────────────────────────────────
class PaperAnalysis(Base):
    """
    Stores all agent outputs for a single uploaded paper.
    One row per paper per chat.
    """
    __tablename__ = "paper_analyses"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id     = Column(UUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    filename    = Column(String(500), nullable=False)
    file_hash   = Column(String(64), nullable=False)     # MD5 — deduplication

    # Stage 3 — Entity Extraction
    entities    = Column(JSON, default=dict)
    # {models, datasets, metrics, methods, hyperparameters, tasks}

    # Stage 4 — Summarization
    summaries   = Column(JSON, default=dict)
    # {abstract, methodology, results, conclusion, overall}

    # Stage 5 — Critic
    quality_score      = Column(Float, default=0.0)
    refined_summary    = Column(Text, default="")
    missing_entities   = Column(JSON, default=list)
    critic_validated   = Column(Boolean, default=False)

    # Stage 6 — Triples (stored here as JSON, also in Neo4j)
    triples     = Column(JSON, default=list)
    # [(subject, relation, object), ...]

    # Stage 8 — Similar papers from arXiv
    similar_papers = Column(JSON, default=list)
    # [{title, arxiv_id, similarity, abstract}, ...]

    # Stage 9 — Gap detection
    research_gaps   = Column(JSON, default=list)
    missing_edges   = Column(JSON, default=list)

    # Stage 10 — Future work
    future_directions = Column(JSON, default=list)
    novelty_score     = Column(Float, default=0.0)

    # Status tracking
    analysis_status = Column(String(50), default="pending")
    # pending | running | complete | failed
    analysis_error  = Column(Text, default="")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now())

    chat     = relationship("Chat", back_populates="paper_analyses")
    triples_rel = relationship("KnowledgeTriple", back_populates="paper",
                               cascade="all, delete-orphan")
    

    __table_args__ = (
        Index("ix_paper_chat", "chat_id"),
        Index("ix_paper_hash", "file_hash"),
    )


class KnowledgeTriple(Base):
    """
    Stores (subject, relation, object) triples per paper.
    Also stored in Neo4j — this is the Postgres mirror.
    """
    __tablename__ = "knowledge_triples"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id   = Column(UUID(as_uuid=True), ForeignKey("paper_analyses.id",
                         ondelete="CASCADE"), nullable=False, index=True)
    chat_id    = Column(UUID(as_uuid=True), nullable=False, index=True)
    subject    = Column(String(500), nullable=False)
    relation   = Column(String(200), nullable=False)
    object     = Column(String(500), nullable=False)
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    paper = relationship("PaperAnalysis", back_populates="triples_rel")

    __table_args__ = (
        Index("ix_triple_paper", "paper_id"),
        Index("ix_triple_chat", "chat_id"),
    )


class PaperComparison(Base):
    """
    Stores comparison results across papers in a chat.
    One row per comparison run.
    """
    __tablename__ = "paper_comparisons"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id         = Column(UUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    paper_ids       = Column(JSON, nullable=False)       # list of paper_analysis UUIDs
    comparison_type = Column(String(50), default="direct")  # direct | web_augmented
    comparison_table = Column(JSON, default=dict)
    ranking         = Column(JSON, default=list)
    evolution_trends = Column(Text, default="")
    positioning     = Column(Text, default="")
    web_papers_used = Column(JSON, default=list)         # arXiv papers used in comparison
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_comparison_chat", "chat_id"),
    )


class LiteratureReview(Base):
    """
    Stores generated literature review for a chat session.
    """
    __tablename__ = "literature_reviews"

    id                   = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id              = Column(UUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"),
                                  nullable=False, index=True)
    paper_ids            = Column(JSON, nullable=False)
    themes               = Column(JSON, default=list)
    review_text          = Column(Text, default="")
    research_gaps_summary = Column(Text, default="")
    future_directions    = Column(Text, default="")
    overall_quality      = Column(Float, default=0.0)
    created_at           = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_litreview_chat", "chat_id"),
    )
