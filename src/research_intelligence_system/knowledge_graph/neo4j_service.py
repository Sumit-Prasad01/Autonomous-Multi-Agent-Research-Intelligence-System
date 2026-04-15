"""
neo4j_service.py — Neo4j driver wrapper
Handles: connect, create nodes/edges, query subgraph, detect missing edges
Thread-safe singleton connection
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── Singleton driver ──────────────────────────────────────────────────────────
class _Neo4jDriver:
    _instance: Optional[Driver] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> Driver:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Connecting to Neo4j …")
                    cls._instance = GraphDatabase.driver(
                        settings.NEO4J_URI,
                        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                        max_connection_pool_size=10,
                    )
                    cls._instance.verify_connectivity()
                    logger.info("Neo4j connected.")
        return cls._instance

    @classmethod
    def close(cls):
        with cls._lock:
            if cls._instance:
                cls._instance.close()
                cls._instance = None


# ── Neo4j Service ─────────────────────────────────────────────────────────────
class Neo4jService:
    """
    All graph operations for the research system.
    Uses chat_id as graph namespace — each chat has its own subgraph.
    """

    def __init__(self):
        self.driver = _Neo4jDriver.get()

    def close(self):
        pass   # driver is singleton — don't close

    def _run(self, query: str, **params) -> List[Dict]:
        """Execute a Cypher query and return results as list of dicts."""
        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except ServiceUnavailable as e:
            logger.error(f"[NEO4J] service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"[NEO4J] query failed: {e}")
            raise

    # ── Constraints + indexes ─────────────────────────────────────────────────
    def create_constraints(self):
        """Create uniqueness constraints — run once on startup."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper)   REQUIRE (p.paper_id) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Model)   REQUIRE (m.name, m.chat_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dataset) REQUIRE (d.name, d.chat_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task)    REQUIRE (t.name, t.chat_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (me:Metric) REQUIRE (me.name, me.chat_id) IS NODE KEY",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (me:Method) REQUIRE (me.name, me.chat_id) IS NODE KEY",
        ]
        for c in constraints:
            try:
                self._run(c)
            except Exception:
                pass   # constraint may already exist

    # ── Node creation ─────────────────────────────────────────────────────────
    def create_paper_node(self, chat_id: str, paper_id: str, filename: str) -> None:
        self._run(
            """
            MERGE (p:Paper {paper_id: $paper_id})
            SET p.chat_id  = $chat_id,
                p.filename = $filename
            """,
            paper_id=paper_id, chat_id=chat_id, filename=filename,
        )

    def create_entity_nodes(
        self, chat_id: str, paper_id: str, entities: Dict[str, Any]
    ) -> None:
        """Create nodes for all entity types in one transaction."""
        type_map = {
            "models":   "Model",
            "datasets": "Dataset",
            "tasks":    "Task",
            "metrics":  "Metric",
            "methods":  "Method",
        }
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                for entity_type, label in type_map.items():
                    for name in entities.get(entity_type, [])[:20]:
                        if not name or len(str(name)) < 2:
                            continue
                        tx.run(
                            f"""
                            MERGE (n:{label} {{name: $name, chat_id: $chat_id}})
                            SET n.paper_id = $paper_id
                            """,
                            name=str(name), chat_id=chat_id, paper_id=paper_id,
                        )
                tx.commit()
        logger.debug(f"[NEO4J] entity nodes created for paper_id={paper_id}")

    # ── Edge creation ─────────────────────────────────────────────────────────
    def create_triple(
        self,
        chat_id:    str,
        paper_id:   str,
        subject:    str,
        relation:   str,
        obj:        str,
        confidence: float = 1.0,
    ) -> None:
        """
        Create a triple as an edge in Neo4j.
        Uses MERGE to avoid duplicates.
        Node labels are inferred from entity type.
        """
        # sanitize relation — must be valid Cypher identifier
        relation = relation.upper().replace(" ", "_").replace("-", "_")
        relation = "".join(c for c in relation if c.isalnum() or c == "_")
        if not relation:
            relation = "RELATED_TO"

        query = f"""
            MERGE (s {{name: $subject, chat_id: $chat_id}})
            MERGE (o {{name: $obj,     chat_id: $chat_id}})
            MERGE (s)-[r:{relation}]->(o)
            SET r.confidence = $confidence,
                r.paper_id   = $paper_id
        """
        try:
            self._run(
                query,
                subject=subject, obj=obj, chat_id=chat_id,
                confidence=confidence, paper_id=paper_id,
            )
        except Exception as e:
            logger.warning(f"[NEO4J] triple failed ({subject}→{relation}→{obj}): {e}")

    def create_triples_batch(
        self,
        chat_id:  str,
        paper_id: str,
        triples:  List[Dict],
    ) -> None:
        """Batch create triples in one transaction."""
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                for t in triples:
                    subject    = str(t.get("subject", "")).strip()
                    relation   = str(t.get("relation", "RELATED_TO")).strip().upper()
                    obj        = str(t.get("object",  "")).strip()
                    confidence = float(t.get("confidence", 1.0))

                    if not subject or not obj:
                        continue

                    relation = "".join(c for c in relation if c.isalnum() or c == "_")
                    if not relation:
                        relation = "RELATED_TO"

                    try:
                        tx.run(
                            f"""
                            MERGE (s {{name: $subject, chat_id: $chat_id}})
                            MERGE (o {{name: $obj,     chat_id: $chat_id}})
                            MERGE (s)-[r:{relation}]->(o)
                            SET r.confidence = $confidence,
                                r.paper_id   = $paper_id
                            """,
                            subject=subject, obj=obj, chat_id=chat_id,
                            confidence=confidence, paper_id=paper_id,
                        )
                    except Exception as e:
                        logger.warning(f"[NEO4J] batch triple failed: {e}")
                        continue
                tx.commit()
        logger.info(f"[NEO4J] {len(triples)} triples stored paper_id={paper_id}")

    # ── Query: subgraph for RAG ───────────────────────────────────────────────
    def get_subgraph(self, chat_id: str, entities: List[str], depth: int = 2) -> List[Dict]:
        """
        Get subgraph around given entity names.
        Used in GraphRAG to add structural context to retrieval.
        """
        if not entities:
            return []

        result = self._run(
            """
            MATCH (n {chat_id: $chat_id})
            WHERE n.name IN $entities
            MATCH path = (n)-[r*1..2]-(m {chat_id: $chat_id})
            RETURN
                n.name     AS source,
                type(r[-1]) AS relation,
                m.name     AS target,
                r[-1].confidence AS confidence
            LIMIT 50
            """,
            chat_id=chat_id, entities=entities,
        )
        return result

    def get_subgraph_text(self, chat_id: str, entities: List[str]) -> str:
        """Convert subgraph to natural language for LLM context."""
        triples = self.get_subgraph(chat_id, entities)
        if not triples:
            return ""

        lines = [
            f"{t['source']} {t['relation'].replace('_', ' ').lower()} {t['target']}"
            for t in triples
            if t.get("source") and t.get("target")
        ]
        return "\n".join(lines[:30])

    # ── Query: gap detection ──────────────────────────────────────────────────
    def get_nodes_by_type(self, chat_id: str, label: str) -> List[str]:
        """Get all node names of a given type for this chat."""
        try:
            results = self._run(
                f"MATCH (n:{label} {{chat_id: $chat_id}}) RETURN n.name AS name",
                chat_id=chat_id,
            )
            return [r["name"] for r in results if r.get("name")]
        except Exception:
            return []

    def get_edges_by_type(self, chat_id: str, relation: str) -> List[Dict]:
        try:
            relation = relation.upper().replace(" ", "_")
            results  = self._run(
                f"""
                MATCH (s {{chat_id: $chat_id}})-[r:{relation}]->(o {{chat_id: $chat_id}})
                RETURN s.name AS model, o.name AS dataset, o.name AS task
                """,
                chat_id=chat_id,
            )
            return results
        except Exception:
            return []

    # ── Query: similar papers ─────────────────────────────────────────────────
    def get_related_entities(self, chat_id: str, entity_name: str) -> List[str]:
        """Find entities related to a given entity in the graph."""
        try:
            results = self._run(
                """
                MATCH (n {name: $name, chat_id: $chat_id})-[r]-(m {chat_id: $chat_id})
                RETURN m.name AS name
                LIMIT 20
                """,
                name=entity_name, chat_id=chat_id,
            )
            return [r["name"] for r in results if r.get("name")]
        except Exception:
            return []

    # ── Delete ────────────────────────────────────────────────────────────────
    def delete_chat_graph(self, chat_id: str) -> None:
        """Delete all nodes and edges for a chat — called on chat delete."""
        try:
            self._run(
                "MATCH (n {chat_id: $chat_id}) DETACH DELETE n",
                chat_id=chat_id,
            )
            logger.info(f"[NEO4J] deleted graph for chat_id={chat_id}")
        except Exception as e:
            logger.warning(f"[NEO4J] delete failed for chat_id={chat_id}: {e}")

    # ── Health check ──────────────────────────────────────────────────────────
    def health_check(self) -> bool:
        try:
            self._run("RETURN 1")
            return True
        except Exception:
            return False


# ── Module-level convenience ──────────────────────────────────────────────────
def get_neo4j() -> Neo4jService:
    return Neo4jService()


async def check_neo4j_health() -> bool:
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: Neo4jService().health_check())
    except Exception:
        return False