"""
graph_builder.py — Orchestrates knowledge graph construction
Takes entities + triples from agents → stores in Neo4j
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from src.research_intelligence_system.knowledge_graph.neo4j_service import Neo4jService
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="graph_builder")


class GraphBuilder:

    def __init__(self):
        self.neo4j = Neo4jService()

    def _build_sync(
        self,
        chat_id:  str,
        paper_id: str,
        filename: str,
        entities: Dict[str, Any],
        triples:  List[Dict],
    ) -> None:
        """Sync graph build — runs in threadpool."""

        # 1. paper node
        self.neo4j.create_paper_node(chat_id, paper_id, filename)
        logger.debug(f"[GRAPH] paper node created paper_id={paper_id}")

        # 2. entity nodes (Model, Dataset, Task, Metric, Method)
        self.neo4j.create_entity_nodes(chat_id, paper_id, entities)
        logger.debug(f"[GRAPH] entity nodes created paper_id={paper_id}")

        # 3. entity → paper edges
        with self.neo4j.driver.session() as session:
            with session.begin_transaction() as tx:
                for model in entities.get("models", [])[:20]:
                    tx.run(
                        """
                        MATCH (p:Paper {paper_id: $paper_id})
                        MATCH (m:Model {name: $name, chat_id: $chat_id})
                        MERGE (p)-[:CONTAINS_MODEL]->(m)
                        """,
                        paper_id=paper_id, name=str(model), chat_id=chat_id,
                    )
                for dataset in entities.get("datasets", [])[:20]:
                    tx.run(
                        """
                        MATCH (p:Paper {paper_id: $paper_id})
                        MATCH (d:Dataset {name: $name, chat_id: $chat_id})
                        MERGE (p)-[:USES_DATASET]->(d)
                        """,
                        paper_id=paper_id, name=str(dataset), chat_id=chat_id,
                    )
                tx.commit()

        # 4. knowledge triples as edges
        if triples:
            self.neo4j.create_triples_batch(chat_id, paper_id, triples)

        logger.info(
            f"[GRAPH] built: paper_id={paper_id} "
            f"entities={sum(len(v) for v in entities.values() if isinstance(v, list))} "
            f"triples={len(triples)}"
        )

    async def build(
        self,
        chat_id:  str,
        paper_id: str,
        entities: Dict[str, Any],
        triples:  List[Dict],
        filename: str = "",
    ) -> None:
        """
        Async entry point — builds full graph for one paper.
        Runs sync Neo4j operations in threadpool.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _POOL,
            self._build_sync,
            chat_id, paper_id, filename, entities, triples,
        )

    async def delete_chat_graph(self, chat_id: str) -> None:
        """Delete all graph data for a chat — called on chat delete."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _POOL,
            self.neo4j.delete_chat_graph,
            chat_id,
        )