"""
graph_evolution_tracker.py — Knowledge Graph Evolution Tracker
Implements Contribution 2: KG Evolution as a Measurable Property of Research Fields

── WHAT THIS IMPLEMENTS ─────────────────────────────────────────────────────
After each paper is ingested and analyzed, this tracker:
  1. Counts the current state of the knowledge graph (nodes, edges) in Neo4j
  2. Compares current research gaps vs gaps from the previous paper
  3. Computes gap closure rate = how many old gaps were resolved by this paper
  4. Computes research velocity = change in closure rate over time
  5. Saves a timestamped snapshot to PostgreSQL

Over multiple papers (especially when ingested chronologically), the snapshots
form a time series that reveals:
  - How the research field fills its own gaps over time
  - Which gaps persist across multiple papers (highest priority future work)
  - Research velocity: does the field accelerate or plateau?

── FORMAL DEFINITIONS ────────────────────────────────────────────────────────
Definition 4: Gap Closure
  Gap Δ(u,v,r) is CLOSED at snapshot t if the gap text no longer appears
  in the current gap list (i.e. the structural missing edge was filled).

Definition 5: Research Velocity
  V(t) = |closed_gaps(t)| / |open_gaps(t-1)|
  Measures the fraction of previously-open gaps resolved by new papers.
  V=0: no gaps closed. V=1: all previous gaps closed. V>1: impossible by design.

Definition 6: Persistent Gap
  Gap Δ is PERSISTENT if it appears in ≥ min_appearances consecutive snapshots.
  Persistent gaps represent the most important unexplored research directions.

── INTEGRATION POINT ─────────────────────────────────────────────────────────
Called in orchestrator_agent.py after Stage 9b (gap evidence scoring):

    tracker = GraphEvolutionTracker()
    await tracker.snapshot(
        db         = db,
        chat_id    = chat_id,
        paper_id   = paper_id,
        paper_year = int(entities.get("year") or 0),
        gaps       = gap_result["research_gaps"],
    )
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from src.research_intelligence_system.knowledge_graph.neo4j_service import Neo4jService
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evolution_tracker")


# ── Graph counting (sync, runs in threadpool) ─────────────────────────────────

def _count_graph_sync(chat_id: str) -> Tuple[int, int]:
    """
    Count total nodes and edges in the Neo4j subgraph for this chat.
    Returns (node_count, edge_count).

    Excludes Paper nodes from node_count — we count entity nodes only
    (Model, Dataset, Task, Metric, Method) because Paper nodes are
    administrative, not part of the knowledge structure.
    """
    neo4j = Neo4jService()
    try:
        # Entity nodes only (not Paper nodes)
        node_result = neo4j._run(
            """
            MATCH (n {chat_id: $chat_id})
            WHERE NOT n:Paper
            RETURN count(n) AS node_count
            """,
            chat_id=chat_id,
        )
        node_count = node_result[0]["node_count"] if node_result else 0

        # All relationship edges between entities in this chat
        edge_result = neo4j._run(
            """
            MATCH (s {chat_id: $chat_id})-[r]->(o {chat_id: $chat_id})
            WHERE NOT s:Paper AND NOT o:Paper
            RETURN count(r) AS edge_count
            """,
            chat_id=chat_id,
        )
        edge_count = edge_result[0]["edge_count"] if edge_result else 0

        return int(node_count), int(edge_count)

    except Exception as e:
        logger.warning(f"[EVOLUTION] graph count failed: {e}")
        return 0, 0


# ── Gap key extraction ────────────────────────────────────────────────────────

def _gap_key(gap: Any) -> str:
    """
    Extract a stable identity key from a gap object.
    Works whether gap is a dict (standard) or string (legacy format).
    Lowercased and stripped for comparison.
    """
    if isinstance(gap, dict):
        text = gap.get("gap", gap.get("description", str(gap)))
    else:
        text = str(gap)
    return text.lower().strip()[:200]


def _gap_keys(gaps: List[Any]) -> set:
    """Extract a set of identity keys from a list of gaps."""
    return {_gap_key(g) for g in gaps if g}


# ── Main tracker ──────────────────────────────────────────────────────────────

class GraphEvolutionTracker:
    """
    Records knowledge graph evolution snapshots after each paper ingestion.

    Each snapshot captures:
      - Graph state: node count, edge count
      - Gap state: open gaps, closed gaps, newly opened gaps
      - Derived metrics: closure rate, research velocity

    The sequence of snapshots across papers forms the evolution curve
    used for Figure 7 and Table 5 in the paper.
    """

    async def snapshot(
        self,
        db:         AsyncSession,
        chat_id:    str,
        paper_id:   str,
        paper_year: int,
        gaps:       List[Any],
    ) -> Dict[str, Any]:
        """
        Take a snapshot of the current graph state after a paper is processed.

        Args:
            db:         SQLAlchemy async session
            chat_id:    Current chat (graph namespace)
            paper_id:   Paper just analyzed
            paper_year: Publication year (0 if unknown) — used for ordering
                        in the chronological ingestion experiment
            gaps:       Research gaps detected for this paper (from GapDetectionAgent)

        Returns:
            Dict with all snapshot metrics (also saved to DB).
        """
        from src.research_intelligence_system.database.paper_repository import (
            get_snapshots,
            save_snapshot,
        )

        # ── 1. Count graph state in Neo4j ─────────────────────────────────────
        loop = asyncio.get_running_loop()
        node_count, edge_count = await loop.run_in_executor(
            _POOL, _count_graph_sync, chat_id
        )

        # ── 2. Load previous snapshots from DB ───────────────────────────────
        prev_snapshots = await get_snapshots(db, chat_id)
        snapshot_order = len(prev_snapshots) + 1

        # ── 3. Compute graph deltas ───────────────────────────────────────────
        if prev_snapshots:
            prev = prev_snapshots[-1]
            node_delta = node_count - prev.node_count
            edge_delta = edge_count - prev.edge_count
        else:
            node_delta = node_count
            edge_delta = edge_count

        # ── 4. Compute gap dynamics ───────────────────────────────────────────
        current_gap_keys = _gap_keys(gaps)
        gap_count        = len(current_gap_keys)

        if prev_snapshots:
            prev_data     = prev_snapshots[-1].snapshot_data or {}
            prev_gap_keys = set(prev_data.get("gap_keys", []))
            prev_gap_count = prev_snapshots[-1].gap_count

            gaps_closed = len(prev_gap_keys - current_gap_keys)
            gaps_opened = len(current_gap_keys - prev_gap_keys)

            # Definition 5: Research Velocity = closed / prev_open
            closure_rate = (gaps_closed / prev_gap_count) if prev_gap_count > 0 else 0.0
            prev_closure  = prev_snapshots[-1].closure_rate
            velocity      = closure_rate - prev_closure
        else:
            # First snapshot — no previous to compare against
            gaps_closed  = 0
            gaps_opened  = gap_count
            closure_rate = 0.0
            velocity     = 0.0

        # ── 5. Build snapshot_data (rich detail for analysis) ─────────────────
        snapshot_data = {
            "gap_keys":      list(current_gap_keys),   # for next snapshot's delta
            "gaps_detail":   [
                {
                    "key":     _gap_key(g),
                    "text":    g.get("gap", str(g)) if isinstance(g, dict) else str(g),
                    "novelty": g.get("novelty_score", 0.0) if isinstance(g, dict) else 0.0,
                }
                for g in gaps[:20]
            ],
        }

        # ── 6. Save to DB ─────────────────────────────────────────────────────
        record = await save_snapshot(
            db            = db,
            chat_id       = chat_id,
            paper_id      = paper_id,
            paper_year    = paper_year,
            snapshot_order = snapshot_order,
            node_count    = node_count,
            edge_count    = edge_count,
            gap_count     = gap_count,
            gaps_closed   = gaps_closed,
            gaps_opened   = gaps_opened,
            closure_rate  = round(closure_rate, 4),
            velocity      = round(velocity, 4),
            node_delta    = node_delta,
            edge_delta    = edge_delta,
            snapshot_data = snapshot_data,
        )

        logger.info(
            f"[EVOLUTION] snapshot #{snapshot_order} chat_id={chat_id} "
            f"nodes={node_count}(+{node_delta}) edges={edge_count}(+{edge_delta}) "
            f"gaps={gap_count} closed={gaps_closed} velocity={velocity:.3f}"
        )

        return {
            "snapshot_order": snapshot_order,
            "node_count":     node_count,
            "edge_count":     edge_count,
            "gap_count":      gap_count,
            "gaps_closed":    gaps_closed,
            "gaps_opened":    gaps_opened,
            "closure_rate":   closure_rate,
            "velocity":       velocity,
            "node_delta":     node_delta,
            "edge_delta":     edge_delta,
        }

    # ── Analysis functions (used by API endpoint + Figure 7) ──────────────────

    @staticmethod
    def get_evolution_curve(snapshots: List[Any]) -> List[Dict]:
        """
        Convert snapshot list into a time series suitable for charting.
        Each point represents one paper's contribution to the graph.

        Used for Figure 7: gap closure rate and research velocity over time.
        """
        return [
            {
                "order":        s.snapshot_order,
                "year":         s.paper_year or 0,
                "nodes":        s.node_count,
                "edges":        s.edge_count,
                "gaps":         s.gap_count,
                "node_delta":   s.node_delta,
                "edge_delta":   s.edge_delta,
                "gaps_closed":  s.gaps_closed,
                "gaps_opened":  s.gaps_opened,
                "closure_rate": s.closure_rate,
                "velocity":     s.velocity,
            }
            for s in snapshots
        ]

    @staticmethod
    def get_persistent_gaps(
        snapshots:        List[Any],
        min_appearances:  int = 2,
    ) -> List[Dict]:
        """
        Identify gaps that appear across multiple consecutive snapshots.

        Definition 6: A gap is PERSISTENT if its key appears in
        ≥ min_appearances snapshots without being closed.

        These are the most important unexplored research directions —
        the field has repeatedly "seen" them but not addressed them.

        Returns list sorted by persistence count descending.
        """
        # Count how many snapshots each gap key appears in
        gap_appearance_count: Dict[str, int] = {}
        gap_text_map:         Dict[str, str] = {}

        for snap in snapshots:
            data = snap.snapshot_data or {}
            for gd in data.get("gaps_detail", []):
                key  = gd.get("key", "")
                text = gd.get("text", key)
                if key:
                    gap_appearance_count[key] = gap_appearance_count.get(key, 0) + 1
                    gap_text_map[key] = text

        # Filter to persistent gaps
        persistent = [
            {
                "gap":          gap_text_map.get(key, key),
                "appearances":  count,
                "persistence":  f"Seen in {count}/{len(snapshots)} papers",
            }
            for key, count in gap_appearance_count.items()
            if count >= min_appearances
        ]

        return sorted(persistent, key=lambda x: x["appearances"], reverse=True)

    @staticmethod
    def get_velocity_stats(snapshots: List[Any]) -> Dict:
        """
        Compute aggregate research velocity statistics.
        Used for Table 5 in the paper.

        Returns:
            avg_velocity:    Mean closure rate across all snapshots
            peak_velocity:   Maximum closure rate achieved
            peak_at_order:   Which paper caused the peak (landmark paper indicator)
            trend:           "increasing" | "decreasing" | "stable" | "insufficient_data"
            total_closed:    Total gaps closed across all papers
            total_opened:    Total gaps opened across all papers
            net_gap_change:  opened - closed (positive = field getting more complex)
        """
        if not snapshots:
            return {
                "avg_velocity":   0.0,
                "peak_velocity":  0.0,
                "peak_at_order":  0,
                "trend":          "insufficient_data",
                "total_closed":   0,
                "total_opened":   0,
                "net_gap_change": 0,
            }

        velocities    = [s.velocity for s in snapshots]
        closure_rates = [s.closure_rate for s in snapshots]

        avg_velocity  = sum(closure_rates) / len(closure_rates)
        peak_velocity = max(closure_rates)
        peak_at_order = next(
            (s.snapshot_order for s in snapshots if s.closure_rate == peak_velocity),
            0,
        )

        total_closed = sum(s.gaps_closed for s in snapshots)
        total_opened = sum(s.gaps_opened for s in snapshots)

        # Trend: compare first half vs second half average velocity
        if len(snapshots) >= 4:
            mid   = len(snapshots) // 2
            first = sum(s.closure_rate for s in snapshots[:mid]) / mid
            second = sum(s.closure_rate for s in snapshots[mid:]) / (len(snapshots) - mid)
            if second > first * 1.1:
                trend = "increasing"
            elif second < first * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "avg_velocity":   round(avg_velocity,  4),
            "peak_velocity":  round(peak_velocity, 4),
            "peak_at_order":  peak_at_order,
            "trend":          trend,
            "total_closed":   total_closed,
            "total_opened":   total_opened,
            "net_gap_change": total_opened - total_closed,
        }