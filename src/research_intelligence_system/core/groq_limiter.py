"""
groq_limiter.py — Centralized Groq API rate limit manager

── WHY notify_complete() EXISTS ─────────────────────────────────────────────
The inter-stage gap must be measured from when the LAST LLM CALL COMPLETED,
not from when we started waiting. Without this, the sequence is:

  T=0:   orchestrator calls wait_for_groq("comparison") → _last_call = T=0
  T=0-5: comparison LLM runs (inside executor, takes 5s)
  T=5:   executor returns
  T=5.03: lit_review _sync_wait fires → gap_wait = 2.5 - (5.03 - 0) = 0 → NO SLEEP → 429

With notify_complete() called at T=5:
  T=5:   executor returns → orchestrator calls notify_complete() → _last_call = T=5
  T=5.03: lit_review _sync_wait fires → gap_wait = 2.5 - (5.03 - 5) = 2.47s → SLEEP ✅

Call pattern:
  await wait_for_groq(model, stage)          # before dispatch
  result = await loop.run_in_executor(...)   # agent runs LLM inside
  notify_groq_complete()                     # after executor returns ← required
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import time
import threading
from typing import Dict, Optional

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── Per-model TPM budgets (85% of free-tier limit) ────────────────────────────
_MODEL_TPM: Dict[str, int] = {
    "llama-3.1-8b-instant":                         5100,
    "llama-3.3-70b-versatile":                      5100,
    "llama3-70b-8192":                              5100,
    "openai/gpt-oss-120b":                          5100,
    "meta-llama/llama-4-scout-17b-16e-instruct":    5100,
}
_DEFAULT_TPM = 5100

# ── Token estimates per stage ─────────────────────────────────────────────────
STAGE_TOKENS: Dict[str, int] = {
    "extraction":          1200,
    "summarization":        900,
    "critic":              1100,
    "critic_refine":        900,
    "triples":             2500,
    "gap_detection":       1900,
    "comparison":          1500,
    "lit_review_themes":    800,
    "lit_review_generate": 2000,
    "qa":                  1000,
}

# ── Minimum gap between consecutive LLM call COMPLETIONS (not starts) ─────────
_MIN_INTER_STAGE_GAP: float = 2.5


class GroqRateLimiter:

    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self._usage:     Dict[str, list] = {}
        self._window     = 60.0
        self._last_call: float = 0.0   # set AFTER each LLM call completes

    def _clean_window(self, model: str) -> None:
        cutoff = time.monotonic() - self._window
        self._usage[model] = [
            (t, tok) for t, tok in self._usage.get(model, []) if t > cutoff
        ]

    def _used_tokens(self, model: str) -> int:
        self._clean_window(model)
        return sum(tok for _, tok in self._usage.get(model, []))

    def _budget(self, model: str) -> int:
        return _MODEL_TPM.get(model, _DEFAULT_TPM)

    def reserve(self, model: str, tokens: int) -> None:
        """Reserve tokens before an LLM call."""
        with self._lock:
            entries = self._usage.setdefault(model, [])
            entries.append((time.monotonic(), tokens))
            entries.sort(key=lambda x: x[0])

    def notify_complete(self) -> None:
        """Update _last_call to NOW. Call immediately after llm.invoke() returns."""
        with self._lock:
            self._last_call = time.monotonic()

    def wait_needed(self, model: str, estimated_tokens: int) -> float:
        with self._lock:
            now    = time.monotonic()
            used   = self._used_tokens(model)
            budget = self._budget(model)

            token_wait = 0.0
            if used + estimated_tokens > budget:
                entries = self._usage.get(model, [])
                if entries:
                    token_wait = max(0.0, self._window - (now - entries[0][0]) + 1.0)

            gap_wait = max(0.0, _MIN_INTER_STAGE_GAP - (now - self._last_call))
            return max(token_wait, gap_wait)

    async def wait_for_budget(self, model: str, stage: str,
                               estimated_tokens: Optional[int] = None) -> None:
        tokens = estimated_tokens or STAGE_TOKENS.get(stage, 2000)
        wait   = self.wait_needed(model, tokens)
        if wait > 0:
            logger.info(f"[GROQ LIMITER] stage={stage} waiting {wait:.1f}s")
            await asyncio.sleep(wait)
        self.reserve(model, tokens)


# ── Singleton ─────────────────────────────────────────────────────────────────
_limiter = GroqRateLimiter()


def get_limiter() -> GroqRateLimiter:
    return _limiter


# ── Public API ────────────────────────────────────────────────────────────────

async def wait_for_groq(model: str, stage: str) -> None:
    """Call BEFORE dispatching to executor. Pair with notify_groq_complete() after."""
    await _limiter.wait_for_budget(model, stage)


def notify_groq_complete() -> None:
    """
    Call AFTER run_in_executor (or llm.invoke) returns.
    This is what makes gap enforcement work correctly.

    Without this, _last_call is set at the START of the wait (before the LLM runs),
    so by the time the next stage fires, the gap appears to have already elapsed.
    """
    _limiter.notify_complete()


def sync_wait_for_groq(model: str, stage: str) -> None:
    """
    For use inside LangGraph nodes (sync executor threads).
    Call BEFORE llm.invoke(). Pair with notify_groq_complete() after.
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(wait_for_groq(model, stage))
    finally:
        loop.close()