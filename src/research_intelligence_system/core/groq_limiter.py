"""
groq_limiter.py — Centralized Groq API rate limit manager

── HOW THIS WORKS ───────────────────────────────────────────────────────────
Two-layer protection against Groq 429s:

Layer 1 — Token bucket (60s rolling window):
  Every llm.invoke() call must call sync_wait_for_groq() BEFORE invoking.
  This reserves tokens and waits if the 60s budget is exhausted.

Layer 2 — Inter-stage gap (4s minimum between call COMPLETIONS):
  Every llm.invoke() call must call notify_groq_complete() AFTER it returns.
  This updates _last_call to the actual completion time so the next call
  measures the gap from when the LLM finished, not when we started waiting.

── TOKEN ESTIMATE CALIBRATION ───────────────────────────────────────────────
STAGE_TOKENS values are reserved before each LLM call. Over-estimating fills
the bucket faster and causes longer token_wait delays.

Calibration basis (observed on gpt-oss-120b):
  triples: 5000-char input (~1200 input tokens) + 12–20 triples output (~400 tokens)
           = ~1600 actual. Reserve 1800 (12% buffer). Was 2500 (56% over).
  The 700-token reduction saves ~8s off the token_wait at lit_review stage.

── CALL PATTERN ─────────────────────────────────────────────────────────────
Inside LangGraph nodes (sync executor thread):
    sync_wait_for_groq(state["llm_id"], "stage_name")
    try:
        response = llm.invoke(prompt)
        notify_groq_complete()
    except Exception as e:
        notify_groq_complete()   # always notify, even on failure
        raise

In orchestrator (for agents throttling via public method):
    result = await SomeAgent().method(...)
    notify_groq_complete()       # after await returns
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

# ── Token estimates per stage (calibrated from observed usage) ────────────────
STAGE_TOKENS: Dict[str, int] = {
    "extraction":          1200,
    "summarization":        900,
    "critic":              1100,
    "critic_refine":        900,
    "triples":             1800,   # was 2500; actual ~1600, reserve 1800 (12% buffer)
    "gap_detection":       1900,
    "comparison":          1500,
    "lit_review_themes":    800,
    "lit_review_generate": 2000,
    "qa":                  1000,
}

_MIN_INTER_STAGE_GAP: float = 4.0


class GroqRateLimiter:

    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self._usage:     Dict[str, list] = {}
        self._window     = 60.0
        self._last_call: float = 0.0

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
        with self._lock:
            entries = self._usage.setdefault(model, [])
            entries.append((time.monotonic(), tokens))
            entries.sort(key=lambda x: x[0])

    def notify_complete(self) -> None:
        """Set _last_call = NOW. Call immediately after llm.invoke() returns."""
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

    async def wait_for_budget(
        self,
        model:            str,
        stage:            str,
        estimated_tokens: Optional[int] = None,
    ) -> None:
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


async def wait_for_groq(model: str, stage: str) -> None:
    """Async: call BEFORE llm.invoke(). Pair with notify_groq_complete() after."""
    await _limiter.wait_for_budget(model, stage)


def notify_groq_complete() -> None:
    """Call AFTER llm.invoke() — in both success AND exception paths."""
    _limiter.notify_complete()


def sync_wait_for_groq(model: str, stage: str) -> None:
    """Sync version for LangGraph nodes in executor threads."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(wait_for_groq(model, stage))
    finally:
        loop.close()