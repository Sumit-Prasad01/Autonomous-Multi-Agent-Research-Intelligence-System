"""
groq_limiter.py — Centralized Groq API rate limit manager
Tracks token usage and enforces per-minute budget across all agents.
Prevents TPM exhaustion by spacing calls based on actual token consumption.

Groq free tier limits:
  llama-3.1-8b-instant:      6,000 TPM
  llama-3.3-70b-versatile:   6,000 TPM
  openai/gpt-oss-120b:       6,000 TPM (estimated)

Dev tier limits (if upgraded):
  All models:                ~30,000 TPM
"""
from __future__ import annotations

import asyncio
import time
import threading
from typing import Dict, Optional

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── Per-model token budgets (conservative — 80% of actual limit) ──────────────
_MODEL_TPM: Dict[str, int] = {
    "llama-3.1-8b-instant":           4800,
    "llama-3.3-70b-versatile":        4800,
    "llama3-70b-8192":                4800,
    "openai/gpt-oss-120b":            4800,
    "openai/gpt-oss-20b":             4800,
    "meta-llama/llama-4-scout-17b-16e-instruct": 4800,
}
_DEFAULT_TPM = 4800

# ── Estimated tokens per stage (input + output) ───────────────────────────────
STAGE_TOKENS: Dict[str, int] = {
    "extraction":          2000,
    "summarization":       1500,
    "critic":              2000,
    "critic_refine":       1500,
    "triples":             2500,
    "gap_detection":       2500,
    "comparison":          2000,
    "lit_review_themes":   1000,
    "lit_review_generate": 2500,
    "qa":                  1500,
}


class GroqRateLimiter:
    """
    Token bucket rate limiter for Groq API.
    Tracks rolling 60-second token window per model.
    Thread-safe — shared across all agents.
    """

    def __init__(self):
        self._lock         = threading.Lock()
        self._usage:  Dict[str, list] = {}  # model → [(timestamp, tokens), ...]
        self._window  = 60.0  # seconds

    def _clean_window(self, model: str) -> None:
        """Remove usage entries older than the rolling window."""
        now     = time.monotonic()
        cutoff  = now - self._window
        entries = self._usage.get(model, [])
        self._usage[model] = [(t, tok) for t, tok in entries if t > cutoff]

    def _used_tokens(self, model: str) -> int:
        """Tokens used in current window."""
        self._clean_window(model)
        return sum(tok for _, tok in self._usage.get(model, []))

    def _budget(self, model: str) -> int:
        return _MODEL_TPM.get(model, _DEFAULT_TPM)

    def record(self, model: str, tokens: int) -> None:
        """Record actual token usage after a Groq call."""
        with self._lock:
            self._usage.setdefault(model, []).append(
                (time.monotonic(), tokens)
            )

    def wait_needed(self, model: str, estimated_tokens: int) -> float:
        """
        Calculate how many seconds to wait before making a call.
        Returns 0.0 if budget allows immediate call.
        """
        with self._lock:
            used     = self._used_tokens(model)
            budget   = self._budget(model)
            combined = used + estimated_tokens

            if combined <= budget:
                return 0.0

            # find oldest entry to determine when window frees up
            entries = self._usage.get(model, [])
            if not entries:
                return 0.0

            oldest_ts = entries[0][0]
            now       = time.monotonic()
            wait      = max(0.0, self._window - (now - oldest_ts) + 1.0)
            return wait

    async def wait_for_budget(
        self,
        model:            str,
        stage:            str,
        estimated_tokens: Optional[int] = None,
    ) -> None:
        """
        Async wait until token budget allows this call.
        Call this BEFORE every Groq API call.
        """
        tokens = estimated_tokens or STAGE_TOKENS.get(stage, 2000)
        wait   = self.wait_needed(model, tokens)

        if wait > 0:
            logger.info(
                f"[GROQ LIMITER] stage={stage} model={model} "
                f"waiting {wait:.1f}s for token budget"
            )
            await asyncio.sleep(wait)

        # optimistically record — will be corrected by actual usage
        self.record(model, tokens)


# ── Singleton ─────────────────────────────────────────────────────────────────
_limiter = GroqRateLimiter()


def get_limiter() -> GroqRateLimiter:
    return _limiter


async def wait_for_groq(model: str, stage: str) -> None:
    """
    Convenience function — call before every Groq API invocation.

    Usage:
        from src.research_intelligence_system.utils.groq_limiter import wait_for_groq
        await wait_for_groq(state["llm_id"], "triples")
        response = llm.invoke(prompt)
    """
    await _limiter.wait_for_budget(model, stage)