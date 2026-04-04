"""
summarizer_agent.py — LangGraph-based summarization agent
Uses facebook/bart-large-cnn locally on GPU
Summarizes each section independently then generates overall summary
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph
from transformers import BartForConditionalGeneration, BartTokenizer

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="summarizer")


# ── Singleton BART model ──────────────────────────────────────────────────────
class _BARTModel:
    _instance  = None
    _tokenizer = None
    _model_obj = None
    _device    = "cpu"
    _lock      = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    import torch
                    cls._device    = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"Loading BART on {cls._device.upper()} …")
                    cls._tokenizer = BartTokenizer.from_pretrained(
                        "facebook/bart-large-cnn",
                        local_files_only = True
                        )
                    cls._model_obj = BartForConditionalGeneration.from_pretrained(
                        "facebook/bart-large-cnn",
                        local_files_only = True
                    ).to(cls._device)
                    cls._model_obj.eval()
                    cls._instance  = True
                    logger.info("BART ready.")
        return cls


# ── Config ────────────────────────────────────────────────────────────────────
_SECTION_MAX_TOKENS = {
    "abstract":     130,
    "introduction": 150,
    "methodology":  200,
    "results":      200,
    "conclusion":   130,
    "overall":      250,
}
_MIN_INPUT_CHARS = 100


# ── State ─────────────────────────────────────────────────────────────────────
class SummaryState(TypedDict):
    paper_id:    str
    sections:    Dict[str, str]
    summaries:   Dict[str, str]
    retry_count: int
    error:       str


# ── Core summarization ────────────────────────────────────────────────────────
def _summarize_text(text: str, max_tokens: int, min_tokens: int = 30) -> str:
    if len(text) < _MIN_INPUT_CHARS:
        return text
    import torch
    m      = _BARTModel.get()
    inputs = m._tokenizer(
        text[:3000],
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(m._device)
    with torch.no_grad():
        ids = m._model_obj.generate(
            inputs["input_ids"],
            max_length    = max_tokens,
            min_length    = min_tokens,
            length_penalty= 2.0,
            num_beams     = 4,
            early_stopping= True,
        )
    return m._tokenizer.decode(ids[0], skip_special_tokens=True)

# ── Nodes ─────────────────────────────────────────────────────────────────────
def _summarize_section_task(args: tuple) -> tuple:
    """Single section summarization — runs in threadpool."""
    section, text, max_tok = args
    summary = _summarize_text(text, max_tokens=max_tok)
    return section, summary


def _summarize_sections_node(state: SummaryState) -> SummaryState:
    """Summarize all sections in parallel using BART."""
    logger.info(f"[SUMMARIZER] paper_id={state['paper_id']}")

    sections      = state["sections"]
    section_order = ["abstract", "introduction", "methodology", "results", "conclusion"]

    tasks = [
        (s, sections[s][:3000], _SECTION_MAX_TOKENS.get(s, 150))
        for s in section_order
        if sections.get(s) and len(sections.get(s, "")) >= _MIN_INPUT_CHARS
    ]

    if not tasks:
        return {**state, "summaries": {}, "error": "no valid sections"}

    try:
        # parallel summarization — all sections at once on GPU
        with ThreadPoolExecutor(max_workers=min(3, len(tasks))) as ex:
            results = list(ex.map(_summarize_section_task, tasks))

        summaries = dict(results)
        logger.info(f"[SUMMARIZER] {len(summaries)} sections summarized in parallel")
        return {**state, "summaries": summaries, "error": ""}

    except Exception as e:
        logger.warning(f"[SUMMARIZER] parallel summarization failed: {e}")
        return {**state, "error": str(e), "retry_count": state["retry_count"] + 1}


def _generate_overall_node(state: SummaryState) -> SummaryState:
    """Combine section summaries into one overall summary."""
    summaries = state.get("summaries", {})
    if not summaries:
        return state

    try:
        combined = " ".join([
            summaries.get("abstract", ""),
            summaries.get("methodology", ""),
            summaries.get("results", ""),
            summaries.get("conclusion", ""),
        ]).strip()

        if len(combined) >= _MIN_INPUT_CHARS:
            overall = _summarize_text(
                combined,
                max_tokens=_SECTION_MAX_TOKENS["overall"],
                min_tokens=50,
            )
            summaries["overall"] = overall

        logger.info(f"[SUMMARIZER] done — {len(summaries)} sections summarized")
        return {**state, "summaries": summaries}

    except Exception as e:
        logger.warning(f"[SUMMARIZER] overall summary failed: {e}")
        summaries["overall"] = summaries.get("abstract", "")
        return {**state, "summaries": summaries}


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry(state: SummaryState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "summarize_sections"
    return "generate_overall"


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph() -> Any:
    graph = StateGraph(SummaryState)

    graph.add_node("summarize_sections", _summarize_sections_node)
    graph.add_node("generate_overall",   _generate_overall_node)

    graph.set_entry_point("summarize_sections")
    graph.add_conditional_edges("summarize_sections", _should_retry, {
        "summarize_sections": "summarize_sections",
        "generate_overall":   "generate_overall",
    })
    graph.add_edge("generate_overall", END)

    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
class SummarizerAgent:

    async def summarize(
        self,
        paper_id: str,
        sections: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Run summarization graph.
        Returns dict of section summaries + overall.
        """
        loop = asyncio.get_running_loop()

        initial_state: SummaryState = {
            "paper_id":    paper_id,
            "sections":    sections,
            "summaries":   {},
            "retry_count": 0,
            "error":       "",
        }

        result = await loop.run_in_executor(
            _POOL, lambda: _graph.invoke(initial_state)
        )

        summaries = result.get("summaries", {})
        if not summaries:
            logger.warning(f"[SUMMARIZER] empty result for paper_id={paper_id}")
            return {"overall": "Summary generation failed."}

        return summaries