"""
summarizer_agent.py — Two-stage summarization
Stage 1: BART extracts section summaries (GPU, fast, grounded)
Stage 2: Groq/Llama synthesizes a single detailed comprehensive summary
This is novel — combines local deep learning + LLM reasoning
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from src.research_intelligence_system.core.groq_limiter import wait_for_groq
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
                    from transformers import BartForConditionalGeneration, BartTokenizer
                    import torch
                    cls._device    = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"Loading BART on {cls._device.upper()} …")
                    cls._tokenizer = BartTokenizer.from_pretrained(
                        "facebook/bart-large-cnn",
                        local_files_only=True,
                    )
                    cls._model_obj = BartForConditionalGeneration.from_pretrained(
                        "facebook/bart-large-cnn",
                        local_files_only=True,
                    ).to(cls._device)
                    cls._model_obj.eval()
                    cls._instance  = True
                    logger.info("BART ready.")
        return cls


# ── Config ────────────────────────────────────────────────────────────────────
_SECTION_MAX_TOKENS = {
    "abstract":     150,
    "introduction": 150,
    "methodology":  200,
    "results":      200,
    "conclusion":   150,
}
_MIN_INPUT_CHARS = 100

_SYNTHESIS_PROMPT = """You are an expert research paper analyst.
Based on the BART-extracted section summaries and key entities below, write a single comprehensive summary of this research paper.

Section Summaries (extracted by BART):
Abstract:    {abstract}
Methodology: {methodology}
Results:     {results}
Conclusion:  {conclusion}

Key Entities:
Models:   {models}
Datasets: {datasets}
Metrics:  {metrics}
Methods:  {methods}

Write a detailed 400-500 word comprehensive summary that covers:
1. Problem Statement: What problem does this paper solve?
2. Proposed Approach: What method/model/technique is proposed?
3. Key Results: What are the main results with specific numbers/metrics?
4. Main Contributions: What are the 2-3 key contributions?
5. Limitations: Any limitations mentioned?

Write in academic style. Be specific — include model names, dataset names, and metric values.
Return ONLY the summary text, no JSON, no headers."""


# ── Core BART summarization ───────────────────────────────────────────────────
def _bart_summarize(text: str, max_tokens: int, min_tokens: int = 30) -> str:
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


def _summarize_section_task(args: tuple) -> tuple:
    section, text, max_tok = args
    summary = _bart_summarize(text, max_tokens=max_tok)
    return section, summary


# ── State ─────────────────────────────────────────────────────────────────────
class SummaryState(TypedDict):
    paper_id:            str
    llm_id:              str
    sections:            Dict[str, str]
    entities:            Dict[str, Any]
    section_summaries:   Dict[str, str]   # BART outputs
    comprehensive:       str              # final LLM synthesis
    retry_count:         int
    error:               str


# ── Node 1: BART section summarization ───────────────────────────────────────
def _bart_sections_node(state: SummaryState) -> SummaryState:
    """Stage 1 — BART extracts section summaries in parallel."""
    logger.info(f"[SUMMARIZER] paper_id={state['paper_id']}")

    sections      = state["sections"]
    section_order = ["abstract", "introduction", "methodology", "results", "conclusion"]

    tasks = [
        (s, sections[s][:3000], _SECTION_MAX_TOKENS.get(s, 150))
        for s in section_order
        if sections.get(s) and len(sections.get(s, "")) >= _MIN_INPUT_CHARS
    ]

    if not tasks:
        return {**state, "section_summaries": {}, "error": "no valid sections"}

    try:
        with ThreadPoolExecutor(max_workers=min(3, len(tasks))) as ex:
            results = list(ex.map(_summarize_section_task, tasks))

        section_summaries = dict(results)
        logger.info(f"[SUMMARIZER] {len(section_summaries)} sections summarized in parallel")
        return {**state, "section_summaries": section_summaries, "error": ""}

    except Exception as e:
        logger.warning(f"[SUMMARIZER] BART failed: {e}")
        return {**state, "error": str(e), "retry_count": state["retry_count"] + 1}


# ── Node 2: LLM comprehensive synthesis ──────────────────────────────────────
def _llm_synthesis_node(state: SummaryState) -> SummaryState:
    """Stage 2 — Groq/Llama synthesizes one detailed comprehensive summary."""
    logger.info(f"[SUMMARIZER] synthesizing comprehensive summary paper_id={state['paper_id']}")

    section_summaries = state.get("section_summaries", {})
    entities          = state.get("entities", {})

    # fallback — if BART failed use raw sections
    sections = state["sections"]

    prompt = _SYNTHESIS_PROMPT.format(
        abstract    = section_summaries.get("abstract", sections.get("abstract", ""))[:500],
        methodology = section_summaries.get("methodology", sections.get("methodology", ""))[:500],
        results     = section_summaries.get("results", sections.get("results", ""))[:500],
        conclusion  = section_summaries.get("conclusion", sections.get("conclusion", ""))[:400],
        models      = ", ".join(entities.get("models",   [])[:8]),
        datasets    = ", ".join(entities.get("datasets", [])[:8]),
        metrics     = ", ".join(entities.get("metrics",  [])[:8]),
        methods     = ", ".join(entities.get("methods",  [])[:8]),
    )

    try:
        llm          = ChatGroq(model=state["llm_id"], temperature=0.1)
        response     = llm.invoke(prompt)
        comprehensive = response.content.strip()

        logger.info(f"[SUMMARIZER] comprehensive summary: {len(comprehensive)} chars")
        return {**state, "comprehensive": comprehensive, "error": ""}

    except Exception as e:
        logger.warning(f"[SUMMARIZER] LLM synthesis failed: {e}")
        # fallback to combined BART outputs
        fallback = " ".join(section_summaries.values())
        return {**state, "comprehensive": fallback, "error": ""}


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry(state: SummaryState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "bart_sections"
    return "llm_synthesis"


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph() -> Any:
    graph = StateGraph(SummaryState)

    graph.add_node("bart_sections", _bart_sections_node)
    graph.add_node("llm_synthesis", _llm_synthesis_node)

    graph.set_entry_point("bart_sections")
    graph.add_conditional_edges("bart_sections", _should_retry, {
        "bart_sections": "bart_sections",
        "llm_synthesis": "llm_synthesis",
    })
    graph.add_edge("llm_synthesis", END)

    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
class SummarizerAgent:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def summarize(
        self,
        paper_id: str,
        sections: Dict[str, str],
        entities: Dict[str, Any] = {},
    ) -> Dict[str, str]:
        """
        Two-stage summarization:
        1. BART → section summaries (grounded, local GPU)
        2. Groq/Llama → single comprehensive summary (400-500 words)

        Returns:
          section_summaries: {abstract, methodology, results, conclusion}
          comprehensive:     single detailed summary
          overall:           alias for comprehensive (backward compat)
        """
        await wait_for_groq(self.llm_id, "summarization")

        loop = asyncio.get_running_loop()

        initial_state: SummaryState = {
            "paper_id":          paper_id,
            "llm_id":            self.llm_id,
            "sections":          sections,
            "entities":          entities,
            "section_summaries": {},
            "comprehensive":     "",
            "retry_count":       0,
            "error":             "",
        }

        result = await loop.run_in_executor(
            _POOL, lambda: _graph.invoke(initial_state)
        )

        section_summaries = result.get("section_summaries", {})
        comprehensive     = result.get("comprehensive", "")

        if not comprehensive:
            logger.warning(f"[SUMMARIZER] empty result for paper_id={paper_id}")
            comprehensive = " ".join(section_summaries.values()) or "Summary generation failed."

        # return both for storage and display
        return {
            **section_summaries,          # abstract, methodology, results, conclusion
            "overall":        comprehensive,
            "comprehensive":  comprehensive,
        }