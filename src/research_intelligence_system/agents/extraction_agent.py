from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.research_intelligence_system.core.groq_limiter import wait_for_groq
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class ExtractionState(TypedDict):
    paper_id:    str
    llm_id:      str
    sections:    Dict[str, str]
    entities:    Dict[str, Any]
    retry_count: int
    error:       str


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    _EXTRACTION_MODEL = "llama-3.1-8b-instant"
    return ChatGroq(model=_EXTRACTION_MODEL, temperature=0)


# ── Math / formula pre-filter ─────────────────────────────────────────────────
_MATH_PATTERN = re.compile(
    r'(\\[a-zA-Z]+\{.*?\}|'   # LaTeX commands \frac{}, \sum{}
    r'\$.*?\$|'                 # inline math $...$
    r'[A-Za-z]\s*=\s*[\d.]+|'  # simple assignments x = 0.5
    r'(?:\d+\s*[\+\-\*/]\s*){2,})',  # repeated arithmetic
    re.DOTALL,
)

def _clean_section(text: str, max_chars: int) -> str:
    """Remove math noise and truncate."""
    text = _MATH_PATTERN.sub(' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()[:max_chars]


# ── Regex fallback ────────────────────────────────────────────────────────────
def _regex_fallback(sections: Dict[str, str]) -> Dict[str, Any]:
    """
    Improved fallback — extracts common paper-specific patterns.
    Handles: CamelCase names, hyphenated names, abbreviations.
    """
    full_text = " ".join(sections.values())

    # Pattern 1: CamelCase names (TurboQuant, RabitQ, ViT-B)
    camel_case = re.findall(
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]*)+\b|\b[A-Z]{2,}[a-z]+\b', 
        full_text
    )

    # Pattern 2: Hyphenated technical names (ViT-B/32, Lloyd-Max, k-means)
    hyphenated = re.findall(
        r'\b[A-Z][a-zA-Z]*(?:-[A-Z0-9][a-zA-Z0-9]*)+\b',
        full_text
    )

    # Pattern 3: Known dataset patterns (MNIST, ImageNet, SQuAD, GLUE)
    datasets = re.findall(
        r'\b(?:ImageNet|CIFAR|MNIST|SQuAD|GLUE|WMT|DBpedia|'
        r'OpenImages|COCO|LibriSpeech|CommonCrawl|C4|WebText)\b',
        full_text
    )

    # Pattern 4: Metric patterns (accuracy, F1, BLEU, Recall@K)
    metrics = re.findall(
        r'\b(?:accuracy|F1|BLEU|ROUGE|NDCG|MRR|Recall@\d+|'
        r'precision|perplexity|MSE|RMSE|AUC|mAP)\b',
        full_text, re.IGNORECASE
    )

    # Pattern 5: Near-keyword extraction for models
    model_context = re.findall(
        r'(?:propose|present|introduce|develop|our)\s+([A-Z][a-zA-Z0-9\-]+)',
        full_text
    )

    # combine and deduplicate
    all_models = list(dict.fromkeys(
        camel_case[:8] + hyphenated[:5] + model_context[:3]
    ))

    return {
        "models":          all_models[:10],
        "datasets":        list(dict.fromkeys(datasets))[:8],
        "metrics":         list(dict.fromkeys(metrics))[:8],
        "methods":         list(dict.fromkeys(hyphenated[:3]))[:5],
        "tasks":           [],
        "hyperparameters": {},
        "authors":         [],
        "year":            "",
    }


# ── Prompt ────────────────────────────────────────────────────────────────────
_EXTRACTION_PROMPT = """You are a scientific paper entity extractor.
Extract ALL named entities from the paper sections below.
This works for ANY scientific domain — not just ML/NLP.

Return ONLY valid JSON with this exact structure:
{{
  "models": ["model, algorithm, or framework names"],
  "datasets": ["dataset, corpus, benchmark, or data source names"],
  "metrics": ["evaluation metrics or performance indicators"],
  "methods": ["methods, techniques, or analytical approaches"],
  "tasks": ["research tasks, objectives, or problems"],
  "hyperparameters": {{}},
  "authors": [],
  "year": ""
}}

If no entities exist for a field, return an empty list [].
Include domain-specific terms — e.g. for biology: genes, proteins, assays.
For finance: indices, instruments, models. For NLP: model architectures, corpora.

Paper sections:
ABSTRACT: {abstract}
INTRODUCTION: {introduction}
METHODOLOGY: {methodology}
RESULTS: {results}
CONCLUSION: {conclusion}

IMPORTANT: Return ONLY the JSON object above. Start with {{ and end with }}.
No markdown, no explanation, no code blocks."""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _extract_node(state: ExtractionState) -> ExtractionState:
    """Call LLM to extract entities from paper sections."""
    logger.info(f"[EXTRACTION] paper_id={state['paper_id']} attempt={state['retry_count']+1}")

    sections = state["sections"]

    # clean + truncate each section
    prompt = _EXTRACTION_PROMPT.format(
        abstract     = _clean_section(sections.get("abstract",     ""), 1500),
        introduction = _clean_section(sections.get("introduction", ""), 1000),
        methodology  = _clean_section(sections.get("methodology",  ""), 1500),
        results      = _clean_section(sections.get("results",      ""), 1500),
        conclusion   = _clean_section(sections.get("conclusion",   ""),  800),
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")

        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
        # Fix single quotes → double quotes
        cleaned = re.sub(r"(?<=[{,])\s*'([^']+)'\s*:", r' "\1":', cleaned)
        cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
        entities = json.loads(cleaned)

        count = sum(len(v) if isinstance(v, list) else 1 for v in entities.values())
        logger.info(f"[EXTRACTION] extracted {count} entities")

        return {**state, "entities": entities, "error": ""}

    except Exception as e:
        logger.warning(f"[EXTRACTION] failed attempt {state['retry_count']+1}: {e}")
        return {**state, "error": str(e), "retry_count": state["retry_count"] + 1}


def _validate_node(state: ExtractionState) -> ExtractionState:
    """
    Validate + clean extracted entities.
    Falls back to regex extraction if entities are empty after retries.
    """
    entities = state.get("entities", {})
    required = ["models", "datasets", "metrics", "methods", "tasks"]

    # if completely empty after all retries — use regex fallback
    if not entities or all(not entities.get(k) for k in required):
        logger.warning(f"[EXTRACTION] LLM failed — using regex fallback paper_id={state['paper_id']}")
        entities = _regex_fallback(state["sections"])

    # ensure all required keys exist and are lists
    for key in required:
        if key not in entities:
            entities[key] = []
        if not isinstance(entities[key], list):
            entities[key] = [str(entities[key])]
        # clean empty strings
        entities[key] = [e.strip() for e in entities[key] if str(e).strip()]
        # deduplicate case-insensitive
        seen = set()
        deduped = []
        for e in entities[key]:
            if e.lower() not in seen:
                seen.add(e.lower())
                deduped.append(e)
        entities[key] = deduped

    if "hyperparameters" not in entities:
        entities["hyperparameters"] = {}
    if "authors" not in entities:
        entities["authors"] = []
    if "year" not in entities:
        entities["year"] = ""

    logger.info(
        f"[EXTRACTION] validated ✓ "
        f"models={len(entities['models'])} "
        f"datasets={len(entities['datasets'])} "
        f"metrics={len(entities['metrics'])}"
    )
    return {**state, "entities": entities}


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry(state: ExtractionState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "extract"
    return "validate"


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph() -> Any:
    graph = StateGraph(ExtractionState)

    graph.add_node("extract",  _extract_node)
    graph.add_node("validate", _validate_node)

    graph.set_entry_point("extract")
    graph.add_conditional_edges("extract", _should_retry, {
        "extract":  "extract",
        "validate": "validate",
    })
    graph.add_edge("validate", END)

    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
class ExtractionAgent:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def extract(
        self,
        paper_id: str,
        sections: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Run entity extraction pipeline.
        Domain-agnostic — works for ML, biology, finance, physics etc.
        Falls back to regex extraction if LLM fails twice.
        Returns entities dict.
        """
        import asyncio

        await wait_for_groq(self.llm_id, "extraction")

        loop = asyncio.get_running_loop()

        initial_state: ExtractionState = {
            "paper_id":    paper_id,
            "llm_id":      self.llm_id,
            "sections":    sections,
            "entities":    {},
            "retry_count": 0,
            "error":       "",
        }

        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        entities = result.get("entities", {})
        if not entities:
            logger.warning(f"[EXTRACTION] empty result paper_id={paper_id}")
            return {
                "models": [], "datasets": [], "metrics": [],
                "methods": [], "tasks": [], "hyperparameters": {},
                "authors": [], "year": "",
            }

        return entities


def get_sections_from_chunks(chunks) -> Dict[str, str]:
    """
    Group parsed chunks by section into a dict.
    Input: list of LangChain Documents with metadata.section
    """
    sections: Dict[str, List[str]] = {
        "abstract": [], "introduction": [], "methodology": [],
        "results":  [], "conclusion":   [], "body":        [],
    }
    for chunk in chunks:
        section = chunk.metadata.get("section", "body")
        sections.setdefault(section, []).append(chunk.page_content)

    return {k: " ".join(v)[:3000] for k, v in sections.items() if v}