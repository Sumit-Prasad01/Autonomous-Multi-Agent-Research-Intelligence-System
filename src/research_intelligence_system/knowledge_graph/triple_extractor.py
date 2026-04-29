"""
triple_extractor.py — LangGraph-based knowledge triple extraction agent
Extracts (subject, relation, object) triples from paper text.
"""
from __future__ import annotations

import json
import re
import asyncio
from typing import Any, Dict, List, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from src.research_intelligence_system.core.groq_limiter import wait_for_groq

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class TripleState(TypedDict):
    paper_id:    str
    llm_id:      str
    sections:    Dict[str, str]
    entities:    Dict[str, Any]
    triples:     List[Dict[str, str]]
    retry_count: int
    error:       str


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=4000)


# ── LaTeX subscript artifact cleaner ─────────────────────────────────────────
# When PDFs contain LaTeX like "TurboQuant_{prod}", PDF parsers strip the braces
# and subscript marker, leaving "TurboQuantprod". This contaminates the entity
# list passed to the triple extraction prompt — the LLM sees "TurboQuantprod"
# instead of "TurboQuant" and can't match it to entities in the paper text.
_MATH_SUBSCRIPTS: frozenset = frozenset({
    'prod', 'mse', 'mae', 'min', 'max', 'opt', 'est', 'ref',
    'val', 'err', 'acc', 'lat', 'mem', 'out', 'init',
})


def _clean_entity_name(name: str) -> str:
    """
    Remove LaTeX subscript artifacts from entity names before triple extraction.

    Examples:
      "TurboQuantprod" → "TurboQuant"
      "TurboQuantmse"  → "TurboQuant"
      "VQprod"         → "VQ"
      "ResNet50"       → "ResNet50"   (unchanged — digit suffix, not subscript)
      "ViTBase"        → "ViTBase"    ('Base' not in math subscript set)
      "BERT"           → "BERT"       (unchanged)
    """
    if not name or not name[0].isupper():
        return name

    for suffix in sorted(_MATH_SUBSCRIPTS, key=len, reverse=True):
        if (name.lower().endswith(suffix)
                and len(name) > len(suffix) + 1):
            return name[:-len(suffix)]

    return name


def _clean_entity_list(entities: List[str]) -> List[str]:
    """Apply _clean_entity_name to a list, deduplicate, preserve order."""
    seen = set()
    result = []
    for e in entities:
        cleaned = _clean_entity_name(e)
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


# ── Text cleaning ─────────────────────────────────────────────────────────────
_MATH_NOISE = re.compile(
    r'(\\[a-zA-Z]+\{[^}]*\}|'       # LaTeX: \frac{}, \sum{}
    r'\$[^$]*\$|'                    # inline math: $...$
    r'\\\([^)]*\\\)|'                # \( ... \)
    r'(?:\d+\s*[\+\-\*/]\s*){3,}|'  # repeated arithmetic
    r'(?<!\w)(\d+\s*){6,}(?!\w)|'   # long number sequences
    r'[\+\-\*/=<>]{3,}|'            # symbol runs
    r'exp\s*\([^)]*\)\s*[\.\,]?)',   # exp(...) patterns
    re.DOTALL,
)


def _clean_text_for_triples(text: str) -> str:
    """
    Remove math formulas while preserving result table rows.
    Result rows like "Transformer (big) | 41.0 | 27.3" are kept because
    they contain at least 2 word-like tokens (Transformer, big).
    Pure symbol/number lines like "= = = =" are stripped.
    """
    text = _MATH_NOISE.sub(' ', text)
    text = re.sub(r'(\.\s*){3,}', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)

    lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        word_like = [w for w in re.split(r'[\s|,;:\-]+', stripped)
                     if sum(c.isalpha() for c in w) >= 2]
        if len(word_like) >= 2:
            lines.append(stripped)
        elif len(stripped) < 100 and any(c.isalpha() for c in stripped):
            lines.append(stripped)

    return '\n'.join(lines).strip()


# ── Prompt ────────────────────────────────────────────────────────────────────
_TRIPLE_PROMPT = """[SYSTEM] You are a scientific knowledge graph construction engine. \
Your sole output is a JSON array of knowledge triples extracted from a research paper. \
You operate across all scientific domains: ML, NLP, biology, chemistry, physics, finance, medicine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — ENTITY INVENTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pre-extracted entities — treat as ground truth:

  Models / Algorithms : {models}
  Datasets / Corpora  : {datasets}
  Metrics             : {metrics}
  Methods / Techniques: {methods}

Rules:
• Use the EXACT string from the list above.
• Only invent a name if it clearly appears in the paper AND is absent from all lists.
• Never abbreviate or expand a known entity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — RELATION SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use ONLY these 13 relation types:

  TRAINED_ON      model was trained on a dataset
  EVALUATED_ON    model was tested on a benchmark
  ACHIEVES        model reaches a result on a metric or benchmark
  USES            method incorporates a technique or component
  IMPROVES_OVER   method surpasses a baseline (no numeric comparison needed)
  PROPOSED_BY     method introduced by named authors
  APPLIED_TO      method deployed on a task or domain
  COMPARED_WITH   method compared with another (neutral)
  BASED_ON        method derived from prior architecture/theory
  REPLACES        method supersedes an older approach
  OUTPERFORMS     method beats another WITH explicit numeric evidence
  BOUNDED_BY      performance constrained by a theoretical limit
  EXTENDS         method broadens or generalises an existing method

Disambiguation:
• OUTPERFORMS requires a numeric score in the text. Otherwise use IMPROVES_OVER.
• EXTENDS = new capability added. BASED_ON = direct implementation of prior work.
• USES = one component. BASED_ON = foundational architecture.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACHIEVES — most important relation for KG utility:
  Include the numeric score in the object when it appears in the text.
  Format: "<score> <metric> on <dataset>"
  ✅ "TurboQuant ACHIEVES 41.0 BLEU on WMT 2014 EN-FR"
  ✅ "BERT ACHIEVES 93.2% F1 on SQuAD"
  ✅ "TurboQuant ACHIEVES Recall@1"   ← acceptable when no score in text

VALID subjects/objects:
  ✅ Named models, datasets, metrics, methods from entity list
  ✅ Named tasks: "machine translation", "image classification"
  ✅ ACHIEVES objects with numeric score: "28.4 BLEU on WMT 2014 EN-DE"

INVALID — reject:
  ✗ Placeholders: "our method", "the proposed approach"
  ✗ Bare numbers: "0.923", "128" (but "28.4 BLEU" is valid — has metric name)
  ✗ Math symbols: "∇θ", "σ(x)", "L2 norm"
  ✗ Overly generic: "deep learning", "neural network"

Quantity: 15–25 triples, hard stop at 25. Prioritise ACHIEVES triples.
Confidence: 1.0=verbatim, 0.9=explicit sentence, 0.85=implied, 0.75=bridged, 0.7=inferred.
Deduplication: same (subject, relation, object) lowercased → keep higher confidence only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT PAPER TEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — JSON ARRAY ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[
  {{"subject": "ModelName", "relation": "ACHIEVES", "object": "41.0 BLEU on WMT 2014 EN-FR", "confidence": 1.0}},
  {{"subject": "ModelName", "relation": "USES", "object": "multi-head attention", "confidence": 0.9}}
]"""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _extract_triples_node(state: TripleState) -> TripleState:
    logger.info(f"[TRIPLES] paper_id={state['paper_id']} attempt={state['retry_count']+1}")

    sections = state["sections"]
    entities = state.get("entities", {})

    # Clean LaTeX subscript artifacts from entity names before building the prompt.
    # "TurboQuantprod" → "TurboQuant" so the LLM can match entities in the paper text.
    clean_models   = _clean_entity_list(entities.get("models",   [])[:12])
    clean_datasets = _clean_entity_list(entities.get("datasets", [])[:12])
    clean_metrics  = _clean_entity_list(entities.get("metrics",  [])[:12])
    clean_methods  = _clean_entity_list(entities.get("methods",  [])[:12])

    if any(len(orig) != len(clean) for orig, clean in [
        (entities.get("models", [])[:12],   clean_models),
        (entities.get("datasets", [])[:12], clean_datasets),
    ]):
        logger.info("[TRIPLES] entity names cleaned (LaTeX subscript artifacts removed)")

    # Section allocation: results first (scores live here)
    raw_text = " ".join([
        sections.get("results",      "")[:3500],
        sections.get("methodology",  "")[:1500],
        sections.get("abstract",     "")[:1000],
        sections.get("conclusion",   "")[:800],
        sections.get("introduction", "")[:700],
    ]).strip()

    text = _clean_text_for_triples(raw_text)[:5000]

    if len(text) < 80:
        logger.warning(f"[TRIPLES] text too noisy after cleaning paper_id={state['paper_id']}")
        return {**state, "triples": [], "error": "text too noisy after cleaning"}

    logger.info(f"[TRIPLES] input text: {len(text)} chars after cleaning")

    prompt = _TRIPLE_PROMPT.format(
        models   = ", ".join(clean_models)   or "none identified",
        datasets = ", ".join(clean_datasets) or "none identified",
        metrics  = ", ".join(clean_metrics)  or "none identified",
        methods  = ", ".join(clean_methods)  or "none identified",
        text     = text,
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()
        logger.info(f"[TRIPLES DEBUG] raw={raw[:300]}")

        if not raw:
            logger.warning(f"[TRIPLES] empty response attempt {state['retry_count']+1}")
            return {**state, "error": "empty response from LLM",
                    "retry_count": state["retry_count"] + 1}

        start = raw.find('[')
        if start == -1:
            raise ValueError("No JSON array found in response")

        end = raw.rfind(']')

        if end == -1 or end <= start:
            logger.warning("[TRIPLES] response truncated — attempting partial recovery")
            last_obj_end = raw.rfind('}')
            if last_obj_end > start:
                partial = raw[start:last_obj_end + 1] + ']'
                try:
                    partial  = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', partial)
                    triples  = json.loads(partial)
                    logger.info(f"[TRIPLES] partial recovery: {len(triples)} triples")
                    return {**state, "triples": triples, "error": ""}
                except Exception:
                    pass
            raise ValueError("Response truncated and partial recovery failed")

        cleaned = raw[start:end + 1]
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
        triples = json.loads(cleaned)
        logger.info(f"[TRIPLES] extracted {len(triples)} triples")
        return {**state, "triples": triples, "error": ""}

    except Exception as e:
        logger.warning(f"[TRIPLES] extraction failed attempt {state['retry_count']+1}: {e}")
        return {**state, "error": str(e), "retry_count": state["retry_count"] + 1}


def _validate_triples_node(state: TripleState) -> TripleState:
    triples = state.get("triples", [])
    valid   = []

    _is_pure_formula = re.compile(r'^[\d\s\+\-\*/=<>\.\\{}()\[\]|,;:]+$')

    valid_relations = {
        "TRAINED_ON", "EVALUATED_ON", "ACHIEVES", "USES",
        "IMPROVES_OVER", "PROPOSED_BY", "APPLIED_TO",
        "COMPARED_WITH", "BASED_ON", "REPLACES",
        "OUTPERFORMS", "BOUNDED_BY", "EXTENDS",
    }

    for t in triples:
        if not isinstance(t, dict):
            continue

        subject  = str(t.get("subject",  "")).strip()
        relation = str(t.get("relation", "")).strip().upper()
        obj      = str(t.get("object",   "")).strip()

        if not subject or not relation or not obj:
            continue
        if len(subject) < 2 or len(obj) < 2:
            continue
        if subject.lower() == obj.lower():
            continue
        if _is_pure_formula.match(subject) or _is_pure_formula.match(obj):
            continue
        if re.sub(r'[\d\s\.]', '', subject) == '' or re.sub(r'[\d\s\.]', '', obj) == '':
            continue
        if relation not in valid_relations:
            continue

        confidence = max(0.0, min(1.0, float(t.get("confidence", 0.8))))

        valid.append({
            "subject":    subject,
            "relation":   relation,
            "object":     obj,
            "confidence": confidence,
        })

    seen   = set()
    unique = []
    for t in valid:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    # ACHIEVES first (most valuable), then by confidence descending
    unique.sort(key=lambda x: (0 if x["relation"] == "ACHIEVES" else 1, -x["confidence"]))

    achieves_count = sum(1 for t in unique if t["relation"] == "ACHIEVES")
    logger.info(
        f"[TRIPLES] validated: {len(unique)} unique triples from {len(triples)} raw "
        f"(ACHIEVES={achieves_count})"
    )
    return {**state, "triples": unique}


# ── Graph ─────────────────────────────────────────────────────────────────────
def _should_retry(state: TripleState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "extract"
    return "validate"


def _build_graph() -> Any:
    graph = StateGraph(TripleState)
    graph.add_node("extract",  _extract_triples_node)
    graph.add_node("validate", _validate_triples_node)
    graph.set_entry_point("extract")
    graph.add_conditional_edges("extract", _should_retry, {
        "extract":  "extract",
        "validate": "validate",
    })
    graph.add_edge("validate", END)
    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
class TripleExtractor:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def extract(self, paper_id: str, sections: Dict, entities: Dict) -> List[Dict]:
        await wait_for_groq("openai/gpt-oss-120b", "triples")

        loop = asyncio.get_running_loop()
        initial_state: TripleState = {
            "paper_id":    paper_id,
            "llm_id":      "openai/gpt-oss-120b",
            "sections":    sections,
            "entities":    entities,
            "triples":     [],
            "retry_count": 0,
            "error":       "",
        }
        result  = await loop.run_in_executor(None, lambda: _graph.invoke(initial_state))
        triples = result.get("triples", [])
        if not triples:
            logger.warning(f"[TRIPLES] no triples extracted for paper_id={paper_id}")
        return triples