"""
triple_extractor.py — LangGraph-based knowledge triple extraction agent
Extracts (subject, relation, object) triples from paper text
These triples are stored in both Postgres and Neo4j
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
    _TRIPLE_MODEL = "openai/gpt-oss-120b"   
    return ChatGroq(model=_TRIPLE_MODEL, temperature=0, max_tokens=4000)


# ── Text cleaning ─────────────────────────────────────────────────────────────
_MATH_NOISE = re.compile(
    r'(\\[a-zA-Z]+\{[^}]*\}|'      # LaTeX: \frac{}, \sum{}
    r'\$[^$]*\$|'                   # inline math $...$
    r'\\\([^)]*\\\)|'               # \( ... \)
    r'(?:\d+\s*[\+\-\*/]\s*){3,}|' # repeated arithmetic
    r'(\d+\s*){5,}|'               # long number sequences
    r'[\+\-\*/=<>]{3,}|'           # symbol runs
    r'exp\s*\([^)]*\)\s*[\.\,]?)',  # exp(...) patterns
    re.DOTALL,
)

def _clean_text_for_triples(text: str) -> str:
    """Remove math formulas, LaTeX, and noise before triple extraction."""
    text = _MATH_NOISE.sub(' ', text)
    text = re.sub(r'(\.\s*){3,}', ' ', text)       # ellipsis sequences
    text = re.sub(r'\s{2,}', ' ', text)             # collapse whitespace
    text = re.sub(r'\n{2,}', '\n', text)            # collapse newlines
    # remove lines that are >50% digits/symbols (formula-heavy lines)
    lines = []
    for line in text.split('\n'):
        if len(line) > 10:
            non_alpha = sum(1 for c in line if not c.isalpha() and not c.isspace())
            if non_alpha / len(line) < 0.65:
                lines.append(line)
    return '\n'.join(lines).strip()


# ── Prompt ────────────────────────────────────────────────────────────────────
_TRIPLE_PROMPT = """[SYSTEM] You are a scientific knowledge graph construction engine. \
Your sole output is a JSON array of knowledge triples extracted from a research paper. \
You operate across all scientific domains: ML, NLP, biology, chemistry, physics, finance, medicine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — ENTITY INVENTORY (do this mentally before extraction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have been given a pre-extracted entity list. Treat it as ground truth.

  Models / Algorithms : {models}
  Datasets / Corpora  : {datasets}
  Metrics             : {metrics}
  Methods / Techniques: {methods}

Rules for entity usage:
- Use the EXACT string from the list above (same casing, same hyphenation).
- Only invent an entity name if it clearly appears in the paper text AND is absent from all lists.
- Never abbreviate or expand a known entity (e.g. do not write "BERT" if the list says "BERT-large").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — RELATION SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use ONLY these 13 relation types. Selection rules are binding:

  TRAINED_ON      model/method was trained using a specific dataset or corpus
  EVALUATED_ON    model/method was tested on a benchmark or held-out dataset
  ACHIEVES        model reaches a named metric result  [object = metric NAME, not value]
  USES            method incorporates a specific technique, module, or component
  IMPROVES_OVER   method surpasses a baseline without a direct numeric comparison
  PROPOSED_BY     method or model is introduced by named authors or a team
  APPLIED_TO      method is deployed on a downstream task or domain
  COMPARED_WITH   method is placed side-by-side with another (neutral, no winner implied)
  BASED_ON        method is directly derived from or built upon a prior architecture/theory
  REPLACES        method is a direct successor that supersedes an older approach
  OUTPERFORMS     method beats another with an explicit numeric score in the paper
  BOUNDED_BY      performance or capacity is constrained by a stated theoretical limit
  EXTENDS         method broadens, generalises, or adds capability to an existing method

Disambiguation rules — apply in order:
  • OUTPERFORMS requires a quoted or tabulated numeric comparison in the source text.
    If the paper only says "better than" without numbers → use IMPROVES_OVER instead.
  • EXTENDS vs BASED_ON: use EXTENDS when new capability is added; BASED_ON when it is
    a direct implementation or replication of prior work.
  • USES vs BASED_ON: USES = one component among many; BASED_ON = the foundational architecture.
  • ACHIEVES object must be the metric NAME only (e.g. "BLEU", "Top-1 Accuracy"), never a number.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID subjects and objects:
  ✅ Specific named models, datasets, metrics, methods from the paper
  ✅ Named theoretical concepts tied to a result (e.g. "Johnson-Lindenstrauss Lemma")
  ✅ Named tasks when used as objects (e.g. "machine translation", "image classification")

INVALID subjects and objects — reject these:
  ✗ Placeholders          → "our method", "the proposed approach", "baseline model"
  ✗ Raw numbers           → "0.923", "128", "3 layers"
  ✗ Mathematical symbols  → "L2 norm", "∇θ", "σ(x)"
  ✗ Partial phrases       → "deep learning", "neural network" (too generic)
  ✗ Authors as objects    → do not create triples like "BERT PROPOSED_BY Devlin"
    unless PROPOSED_BY is the primary contribution of the section

Quantity:
  • Target 10–25 triples. Hard stop at 25.
  • Never pad to reach 10. Fewer precise triples are better than more vague ones.
  • If the paper has fewer than 5 extractable facts, return what you find.

Confidence scoring — use the exact anchors below:
  1.0   The paper states this triple verbatim in text or a table.
  0.9   The triple is the clear logical reading of an explicit sentence.
  0.85  The triple is strongly implied by adjacent sentences in context.
  0.75  The triple requires bridging two separate sections of the paper.
  0.7   The triple is a reasonable inference but not directly stated.
  (Do not use values outside 0.7–1.0. Do not use the same value for every triple.)

Deduplication:
  • If two triples share the same (subject, relation, object) after lowercasing → keep only the higher-confidence one.
  • Semantically identical triples with different surface forms count as duplicates.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT PAPER TEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — JSON ARRAY ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return a single JSON array. No preamble, no markdown, no explanation.
First character must be [  •  Last character must be ]

[
  {{"subject": "ExactEntityName", "relation": "RELATION_TYPE", "object": "ExactEntityName", "confidence": 0.95}},
  {{"subject": "ExactEntityName", "relation": "RELATION_TYPE", "object": "ExactEntityName", "confidence": 0.85}}
]"""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _extract_triples_node(state: TripleState) -> TripleState:
    """Extract knowledge triples using LLM."""
    logger.info(f"[TRIPLES] paper_id={state['paper_id']} attempt={state['retry_count']+1}")

    sections = state["sections"]
    entities = state.get("entities", {})

    # combine most informative sections — trim aggressively
    raw_text = " ".join([
        sections.get("abstract",     "")[:1500],
        sections.get("introduction", "")[:1800],  # ADD THIS — clean entity text
        sections.get("results",      "")[:2500],  # INCREASE — comparisons live here
        sections.get("methodology",  "")[:1200],   # REDUCE — mostly math
        sections.get("conclusion",   "")[:1000],
    ]).strip()

    text = _clean_text_for_triples(raw_text)[:3500]

    if len(text) < 80:
        logger.warning(f"[TRIPLES] text too noisy after cleaning paper_id={state['paper_id']}")
        return {**state, "triples": [], "error": "text too noisy after cleaning"}

    prompt = _TRIPLE_PROMPT.format(
        models   = ", ".join(entities.get("models",   [])[:12]) or "none identified",
        datasets = ", ".join(entities.get("datasets", [])[:12]) or "none identified",
        metrics  = ", ".join(entities.get("metrics",  [])[:12]) or "none identified",
        methods  = ", ".join(entities.get("methods",  [])[:12]) or "none identified",
        text     = text,
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()
        logger.info(f"[TRIPLES DEBUG] raw={raw[:300]}")

        if not raw:
            logger.warning(f"[TRIPLES] LLM returned empty response attempt {state['retry_count']+1}")
            return {**state, "error": "empty response from LLM",
                    "retry_count": state["retry_count"] + 1}

        start = raw.find('[')
        if start == -1:
            raise ValueError("No JSON array found in response")

        end = raw.rfind(']')

        # ── Bug 1 fix: handle truncated JSON (no closing ]) ───────────────
        if end == -1 or end <= start:
            logger.warning(f"[TRIPLES] response truncated — attempting partial recovery")
            # find the last complete object: last occurrence of '}' before truncation
            last_obj_end = raw.rfind('}')
            if last_obj_end > start:
                # reconstruct a valid array from whatever complete objects exist
                partial = raw[start:last_obj_end + 1] + ']'
                try:
                    partial = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', partial)
                    triples = json.loads(partial)
                    logger.info(f"[TRIPLES] partial recovery: {len(triples)} triples from truncated response")
                    return {**state, "triples": triples, "error": ""}
                except Exception:
                    pass  # fall through to retry
            raise ValueError("Response truncated and partial recovery failed")
        # ─────────────────────────────────────────────────────────────────

        cleaned = raw[start:end + 1]
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
        triples = json.loads(cleaned)
        logger.info(f"[TRIPLES] extracted {len(triples)} triples")
        return {**state, "triples": triples, "error": ""}

    except Exception as e:
        logger.warning(f"[TRIPLES] extraction failed attempt {state['retry_count']+1}: {e}")
        return {
            **state,
            "error":       str(e),
            "retry_count": state["retry_count"] + 1,
        }


def _validate_triples_node(state: TripleState) -> TripleState:
    """Validate, clean, and deduplicate extracted triples."""
    triples = state.get("triples", [])
    valid   = []

    # regex to detect math/formula strings
    _is_formula = re.compile(r'^[\d\s\+\-\*/=<>\.\\{}()\[\]]+$')

    for t in triples:
        if not isinstance(t, dict):
            continue

        subject  = str(t.get("subject",  "")).strip()
        relation = str(t.get("relation", "")).strip().upper()
        obj      = str(t.get("object",   "")).strip()

        # skip incomplete
        if not subject or not relation or not obj:
            continue
        if len(subject) < 2 or len(obj) < 2:
            continue
        # skip self-referential
        if subject.lower() == obj.lower():
            continue
        # skip formula subjects/objects
        if _is_formula.match(subject) or _is_formula.match(obj):
            continue
        # skip if subject or object is just digits
        if subject.replace(' ', '').isdigit() or obj.replace(' ', '').isdigit():
            continue
        # skip unknown relation types
        valid_relations = {
            "TRAINED_ON", "EVALUATED_ON", "ACHIEVES", "USES",
            "IMPROVES_OVER", "PROPOSED_BY", "APPLIED_TO",
            "COMPARED_WITH", "BASED_ON", "REPLACES",
            "OUTPERFORMS", "BOUNDED_BY", "EXTENDS",  
        }
        if relation not in valid_relations:
            continue

        confidence = float(t.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))

        valid.append({
            "subject":    subject,
            "relation":   relation,
            "object":     obj,
            "confidence": confidence,
        })

    # deduplicate — case-insensitive on subject + object
    seen   = set()
    unique = []
    for t in valid:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    # sort by confidence descending
    unique.sort(key=lambda x: x["confidence"], reverse=True)

    logger.info(
        f"[TRIPLES] validated: {len(unique)} unique triples "
        f"from {len(triples)} raw"
    )
    return {**state, "triples": unique}


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry(state: TripleState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "extract"
    return "validate"


# ── Graph ─────────────────────────────────────────────────────────────────────
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

    async def extract(self, paper_id, sections, entities):
        await wait_for_groq("openai/gpt-oss-20b", "triples")
        loop = asyncio.get_running_loop()
        initial_state: TripleState = {
            "paper_id":    paper_id,
            "llm_id":      "openai/gpt-oss-120b",   # ← was gpt-oss-20b
            "sections":    sections,
            "entities":    entities,
            "triples":     [],
            "retry_count": 0,
            "error":       "",
        }
        result = await loop.run_in_executor(None, lambda: _graph.invoke(initial_state))
        triples = result.get("triples", [])
        if not triples:
            logger.warning(f"[TRIPLES] no triples extracted for paper_id={paper_id}")
        return triples