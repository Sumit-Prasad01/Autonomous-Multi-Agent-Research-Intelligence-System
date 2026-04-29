"""
triple_extractor.py — LangGraph-based knowledge triple extraction agent
Extracts (subject, relation, object) triples from paper text.
These triples are stored in both Postgres and Neo4j.

── QUALITY DESIGN ────────────────────────────────────────────────────────────
Five decisions that determine KG quality:

1. Text cleaning: _MATH_NOISE removes LaTeX/formulas. The LINE FILTER
   preserves result-table rows (they have entity names + numbers) while
   stripping pure symbol/formula lines. The old 65% threshold was too
   aggressive and stripped result tables like "Transformer (big) | 41.0".

2. Section allocation: results section gets the most budget (3500 chars)
   because that's where model-dataset-metric relationships live. Introduction
   is reduced — it's mostly motivation text, not facts.

3. Text cap: 5000 chars after cleaning (was 3500). Gives the LLM enough
   context to see multiple result rows without hitting token limits.

4. ACHIEVES relation: object can include the numeric score alongside the
   metric name (e.g. "41.0 BLEU on WMT 2014 EN-FR"). This makes the KG
   actually useful for comparison — "Transformer ACHIEVES BLEU" tells you
   nothing; "Transformer ACHIEVES 41.0 BLEU on WMT 2014 EN-FR" does.

5. Model in limiter: wait_for_groq must use the ACTUAL model being called
   (gpt-oss-120b), not the old model name (gpt-oss-20b), so the token
   bucket tracks the right model's budget.
─────────────────────────────────────────────────────────────────────────────
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


# ── Text cleaning ─────────────────────────────────────────────────────────────
_MATH_NOISE = re.compile(
    r'(\\[a-zA-Z]+\{[^}]*\}|'       # LaTeX commands: \frac{}, \sum{}
    r'\$[^$]*\$|'                    # inline math: $...$
    r'\\\([^)]*\\\)|'                # display math: \( ... \)
    r'(?:\d+\s*[\+\-\*/]\s*){3,}|'  # repeated arithmetic: a+b+c+d
    r'(?<!\w)(\d+\s*){6,}(?!\w)|'   # long number sequences (6+) not part of words
    r'[\+\-\*/=<>]{3,}|'            # symbol runs: ===, ---, >>>
    r'exp\s*\([^)]*\)\s*[\.\,]?)',   # exp(...) patterns
    re.DOTALL,
)


def _clean_text_for_triples(text: str) -> str:
    """
    Remove math formulas and LaTeX noise while PRESERVING result table rows.

    Key design: result table rows like "Transformer (big) | 41.0 | 27.3"
    have high non-alpha density but contain the most valuable triple
    information. The old 65% threshold stripped these.

    New rule: keep a line if it has ≥2 tokens with at least 2 alpha chars.
    This preserves:
      "Transformer (big) | 41.0 | 27.3 | 6 | 65M"  → kept (Transformer, big)
      "BLEU scores: EN-DE 28.4, EN-FR 41.0"          → kept (BLEU, scores)
    And strips:
      "= = = = = = = = = ="                           → stripped (no alpha words)
      "0.1 0.2 0.3 0.4 0.5 0.6 0.7"                 → stripped (digits only)
    """
    # 1. Remove LaTeX formulas and math noise
    text = _MATH_NOISE.sub(' ', text)

    # 2. Remove ellipsis sequences
    text = re.sub(r'(\.\s*){3,}', ' ', text)

    # 3. Collapse whitespace
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)

    # 4. Line-level filter: keep lines with ≥2 word-like tokens (≥2 alpha chars each)
    lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        # count tokens that have at least 2 alphabetic characters
        word_like = [w for w in re.split(r'[\s|,;:\-]+', stripped)
                     if sum(c.isalpha() for c in w) >= 2]
        if len(word_like) >= 2:
            lines.append(stripped)
        elif len(stripped) < 100 and any(c.isalpha() for c in stripped):
            # short lines (headers, captions) with any letters — keep
            lines.append(stripped)

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
• Use the EXACT string from the list above (same casing, same hyphenation).
• Only invent an entity name if it clearly appears in the paper text AND is absent from all lists.
• Never abbreviate or expand a known entity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — RELATION SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use ONLY these 13 relation types. Selection rules are binding:

  TRAINED_ON      model/method was trained using a specific dataset or corpus
  EVALUATED_ON    model/method was tested on a benchmark or held-out dataset
  ACHIEVES        model reaches a specific result on a metric or benchmark
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACHIEVES — CRITICAL RULE (read carefully):
  The ACHIEVES object MUST include the numeric score when it appears in the paper.
  This is the most important triple type for knowledge graph utility.

  Format: "<score> <metric> on <dataset>" when all three are in the text.
  Examples:
    ✅ "Transformer (big) ACHIEVES 41.0 BLEU on WMT 2014 English-French"
    ✅ "Transformer (big) ACHIEVES 28.4 BLEU on WMT 2014 English-German"
    ✅ "BERT-large ACHIEVES 93.2% F1 on SQuAD"
    ✅ "ResNet-50 ACHIEVES 76.1% Top-1 on ImageNet"
    ✅ "Transformer ACHIEVES BLEU"   ← acceptable if no score in text
  
  If the paper has a results table, extract one ACHIEVES triple per
  (model, dataset) pair that has a numeric score.

VALID subjects and objects:
  ✅ Specific named models, datasets, metrics, methods from the paper
  ✅ Named theoretical concepts (e.g. "Johnson-Lindenstrauss Lemma")
  ✅ Named tasks as objects (e.g. "machine translation", "image classification")
  ✅ Numeric results as ACHIEVES objects: "41.0 BLEU on WMT 2014 EN-FR"

INVALID subjects and objects — reject these:
  ✗ Placeholders          → "our method", "the proposed approach", "baseline model"
  ✗ Bare numbers          → "0.923", "128" (but "41.0 BLEU" is valid — has metric name)
  ✗ Mathematical symbols  → "L2 norm", "∇θ", "σ(x)"
  ✗ Overly generic terms  → "deep learning", "neural network"

Quantity:
  • Target 15–25 triples. Hard stop at 25. Prioritise ACHIEVES triples.
  • Never pad to reach 15. Fewer precise triples > more vague ones.

Confidence scoring:
  1.0   Verbatim in text or table (numeric result with source)
  0.9   Clear logical reading of an explicit sentence
  0.85  Strongly implied by adjacent sentences
  0.75  Bridges two separate sections
  0.7   Reasonable inference, not directly stated
  (Do not use values outside 0.7–1.0.)

Deduplication:
  • Same (subject, relation, object) lowercased → keep higher-confidence one.
  • Semantically identical different surface forms → one triple only.

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
  {{"subject": "Transformer (big)", "relation": "ACHIEVES", "object": "41.0 BLEU on WMT 2014 English-French", "confidence": 1.0}},
  {{"subject": "Transformer", "relation": "USES", "object": "multi-head attention", "confidence": 0.9}}
]"""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _extract_triples_node(state: TripleState) -> TripleState:
    """Extract knowledge triples using LLM."""
    logger.info(f"[TRIPLES] paper_id={state['paper_id']} attempt={state['retry_count']+1}")

    sections = state["sections"]
    entities = state.get("entities", {})

    # Section allocation: results first (most triples live here), then methodology,
    # then abstract (defines what the paper is), then conclusion, then intro (motivation only).
    raw_text = " ".join([
        sections.get("results",      "")[:3500],   # ↑ highest — BLEU tables, ablations
        sections.get("methodology",  "")[:1500],   # architecture details
        sections.get("abstract",     "")[:1000],   # what the paper is
        sections.get("conclusion",   "")[:800],    # summary of contributions
        sections.get("introduction", "")[:700],    # motivation (least useful for triples)
    ]).strip()

    text = _clean_text_for_triples(raw_text)[:5000]   # ↑ was 3500

    if len(text) < 80:
        logger.warning(f"[TRIPLES] text too noisy after cleaning paper_id={state['paper_id']}")
        return {**state, "triples": [], "error": "text too noisy after cleaning"}

    logger.info(f"[TRIPLES] input text: {len(text)} chars after cleaning")

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
            return {
                **state,
                "error":       "empty response from LLM",
                "retry_count": state["retry_count"] + 1,
            }

        start = raw.find('[')
        if start == -1:
            raise ValueError("No JSON array found in response")

        end = raw.rfind(']')

        # Handle truncated JSON (no closing ])
        if end == -1 or end <= start:
            logger.warning("[TRIPLES] response truncated — attempting partial recovery")
            last_obj_end = raw.rfind('}')
            if last_obj_end > start:
                partial = raw[start:last_obj_end + 1] + ']'
                try:
                    partial = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', partial)
                    triples = json.loads(partial)
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
        return {
            **state,
            "error":       str(e),
            "retry_count": state["retry_count"] + 1,
        }


def _validate_triples_node(state: TripleState) -> TripleState:
    """Validate, clean, and deduplicate extracted triples."""
    triples = state.get("triples", [])
    valid   = []

    # Detects pure formula strings (no real words).
    # Note: "41.0 BLEU on WMT 2014 EN-FR" has alpha chars → NOT matched → kept.
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
        # reject if subject or object is only digits/spaces
        if re.sub(r'[\d\s\.]', '', subject) == '' or re.sub(r'[\d\s\.]', '', obj) == '':
            continue
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

    # Deduplicate case-insensitively on (subject, relation, object)
    seen   = set()
    unique = []
    for t in valid:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    # Sort: ACHIEVES triples first (most valuable for comparison), then by confidence
    unique.sort(key=lambda x: (0 if x["relation"] == "ACHIEVES" else 1, -x["confidence"]))
    achieves_count = sum(1 for t in unique if t["relation"] == "ACHIEVES")

    logger.info(
        f"[TRIPLES] validated: {len(unique)} unique triples from {len(triples)} raw "
        f"(ACHIEVES={achieves_count})"
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

    async def extract(self, paper_id: str, sections: Dict, entities: Dict) -> List[Dict]:
        # Fix: use the actual model being called (gpt-oss-120b), not the old name.
        # This ensures the token bucket tracks the right model's quota.
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