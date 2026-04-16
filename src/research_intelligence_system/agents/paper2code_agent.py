"""
paper2code_agent.py — LangGraph-based implementation planner
Extracts algorithm steps → generates pseudocode → generates Python skeleton
Token efficient: reuses entities + sections from orchestrator pipeline
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class CodeState(TypedDict):
    paper_id:         str
    llm_id:           str
    sections:         Dict[str, str]
    entities:         Dict[str, Any]
    algorithm_steps:  List[str]
    pseudocode:       str
    python_skeleton:  str
    time_complexity:  str
    space_complexity: str
    key_components:   List[Dict]
    retry_count:      int
    error:            str


# ── LLM ──────────────────────────────────────────────────────────────────────
def _get_llm(llm_id: str) -> ChatGroq:
    return ChatGroq(model=llm_id, temperature=0.1)


# ── Prompts ───────────────────────────────────────────────────────────────────
_ALGORITHM_EXTRACTION_PROMPT = """You are an algorithm extraction expert.
Extract the core algorithm or method from this research paper.

Paper methodology:
{methodology}

Paper results:
{results}

Known entities:
Models:  {models}
Methods: {methods}

Return ONLY valid JSON with NO newlines inside strings:
{{
  "algorithm_steps": [
    "Step 1: Clear description of first step",
    "Step 2: Clear description of second step"
  ],
  "key_components": [
    {{"name": "component_name", "description": "what it does", "type": "class|function|module"}}
  ],
  "time_complexity": "O(n log n) — brief explanation",
  "space_complexity": "O(n) — brief explanation",
  "algorithm_name": "Name of the main algorithm/method"
}}

Rules:
- Extract 3-8 concrete algorithm steps
- Identify 3-6 key implementable components
- Be specific about data structures and operations used
- Return only JSON, no explanation"""


_CODE_GENERATION_PROMPT = """You are an expert Python developer implementing a research paper algorithm.

Algorithm steps:
{algorithm_steps}

Key components to implement:
{key_components}

Paper entities:
Models:   {models}
Methods:  {methods}
Metrics:  {metrics}

Generate TWO things and return as JSON:

1. PSEUDOCODE: Clear pseudocode showing the algorithm logic
2. PYTHON SKELETON: Python code with structure, type hints, and detailed docstrings

Return ONLY valid JSON — use \\n for line breaks inside code strings:
{{
  "pseudocode": "ALGORITHM Name\\nINPUT: description\\nOUTPUT: description\\nBEGIN\\n  1. step one\\n  2. step two\\nEND",
  "python_skeleton": "from typing import List, Optional\\nimport numpy as np\\n\\nclass MainClass:\\n    def __init__(self):\\n        pass"
}}

Python skeleton rules:
- Include all necessary imports
- Create proper class/function structure matching the paper
- Add detailed docstrings with Args, Returns, References sections
- Add type hints everywhere
- Add TODO comments where implementation details are needed
- Return only JSON, no markdown, no explanation"""


# ── Nodes ─────────────────────────────────────────────────────────────────────
def _extract_algorithm_node(state: CodeState) -> CodeState:
    """Stage 1 — extract algorithm steps and key components."""
    logger.info(f"[PAPER2CODE] extracting algorithm paper_id={state['paper_id']}")

    sections = state["sections"]
    entities = state.get("entities", {})

    methodology = sections.get("methodology", sections.get("body", ""))[:2000]
    results     = sections.get("results", "")[:1000]

    if not methodology:
        return {
            **state,
            "error":       "no methodology section found",
            "retry_count": state["retry_count"] + 1,
        }

    prompt = _ALGORITHM_EXTRACTION_PROMPT.format(
        methodology = methodology,
        results     = results,
        models      = ", ".join(entities.get("models",  [])[:8]),
        methods     = ", ".join(entities.get("methods", [])[:8]),
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in algorithm extraction response")

        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
        result  = json.loads(cleaned)

        algorithm_steps  = result.get("algorithm_steps", [])
        key_components   = result.get("key_components", [])
        time_complexity  = result.get("time_complexity", "")
        space_complexity = result.get("space_complexity", "")

        logger.info(
            f"[PAPER2CODE] extracted {len(algorithm_steps)} steps "
            f"{len(key_components)} components"
        )

        return {
            **state,
            "algorithm_steps":  algorithm_steps,
            "key_components":   key_components,
            "time_complexity":  time_complexity,
            "space_complexity": space_complexity,
            "error":            "",
        }

    except Exception as e:
        logger.warning(f"[PAPER2CODE] extraction failed: {e}")
        return {**state, "error": str(e), "retry_count": state["retry_count"] + 1}


def _generate_code_node(state: CodeState) -> CodeState:
    """Stage 2 — generate pseudocode + Python skeleton."""
    logger.info(f"[PAPER2CODE] generating code paper_id={state['paper_id']}")

    algorithm_steps = state.get("algorithm_steps", [])
    key_components  = state.get("key_components", [])
    entities        = state.get("entities", {})

    if not algorithm_steps:
        return {**state, "error": "no algorithm steps to generate code from"}

    steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(algorithm_steps)])
    comp_text  = "\n".join([
        f"- {c.get('name', '')}: {c.get('description', '')} ({c.get('type', '')})"
        for c in key_components[:6]
    ])

    prompt = _CODE_GENERATION_PROMPT.format(
        algorithm_steps = steps_text,
        key_components  = comp_text,
        models          = ", ".join(entities.get("models",   [])[:6]),
        methods         = ", ".join(entities.get("methods",  [])[:6]),
        metrics         = ", ".join(entities.get("metrics",  [])[:6]),
    )

    try:
        llm      = _get_llm(state["llm_id"])
        response = llm.invoke(prompt)
        raw      = response.content.strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in code generation response")

        cleaned = json_match.group()
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
        result  = json.loads(cleaned)

        pseudocode      = result.get("pseudocode", "").replace("\\n", "\n")
        python_skeleton = result.get("python_skeleton", "").replace("\\n", "\n")

        logger.info(
            f"[PAPER2CODE] generated pseudocode={len(pseudocode)} chars "
            f"skeleton={len(python_skeleton)} chars"
        )

        return {
            **state,
            "pseudocode":      pseudocode,
            "python_skeleton": python_skeleton,
            "error":           "",
        }

    except Exception as e:
        logger.warning(f"[PAPER2CODE] code generation failed: {e}")
        return {**state, "error": str(e), "retry_count": state["retry_count"] + 1}


# ── Conditional edges ─────────────────────────────────────────────────────────
def _should_retry_extraction(state: CodeState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "extract_algorithm"
    return "generate_code"


def _should_retry_code(state: CodeState) -> str:
    if state.get("error") and state["retry_count"] < 2:
        return "generate_code"
    return END


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph():
    graph = StateGraph(CodeState)

    graph.add_node("extract_algorithm", _extract_algorithm_node)
    graph.add_node("generate_code",     _generate_code_node)

    graph.set_entry_point("extract_algorithm")
    graph.add_conditional_edges("extract_algorithm", _should_retry_extraction, {
        "extract_algorithm": "extract_algorithm",
        "generate_code":     "generate_code",
    })
    graph.add_conditional_edges("generate_code", _should_retry_code, {
        "generate_code": "generate_code",
        END:             END,
    })

    return graph.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
class Paper2CodeAgent:

    def __init__(self, llm_id: str = "llama-3.3-70b-versatile"):
        self.llm_id = llm_id

    async def generate(
        self,
        paper_id: str,
        sections: Dict[str, str],
        entities: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate pseudocode + Python skeleton from paper.
        Token efficient — reuses sections + entities from orchestrator.

        Returns:
            algorithm_steps, pseudocode, python_skeleton,
            time_complexity, space_complexity, key_components
        """
        import asyncio
        loop = asyncio.get_running_loop()

        initial_state: CodeState = {
            "paper_id":         paper_id,
            "llm_id":           self.llm_id,
            "sections":         sections,
            "entities":         entities,
            "algorithm_steps":  [],
            "pseudocode":       "",
            "python_skeleton":  "",
            "time_complexity":  "",
            "space_complexity": "",
            "key_components":   [],
            "retry_count":      0,
            "error":            "",
        }

        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        return {
            "algorithm_steps":  result.get("algorithm_steps", []),
            "pseudocode":       result.get("pseudocode", ""),
            "python_skeleton":  result.get("python_skeleton", ""),
            "time_complexity":  result.get("time_complexity", ""),
            "space_complexity": result.get("space_complexity", ""),
            "key_components":   result.get("key_components", []),
        }