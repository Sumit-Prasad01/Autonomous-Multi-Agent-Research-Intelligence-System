"""export_ui.py — ZIP export of all analysis results"""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from typing import Dict, List


def _clean_name(filename: str) -> str:
    name = re.sub(r'^[a-f0-9]{32}_', '', filename or "paper")
    name = re.sub(r'\.pdf$', '', name, flags=re.IGNORECASE)
    return name.replace("_", " ").strip() or "paper"


def _build_summary_txt(paper: Dict) -> str:
    name    = _clean_name(paper.get("filename", ""))
    quality = paper.get("quality_score", 0)
    faith   = paper.get("faithfulness_score", 1.0)
    hall    = paper.get("hallucination_score", 0.0)
    summary = paper.get("refined_summary", "No summary available.")

    lines = [
        f"RESEARCH PAPER ANALYSIS",
        f"{'='*60}",
        f"Paper:              {name}",
        f"Quality Score:      {quality:.1f}/10",
        f"Faithfulness:       {faith:.0%}",
        f"Hallucination Rate: {hall:.0%}",
        f"Generated:          {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'='*60}",
        f"",
        f"COMPREHENSIVE SUMMARY",
        f"{'-'*40}",
        summary,
        f"",
        f"SECTION SUMMARIES",
        f"{'-'*40}",
    ]
    summaries = paper.get("summaries", {})
    for section, text in summaries.items():
        if text and section not in ("overall", "comprehensive"):
            lines.append(f"\n{section.upper()}:\n{text}")

    return "\n".join(lines)


def _build_entities_csv(paper: Dict) -> str:
    entities = paper.get("entities", {})
    rows     = ["Type,Entity"]
    for etype in ("models", "datasets", "metrics", "methods", "tasks"):
        for e in entities.get(etype, []):
            rows.append(f"{etype.rstrip('s').title()},{e}")
    return "\n".join(rows)


def _build_gaps_csv(paper: Dict) -> str:
    gaps = paper.get("research_gaps", [])
    rows = ["Rank,Novelty Score,Gap,Supporting Evidence,Suggested Experiment"]
    for i, g in enumerate(gaps, 1):
        if isinstance(g, dict):
            gap   = g.get("gap", "").replace(",", ";")
            evid  = g.get("supporting_evidence", "").replace(",", ";")
            exp   = g.get("suggested_experiment", "").replace(",", ";")
            score = g.get("novelty_score", 0)
            rows.append(f'{i},{score:.1f},"{gap}","{evid}","{exp}"')
        else:
            rows.append(f'{i},5.0,"{str(g).replace(",", ";")}","",""')
    return "\n".join(rows)


def _build_comparison_csv(comp: Dict) -> str:
    table   = comp.get("comparison_table", {})
    headers = table.get("headers", [])
    rows_   = table.get("rows", [])
    if not headers or not rows_:
        return "No comparison data available."
    rows = [",".join(f'"{h}"' for h in headers)]
    for row in rows_:
        rows.append(",".join(f'"{str(c)}"' for c in row))
    return "\n".join(rows)


def _build_lit_review_txt(lit: Dict) -> str:
    themes  = lit.get("themes", [])
    review  = lit.get("review_text", "")
    gaps    = lit.get("research_gaps_summary", "")
    future  = lit.get("future_directions", "")
    quality = lit.get("overall_quality", 0)

    themes_text = " · ".join([
        t if isinstance(t, str) else t.get("gap", str(t))
        for t in themes
    ])

    return "\n".join([
        "LITERATURE REVIEW",
        "="*60,
        f"Overall Quality: {quality:.1f}/10",
        f"Themes: {themes_text}",
        "-"*40,
        review,
        "",
        "RESEARCH GAPS SUMMARY",
        "-"*40,
        gaps,
        "",
        "FUTURE DIRECTIONS",
        "-"*40,
        future,
    ])


def _build_graphml(papers: List[Dict]) -> str:
    """Export knowledge triples as GraphML."""
    all_triples = []
    for paper in papers:
        all_triples.extend(paper.get("triples", []))

    nodes = {}
    edges = []
    for t in all_triples:
        s = str(t.get("subject", "")).strip()
        o = str(t.get("object",  "")).strip()
        r = str(t.get("relation","")).strip()
        c = float(t.get("confidence", 1.0))
        if s: nodes[s] = True
        if o: nodes[o] = True
        if s and o and r:
            edges.append((s, r, o, c))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/graphml">',
        '  <key id="relation"   for="edge" attr.name="relation"   attr.type="string"/>',
        '  <key id="confidence" for="edge" attr.name="confidence" attr.type="double"/>',
        '  <graph id="G" edgedefault="directed">',
    ]
    for node in nodes:
        safe = node.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        lines.append(f'    <node id="{safe}"/>')
    for i, (s, r, o, c) in enumerate(edges):
        ss = s.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        oo = o.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        lines.append(f'    <edge id="e{i}" source="{ss}" target="{oo}">')
        lines.append(f'      <data key="relation">{r}</data>')
        lines.append(f'      <data key="confidence">{c}</data>')
        lines.append(f'    </edge>')
    lines += ['  </graph>', '</graphml>']
    return "\n".join(lines)


def _build_metadata_json(analysis: Dict) -> str:
    papers = analysis.get("papers", [])
    meta = {
        "generated_at": datetime.now().isoformat(),
        "papers": [
            {
                "filename":          p.get("filename", ""),
                "quality_score":     p.get("quality_score", 0),
                "faithfulness":      p.get("faithfulness_score", 1.0),
                "hallucination":     p.get("hallucination_score", 0.0),
                "entity_count":      sum(
                    len(p.get("entities", {}).get(k, []))
                    for k in ("models","datasets","metrics","methods")
                ),
                "triple_count":      len(p.get("triples", [])),
                "gap_count":         len(p.get("research_gaps", [])),
                "novelty_score":     p.get("novelty_score", 0.0),
                "analysis_status":   p.get("status", ""),
            }
            for p in papers
        ],
        "has_comparison":       bool(analysis.get("comparison")),
        "has_literature_review":bool(analysis.get("literature_review", {}).get("review_text")),
    }
    return json.dumps(meta, indent=2)


def build_export_zip(analysis: Dict) -> bytes:
    """
    Build a ZIP archive with all analysis outputs.
    Returns bytes ready for st.download_button.
    """
    papers = analysis.get("papers", [])
    comp   = analysis.get("comparison", {})
    lit    = analysis.get("literature_review", {})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # per-paper files
        for paper in papers:
            name = _clean_name(paper.get("filename", "paper")).replace(" ", "_")
            prefix = f"papers/{name}/"

            zf.writestr(f"{prefix}summary.txt",  _build_summary_txt(paper))
            zf.writestr(f"{prefix}entities.csv", _build_entities_csv(paper))
            zf.writestr(f"{prefix}gaps.csv",     _build_gaps_csv(paper))

        # cross-paper files
        if comp:
            zf.writestr("comparison.csv",       _build_comparison_csv(comp))
        if lit and lit.get("review_text"):
            zf.writestr("literature_review.txt", _build_lit_review_txt(lit))

        # knowledge graph
        if any(p.get("triples") for p in papers):
            zf.writestr("knowledge_graph.graphml", _build_graphml(papers))

        # metadata
        zf.writestr("metadata.json", _build_metadata_json(analysis))

    buf.seek(0)
    return buf.read()


def render_export_button(analysis: Dict) -> None:
    """Render the download ZIP button."""
    import streamlit as st

    st.markdown(
        "<div style='height:8px'></div>",
        unsafe_allow_html=True,
    )
    try:
        zip_bytes = build_export_zip(analysis)
        papers    = analysis.get("papers", [])
        name      = _clean_name(
            papers[0].get("filename", "analysis") if papers else "analysis"
        ).replace(" ", "_")
        filename  = f"research_analysis_{name}_{datetime.now().strftime('%Y%m%d')}.zip"

        with st.expander("📦 Export Results"):
            st.markdown(
                "<div style='color:#8888aa;font-size:0.85rem;margin-bottom:12px'>"
                "Download all analysis results as a ZIP archive containing "
                "summaries, entities, gaps, comparison, literature review, "
                "and knowledge graph.</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns([1, 2])
            with cols[0]:
                st.markdown(
                    f"<div style='font-size:0.8rem;color:#55556a'>"
                    f"📄 {len(papers)} paper(s)<br>"
                    f"{'📊 Comparison included<br>' if analysis.get('comparison') else ''}"
                    f"{'📚 Literature review included' if analysis.get('literature_review', {}).get('review_text') else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.download_button(
                    label      = "⬇ Download ZIP",
                    data       = zip_bytes,
                    file_name  = filename,
                    mime       = "application/zip",
                    type       = "primary",
                    use_container_width=True,
                )
    except Exception as e:
        st.error(f"Export failed: {e}")