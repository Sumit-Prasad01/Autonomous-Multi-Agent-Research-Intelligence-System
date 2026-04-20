"""analysis_ui.py — Analysis results rendering"""
from __future__ import annotations

import re
from typing import Dict, List

import streamlit as st
import streamlit.components.v1 as components


def _clean_name(filename: str) -> str:
    name = re.sub(r'^[a-f0-9]{32}_', '', filename or "Paper")
    return re.sub(r'\.pdf$', '', name, flags=re.IGNORECASE).replace("_", " ").strip()


def _render_knowledge_graph(chat_id: str, papers: List[Dict]) -> None:
    try:
        from pyvis.network import Network

        all_triples = []
        for paper in papers:
            all_triples.extend(paper.get("triples", []))

        if not all_triples:
            st.info("No knowledge graph — triples not extracted yet.")
            return

        net = Network(
            height="480px", width="100%",
            bgcolor="#0a0a0f", font_color="#f0f0f8",
            directed=True,
        )
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=130)

        COLORS = {
            "TRAINED_ON":    "#6366f1",
            "EVALUATED_ON":  "#10b981",
            "ACHIEVES":      "#f59e0b",
            "USES":          "#8b5cf6",
            "IMPROVES_OVER": "#ef4444",
            "PROPOSED_BY":   "#f97316",
            "APPLIED_TO":    "#06b6d4",
            "COMPARED_WITH": "#ec4899",
            "BASED_ON":      "#84cc16",
            "REPLACES":      "#f43f5e",
        }

        added = set()
        for t in all_triples[:60]:
            s = str(t.get("subject", "")).strip()
            r = str(t.get("relation","")).strip().upper()
            o = str(t.get("object",  "")).strip()
            if not s or not o:
                continue
            if s not in added:
                net.add_node(s, label=s, color="#6366f1", size=22, title=s,
                             font={"color": "#f0f0f8"})
                added.add(s)
            if o not in added:
                net.add_node(o, label=o, color="#10b981", size=16, title=o,
                             font={"color": "#f0f0f8"})
                added.add(o)
            net.add_edge(s, o, label=r,
                         color=COLORS.get(r, "#555566"),
                         title=r, arrows="to")

        html = net.generate_html()
        components.html(html, height=500, scrolling=False)
        st.caption(
            f"⬡ {len(added)} nodes · "
            f"{min(len(all_triples), 60)} edges · "
            "color-coded by relation type"
        )

    except ImportError:
        st.warning("`uv pip install pyvis` to enable graph visualization.")
    except Exception as e:
        st.error(f"Graph render error: {e}")


def render_analysis(analysis: Dict) -> None:
    papers = analysis.get("papers", [])
    if not papers:
        return

    # ── Per-paper sections ────────────────────────────────────────────────────
    for paper in papers:
        fname       = paper.get("filename", "Paper")
        entities    = paper.get("entities", {})
        summary     = paper.get("refined_summary", "")
        summaries   = paper.get("summaries", {})
        gaps        = paper.get("research_gaps", [])
        dirs        = paper.get("future_directions", [])
        quality     = paper.get("quality_score", 0)
        hall_score  = paper.get("hallucination_score", 0.0)
        faith_score = paper.get("faithfulness_score", 1.0)
        hall_sents  = paper.get("hallucinated_sentences", [])
        similar     = paper.get("similar_papers", [])
        clean_name  = _clean_name(fname)

        # paper header
        st.markdown(
            f"<div class='paper-header'>"
            f"<span style='font-size:1.3rem'>📄</span>"
            f"<span class='paper-title'>{clean_name}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("Quality Score",  f"{quality:.1f} / 10")
        c2.metric("Faithfulness",   f"{faith_score:.0%}")
        c3.metric(
            "Hallucination",
            f"{hall_score:.0%}",
            delta     = f"{'⚠ High' if hall_score > 0.3 else None}",
            delta_color="inverse" if hall_score > 0.3 else "off",
        )

        # ── Summary ───────────────────────────────────────────────────────────
        with st.expander("📋 Summary", expanded=True):
            if summary:
                st.markdown(summary)
            if summaries:
                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                for section, text in summaries.items():
                    if text and section not in ("overall", "comprehensive"):
                        st.markdown(
                            f"<div style='margin-bottom:10px'>"
                            f"<span style='font-size:0.72rem;color:#8888aa;"
                            f"text-transform:uppercase;letter-spacing:0.08em'>"
                            f"{section}</span><br>"
                            f"<span style='font-size:0.9rem;color:#d0d0e8'>{text}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
            if hall_score > 0.3 and hall_sents:
                st.warning(
                    f"⚠️ {len(hall_sents)} sentence(s) may not be fully "
                    f"supported by the source paper."
                )
                for s in hall_sents[:3]:
                    st.markdown(
                        f'<div class="gap-item">❓ {s}</div>',
                        unsafe_allow_html=True,
                    )

        # ── Entities ──────────────────────────────────────────────────────────
        with st.expander("🔬 Entities"):
            cols = st.columns(3)
            for col, (label, key) in zip(
                cols,
                [("🤖 Models","models"), ("📊 Datasets","datasets"), ("📏 Metrics","metrics")]
            ):
                with col:
                    st.markdown(
                        f"<div style='font-size:0.72rem;color:#8888aa;"
                        f"text-transform:uppercase;letter-spacing:0.08em;"
                        f"margin-bottom:8px'>{label}</div>",
                        unsafe_allow_html=True,
                    )
                    for e in entities.get(key, []):
                        st.markdown(
                            f'<span class="entity-tag">{e}</span>',
                            unsafe_allow_html=True,
                        )

            for label, key in [("🔧 Methods","methods"), ("📋 Tasks","tasks")]:
                items = entities.get(key, [])
                if items:
                    st.markdown(
                        f"<div style='margin-top:12px;font-size:0.72rem;"
                        f"color:#8888aa;text-transform:uppercase;"
                        f"letter-spacing:0.08em;margin-bottom:8px'>{label}</div>",
                        unsafe_allow_html=True,
                    )
                    for e in items:
                        st.markdown(
                            f'<span class="entity-tag">{e}</span>',
                            unsafe_allow_html=True,
                        )

        # ── Research Gaps ─────────────────────────────────────────────────────
        with st.expander("🔍 Research Gaps & Future Directions"):
            if gaps:
                st.markdown(
                    "<div style='font-size:0.72rem;color:#8888aa;"
                    "text-transform:uppercase;letter-spacing:0.08em;"
                    "margin-bottom:12px'>Ranked by novelty score</div>",
                    unsafe_allow_html=True,
                )
                for i, gap in enumerate(gaps, 1):
                    if isinstance(gap, dict):
                        gs = gap.get("novelty_score", 0)
                        color = (
                            "#ef4444" if gs >= 8 else
                            "#f59e0b" if gs >= 6 else
                            "#6366f1"
                        )
                        st.markdown(
                            f'<div class="gap-item">'
                            f'<div style="display:flex;justify-content:space-between;'
                            f'margin-bottom:4px">'
                            f'<span style="font-weight:600;font-size:0.88rem">#{i} {gap["gap"]}</span>'
                            f'<span style="font-family:Space Mono,monospace;font-size:0.8rem;'
                            f'color:{color}">{gs:.1f}</span>'
                            f'</div>'
                            f'<div style="font-size:0.78rem;color:#8888aa;margin-top:4px">'
                            f'📎 {gap.get("supporting_evidence","")}</div>'
                            f'<div style="font-size:0.78rem;color:#6366f1;margin-top:2px">'
                            f'🧪 {gap.get("suggested_experiment","")}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="gap-item">{gap}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No research gaps detected.")

            if dirs:
                st.markdown(
                    "<div style='margin-top:16px;font-size:0.72rem;color:#8888aa;"
                    "text-transform:uppercase;letter-spacing:0.08em;"
                    "margin-bottom:8px'>Future Directions</div>",
                    unsafe_allow_html=True,
                )
                for d in dirs:
                    st.markdown(f"→ {d}")

        # ── Similar Papers ────────────────────────────────────────────────────
        if similar:
            with st.expander("🔗 Similar Papers (arXiv)"):
                for p in similar[:5]:
                    st.markdown(
                        f"<div style='padding:12px 0;border-bottom:1px solid #1e1e2a'>"
                        f"<div style='font-weight:600;font-size:0.9rem;margin-bottom:4px'>"
                        f"{p.get('title','Unknown')}"
                        f"<span style='color:#55556a;font-weight:400;margin-left:8px'>"
                        f"({p.get('year','')})</span></div>"
                        f"<div style='font-size:0.82rem;color:#8888aa;margin-bottom:6px'>"
                        f"{p.get('abstract','')[:200]}…</div>"
                        f"<a href='{p.get('url','#')}' target='_blank' "
                        f"style='font-size:0.8rem;color:#6366f1;text-decoration:none'>"
                        f"View on arXiv →</a>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── Cross-paper: Knowledge Graph ──────────────────────────────────────────
    with st.expander("🕸️ Knowledge Graph"):
        _render_knowledge_graph(
            chat_id=st.session_state.current_chat_id,
            papers=papers,
        )

    # ── Cross-paper: Comparison ───────────────────────────────────────────────
    comp = analysis.get("comparison", {})
    if comp:
        with st.expander("📊 Comparison"):
            if comp.get("positioning"):
                st.markdown(
                    f"<div style='margin-bottom:12px;padding:12px 16px;"
                    f"background:rgba(99,102,241,0.06);"
                    f"border-radius:8px;font-size:0.88rem'>"
                    f"<b style='color:#a5b4fc'>Positioning:</b> "
                    f"{comp['positioning']}</div>",
                    unsafe_allow_html=True,
                )
            if comp.get("evolution_trends"):
                st.markdown(
                    f"<div style='margin-bottom:16px;font-size:0.88rem;"
                    f"color:#8888aa'><b style='color:#d0d0e8'>Evolution:</b> "
                    f"{comp['evolution_trends']}</div>",
                    unsafe_allow_html=True,
                )
            table = comp.get("comparison_table", {})
            if table and table.get("headers") and table.get("rows"):
                import pandas as pd
                df = pd.DataFrame(table["rows"], columns=table["headers"])
                st.dataframe(df, use_container_width=True)
            if comp.get("ranking"):
                st.markdown(
                    "<div style='margin-top:12px;font-size:0.85rem'>"
                    + " <span style='color:#55556a;margin:0 6px'>›</span> ".join(
                        f"<span style='color:#a5b4fc;font-weight:600'>{r}</span>"
                        for r in comp["ranking"]
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

    # ── Cross-paper: Literature Review ────────────────────────────────────────
    lit = analysis.get("literature_review", {})
    if lit and lit.get("review_text"):
        with st.expander("📚 Literature Review"):
            themes = lit.get("themes", [])
            if themes:
                theme_html = " <span style='color:#55556a'>·</span> ".join([
                    f"<span style='color:#a5b4fc'>"
                    f"{t if isinstance(t, str) else t.get('gap', str(t))}"
                    f"</span>"
                    for t in themes
                ])
                st.markdown(
                    f"<div style='margin-bottom:14px;font-size:0.82rem'>"
                    f"<span style='color:#55556a;text-transform:uppercase;"
                    f"font-size:0.7rem;letter-spacing:0.08em'>Themes</span><br>"
                    f"{theme_html}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(lit["review_text"])
            if lit.get("research_gaps_summary"):
                st.markdown(
                    f"<div style='margin-top:14px;padding:12px 16px;"
                    f"background:rgba(99,102,241,0.06);border-radius:8px;"
                    f"font-size:0.85rem'><b style='color:#a5b4fc'>Gaps:</b> "
                    f"{lit['research_gaps_summary']}</div>",
                    unsafe_allow_html=True,
                )
            if lit.get("future_directions"):
                st.markdown(
                    f"<div style='margin-top:10px;font-size:0.85rem;"
                    f"color:#8888aa'><b style='color:#d0d0e8'>Future:</b> "
                    f"{lit['future_directions']}</div>",
                    unsafe_allow_html=True,
                )

    # ── Export ────────────────────────────────────────────────────────────────
    from frontend.export_ui import render_export_button
    render_export_button(analysis)